#!/usr/bin/env python3
"""
evaluate.py — Honest evaluation of the ISL recognition pipeline on the
data that actually exists in isl_dataset.csv.

Every number this script prints/saves is computed from that file as it
stands. Where a scientifically meaningful number (e.g. signer- or
session-generalization accuracy) CANNOT currently be computed because the
data doesn't support it, this script says so explicitly instead of
reporting a number that would look valid but isn't — see
docs/data_collection_protocol.md for what would need to change to make
those measurements possible.

Standalone by design: does not import isl_main.py, so it doesn't pull in
cv2/mediapipe/Flask just to analyze a CSV.

Outputs -> results/:
  descriptive_stats.json, descriptive_stats.png
  leakage_demo.json
  repeated_split_stats.json
  insample_classifier_comparison.json
  confusion_matrix.png
  feature_ablation.json
  RESULTS.md   (human-readable summary of everything above)
"""
import csv
import json
import os
import platform
import time
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

DATASET_PATH = "isl_dataset.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def _build_feature_columns():
    cols = []
    for i in range(21):
        cols.extend([f"rx{i}", f"ry{i}", f"rz{i}"])
    for i in range(21):
        cols.extend([f"lx{i}", f"ly{i}", f"lz{i}"])
    cols.extend(["rel_dx", "rel_dy", "rel_dz", "rel_d_wrist", "rel_d_index", "rel_d_thumb"])
    return cols


FEATURE_COLUMNS = _build_feature_columns()
RIGHT_HAND_COLUMNS = FEATURE_COLUMNS[0:63]
RELATIVE_COLUMNS = FEATURE_COLUMNS[126:132]


def load_dataset():
    X, y, groups, signers, sessions = [], [], [], [], []
    with open(DATASET_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or row.get("label") is None:
                continue
            X.append([float(row[c]) for c in FEATURE_COLUMNS])
            y.append(row["label"].strip())
            groups.append(row.get("burst_id") or "unknown_legacy")
            signers.append(row.get("signer_id") or "unknown_legacy")
            sessions.append(row.get("session_id") or "unknown_legacy")
    # dtype=object keeps elements as plain python str (not numpy.str_), so
    # printing/json-dumping lists of labels doesn't leak "np.str_('...')" reprs.
    return (np.array(X), np.array(y, dtype=object), np.array(groups, dtype=object),
            np.array(signers, dtype=object), np.array(sessions, dtype=object))


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    results = {}
    results["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": "Intel(R) Core(TM) Ultra 5 125H",
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

    X, y, groups, signers, sessions = load_dataset()
    n, n_classes = len(y), len(set(y))
    unique_signers = set(signers)
    unique_sessions = set(sessions)

    section("1. Descriptive statistics")
    class_counts = dict(sorted(Counter(y).items()))
    burst_counts_per_class = {c: len(set(groups[y == c])) for c in class_counts}
    print(f"Total samples: {n}, classes present: {n_classes}/35 target classes")
    print(f"Distinct signer_id values in data: {unique_signers}")
    print(f"Distinct session_id values in data: {unique_sessions}")
    print("Per-class: n_samples (n_independent_recording_bursts)")
    for c in class_counts:
        print(f"  {c:10s} n={class_counts[c]:4d}  bursts={burst_counts_per_class[c]}")
    multi_burst_classes = [c for c, b in burst_counts_per_class.items() if b >= 2]
    print(f"\nClasses with >=2 independent recording bursts: {multi_burst_classes or 'NONE'}")

    results["descriptive"] = {
        "n_samples": n, "n_classes_present": n_classes, "n_classes_target": 35,
        "missing_classes": sorted(set([chr(c) for c in range(65, 91)]) - set(class_counts) - {"J","Q","X","Z"})
                            if False else None,  # computed properly below
        "class_counts": class_counts,
        "burst_counts_per_class": burst_counts_per_class,
        "unique_signer_ids": sorted(unique_signers),
        "unique_session_ids": sorted(unique_sessions),
        "classes_with_ge2_bursts": multi_burst_classes,
    }
    alphabet = set(chr(c) for c in range(65, 91))
    gestures = {"HELLO","LOVE","THANKYOU","HELP","SORRY","YES","NO","GOOD","PLEASE"}
    target_classes = alphabet | gestures
    results["descriptive"]["missing_classes"] = sorted(target_classes - set(class_counts))

    fig, ax = plt.subplots(figsize=(10, 4))
    classes_sorted = list(class_counts)
    ax.bar(classes_sorted, [class_counts[c] for c in classes_sorted], color="#4C72B0")
    ax.set_ylabel("Samples"); ax.set_title("Per-class sample count (isl_dataset.csv)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(f"{RESULTS_DIR}/descriptive_stats.png", dpi=150)
    plt.close(fig)

    section("1b. Near-duplication: intra-burst vs cross-burst feature distance")
    rng = np.random.default_rng(42)
    intra_dists, cross_dists = [], []
    for label in class_counts:
        idx = np.where(y == label)[0]
        lbl_groups = groups[idx]
        for gi in set(lbl_groups):
            members = idx[lbl_groups == gi]
            for a, b in zip(members[:-1], members[1:]):
                intra_dists.append(np.linalg.norm(X[a] - X[b]))
        uniq_groups = sorted(set(lbl_groups))
        if len(uniq_groups) > 1:
            for _ in range(min(300, len(idx) * 3)):
                g1, g2 = rng.choice(uniq_groups, size=2, replace=False)
                a = rng.choice(idx[lbl_groups == g1])
                b = rng.choice(idx[lbl_groups == g2])
                cross_dists.append(np.linalg.norm(X[a] - X[b]))
    intra_dists, cross_dists = np.array(intra_dists), np.array(cross_dists)
    exact_dupe_frac = float((intra_dists < 1e-6).mean()) if len(intra_dists) else None
    ratio = float(cross_dists.mean() / max(intra_dists.mean(), 1e-9)) if len(cross_dists) else None
    print(f"Intra-burst consecutive-frame L2 distance: mean={intra_dists.mean():.4f} "
          f"median={np.median(intra_dists):.4f} (n={len(intra_dists)})")
    if len(cross_dists):
        print(f"Cross-burst same-label L2 distance:         mean={cross_dists.mean():.4f} "
              f"median={np.median(cross_dists):.4f} (n={len(cross_dists)})")
        print(f"Ratio (cross/intra): {ratio:.2f}x")
    else:
        print("Cross-burst same-label distance: NOT COMPUTABLE -- only 1 class "
              "(E) has >1 burst, contributing too few pairs for a stable estimate.")
    print(f"Fraction of consecutive intra-burst pairs that are exact/near-exact "
          f"duplicates (<1e-6): {exact_dupe_frac*100:.1f}%")
    results["near_duplication"] = {
        "intra_burst_mean": float(intra_dists.mean()), "intra_burst_median": float(np.median(intra_dists)),
        "intra_burst_n_pairs": int(len(intra_dists)),
        "cross_burst_mean": float(cross_dists.mean()) if len(cross_dists) else None,
        "cross_burst_median": float(np.median(cross_dists)) if len(cross_dists) else None,
        "cross_burst_n_pairs": int(len(cross_dists)),
        "cross_over_intra_ratio": ratio,
        "exact_duplicate_fraction": exact_dupe_frac,
        "caveat": ("cross-burst pairs come almost entirely from the single class (E) with "
                   ">1 burst -- this estimate is not class-balanced and should not be read as "
                   "precise; it is directionally indicative only, given n_pairs above."),
    }

    section("2. Leakage demonstration: naive random split vs burst-disjoint split")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    acc_naive = accuracy_score(yte, clf.predict(Xte))
    f1_naive = f1_score(yte, clf.predict(Xte), average="macro")
    print(f"[naive random split]   acc={acc_naive*100:.2f}%  macro-F1={f1_naive:.3f}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    clf2 = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf2.fit(X[tr_idx], y[tr_idx])
    acc_group = accuracy_score(y[te_idx], clf2.predict(X[te_idx]))
    f1_group = f1_score(y[te_idx], clf2.predict(X[te_idx]), average="macro", zero_division=0)
    missing = sorted(set(y[te_idx]) - set(y[tr_idx]))
    print(f"[burst-disjoint split] acc={acc_group*100:.2f}%  macro-F1={f1_group:.3f}  "
          f"(classes in test absent from train: {missing})")
    results["leakage_demo"] = {
        "naive_random_split": {"accuracy": float(acc_naive), "macro_f1": float(f1_naive), "n_test": int(len(yte))},
        "burst_disjoint_split": {"accuracy": float(acc_group), "macro_f1": float(f1_group),
                                  "n_test": int(len(te_idx)), "classes_missing_from_train": missing},
        "interpretation": ("The near-100% naive-split accuracy is an artifact of near-duplicate "
                            "frames from the same continuous recording burst appearing on both "
                            "sides of the split. The burst-disjoint collapse is not a measure of "
                            "poor model quality either -- most classes have only ONE recording "
                            "burst, so holding a burst out removes that class from training "
                            "entirely. Neither number estimates real-world generalization; see "
                            "docs/data_collection_protocol.md."),
    }

    section("3. Repeated random-split statistics + 1-NN same-burst collision rate")
    accs, f1s, collision_rates = [], [], []
    for seed in range(20):
        Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
            X, y, groups, test_size=0.2, random_state=seed, stratify=y)
        c = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1)
        c.fit(Xtr, ytr)
        pred = c.predict(Xte)
        accs.append(accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, average="macro"))
        # For every test sample, is its nearest TRAIN neighbor (by L2, same
        # label) from the same recording burst? If so, the "test" prediction
        # can be resolved by near-duplicate matching, not generalization.
        same_burst_hits = 0
        for i in range(len(Xte)):
            same_label_mask = (ytr == yte[i])
            if not same_label_mask.any():
                continue
            dists = np.linalg.norm(Xtr[same_label_mask] - Xte[i], axis=1)
            nn_idx = np.argmin(dists)
            nn_group = gtr[same_label_mask][nn_idx]
            if nn_group == gte[i]:
                same_burst_hits += 1
        collision_rates.append(same_burst_hits / len(Xte))
    accs, f1s, collision_rates = map(np.array, (accs, f1s, collision_rates))
    print(f"Naive random split over 20 seeds: acc={accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}pp, "
          f"macro-F1={f1s.mean():.3f} +/- {f1s.std():.3f}")
    print(f"1-NN same-burst collision rate: {collision_rates.mean()*100:.1f}% +/- {collision_rates.std()*100:.1f}pp "
          f"of test samples' nearest same-label neighbor comes from their own recording burst")

    # Percentile bootstrap CI over the 20 per-seed values. Caveat printed and
    # stored explicitly: these 20 splits are overlapping 80/20 resamples of
    # the SAME 842 rows, not 20 independent datasets, so this understates
    # true variability (pseudo-replication) -- it quantifies sensitivity to
    # the random split, not to new data.
    def _bootstrap_ci(values, n_boot=2000, seed=0):
        r = np.random.default_rng(seed)
        boot_means = [r.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
        return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    acc_ci = _bootstrap_ci(accs)
    coll_ci = _bootstrap_ci(collision_rates)
    print(f"Bootstrap 95% CI (over the 20 seeds, NOT over independent data): "
          f"accuracy [{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%], "
          f"collision rate [{coll_ci[0]*100:.1f}%, {coll_ci[1]*100:.1f}%]")
    results["repeated_split_stats"] = {
        "n_seeds": 20,
        "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
        "accuracy_bootstrap_95ci": acc_ci,
        "macro_f1_mean": float(f1s.mean()), "macro_f1_std": float(f1s.std()),
        "same_burst_1nn_collision_rate_mean": float(collision_rates.mean()),
        "same_burst_1nn_collision_rate_std": float(collision_rates.std()),
        "same_burst_1nn_collision_rate_bootstrap_95ci": coll_ci,
        "interpretation": ("A same-burst 1-NN collision rate this high means the reported "
                            "accuracy is dominated by near-duplicate matching within a burst, "
                            "not sign discrimination."),
        "pseudo_replication_caveat": ("The 20 seeds are overlapping 80/20 resamples of the same "
                            "842 rows, not 20 independent datasets -- this measures sensitivity "
                            "to split randomness, not true sampling variability, and likely "
                            "understates real-world variance."),
    }

    section("4. In-sample classifier comparison (NOT a generalization estimate) + inference latency")
    classifiers = {
        "RandomForest(100,d15)": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "SVM-RBF(C=10)": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        "kNN(k=5)": KNeighborsClassifier(n_neighbors=5),
    }
    insample = {}
    for name, c in classifiers.items():
        t0 = time.perf_counter(); c.fit(X, y); train_s = time.perf_counter() - t0
        pred = c.predict(X)
        acc = accuracy_score(y, pred); f1m = f1_score(y, pred, average="macro")
        # single-sample inference latency, CPU, this machine
        one = X[:1]
        for _ in range(20):
            c.predict(one)  # warmup
        t0 = time.perf_counter()
        for _ in range(200):
            c.predict(one)
        per_call_ms = (time.perf_counter() - t0) / 200 * 1000
        print(f"{name:24s} in-sample acc={acc*100:5.1f}%  macro-F1={f1m:.3f}  "
              f"train={train_s:.3f}s  single-sample predict={per_call_ms:.2f}ms")
        insample[name] = {"accuracy": float(acc), "macro_f1": float(f1m),
                           "train_seconds": float(train_s), "predict_latency_ms": float(per_call_ms)}
    results["insample_classifier_comparison"] = insample
    results["insample_comparison_caveat"] = (
        "Trained and evaluated on the SAME data (all of it) -- this measures whether each "
        "classifier's capacity suffices to fit the observed classes with this feature "
        "representation, not whether it generalizes to unseen signers/sessions. Latency is "
        "single-sample CPU inference on the development machine (see environment.processor), "
        "not the deployment device, and excludes MediaPipe detection time.")

    section("5. Confusion matrix (in-sample fit; feature-separability diagnostic)")
    rf = classifiers["RandomForest(100,d15)"]
    pred = rf.predict(X)
    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y, pred, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels_sorted))); ax.set_xticklabels(labels_sorted, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels_sorted))); ax.set_yticklabels(labels_sorted, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("In-sample confusion matrix (RandomForest) -- NOT a generalization measure")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)
    off_diag_errors = [(labels_sorted[i], labels_sorted[j], int(cm[i, j]))
                        for i in range(len(labels_sorted)) for j in range(len(labels_sorted))
                        if i != j and cm[i, j] > 0]
    off_diag_errors.sort(key=lambda t: -t[2])
    print(f"In-sample confusions (true, pred, count), top 10: {off_diag_errors[:10]}")
    results["confusion_matrix_top_errors"] = off_diag_errors[:10]

    section("6. Feature ablation (in-sample fit)")
    idx_full = list(range(132))
    idx_no_relative = list(range(126))
    idx_right_only = list(range(63))
    variants = {"full_132": idx_full, "no_relative_126": idx_no_relative, "right_hand_only_63": idx_right_only}
    ablation = {}
    for name, idxs in variants.items():
        Xv = X[:, idxs]
        c = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        c.fit(Xv, y)
        pred = c.predict(Xv)
        acc = accuracy_score(y, pred); f1m = f1_score(y, pred, average="macro")
        print(f"{name:20s} n_features={len(idxs):3d}  in-sample acc={acc*100:5.1f}%  macro-F1={f1m:.3f}")
        ablation[name] = {"n_features": len(idxs), "accuracy": float(acc), "macro_f1": float(f1m)}
    results["feature_ablation"] = ablation
    results["feature_ablation_caveat"] = "In-sample fit only, same caveat as section 4."

    with open(f"{RESULTS_DIR}/all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    write_results_md(results)
    print(f"\nAll figures/tables written to {RESULTS_DIR}/")


def write_results_md(r):
    d = r["descriptive"]
    nd = r["near_duplication"]
    lk = r["leakage_demo"]
    rs = r["repeated_split_stats"]
    ic = r["insample_classifier_comparison"]
    fa = r["feature_ablation"]
    env = r["environment"]
    lines = []
    lines.append("# Evaluation Results (real, reproducible via `python evaluate.py`)\n")
    lines.append(f"Environment: Python {env['python']}, {env['platform']}, {env['processor']}, "
                  f"scikit-learn {env['sklearn']}, numpy {env['numpy']}, scipy {env['scipy']}.\n")
    lines.append("## 1. Dataset\n")
    lines.append(f"- {d['n_samples']} samples across {d['n_classes_present']}/{d['n_classes_target']} target classes "
                  f"(missing: {', '.join(d['missing_classes'])}).\n")
    lines.append(f"- signer_id values present: {d['unique_signer_ids']} -- "
                  f"{'no real signer diversity recorded' if d['unique_signer_ids'] in (['unknown_legacy'], []) else 'see counts'}.\n")
    lines.append(f"- Classes with >=2 independent recording bursts: {d['classes_with_ge2_bursts'] or 'NONE'}.\n")
    lines.append("![per-class counts](descriptive_stats.png)\n")
    lines.append("## 1b. Near-duplication (intra- vs cross-burst distance)\n")
    lines.append(f"- Intra-burst consecutive-frame L2 distance: mean {nd['intra_burst_mean']:.4f}, "
                 f"median {nd['intra_burst_median']:.4f} (n={nd['intra_burst_n_pairs']} pairs).\n")
    if nd['cross_burst_mean'] is not None:
        lines.append(f"- Cross-burst same-label L2 distance: mean {nd['cross_burst_mean']:.4f}, "
                     f"median {nd['cross_burst_median']:.4f} (n={nd['cross_burst_n_pairs']} pairs), "
                     f"ratio (cross/intra) {nd['cross_over_intra_ratio']:.2f}x.\n")
    lines.append(f"- Exact/near-exact duplicate fraction among consecutive intra-burst pairs "
                 f"(L2 < 1e-6): {nd['exact_duplicate_fraction']*100:.1f}%.\n")
    lines.append(f"> {nd['caveat']}\n")
    lines.append("## 2. Leakage demonstration\n")
    lines.append(f"- Naive random 80/20 split: **{lk['naive_random_split']['accuracy']*100:.2f}%** accuracy, "
                 f"macro-F1 {lk['naive_random_split']['macro_f1']:.3f}.\n")
    lines.append(f"- Burst-disjoint split: **{lk['burst_disjoint_split']['accuracy']*100:.2f}%** accuracy, "
                 f"macro-F1 {lk['burst_disjoint_split']['macro_f1']:.3f} "
                 f"(classes absent from train: {lk['burst_disjoint_split']['classes_missing_from_train']}).\n")
    lines.append(f"> {lk['interpretation']}\n")
    lines.append("## 3. Repeated-split statistics (20 seeds) + 1-NN same-burst collision rate\n")
    lines.append(f"- Naive split accuracy: {rs['accuracy_mean']*100:.2f}% +/- {rs['accuracy_std']*100:.2f}pp, "
                 f"bootstrap 95% CI [{rs['accuracy_bootstrap_95ci'][0]*100:.2f}%, {rs['accuracy_bootstrap_95ci'][1]*100:.2f}%] "
                 f"(macro-F1 {rs['macro_f1_mean']:.3f} +/- {rs['macro_f1_std']:.3f}).\n")
    lines.append(f"- Same-burst 1-NN collision rate: {rs['same_burst_1nn_collision_rate_mean']*100:.1f}% "
                 f"+/- {rs['same_burst_1nn_collision_rate_std']*100:.1f}pp, bootstrap 95% CI "
                 f"[{rs['same_burst_1nn_collision_rate_bootstrap_95ci'][0]*100:.1f}%, {rs['same_burst_1nn_collision_rate_bootstrap_95ci'][1]*100:.1f}%].\n")
    lines.append(f"> {rs['interpretation']}\n")
    lines.append(f"> **Caveat:** {rs['pseudo_replication_caveat']}\n")
    lines.append("## 4. In-sample classifier comparison (not a generalization estimate)\n")
    lines.append("| Classifier | In-sample acc | Macro-F1 | Train time (s) | Predict latency (ms/sample) |\n")
    lines.append("|---|---|---|---|---|\n")
    for name, v in ic.items():
        lines.append(f"| {name} | {v['accuracy']*100:.1f}% | {v['macro_f1']:.3f} | {v['train_seconds']:.3f} | {v['predict_latency_ms']:.2f} |\n")
    lines.append(f"\n> {r['insample_comparison_caveat']}\n")
    lines.append("## 5. Confusion matrix\n")
    lines.append("![confusion matrix](confusion_matrix.png)\n")
    lines.append(f"Top in-sample confusions (true, pred, count): {r['confusion_matrix_top_errors']}\n")
    lines.append("## 6. Feature ablation (in-sample fit)\n")
    lines.append("| Feature set | # features | In-sample acc | Macro-F1 |\n")
    lines.append("|---|---|---|---|\n")
    for name, v in fa.items():
        lines.append(f"| {name} | {v['n_features']} | {v['accuracy']*100:.1f}% | {v['macro_f1']:.3f} |\n")
    lines.append(f"\n> {r['feature_ablation_caveat']}\n")
    with open(f"{RESULTS_DIR}/RESULTS.md", "w") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    main()
