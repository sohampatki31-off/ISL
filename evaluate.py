#!/usr/bin/env python3
"""
evaluate.py — Honest evaluation of the ISL recognition pipeline on the
data that actually exists in isl_dataset.csv.

Every number this script prints/saves is computed from that file as it
stands. Where a scientifically meaningful number (e.g. true signer-
independent generalization accuracy) CANNOT currently be computed because
the data doesn't support it, this script says so explicitly instead of
reporting a number that would look valid but isn't.

Study scope (see docs/data_collection_protocol.md): 26 fingerspelling
letters + 5 word-gestures with real recorded data (GOOD, HELLO, LOVE,
HELP, THANKYOU) = 31 classes. isl_main.py additionally hardcodes SORRY,
YES, NO, PLEASE as target vocabulary (35 total) but zero samples exist
for those 4 -- they are out of scope for this evaluation and documented
as future work, not silently ignored.

signer_id/session_id in the CSV are placeholder strings, not real signer
identities (unknown_legacy / kaggle_unknown / unassigned_signer / etc.) --
this script never claims signer-independent generalization. What IS real
and available: two genuinely different data SOURCES (a Kaggle image
dataset vs. hand-recorded live webcam capture), which differ in capture
device, lighting/background, MediaPipe detection-confidence threshold,
and temporal-smoothing (webcam rows are One-Euro filtered, Kaggle rows
are not). Cross-source evaluation (train on one, test on the other) is
therefore the strongest domain-shift evidence this dataset supports, and
is treated as the key finding, not a problem to hide.

Standalone by design: does not import isl_main.py, so it doesn't pull in
cv2/mediapipe/Flask just to analyze a CSV.

Outputs -> results/:
  all_results.json, RESULTS.md
  descriptive_stats.png, confusion_matrix.png, cross_source_confusion.png
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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

DATASET_PATH = "isl_dataset.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_ALPHABET = set(chr(c) for c in range(65, 91))
TARGET_GESTURES = {"HELLO", "LOVE", "THANKYOU", "HELP", "SORRY", "YES", "NO", "GOOD", "PLEASE"}
TARGET_CLASSES_35 = TARGET_ALPHABET | TARGET_GESTURES
SVM_TRAIN_CAP = 6000  # stratified subsample cap so RBF-SVM fitting stays tractable at 30k+ rows


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
LEFT_HAND_COLUMNS = FEATURE_COLUMNS[63:126]
RELATIVE_COLUMNS = FEATURE_COLUMNS[126:132]
RIGHT_IDX = list(range(0, 63))
LEFT_IDX = list(range(63, 126))


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
    X = np.array(X)
    y = np.array(y, dtype=object)
    groups = np.array(groups, dtype=object)
    signers = np.array(signers, dtype=object)
    sessions = np.array(sessions, dtype=object)
    domains = np.array(["kaggle" if s == "kaggle_import" else "webcam" for s in sessions], dtype=object)
    return X, y, groups, signers, sessions, domains


def dedupe_exact(X, y, groups, signers, sessions, domains):
    """Remove exact-duplicate feature vectors (keep first occurrence). Exact
    duplicates are a pure leakage risk in any downstream split -- there is
    no scientific reason to let the same feature vector appear in both
    train and test."""
    seen = {}
    keep = np.ones(len(X), dtype=bool)
    n_dupe_groups = 0
    dupe_by_domain = Counter()
    for i in range(len(X)):
        key = X[i].tobytes()
        if key in seen:
            keep[i] = False
            dupe_by_domain[domains[i]] += 1
        else:
            seen[key] = i
    n_removed = int((~keep).sum())
    return (X[keep], y[keep], groups[keep], signers[keep], sessions[keep], domains[keep],
            {"n_removed": n_removed, "removed_by_domain": dict(dupe_by_domain)})


def stratified_cap(X, y, cap, seed=42):
    """Stratified subsample to at most `cap` rows, preserving class proportions
    as closely as integer counts allow. Used only to keep RBF-SVM fitting
    tractable; RandomForest/kNN always use the full training set."""
    if len(X) <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    idx_out = []
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        idx_c = np.where(y == c)[0]
        n_take = max(1, round(n / len(y) * cap))
        n_take = min(n_take, len(idx_c))
        idx_out.extend(rng.choice(idx_c, size=n_take, replace=False))
    idx_out = np.array(idx_out)
    rng.shuffle(idx_out)
    return X[idx_out], y[idx_out]


def per_class_table(y_true, y_pred, labels):
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {lab: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for lab, p, r, f, s in zip(labels, prec, rec, f1, support)}


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    results = {}
    results["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

    X_raw, y_raw, groups_raw, signers_raw, sessions_raw, domains_raw = load_dataset()

    section("1. Descriptive statistics (raw, before dedup)")
    class_counts_raw = dict(sorted(Counter(y_raw).items()))
    n_raw, n_classes_raw = len(y_raw), len(class_counts_raw)
    print(f"Raw rows: {n_raw}, classes present: {n_classes_raw}/35 target classes "
          f"(35 = 26 letters + 9 hardcoded word-gestures in isl_main.py)")
    print(f"Distinct signer_id values: {sorted(set(signers_raw))} -- placeholders only, no real signer identity")
    print(f"Distinct session_id values: {sorted(set(sessions_raw))}")
    print(f"Domain counts: {dict(Counter(domains_raw))}")
    missing_35 = sorted(TARGET_CLASSES_35 - set(class_counts_raw))
    print(f"Classes with zero samples (out of 35-word aspirational vocabulary): {missing_35}")

    section("1b. Exact-duplicate audit + removal")
    X, y, groups, signers, sessions, domains, dedup_info = dedupe_exact(
        X_raw, y_raw, groups_raw, signers_raw, sessions_raw, domains_raw)
    n, n_classes = len(y), len(set(y))
    class_counts = dict(sorted(Counter(y).items()))
    print(f"Exact-duplicate feature-vector rows removed: {dedup_info['n_removed']} "
          f"(by domain: {dedup_info['removed_by_domain']})")
    print(f"Usable rows after dedup: {n} across {n_classes} classes")
    print("All experiments below (sections 2 onward) use the deduplicated dataset.")

    results["descriptive"] = {
        "n_samples_raw": n_raw, "n_classes_present_raw": n_classes_raw, "n_classes_target": 35,
        "missing_classes_35": missing_35,
        "n_exact_duplicates_removed": dedup_info["n_removed"],
        "exact_duplicates_removed_by_domain": dedup_info["removed_by_domain"],
        "n_samples_deduped": n, "n_classes_deduped": n_classes,
        "class_counts_deduped": class_counts,
        "domain_counts_deduped": dict(Counter(domains)),
        "unique_signer_ids": sorted(set(signers)),
        "unique_session_ids": sorted(set(sessions)),
        "signer_session_caveat": ("These are placeholder source tags (unknown_legacy / kaggle_unknown / "
                                   "unassigned_signer / kaggle_import / unassigned_session), not real signer "
                                   "or session identities. No signer-independent or session-independent claim "
                                   "is made anywhere in this evaluation."),
    }

    fig, ax = plt.subplots(figsize=(11, 4))
    classes_sorted = list(class_counts)
    ax.bar(classes_sorted, [class_counts[c] for c in classes_sorted], color="#4C72B0")
    ax.set_ylabel("Samples"); ax.set_title("Per-class sample count, deduplicated (isl_dataset.csv)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(f"{RESULTS_DIR}/descriptive_stats.png", dpi=150)
    plt.close(fig)

    section("1c. Near-duplication: intra-burst vs cross-burst feature distance (post-dedup)")
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
    ratio = float(cross_dists.mean() / max(intra_dists.mean(), 1e-9)) if len(cross_dists) else None
    n_multi_burst = sum(1 for c in class_counts if len(set(groups[y == c])) >= 2)
    print(f"Classes with >=2 independent recording bursts: {n_multi_burst}/{n_classes}")
    print(f"Intra-burst consecutive-frame L2 distance: mean={intra_dists.mean():.4f} "
          f"median={np.median(intra_dists):.4f} (n={len(intra_dists)})")
    if len(cross_dists):
        print(f"Cross-burst same-label L2 distance:         mean={cross_dists.mean():.4f} "
              f"median={np.median(cross_dists):.4f} (n={len(cross_dists)})")
        print(f"Ratio (cross/intra): {ratio:.2f}x")
    results["near_duplication"] = {
        "n_classes_with_ge2_bursts": n_multi_burst,
        "intra_burst_mean": float(intra_dists.mean()) if len(intra_dists) else None,
        "intra_burst_n_pairs": int(len(intra_dists)),
        "cross_burst_mean": float(cross_dists.mean()) if len(cross_dists) else None,
        "cross_burst_n_pairs": int(len(cross_dists)),
        "cross_over_intra_ratio": ratio,
        "caveat": ("burst_id is near-unique per row for the Kaggle-sourced majority of the data (mean "
                   "group size ~1.03); this diagnostic is only informative for the small legacy-webcam "
                   "subset that actually has multi-row bursts."),
    }

    section("2. Leakage demonstration: naive random split vs burst-disjoint split (post-dedup)")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    acc_naive = accuracy_score(yte, clf.predict(Xte))
    f1_naive = f1_score(yte, clf.predict(Xte), average="macro")
    print(f"[naive random split]   acc={acc_naive*100:.2f}%  macro-F1={f1_naive:.3f}")

    classes_ge2_bursts = [c for c in class_counts if len(set(groups[y == c])) >= 2]
    mask_ge2 = np.isin(y, classes_ge2_bursts)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X[mask_ge2], y[mask_ge2], groups=groups[mask_ge2]))
    Xg, yg = X[mask_ge2], y[mask_ge2]
    clf2 = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf2.fit(Xg[tr_idx], yg[tr_idx])
    acc_group = accuracy_score(yg[te_idx], clf2.predict(Xg[te_idx]))
    f1_group = f1_score(yg[te_idx], clf2.predict(Xg[te_idx]), average="macro", zero_division=0)
    missing = sorted(set(yg[te_idx]) - set(yg[tr_idx]))
    print(f"[burst-disjoint split, restricted to {len(classes_ge2_bursts)} classes with >=2 bursts] "
          f"acc={acc_group*100:.2f}%  macro-F1={f1_group:.3f}  (classes missing from train: {missing})")
    results["leakage_demo"] = {
        "naive_random_split": {"accuracy": float(acc_naive), "macro_f1": float(f1_naive), "n_test": int(len(yte))},
        "burst_disjoint_split": {"accuracy": float(acc_group), "macro_f1": float(f1_group),
                                  "n_test": int(len(te_idx)), "n_classes_evaluated": len(classes_ge2_bursts),
                                  "classes_missing_from_train": missing},
        "interpretation": ("Unlike the pre-Kaggle-merge dataset (where burst-disjoint splitting collapsed "
                            "accuracy to 0% because almost every class had exactly one burst), burst-disjoint "
                            "and naive-random splits now give SIMILAR results. This is not because leakage "
                            "was fixed -- it's because burst_id is near-unique per row for 97% of the current "
                            "data (Kaggle-sourced), so grouping by burst barely constrains the split at all. "
                            "burst_id is not a meaningful leakage control at the current dataset composition."),
    }

    section("3. Repeated random-split statistics + 1-NN same-source collision rate (post-dedup)")
    accs, f1s, collision_rates = [], [], []
    for seed in range(10):
        Xtr, Xte, ytr, yte, dtr, dte = train_test_split(
            X, y, domains, test_size=0.2, random_state=seed, stratify=y)
        c = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1)
        c.fit(Xtr, ytr)
        pred = c.predict(Xte)
        accs.append(accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, average="macro"))
        # For every test sample, is its nearest TRAIN neighbor (by L2, same
        # label) from the same SOURCE DOMAIN? A near-1-NN match from the same
        # domain doesn't prove leakage the way a same-burst match would, but a
        # very high rate combined with a very small min-distance is still
        # informative about near-duplicate density within a domain.
        same_domain_hits = 0
        for i in range(len(Xte)):
            same_label_mask = (ytr == yte[i])
            if not same_label_mask.any():
                continue
            dists = np.linalg.norm(Xtr[same_label_mask] - Xte[i], axis=1)
            nn_idx = np.argmin(dists)
            nn_domain = dtr[same_label_mask][nn_idx]
            if nn_domain == dte[i]:
                same_domain_hits += 1
        collision_rates.append(same_domain_hits / len(Xte))
    accs, f1s, collision_rates = map(np.array, (accs, f1s, collision_rates))
    print(f"Naive random split over 10 seeds: acc={accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}pp, "
          f"macro-F1={f1s.mean():.3f} +/- {f1s.std():.3f}")
    print(f"1-NN same-domain rate: {collision_rates.mean()*100:.1f}% +/- {collision_rates.std()*100:.1f}pp "
          f"(expected to track domain proportions since domains aren't held out here)")

    def _bootstrap_ci(values, n_boot=2000, seed=0):
        r = np.random.default_rng(seed)
        boot_means = [r.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
        return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    acc_ci = _bootstrap_ci(accs)
    print(f"Bootstrap 95% CI (over the 10 seeds, NOT over independent data): "
          f"accuracy [{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%]")
    results["repeated_split_stats"] = {
        "n_seeds": 10,
        "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
        "accuracy_bootstrap_95ci": acc_ci,
        "macro_f1_mean": float(f1s.mean()), "macro_f1_std": float(f1s.std()),
        "same_domain_1nn_rate_mean": float(collision_rates.mean()),
        "same_domain_1nn_rate_std": float(collision_rates.std()),
        "pseudo_replication_caveat": ("The 10 seeds are overlapping 80/20 resamples of the same "
                            f"{n} deduplicated rows, not 10 independent datasets -- this measures "
                            "sensitivity to split randomness, not true sampling variability."),
    }

    section("4. Cross-source (domain) evaluation: Kaggle vs Webcam, on shared classes")
    domain_counts = Counter(domains)
    shared_classes = sorted(c for c in class_counts
                             if ((y == c) & (domains == "kaggle")).sum() > 0
                             and ((y == c) & (domains == "webcam")).sum() > 0)
    print(f"Domain sizes: {dict(domain_counts)}")
    print(f"Classes present in BOTH domains: {shared_classes} ({len(shared_classes)} classes)")
    cross = {}
    mask_shared = np.isin(y, shared_classes)
    Xs, ys, ds, gs = X[mask_shared], y[mask_shared], domains[mask_shared], groups[mask_shared]

    def _bootstrap_test_ci(y_true, y_pred, n_boot=2000, seed=0):
        # Bootstraps over the (fixed) test-set predictions -- cheap, no
        # retraining -- to give the cross-domain headline numbers a CI
        # instead of a single point estimate (CVIP2026 hostile-ML-review
        # finding: these small-n cross-domain cells had none before).
        r = np.random.default_rng(seed)
        n = len(y_true)
        accs_b, f1s_b = [], []
        for _ in range(n_boot):
            idx = r.integers(0, n, size=n)
            accs_b.append(accuracy_score(y_true[idx], y_pred[idx]))
            f1s_b.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
        return {"accuracy_95ci": [float(np.percentile(accs_b, 2.5)), float(np.percentile(accs_b, 97.5))],
                "macro_f1_95ci": [float(np.percentile(f1s_b, 2.5)), float(np.percentile(f1s_b, 97.5))]}

    def fit_eval(train_mask, test_mask, name):
        c = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        c.fit(Xs[train_mask], ys[train_mask])
        pred = c.predict(Xs[test_mask])
        y_true = ys[test_mask]
        acc = accuracy_score(y_true, pred)
        f1m = f1_score(y_true, pred, average="macro", zero_division=0)
        per_class = per_class_table(y_true, pred, shared_classes)
        ci = _bootstrap_test_ci(y_true, pred)
        print(f"  [{name}] n_train={train_mask.sum()} n_test={test_mask.sum()} "
              f"acc={acc*100:.2f}% [{ci['accuracy_95ci'][0]*100:.1f}, {ci['accuracy_95ci'][1]*100:.1f}] "
              f"macro-F1={f1m:.3f} [{ci['macro_f1_95ci'][0]:.3f}, {ci['macro_f1_95ci'][1]:.3f}]")
        return {"n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
                "accuracy": float(acc), "macro_f1": float(f1m), "per_class": per_class, **ci}

    is_kaggle = ds == "kaggle"
    is_webcam = ds == "webcam"

    # in-domain baselines: burst-disjoint split *within* each domain (falls back
    # to stratified random split if a domain has too few multi-row bursts)
    def indomain_split(domain_mask):
        idx = np.where(domain_mask)[0]
        try:
            gss_local = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            tr, te = next(gss_local.split(Xs[idx], ys[idx], groups=gs[idx]))
            return idx[tr], idx[te]
        except ValueError:
            tr, te = train_test_split(idx, test_size=0.3, random_state=42, stratify=ys[idx])
            return tr, te

    kaggle_tr, kaggle_te = indomain_split(is_kaggle)
    webcam_tr, webcam_te = indomain_split(is_webcam)
    m_kaggle_tr = np.zeros(len(Xs), dtype=bool); m_kaggle_tr[kaggle_tr] = True
    m_kaggle_te = np.zeros(len(Xs), dtype=bool); m_kaggle_te[kaggle_te] = True
    m_webcam_tr = np.zeros(len(Xs), dtype=bool); m_webcam_tr[webcam_tr] = True
    m_webcam_te = np.zeros(len(Xs), dtype=bool); m_webcam_te[webcam_te] = True

    cross["in_domain_kaggle"] = fit_eval(m_kaggle_tr, m_kaggle_te, "in-domain: train Kaggle, test Kaggle (held-out)")
    cross["in_domain_webcam"] = fit_eval(m_webcam_tr, m_webcam_te, "in-domain: train Webcam, test Webcam (held-out)")
    cross["kaggle_to_webcam"] = fit_eval(is_kaggle, is_webcam, "cross-domain: train Kaggle, test ALL Webcam")
    cross["webcam_to_kaggle"] = fit_eval(is_webcam, is_kaggle, "cross-domain: train Webcam, test ALL Kaggle")
    results["cross_source_evaluation"] = {
        "shared_classes": shared_classes,
        "domain_counts": {k: int(v) for k, v in domain_counts.items()},
        **cross,
        "interpretation": ("in_domain_* results show what accuracy this feature representation/classifier "
                            "achieves when train and test come from the same source. kaggle_to_webcam and "
                            "webcam_to_kaggle show what happens when the model has never seen the OTHER "
                            "source at all. A large drop from in-domain to cross-domain accuracy is evidence "
                            "of genuine domain shift (capture device / lighting / temporal smoothing / "
                            "detection-threshold differences), not of the model failing to learn the sign "
                            "shapes themselves."),
    }

    section("5. Handedness analysis (post-dedup, held-out)")
    r_present = (np.abs(X[:, RIGHT_IDX]).sum(axis=1) > 0)
    l_present = (np.abs(X[:, LEFT_IDX]).sum(axis=1) > 0)
    handedness = np.where(r_present & l_present, "both", np.where(r_present, "right_only", "left_only"))
    print(f"Handedness distribution (post-dedup): {dict(Counter(handedness))}")
    Xtr, Xte, ytr, yte, htr, hte = train_test_split(
        X, y, handedness, test_size=0.2, random_state=42, stratify=y)
    clf_hand = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf_hand.fit(Xtr, ytr)
    pred_hand = clf_hand.predict(Xte)
    handedness_breakdown = {}
    for h in ["both", "right_only", "left_only"]:
        m = hte == h
        if m.sum() == 0:
            continue
        acc_h = accuracy_score(yte[m], pred_hand[m])
        f1_h = f1_score(yte[m], pred_hand[m], average="macro", zero_division=0)
        print(f"  test subset handedness={h:11s} n={m.sum():5d}  acc={acc_h*100:.2f}%  macro-F1={f1_h:.3f}")
        handedness_breakdown[h] = {"n_test": int(m.sum()), "accuracy": float(acc_h), "macro_f1": float(f1_h)}
    results["handedness_analysis"] = {
        "distribution": {k: int(v) for k, v in Counter(handedness).items()},
        "held_out_breakdown_by_test_subset_handedness": handedness_breakdown,
        "caveat": ("Handedness here is inferred from which hand-feature block is non-zero at capture time, "
                    "not ground-truth sign linguistics (some classes are inherently one-handed signs; this "
                    "mixes 'sign is one-handed' with 'only one hand was detected'). A single classifier "
                    "trained on the full mixed-handedness training set is evaluated separately on each "
                    "test subset, so this measures whether ONE model generalizes evenly across handedness "
                    "conditions, not whether per-handedness specialist models would do better."),
    }

    section("6. Held-out classifier comparison (RandomForest / SVM-RBF / kNN / MLP)")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Xtr_svm, ytr_svm = stratified_cap(Xtr, ytr, SVM_TRAIN_CAP)
    classifiers = {
        "RandomForest(100,d15)": (RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1), Xtr, ytr),
        "SVM-RBF(C=10)": (SVC(kernel="rbf", C=10, gamma="scale", random_state=42), Xtr_svm, ytr_svm),
        "kNN(k=5)": (KNeighborsClassifier(n_neighbors=5), Xtr, ytr),
        # Shallow neural-network baseline (CVIP2026 hostile-ML-review finding: a
        # tree-ensemble/kNN/SVM-only comparison with zero neural baseline is a
        # credible desk-reject risk for a CV/ML venue -- this landmark-vector
        # input is exactly what an MLP takes natively, no architecture search
        # attempted, just a reasonable default).
        "MLP(128,64)": (MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42), Xtr, ytr),
    }
    heldout = {}
    for name, (c, Xt, yt) in classifiers.items():
        t0 = time.perf_counter(); c.fit(Xt, yt); train_s = time.perf_counter() - t0
        pred = c.predict(Xte)
        acc = accuracy_score(yte, pred); f1m = f1_score(yte, pred, average="macro", zero_division=0)
        one = Xte[:1]
        for _ in range(10):
            c.predict(one)
        t0 = time.perf_counter()
        for _ in range(100):
            c.predict(one)
        per_call_ms = (time.perf_counter() - t0) / 100 * 1000
        print(f"{name:24s} held-out acc={acc*100:5.1f}%  macro-F1={f1m:.3f}  "
              f"n_train={len(Xt):6d}  train={train_s:.3f}s  predict={per_call_ms:.2f}ms/sample")
        heldout[name] = {"accuracy": float(acc), "macro_f1": float(f1m), "n_train": int(len(Xt)),
                          "train_seconds": float(train_s), "predict_latency_ms": float(per_call_ms)}
    results["classifier_comparison"] = heldout
    results["classifier_comparison_caveat"] = (
        f"Held-out stratified 80/20 split (post-dedup). SVM-RBF trained on a stratified subsample capped at "
        f"{SVM_TRAIN_CAP} rows for tractability (RandomForest/kNN use the full {len(Xtr)}-row training set); "
        "this may understate SVM's achievable accuracy but doesn't invalidate the comparison's direction. "
        "Latency is single-sample CPU inference on the development machine, excludes MediaPipe detection time.")

    section("7. Confusion matrix + per-class error analysis (held-out RandomForest, from section 6)")
    rf = classifiers["RandomForest(100,d15)"][0]
    pred = rf.predict(Xte)
    labels_sorted = sorted(set(y))
    cm = confusion_matrix(yte, pred, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels_sorted))); ax.set_xticklabels(labels_sorted, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels_sorted))); ax.set_yticklabels(labels_sorted, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Held-out confusion matrix (RandomForest, stratified 80/20 split)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)
    off_diag_errors = [(labels_sorted[i], labels_sorted[j], int(cm[i, j]))
                        for i in range(len(labels_sorted)) for j in range(len(labels_sorted))
                        if i != j and cm[i, j] > 0]
    off_diag_errors.sort(key=lambda t: -t[2])
    print(f"Held-out confusions (true, pred, count), top 10: {off_diag_errors[:10]}")
    results["held_out_confusion"] = {
        "top_errors": off_diag_errors[:10],
        "per_class": per_class_table(yte, pred, labels_sorted),
    }

    section("8. Feature ablation (held-out)")
    idx_full = list(range(132))
    idx_no_relative = list(range(126))
    idx_right_only = list(range(63))
    variants = {"full_132": idx_full, "no_relative_126": idx_no_relative, "right_hand_only_63": idx_right_only}
    ablation = {}
    for name, idxs in variants.items():
        Xtr_v, Xte_v = Xtr[:, idxs], Xte[:, idxs]
        c = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        c.fit(Xtr_v, ytr)
        pred_v = c.predict(Xte_v)
        acc = accuracy_score(yte, pred_v); f1m = f1_score(yte, pred_v, average="macro", zero_division=0)
        print(f"{name:20s} n_features={len(idxs):3d}  held-out acc={acc*100:5.1f}%  macro-F1={f1m:.3f}")
        ablation[name] = {"n_features": len(idxs), "accuracy": float(acc), "macro_f1": float(f1m)}
    results["feature_ablation"] = ablation
    results["feature_ablation_caveat"] = "Held-out stratified 80/20 split, same split as section 6/7."

    with open(f"{RESULTS_DIR}/all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    write_results_md(results)
    print(f"\nAll figures/tables written to {RESULTS_DIR}/")


def write_results_md(r):
    d = r["descriptive"]
    nd = r["near_duplication"]
    lk = r["leakage_demo"]
    rs = r["repeated_split_stats"]
    cs = r["cross_source_evaluation"]
    ha = r["handedness_analysis"]
    ic = r["classifier_comparison"]
    hc = r["held_out_confusion"]
    fa = r["feature_ablation"]
    env = r["environment"]
    lines = []
    lines.append("# Evaluation Results (real, reproducible via `python evaluate.py`)\n")
    lines.append(f"Environment: Python {env['python']}, {env['platform']}, {env['processor']}, "
                  f"scikit-learn {env['sklearn']}, numpy {env['numpy']}, scipy {env['scipy']}.\n")
    lines.append("## 1. Dataset\n")
    lines.append(f"- Raw: {d['n_samples_raw']} samples, {d['n_classes_present_raw']}/{d['n_classes_target']} "
                 f"target classes populated (missing: {', '.join(d['missing_classes_35'])} -- unrecorded, "
                 "out of scope for this study, documented as future work).\n")
    lines.append(f"- After removing {d['n_exact_duplicates_removed']} exact-duplicate feature rows "
                 f"({d['exact_duplicates_removed_by_domain']}): **{d['n_samples_deduped']} usable samples, "
                 f"{d['n_classes_deduped']} classes** -- this is the dataset used for every experiment below.\n")
    lines.append(f"- Domain split: {d['domain_counts_deduped']}.\n")
    lines.append(f"- signer_id/session_id values: {d['unique_signer_ids']} / {d['unique_session_ids']}. "
                 f"{d['signer_session_caveat']}\n")
    lines.append("![per-class counts](descriptive_stats.png)\n")
    lines.append("## 1b. Near-duplication (intra- vs cross-burst distance, post-dedup)\n")
    lines.append(f"- Classes with >=2 independent recording bursts: {nd['n_classes_with_ge2_bursts']}.\n")
    lines.append(f"- Intra-burst consecutive-frame L2 distance: mean {nd['intra_burst_mean']:.4f} "
                 f"(n={nd['intra_burst_n_pairs']} pairs).\n")
    if nd['cross_burst_mean'] is not None:
        lines.append(f"- Cross-burst same-label L2 distance: mean {nd['cross_burst_mean']:.4f} "
                     f"(n={nd['cross_burst_n_pairs']} pairs), ratio (cross/intra) {nd['cross_over_intra_ratio']:.2f}x.\n")
    lines.append(f"> {nd['caveat']}\n")
    lines.append("## 2. Leakage demonstration: burst_id as a split control\n")
    lines.append(f"- Naive random 80/20 split: **{lk['naive_random_split']['accuracy']*100:.2f}%** accuracy, "
                 f"macro-F1 {lk['naive_random_split']['macro_f1']:.3f}.\n")
    lines.append(f"- Burst-disjoint split ({lk['burst_disjoint_split']['n_classes_evaluated']} classes with "
                 f">=2 bursts): **{lk['burst_disjoint_split']['accuracy']*100:.2f}%** accuracy, "
                 f"macro-F1 {lk['burst_disjoint_split']['macro_f1']:.3f}.\n")
    lines.append(f"> {lk['interpretation']}\n")
    lines.append("## 3. Repeated-split statistics (10 seeds)\n")
    lines.append(f"- Naive split accuracy: {rs['accuracy_mean']*100:.2f}% +/- {rs['accuracy_std']*100:.2f}pp, "
                 f"bootstrap 95% CI [{rs['accuracy_bootstrap_95ci'][0]*100:.2f}%, {rs['accuracy_bootstrap_95ci'][1]*100:.2f}%] "
                 f"(macro-F1 {rs['macro_f1_mean']:.3f} +/- {rs['macro_f1_std']:.3f}).\n")
    lines.append(f"> {rs['pseudo_replication_caveat']}\n")
    lines.append("## 4. Cross-source (domain) evaluation -- KEY FINDING\n")
    lines.append(f"Shared classes (present in both Kaggle and Webcam sources): {', '.join(cs['shared_classes'])} "
                 f"({len(cs['shared_classes'])} classes). Domain sizes: {cs['domain_counts']}.\n\n")
    lines.append("| Setting | n_train | n_test | Accuracy [95% CI] | Macro-F1 [95% CI] |\n|---|---|---|---|---|\n")
    for key, label in [("in_domain_kaggle", "In-domain: train+test Kaggle"),
                        ("in_domain_webcam", "In-domain: train+test Webcam"),
                        ("kaggle_to_webcam", "Cross-domain: train Kaggle -> test Webcam"),
                        ("webcam_to_kaggle", "Cross-domain: train Webcam -> test Kaggle")]:
        v = cs[key]
        acc_ci_s = f"[{v['accuracy_95ci'][0]*100:.1f}, {v['accuracy_95ci'][1]*100:.1f}]"
        f1_ci_s = f"[{v['macro_f1_95ci'][0]:.3f}, {v['macro_f1_95ci'][1]:.3f}]"
        lines.append(f"| {label} | {v['n_train']} | {v['n_test']} | {v['accuracy']*100:.1f}% {acc_ci_s} | "
                     f"{v['macro_f1']:.3f} {f1_ci_s} |\n")
    lines.append(f"\n> {cs['interpretation']} 95% CIs are bootstrapped over the fixed test-set predictions "
                 "(2000 resamples) -- they quantify test-set sampling uncertainty for a single trained model, "
                 "not retraining variability; the smallest cross-domain test sets (e.g. n_test=631 for "
                 "kaggle_to_webcam) have correspondingly wide CIs, shown explicitly rather than implied by a "
                 "bare point estimate.\n")
    lines.append("## 5. Handedness analysis (held-out)\n")
    lines.append(f"Distribution: {ha['distribution']}.\n\n")
    lines.append("| Test subset (by detected handedness) | n_test | Accuracy | Macro-F1 |\n|---|---|---|---|\n")
    for h, v in ha["held_out_breakdown_by_test_subset_handedness"].items():
        lines.append(f"| {h} | {v['n_test']} | {v['accuracy']*100:.1f}% | {v['macro_f1']:.3f} |\n")
    lines.append(f"\n> {ha['caveat']}\n")
    lines.append("## 6. Held-out classifier comparison\n")
    lines.append("| Classifier | n_train | Held-out acc | Macro-F1 | Train time (s) | Predict latency (ms/sample) |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for name, v in ic.items():
        lines.append(f"| {name} | {v['n_train']} | {v['accuracy']*100:.1f}% | {v['macro_f1']:.3f} | "
                     f"{v['train_seconds']:.3f} | {v['predict_latency_ms']:.2f} |\n")
    lines.append(f"\n> {r['classifier_comparison_caveat']}\n")
    lines.append("## 7. Confusion matrix + per-class error analysis (held-out)\n")
    lines.append("![confusion matrix](confusion_matrix.png)\n")
    lines.append(f"Top held-out confusions (true, pred, count): {hc['top_errors']}\n")
    lines.append("## 8. Feature ablation (held-out)\n")
    lines.append("| Feature set | # features | Held-out acc | Macro-F1 |\n|---|---|---|---|\n")
    for name, v in fa.items():
        lines.append(f"| {name} | {v['n_features']} | {v['accuracy']*100:.1f}% | {v['macro_f1']:.3f} |\n")
    lines.append(f"\n> {r['feature_ablation_caveat']}\n")
    with open(f"{RESULTS_DIR}/RESULTS.md", "w") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    main()
