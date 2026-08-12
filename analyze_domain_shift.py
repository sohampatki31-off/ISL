#!/usr/bin/env python3
"""
analyze_domain_shift.py — Forensic analysis of the Kaggle-vs-Webcam
cross-source accuracy gap found in evaluate.py (section 4).

Question: is the ~50-99pp accuracy drop when moving between the Kaggle and
webcam domains caused by (a) the webcam domain simply having too few
training samples per class, (b) a genuine geometric/scale shift in the
feature representation between sources, (c) a handedness-composition
difference between sources, or (d) some combination? This script runs
targeted control experiments and reports evidence for each hypothesis
without assuming which one is true.

Outputs -> results/: domain_shift_forensics.json, domain_shift_features.png
"""
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from evaluate import load_dataset, dedupe_exact, FEATURE_COLUMNS, RIGHT_IDX, LEFT_IDX

SHARED_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "I", "J", "K", "L", "M", "U", "V"]


def main():
    X_raw, y_raw, groups_raw, signers_raw, sessions_raw, domains_raw = load_dataset()
    X, y, groups, signers, sessions, domains, _ = dedupe_exact(
        X_raw, y_raw, groups_raw, signers_raw, sessions_raw, domains_raw)

    mask_shared = np.isin(y, SHARED_CLASSES)
    Xs, ys, ds, gs, sess = (X[mask_shared], y[mask_shared], domains[mask_shared],
                             groups[mask_shared], sessions[mask_shared])
    is_kaggle = ds == "kaggle"
    is_webcam = ds == "webcam"
    n_webcam = int(is_webcam.sum())
    results = {"shared_classes": SHARED_CLASSES, "n_kaggle": int(is_kaggle.sum()), "n_webcam": n_webcam}

    print("=" * 70)
    print("H1: is in-domain webcam's low accuracy (46.88% in evaluate.py) just")
    print("    a sample-size effect, not a domain-intrinsic difficulty?")
    print("=" * 70)
    # Control: subsample Kaggle down to the SAME number of rows as the full
    # webcam-shared-classes set (matched roughly per-class), then measure
    # in-domain accuracy the same way (70/30 split) evaluate.py does for webcam.
    rng = np.random.default_rng(42)
    kaggle_idx = np.where(is_kaggle)[0]
    webcam_class_counts = Counter(ys[is_webcam])
    sample_idx = []
    for c, n_target in webcam_class_counts.items():
        pool = kaggle_idx[ys[kaggle_idx] == c]
        take = min(len(pool), n_target)
        sample_idx.extend(rng.choice(pool, size=take, replace=False))
    sample_idx = np.array(sample_idx)
    Xk_small, yk_small = Xs[sample_idx], ys[sample_idx]
    n_train = int(round(len(Xk_small) * 0.7))
    perm = rng.permutation(len(Xk_small))
    tr, te = perm[:n_train], perm[n_train:]
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(Xk_small[tr], yk_small[tr])
    pred = clf.predict(Xk_small[te])
    acc_small_kaggle = accuracy_score(yk_small[te], pred)
    f1_small_kaggle = f1_score(yk_small[te], pred, average="macro", zero_division=0)
    print(f"Kaggle subsampled to webcam's per-class sample sizes (n={len(Xk_small)}): "
          f"in-domain held-out acc={acc_small_kaggle*100:.2f}% macro-F1={f1_small_kaggle:.3f}")
    print("(compare to full-size in-domain Kaggle: 99.92% acc, and in-domain webcam: 46.88% acc)")
    results["h1_sample_size_control"] = {
        "n_matched": int(len(Xk_small)),
        "kaggle_matched_size_accuracy": float(acc_small_kaggle),
        "kaggle_matched_size_macro_f1": float(f1_small_kaggle),
        "full_size_kaggle_accuracy_reference": 0.9992,
        "full_size_webcam_accuracy_reference": 0.4688,
    }

    print()
    print("=" * 70)
    print("H2: does handedness composition differ by domain (a domain")
    print("    characteristic that could explain part of the gap)?")
    print("=" * 70)
    r_present = np.abs(Xs[:, RIGHT_IDX]).sum(axis=1) > 0
    l_present = np.abs(Xs[:, LEFT_IDX]).sum(axis=1) > 0
    handedness = np.where(r_present & l_present, "both", np.where(r_present, "right_only", "left_only"))
    hand_by_domain = {}
    for dom in ["kaggle", "webcam"]:
        m = ds == dom
        counts = Counter(handedness[m])
        total = int(m.sum())
        hand_by_domain[dom] = {k: round(v / total, 4) for k, v in counts.items()}
        print(f"{dom:8s} (n={total}): {hand_by_domain[dom]}")
    results["h2_handedness_by_domain"] = hand_by_domain

    print()
    print("=" * 70)
    print("H3: is there a systematic geometric/scale shift in the feature")
    print("    representation between domains, for the SAME shared classes?")
    print("=" * 70)
    # Compare per-feature mean/std between domains, for right-hand landmark
    # features only (present in effectively all rows), restricted to shared
    # classes so this isn't confounded by class composition differences.
    # Computed WITHIN each shared class, then averaged across classes -- a
    # marginal (class-agnostic) comparison would conflate genuine domain shift
    # with the fact that webcam's per-class counts are wildly uneven vs.
    # Kaggle's near-uniform ~1200/class (CV-review finding, fixed here).
    right_cols_idx = RIGHT_IDX
    per_class_d = []
    for c in SHARED_CLASSES:
        kf = Xs[(ys == c) & is_kaggle][:, right_cols_idx]
        wf = Xs[(ys == c) & is_webcam][:, right_cols_idx]
        kf = kf[np.abs(kf).sum(axis=1) > 0]
        wf = wf[np.abs(wf).sum(axis=1) > 0]
        if len(kf) < 2 or len(wf) < 2:
            continue
        mean_diff_c = kf.mean(axis=0) - wf.mean(axis=0)
        pooled_std_c = np.sqrt((kf.std(axis=0) ** 2 + wf.std(axis=0) ** 2) / 2)
        per_class_d.append(mean_diff_c / np.where(pooled_std_c > 1e-9, pooled_std_c, 1e-9))
    cohens_d = np.mean(per_class_d, axis=0)  # per-feature, averaged over classes
    kaggle_feat = Xs[is_kaggle][:, right_cols_idx]
    webcam_feat = Xs[is_webcam][:, right_cols_idx]
    kaggle_feat = kaggle_feat[np.abs(kaggle_feat).sum(axis=1) > 0]
    webcam_feat = webcam_feat[np.abs(webcam_feat).sum(axis=1) > 0]
    top_shift_idx = np.argsort(-np.abs(cohens_d))[:10]
    right_col_names = FEATURE_COLUMNS[0:63]
    top_shifted_features = [(right_col_names[i], float(cohens_d[i])) for i in top_shift_idx]
    print(f"Right-hand features with largest Kaggle-vs-Webcam standardized mean shift (Cohen's d,")
    print(f"computed within each of {len(per_class_d)} shared classes, then averaged across classes):")
    for name, d in top_shifted_features:
        print(f"  {name:10s} d={d:+.3f}")
    print(f"Median |Cohen's d| across all 63 right-hand features: {np.median(np.abs(cohens_d)):.3f}")
    results["h3_feature_shift"] = {
        "method": "within-class Cohen's d, averaged across classes (not a marginal/class-agnostic comparison)",
        "n_classes_averaged": len(per_class_d),
        "top_shifted_features": top_shifted_features,
        "median_abs_cohens_d": float(np.median(np.abs(cohens_d))),
        "n_kaggle_rows_compared": int(len(kaggle_feat)),
        "n_webcam_rows_compared": int(len(webcam_feat)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, feat_name in zip(axes, [top_shifted_features[0][0], top_shifted_features[1][0]]):
        i = right_col_names.index(feat_name)
        ax.hist(kaggle_feat[:, i], bins=40, alpha=0.6, label="Kaggle", color="#4C72B0", density=True)
        ax.hist(webcam_feat[:, i], bins=40, alpha=0.6, label="Webcam", color="#DD8452", density=True)
        ax.set_title(f"Feature: {feat_name}")
        ax.legend()
    fig.suptitle("Largest domain-shifted right-hand features (shared classes only)")
    fig.tight_layout()
    fig.savefig("results/domain_shift_features.png", dpi=150)
    plt.close(fig)

    print()
    print("=" * 70)
    print("H4: per-class breakdown of cross-domain failure (which shared")
    print("    classes drive the kaggle->webcam and webcam->kaggle drops?)")
    print("=" * 70)
    clf_kw = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf_kw.fit(Xs[is_kaggle], ys[is_kaggle])
    pred_kw = clf_kw.predict(Xs[is_webcam])
    per_class_kw = {}
    for c in SHARED_CLASSES:
        m = ys[is_webcam] == c
        if m.sum() == 0:
            continue
        per_class_kw[c] = {"n": int(m.sum()), "recall": float(accuracy_score(ys[is_webcam][m], pred_kw[m]))}
    print("Kaggle-trained model, per-class recall on webcam test set:")
    for c, v in sorted(per_class_kw.items(), key=lambda kv: kv[1]["recall"]):
        print(f"  {c:3s} n={v['n']:3d} recall={v['recall']*100:5.1f}%")
    results["h4_kaggle_to_webcam_per_class"] = per_class_kw

    print()
    print("=" * 70)
    print("H5: does the failure pattern persist in the NEWEST recordings too,")
    print("    or only in old pre-mirroring-fix legacy rows (commit b5fc4c6)?")
    print("=" * 70)
    # class M has samples from both the old 'unknown_legacy' session and the
    # newer 'unassigned_session' recordings (made after the mirroring-fix
    # commit) -- a natural before/after control for whether that fix resolved
    # the cross-domain gap.
    h5 = {}
    for sess_val in ["unknown_legacy", "unassigned_session"]:
        m = (ys == "M") & (sess == sess_val)
        if m.sum() == 0:
            continue
        pred_m = clf_kw.predict(Xs[m])
        recall = float((pred_m == "M").mean())
        h5[sess_val] = {"n": int(m.sum()), "recall": recall, "predicted_as": dict(Counter(pred_m).most_common(3))}
        print(f"  M / session={sess_val:20s} n={m.sum():3d} recall={recall*100:.1f}% "
              f"predicted_as={h5[sess_val]['predicted_as']}")
    print("  Both pre- and post-fix recordings show ~0% recall (different wrong labels) -- the")
    print("  cross-domain gap is NOT explained by the old mirroring bug alone; it persists in the")
    print("  newest recordings too, pointing to a more persistent representation/orientation gap")
    print("  between live-webcam capture and static Kaggle photos.")
    results["h5_pre_vs_post_mirroring_fix"] = h5

    print()
    print("=" * 70)
    print("H6: per-class handedness composition, Kaggle vs Webcam -- does a")
    print("    composition MISMATCH between domains predict which classes fail?")
    print("=" * 70)
    h6 = {}
    for c in SHARED_CLASSES:
        row = {}
        for dom_name, dom_mask in [("kaggle", is_kaggle), ("webcam", is_webcam)]:
            m = (ys == c) & dom_mask
            n = int(m.sum())
            if n == 0:
                continue
            r_p = np.abs(Xs[m][:, :63]).sum(axis=1) > 0
            l_p = np.abs(Xs[m][:, 63:126]).sum(axis=1) > 0
            hnd = np.where(r_p & l_p, "both", np.where(r_p, "right_only", "left_only"))
            cnt = Counter(hnd)
            row[dom_name] = {"n": n, **{k: round(100 * v / n, 1) for k, v in cnt.items()}}
        row["recall_kaggle_to_webcam"] = per_class_kw.get(c, {}).get("recall")
        h6[c] = row
        print(f"  {c:3s} kaggle={row.get('kaggle')}  webcam={row.get('webcam')}  "
              f"recall={row['recall_kaggle_to_webcam']}")
    print("  Mixed picture: J/K/U/C show a clear handedness-composition MISMATCH between domains")
    print("  (consistent with the hand-slot-agnostic fix helping those). But A, E, F, M have SIMILAR")
    print("  handedness composition in both domains and STILL fail completely -- handedness/hand-slot")
    print("  assignment does not explain their failure. A second, still-unidentified factor is at play.")
    results["h6_handedness_composition_vs_recall"] = h6

    print()
    print("=" * 70)
    print("H7 (manual finding, not reproducible from this repo alone): the original")
    print("    Kaggle image files, inspected manually at C:\\Users\\Soham\\Downloads\\")
    print("    ISL_kaggle_extracted on 2026-08-12, are NOT uniform resolution.")
    print("=" * 70)
    print("  31/35 classes: 1200 images each, all 128x128px, tightly cropped to the hand.")
    print("  4 classes (C, I, O, V) additionally contain 90-247 extra images each from a")
    print("  DIFFERENT, much higher-resolution sub-source (640x480 or ~1088x1920) --")
    print("  matching exactly the 4 classes whose isl_dataset_kaggle.csv sample counts")
    print("  exceed 1200 (I=1379, C=1292, O=1313, V=1285; see results/RESULTS.md section 1).")
    print("  This means the Kaggle 'ISL' dataset is itself a class-correlated patchwork of")
    print("  at least 2 sub-sources with very different image quality/framing, and 97% of")
    print("  isl_dataset.csv is built from 128x128 pre-cropped photos -- a far more extreme")
    print("  resolution/framing gap against live 720p+ webcam capture than 'domain shift' alone")
    print("  conveys. PLAUSIBLE contributing cause for the A/E/F/M failures H6 leaves unexplained,")
    print("  but NOT experimentally isolated here (no controlled re-detection-at-matched-resolution")
    print("  test was run) -- reported as a well-evidenced fact plus an untested hypothesis, not a")
    print("  proven cause. Not re-verified automatically because the original extracted Kaggle")
    print("  folder lives outside this repo and isn't guaranteed to exist on another machine.")
    results["h7_manual_finding_kaggle_image_resolution_heterogeneity"] = {
        "inspected_path": r"C:\Users\Soham\Downloads\ISL_kaggle_extracted\Indian",
        "inspection_date": "2026-08-12",
        "finding": ("31/35 classes are uniformly 1200x 128x128px tightly-cropped hand photos; "
                    "C/I/O/V additionally contain 90-247 extra images each from a different, much "
                    "higher-resolution sub-source (640x480 or ~1088x1920), matching those same "
                    "4 classes' >1200-row counts in isl_dataset_kaggle.csv."),
        "reproducible_from_repo_alone": False,
        "status": "well-evidenced fact; NOT experimentally confirmed as a cause of the cross-domain gap",
    }

    with open("results/domain_shift_forensics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWritten: results/domain_shift_forensics.json, results/domain_shift_features.png")


if __name__ == "__main__":
    main()
