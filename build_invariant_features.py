#!/usr/bin/env python3
"""
build_invariant_features.py — Phase 5 candidate fix for the cross-source
degradation found in analyze_domain_shift.py.

Hypothesis under test: the Kaggle-vs-webcam cross-domain failure is
dominated by a systematic hand-orientation/mirroring-convention mismatch
between the two capture pipelines (evidence: x-coordinate-dominated
feature shift, deterministic per-class letter-pair confusions, persists
across the pre/post mirroring-fix boundary -- see analyze_domain_shift.py).

Proposed fix: replace the raw per-landmark (x, y, z) coordinates -- which
encode absolute hand orientation and are therefore sensitive to any
mirroring/rotation-convention mismatch -- with joint angles and inter-
finger spread angles computed from the SAME landmark positions already in
isl_dataset.csv. Angles computed via arccos(dot product) are invariant to
BOTH rotation and reflection (mirroring), so if the failure really is an
orientation/mirroring artifact, this representation should be far less
sensitive to it. If the failure is instead caused by something else
(e.g. genuinely different hand shapes/backgrounds), this representation
should NOT recover cross-domain accuracy -- that would be evidence against
the mirroring hypothesis, and this script reports whichever result occurs.

19 angles per hand (15 finger-joint angles + 4 inter-finger spread angles)
+ 2 hands + the 3 already-invariant inter-hand distance features
(rel_d_wrist, rel_d_index, rel_d_thumb) = 41 features, vs. 132 raw.
"""
import json
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from evaluate import load_dataset, dedupe_exact, FEATURE_COLUMNS

SHARED_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "I", "J", "K", "L", "M", "U", "V"]

FINGER_CHAINS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}
FINGER_BASES = [1, 5, 9, 13, 17]


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos))


def hand_angles(landmarks_21x3):
    angles = []
    for chain in FINGER_CHAINS.values():
        for i in range(1, len(chain) - 1):
            angles.append(_angle(landmarks_21x3[chain[i - 1]], landmarks_21x3[chain[i]], landmarks_21x3[chain[i + 1]]))
    wrist = landmarks_21x3[0]
    for i in range(len(FINGER_BASES) - 1):
        angles.append(_angle(landmarks_21x3[FINGER_BASES[i]], wrist, landmarks_21x3[FINGER_BASES[i + 1]]))
    return angles  # 15 + 4 = 19


def rows_to_landmarks(X_block_63):
    """(n, 63) flat rx/ry/rz-style columns -> (n, 21, 3)."""
    return X_block_63.reshape(-1, 21, 3)


def compute_invariant_features(X):
    """X: (n, 132) raw feature matrix (evaluate.py's FEATURE_COLUMNS order) ->
    (n, 41) rotation+reflection-invariant feature matrix. Right/left slots kept
    as-is (this is the version that failed to fix cross-domain accuracy)."""
    right = rows_to_landmarks(X[:, 0:63])
    left = rows_to_landmarks(X[:, 63:126])
    d_wrist, d_index, d_thumb = X[:, 129], X[:, 130], X[:, 131]
    out = np.zeros((len(X), 19 + 19 + 3), dtype=float)
    for i in range(len(X)):
        out[i, 0:19] = hand_angles(right[i])
        out[i, 19:38] = hand_angles(left[i])
    out[:, 38] = d_wrist
    out[:, 39] = d_index
    out[:, 40] = d_thumb
    return out


def compute_handslot_agnostic_features(X):
    """Same 41-d angle representation, but for single-handed rows the active
    hand's 19 angles always go in the 'primary' slot regardless of whether it
    was stored in the right-hand or left-hand columns -- tests whether the
    cross-domain failure is caused by which FEATURE SLOT a one-handed sign
    lands in (right-slot in Kaggle vs. left-slot in webcam) rather than by
    hand orientation itself."""
    right = rows_to_landmarks(X[:, 0:63])
    left = rows_to_landmarks(X[:, 63:126])
    r_present = np.abs(X[:, 0:63]).sum(axis=1) > 0
    l_present = np.abs(X[:, 63:126]).sum(axis=1) > 0
    d_wrist, d_index, d_thumb = X[:, 129], X[:, 130], X[:, 131]
    out = np.zeros((len(X), 19 + 19 + 3), dtype=float)
    for i in range(len(X)):
        r_ang = hand_angles(right[i]) if r_present[i] else None
        l_ang = hand_angles(left[i]) if l_present[i] else None
        if r_present[i] and l_present[i]:
            out[i, 0:19] = r_ang
            out[i, 19:38] = l_ang
        elif r_present[i]:
            out[i, 0:19] = r_ang  # only hand present -> always primary slot
        elif l_present[i]:
            out[i, 0:19] = l_ang  # only hand present -> always primary slot
    out[:, 38] = d_wrist
    out[:, 39] = d_index
    out[:, 40] = d_thumb
    return out


def _bootstrap_test_ci(y_true, y_pred, n_boot=2000, seed=0):
    r = np.random.default_rng(seed)
    n = len(y_true)
    accs_b, f1s_b = [], []
    for _ in range(n_boot):
        idx = r.integers(0, n, size=n)
        accs_b.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s_b.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return {"accuracy_95ci": [float(np.percentile(accs_b, 2.5)), float(np.percentile(accs_b, 97.5))],
            "macro_f1_95ci": [float(np.percentile(f1s_b, 2.5)), float(np.percentile(f1s_b, 97.5))]}


def fit_eval(Xtr, ytr, Xte, yte, labels_for_percls=None, with_ci=False):
    c = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    c.fit(Xtr, ytr)
    pred = c.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro", zero_division=0)
    per_class = None
    if labels_for_percls is not None:
        per_class = {lab: float(accuracy_score(yte[yte == lab], pred[yte == lab]))
                     for lab in labels_for_percls if (yte == lab).sum() > 0}
    if with_ci:
        return acc, f1m, per_class, pred, _bootstrap_test_ci(yte, pred)
    return acc, f1m, per_class, pred


def main():
    X_raw, y_raw, groups_raw, signers_raw, sessions_raw, domains_raw = load_dataset()
    X, y, groups, signers, sessions, domains, _ = dedupe_exact(
        X_raw, y_raw, groups_raw, signers_raw, sessions_raw, domains_raw)

    mask = np.isin(y, SHARED_CLASSES)
    Xs, ys, ds = X[mask], y[mask], domains[mask]
    is_kaggle, is_webcam = ds == "kaggle", ds == "webcam"

    print("Building invariant (angle-based) feature representations...")
    Xi = compute_invariant_features(Xs)
    Xh = compute_handslot_agnostic_features(Xs)
    print(f"Raw features: {Xs.shape[1]}, invariant features: {Xi.shape[1]}, handslot-agnostic: {Xh.shape[1]}")

    results = {}
    preds_for_mcnemar = {}
    print("\n" + "=" * 70)
    print("Cross-domain evaluation: RAW vs INVARIANT vs HANDSLOT-AGNOSTIC features")
    print("=" * 70)
    for feat_name, Xf in [("raw_132", Xs), ("invariant_41", Xi), ("handslot_agnostic_41", Xh)]:
        acc_kk, f1_kk, _, _ = fit_eval(Xf[is_kaggle], ys[is_kaggle], Xf[is_kaggle], ys[is_kaggle])
        # in-domain kaggle reported here is in-sample only for a quick sanity check;
        # the headline numbers are the cross-domain ones below.
        acc_kw, f1_kw, per_class_kw, pred_kw, ci_kw = fit_eval(
            Xf[is_kaggle], ys[is_kaggle], Xf[is_webcam], ys[is_webcam], SHARED_CLASSES, with_ci=True)
        acc_wk, f1_wk, _, pred_wk, ci_wk = fit_eval(
            Xf[is_webcam], ys[is_webcam], Xf[is_kaggle], ys[is_kaggle], with_ci=True)
        print(f"\n[{feat_name}]")
        print(f"  train Kaggle -> test Webcam: acc={acc_kw*100:.2f}% {ci_kw['accuracy_95ci']} "
              f"macro-F1={f1_kw:.3f} {ci_kw['macro_f1_95ci']}")
        print(f"  train Webcam -> test Kaggle: acc={acc_wk*100:.2f}% {ci_wk['accuracy_95ci']} "
              f"macro-F1={f1_wk:.3f} {ci_wk['macro_f1_95ci']}")
        print(f"  (95% CIs are bootstrapped over the fixed test-set predictions, 2000 resamples --")
        print(f"   quantify test-set sampling uncertainty for this one trained model, not retraining variance)")
        print(f"  per-class recall (Kaggle->Webcam), previously-failing classes:")
        for c_name in ["A", "E", "F", "J", "K", "M", "U"]:
            if c_name in per_class_kw:
                print(f"    {c_name}: {per_class_kw[c_name]*100:.1f}%")
        results[feat_name] = {
            "kaggle_to_webcam_accuracy": acc_kw, "kaggle_to_webcam_macro_f1": f1_kw,
            "kaggle_to_webcam_ci": ci_kw,
            "webcam_to_kaggle_accuracy": acc_wk, "webcam_to_kaggle_macro_f1": f1_wk,
            "webcam_to_kaggle_ci": ci_wk,
            "per_class_recall_kaggle_to_webcam": per_class_kw,
        }
        # y_webcam/y_kaggle row order is identical across feat_name (same mask,
        # same ys), so predictions are directly pairable across representations.
        preds_for_mcnemar[feat_name] = {"kaggle_to_webcam": pred_kw, "webcam_to_kaggle": pred_wk}

    print("\n" + "=" * 70)
    print("Paired significance test: raw_132 vs handslot_agnostic_41, SAME test rows")
    print("(McNemar's test on correct/incorrect per-sample, exact binomial version --")
    print(" a stronger claim than non-overlapping bootstrap CIs alone, since the two")
    print(" models are compared on paired predictions rather than independent samples)")
    print("=" * 70)
    from scipy.stats import binomtest
    mcnemar = {}
    y_true_by_direction = {"kaggle_to_webcam": ys[is_webcam], "webcam_to_kaggle": ys[is_kaggle]}
    for direction in ["kaggle_to_webcam", "webcam_to_kaggle"]:
        y_true = y_true_by_direction[direction]
        p_raw = preds_for_mcnemar["raw_132"][direction]
        p_hs = preds_for_mcnemar["handslot_agnostic_41"][direction]
        raw_correct = (p_raw == y_true)
        hs_correct = (p_hs == y_true)
        n01 = int(((~raw_correct) & hs_correct).sum())   # raw wrong, handslot right
        n10 = int((raw_correct & (~hs_correct)).sum())   # raw right, handslot wrong
        n_discordant = n01 + n10
        if n_discordant == 0:
            pval = 1.0
        else:
            pval = binomtest(n01, n_discordant, 0.5, alternative="greater").pvalue
        print(f"  [{direction}] raw-wrong/handslot-right: {n01}   raw-right/handslot-wrong: {n10}   "
              f"discordant pairs: {n_discordant}   McNemar exact p-value: {pval:.2e}")
        mcnemar[direction] = {"raw_wrong_handslot_right": n01, "raw_right_handslot_wrong": n10,
                               "n_discordant_pairs": n_discordant, "p_value": float(pval),
                               "significant_at_0.05": bool(pval < 0.05)}
    results["mcnemar_raw_vs_handslot_agnostic"] = mcnemar

    print("\n" + "=" * 70)
    print("Sanity check: does the invariant representation hurt WITHIN-domain accuracy?")
    print("=" * 70)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, random_state=42, stratify=ys)
    Xitr, Xite = compute_invariant_features(Xtr), compute_invariant_features(Xte)
    Xhtr, Xhte = compute_handslot_agnostic_features(Xtr), compute_handslot_agnostic_features(Xte)
    for feat_name, (Xt, Xe) in [("raw_132", (Xtr, Xte)), ("invariant_41", (Xitr, Xite)),
                                 ("handslot_agnostic_41", (Xhtr, Xhte))]:
        acc, f1m, _, _ = fit_eval(Xt, ytr, Xe, yte)
        print(f"  [{feat_name}] mixed-domain held-out (shared classes only): acc={acc*100:.2f}% macro-F1={f1m:.3f}")
        results.setdefault("within_domain_sanity_check", {})[feat_name] = {"accuracy": acc, "macro_f1": f1m}

    print("\n" + "=" * 70)
    print("Multi-seed robustness check: does handslot-agnostic > raw survive across")
    print("multiple RandomForest random_state seeds, not just seed=42 (rules out a")
    print("single-lucky-seed artifact -- CVIP2026-v2 hostile-ML-review requirement)")
    print("=" * 70)
    SEEDS = [0, 1, 2, 3, 4, 42, 100]
    multiseed = {}
    for feat_name, Xf in [("raw_132", Xs), ("invariant_41", Xi), ("handslot_agnostic_41", Xh)]:
        kw_f1s, wk_f1s = [], []
        for seed in SEEDS:
            c1 = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1)
            c1.fit(Xf[is_kaggle], ys[is_kaggle])
            kw_f1s.append(f1_score(ys[is_webcam], c1.predict(Xf[is_webcam]), average="macro", zero_division=0))
            c2 = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1)
            c2.fit(Xf[is_webcam], ys[is_webcam])
            wk_f1s.append(f1_score(ys[is_kaggle], c2.predict(Xf[is_kaggle]), average="macro", zero_division=0))
        kw_f1s, wk_f1s = np.array(kw_f1s), np.array(wk_f1s)
        print(f"  [{feat_name}] Kaggle->Webcam macro-F1 across {len(SEEDS)} RF seeds: "
              f"{kw_f1s.mean():.3f} +/- {kw_f1s.std():.3f}  (min {kw_f1s.min():.3f}, max {kw_f1s.max():.3f})")
        print(f"  [{feat_name}] Webcam->Kaggle macro-F1 across {len(SEEDS)} RF seeds: "
              f"{wk_f1s.mean():.3f} +/- {wk_f1s.std():.3f}  (min {wk_f1s.min():.3f}, max {wk_f1s.max():.3f})")
        multiseed[feat_name] = {
            "seeds": SEEDS,
            "kaggle_to_webcam_macro_f1_mean": float(kw_f1s.mean()), "kaggle_to_webcam_macro_f1_std": float(kw_f1s.std()),
            "kaggle_to_webcam_macro_f1_min": float(kw_f1s.min()), "kaggle_to_webcam_macro_f1_max": float(kw_f1s.max()),
            "webcam_to_kaggle_macro_f1_mean": float(wk_f1s.mean()), "webcam_to_kaggle_macro_f1_std": float(wk_f1s.std()),
            "webcam_to_kaggle_macro_f1_min": float(wk_f1s.min()), "webcam_to_kaggle_macro_f1_max": float(wk_f1s.max()),
        }
    raw_kw_max = multiseed["raw_132"]["kaggle_to_webcam_macro_f1_max"]
    hs_kw_min = multiseed["handslot_agnostic_41"]["kaggle_to_webcam_macro_f1_min"]
    survives = hs_kw_min > raw_kw_max
    print(f"\n  Handslot-agnostic's WORST seed ({hs_kw_min:.3f}) vs raw's BEST seed ({raw_kw_max:.3f}) on "
          f"Kaggle->Webcam: {'NO OVERLAP -- improvement survives worst-case seed variation' if survives else 'OVERLAP -- improvement is not seed-robust at the extremes'}")
    multiseed["verdict_kaggle_to_webcam_seed_robust"] = bool(survives)
    results["multiseed_robustness_check"] = multiseed

    with open("results/invariant_features_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWritten: results/invariant_features_test.json")


if __name__ == "__main__":
    main()
