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
    print("\n" + "=" * 70)
    print("Cross-domain evaluation: RAW vs INVARIANT vs HANDSLOT-AGNOSTIC features")
    print("=" * 70)
    for feat_name, Xf in [("raw_132", Xs), ("invariant_41", Xi), ("handslot_agnostic_41", Xh)]:
        acc_kk, f1_kk, _, _ = fit_eval(Xf[is_kaggle], ys[is_kaggle], Xf[is_kaggle], ys[is_kaggle])
        # in-domain kaggle reported here is in-sample only for a quick sanity check;
        # the headline numbers are the cross-domain ones below.
        acc_kw, f1_kw, per_class_kw, pred_kw, ci_kw = fit_eval(
            Xf[is_kaggle], ys[is_kaggle], Xf[is_webcam], ys[is_webcam], SHARED_CLASSES, with_ci=True)
        acc_wk, f1_wk, _, _, ci_wk = fit_eval(
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

    with open("results/invariant_features_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWritten: results/invariant_features_test.json")


if __name__ == "__main__":
    main()
