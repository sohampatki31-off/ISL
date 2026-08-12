#!/usr/bin/env python3
"""
test_handswap_hypothesis.py — CVIP2026-v2 targeted test.

New hypothesis surfaced during the final-validation audit (not previously
tested in analyze_domain_shift.py's H1-H9): isl_main.py (live/webcam
pipeline) deliberately swaps MediaPipe's handedness label and un-mirrors
x-coordinates before building the feature vector (see _HANDEDNESS_UNMIRROR
in isl_main.py), specifically so its output "line[s] up with the
Kaggle-imported training rows (extracted from un-mirrored photos, never
flipped)" -- a documented but NEVER experimentally verified assumption
about the Kaggle images' mirroring convention. import_kaggle_dataset.py /
process_images.py take MediaPipe's raw handedness label with no swap.

If that assumption is wrong for two-handed signs, physical left and right
hands would land in swapped feature columns (0:63 vs 63:126) between the
two domains -- invisible to the H6 handedness-COMPOSITION check (which only
counts how many/which hands are present, not which physical hand occupies
which column), but exactly the kind of thing that could explain why A, E, F
(two-handed in BOTH domains per H6, so composition-matched, yet still 0%
cross-domain recall) remain unexplained after every other H1-H9 test.

This script tests it directly and cheaply, with no new data collection:
take the SAME Kaggle-trained raw-132 model from analyze_domain_shift.py,
and re-evaluate it on webcam rows with the right-hand (cols 0:63) and
left-hand (cols 63:126) blocks SWAPPED. The 6 inter-hand relative features
(dx,dy,dz,d_wrist,d_index,d_thumb, cols 126:132) are left as-is -- an
approximation, since re-deriving them exactly after a swap would require
the pre-normalization raw landmark positions, which are not stored in
isl_dataset.csv (only the post-normalization feature vector is saved).
This approximation is stated explicitly, not hidden: it under-tests the
hypothesis for the 3 relative-delta features only, not the 126 raw
per-landmark features where the swap is exact.

If swapping recovers accuracy for A/E/F/etc. (two-handed classes) without
hurting classes that already work, that's real evidence the un-mirror
convention mismatch (not unexplained "domain shift") drives their failure.
If it does nothing or hurts, that's evidence against the hypothesis.

Outputs -> results/handswap_hypothesis_test.json
"""
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

from evaluate import load_dataset, dedupe_exact, FEATURE_COLUMNS

SHARED_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "I", "J", "K", "L", "M", "U", "V"]


def swap_hand_blocks(X):
    """Swap columns 0:63 <-> 63:126 (right<->left landmark blocks). Relative
    features (126:132) are left unchanged -- see module docstring caveat."""
    Xs = X.copy()
    Xs[:, 0:63], Xs[:, 63:126] = X[:, 63:126].copy(), X[:, 0:63].copy()
    return Xs


def both_hands_mask(X):
    r_present = np.abs(X[:, 0:63]).sum(axis=1) > 0
    l_present = np.abs(X[:, 63:126]).sum(axis=1) > 0
    return r_present & l_present


def per_class_recall(y_true, y_pred, labels):
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {lab: {"recall": float(r), "support": int(s)} for lab, r, s in zip(labels, rec, support)}


def main():
    X_raw, y_raw, groups_raw, signers_raw, sessions_raw, domains_raw = load_dataset()
    X, y, groups, signers, sessions, domains, _ = dedupe_exact(
        X_raw, y_raw, groups_raw, signers_raw, sessions_raw, domains_raw)

    mask = np.isin(y, SHARED_CLASSES)
    Xs, ys, ds = X[mask], y[mask], domains[mask]
    is_kaggle, is_webcam = ds == "kaggle", ds == "webcam"

    print("Training RandomForest on ALL Kaggle rows (raw 132-d), same setup as analyze_domain_shift.py...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(Xs[is_kaggle], ys[is_kaggle])

    Xw, yw = Xs[is_webcam], ys[is_webcam]
    both_mask = both_hands_mask(Xw)
    print(f"Webcam test set: {len(Xw)} rows, {both_mask.sum()} are two-handed (both hands detected)")

    pred_baseline = clf.predict(Xw)
    acc_baseline = accuracy_score(yw, pred_baseline)
    f1_baseline = f1_score(yw, pred_baseline, average="macro", zero_division=0)
    rec_baseline = per_class_recall(yw, pred_baseline, SHARED_CLASSES)

    # Swap-hypothesis test: swap hand blocks ONLY for two-handed rows (a
    # one-handed row swapped would move its single hand into the OTHER
    # slot, which is a different, already-tested question -- see
    # build_invariant_features.py's handslot-agnostic representation).
    Xw_swapped = Xw.copy()
    Xw_swapped[both_mask] = swap_hand_blocks(Xw[both_mask])
    pred_swapped = clf.predict(Xw_swapped)
    acc_swapped = accuracy_score(yw, pred_swapped)
    f1_swapped = f1_score(yw, pred_swapped, average="macro", zero_division=0)
    rec_swapped = per_class_recall(yw, pred_swapped, SHARED_CLASSES)

    print(f"\nBASELINE   (as-stored):        acc={acc_baseline*100:.2f}% macro-F1={f1_baseline:.3f}")
    print(f"HAND-SWAPPED (two-handed rows only): acc={acc_swapped*100:.2f}% macro-F1={f1_swapped:.3f}")
    print(f"\nPer-class recall, baseline vs. hand-swapped (two-handed rows only):")
    per_class_compare = {}
    for c in SHARED_CLASSES:
        b, s = rec_baseline[c], rec_swapped[c]
        n_both = int(both_mask[yw == c].sum())
        print(f"  {c:3s} support={b['support']:3d} (two-handed n={n_both:3d})  "
              f"baseline_recall={b['recall']*100:5.1f}%  swapped_recall={s['recall']*100:5.1f}%  "
              f"delta={100*(s['recall']-b['recall']):+.1f}pp")
        per_class_compare[c] = {"support": b["support"], "n_two_handed": n_both,
                                 "baseline_recall": b["recall"], "swapped_recall": s["recall"],
                                 "delta_pp": 100 * (s["recall"] - b["recall"])}

    # Verdict logic: hypothesis is SUPPORTED if two-handed-dominant classes
    # that were failing (baseline recall == 0, n_two_handed large) show a
    # material recall increase after swapping; REFUTED/UNCONFIRMED if not,
    # or if swapping instead damages classes that were already working.
    failing_two_handed = [c for c in SHARED_CLASSES
                           if rec_baseline[c]["recall"] == 0.0 and per_class_compare[c]["n_two_handed"] >= per_class_compare[c]["support"] * 0.5
                           and per_class_compare[c]["support"] > 0]
    recovered = [c for c in failing_two_handed if per_class_compare[c]["delta_pp"] > 10]
    newly_broken = [c for c in SHARED_CLASSES
                    if rec_baseline[c]["recall"] > 0.5 and per_class_compare[c]["delta_pp"] < -10]

    print(f"\nPreviously-failing, two-handed-dominant classes: {failing_two_handed}")
    print(f"Of those, materially recovered (>10pp recall gain) by swapping: {recovered}")
    print(f"Previously-working classes damaged (>10pp recall loss) by swapping: {newly_broken}")

    if recovered and not newly_broken:
        verdict = ("SUPPORTED -- swapping right/left hand blocks for two-handed webcam rows recovers material "
                    f"recall for {recovered} without damaging other classes. Consistent with a systematic "
                    "hand-column mislabeling between the live and offline pipelines' handedness conventions.")
    elif recovered and newly_broken:
        verdict = ("MIXED -- swapping helps some previously-failing classes " + str(recovered) +
                    " but damages previously-working classes " + str(newly_broken) +
                    " -- not a clean global fix; at most a partial, class-specific effect.")
    else:
        verdict = ("NOT SUPPORTED -- swapping hand blocks does not materially recover any previously-failing "
                    "two-handed class. The un-mirror/handedness-swap convention mismatch is not shown to be a "
                    "cause of the A/E/F/K residual failures by this test.")
    print(f"\nVERDICT: {verdict}")

    out = {
        "hypothesis": ("Systematic right/left hand-column mislabeling between isl_main.py's un-mirror "
                        "correction and import_kaggle_dataset.py's raw-MediaPipe-label convention explains "
                        "some of the residual (H6-unexplained) cross-domain failures for two-handed classes."),
        "method_caveat": ("Only the 126 raw per-landmark columns were swapped exactly; the 6 inter-hand "
                           "relative-delta features (126:132) were left unchanged because their pre-normalization "
                           "raw inputs are not recoverable from the stored, already-normalized CSV -- this "
                           "under-tests the hypothesis for those 6 features only."),
        "baseline": {"accuracy": acc_baseline, "macro_f1": f1_baseline, "per_class": rec_baseline},
        "hand_swapped_two_handed_rows_only": {"accuracy": acc_swapped, "macro_f1": f1_swapped, "per_class": rec_swapped},
        "per_class_comparison": per_class_compare,
        "previously_failing_two_handed_dominant_classes": failing_two_handed,
        "materially_recovered_by_swap": recovered,
        "previously_working_classes_damaged_by_swap": newly_broken,
        "verdict": verdict,
    }
    with open("results/handswap_hypothesis_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWritten: results/handswap_hypothesis_test.json")


if __name__ == "__main__":
    main()
