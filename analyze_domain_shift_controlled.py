#!/usr/bin/env python3
"""
analyze_domain_shift_controlled.py — CVIP2026-v2 controlled cross-source
experiment.

Question: does the Kaggle-vs-webcam cross-domain accuracy gap (see
analyze_domain_shift.py, results/RESULTS.md section 4) survive once the
single confound we CAN retroactively fix is fixed -- the MediaPipe
detection-confidence threshold?

What this controls: isl_dataset.csv's Kaggle-sourced rows were extracted
at confidence=0.5 (import_kaggle_dataset.py's old default); isl_main.py's
live/webcam pipeline uses confidence=0.7. This script re-extracts the SAME
raw Kaggle images at confidence=0.7 (see import_kaggle_dataset.py
--confidence 0.7) and reruns the 4 cross-domain settings against that
controlled extraction, compared side-by-side with the original.

What this does NOT and CANNOT control (stated plainly, not glossed over):
One-Euro temporal filtering is applied to webcam rows at capture time,
before they were ever written to isl_dataset.csv -- the raw, unfiltered
webcam video frames were never saved (docs/data_collection_protocol.md:
"no raw video or images are saved to isl_dataset.csv, only the derived
feature vector"). There is therefore no way to retroactively produce
"unfiltered webcam" or "filtered Kaggle" rows without fabricating data.
Kaggle images are static single photos, which have no temporal dimension
to filter in the first place -- this asymmetry is partly an INHERENT
property of comparing a photo dataset to a video-capture dataset, not
purely a fixable pipeline bug. Documented as an unresolved, unfixable-
without-new-data limitation, not silently ignored.

Requires: results/isl_dataset_kaggle_controlled.csv (or a path passed via
--controlled-csv), produced by:
    python import_kaggle_dataset.py --dir <extracted Kaggle root> \
        --out-csv isl_dataset_kaggle_controlled.csv --confidence 0.7

Outputs -> results/: controlled_cross_source_comparison.json,
fig_controlled_vs_original.png
"""
import argparse
import csv
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from evaluate import load_dataset, dedupe_exact, FEATURE_COLUMNS

SHARED_CLASSES_ORIGINAL = ["A", "B", "C", "D", "E", "F", "G", "I", "J", "K", "L", "M", "U", "V"]


def load_controlled_kaggle(csv_path):
    X, y, groups = [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or row.get("label") is None:
                continue
            X.append([float(row[c]) for c in FEATURE_COLUMNS])
            y.append(row["label"].strip())
            groups.append(row.get("burst_id") or "unknown")
    return np.array(X), np.array(y, dtype=object), np.array(groups, dtype=object)


def dedupe(X, y, groups):
    seen, keep = set(), []
    for i in range(len(X)):
        key = X[i].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return X[keep], y[keep], groups[keep]


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


def per_class_table(y_true, y_pred, labels):
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {lab: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for lab, p, r, f, s in zip(labels, prec, rec, f1, support)}


def fit_eval(Xtr, ytr, Xte, yte, labels_for_percls, name):
    c = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    c.fit(Xtr, ytr)
    pred = c.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro", zero_division=0)
    ci = _bootstrap_test_ci(yte, pred)
    per_class = per_class_table(yte, pred, labels_for_percls)
    print(f"  [{name}] n_train={len(Xtr)} n_test={len(Xte)} acc={acc*100:.2f}% {ci['accuracy_95ci']} "
          f"macro-F1={f1m:.3f} {ci['macro_f1_95ci']}")
    return {"n_train": int(len(Xtr)), "n_test": int(len(Xte)), "accuracy": float(acc), "macro_f1": float(f1m),
            **ci, "per_class": per_class}


def run_four_settings(Xk, yk, gk, Xw, yw, gw, shared_classes, label):
    mk = np.isin(yk, shared_classes)
    mw = np.isin(yw, shared_classes)
    Xk, yk, gk = Xk[mk], yk[mk], gk[mk]
    Xw, yw, gw = Xw[mw], yw[mw], gw[mw]

    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    def indomain_split(Xd, yd, gd):
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            tr, te = next(gss.split(Xd, yd, groups=gd))
            return tr, te
        except ValueError:
            idx = np.arange(len(Xd))
            tr, te = train_test_split(idx, test_size=0.3, random_state=42, stratify=yd)
            return tr, te

    k_tr, k_te = indomain_split(Xk, yk, gk)
    w_tr, w_te = indomain_split(Xw, yw, gw)

    results = {}
    print(f"\n[{label}] shared classes: {shared_classes} ({len(shared_classes)})")
    print(f"  domain sizes: kaggle={len(Xk)} webcam={len(Xw)}")
    results["in_domain_kaggle"] = fit_eval(Xk[k_tr], yk[k_tr], Xk[k_te], yk[k_te], shared_classes, "in-domain Kaggle")
    results["in_domain_webcam"] = fit_eval(Xw[w_tr], yw[w_tr], Xw[w_te], yw[w_te], shared_classes, "in-domain Webcam")
    results["kaggle_to_webcam"] = fit_eval(Xk, yk, Xw, yw, shared_classes, "Kaggle -> Webcam")
    results["webcam_to_kaggle"] = fit_eval(Xw, yw, Xk, yk, shared_classes, "Webcam -> Kaggle")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-csv", default="isl_dataset_kaggle_controlled.csv")
    args = parser.parse_args()

    print("=" * 70)
    print("Loading ORIGINAL webcam rows from isl_dataset.csv, and the ORIGINAL")
    print("(confidence=0.5) vs CONTROLLED (confidence=0.7) Kaggle extractions")
    print("=" * 70)
    X_all, y_all, groups_all, signers_all, sessions_all, domains_all = load_dataset()
    X_all, y_all, groups_all, *_ , domains_all, _ = dedupe_exact(
        X_all, y_all, groups_all, signers_all, sessions_all, domains_all)
    webcam_mask = domains_all == "webcam"
    Xw, yw, gw = X_all[webcam_mask], y_all[webcam_mask], groups_all[webcam_mask]

    kaggle_mask = domains_all == "kaggle"
    Xk_orig, yk_orig, gk_orig = X_all[kaggle_mask], y_all[kaggle_mask], groups_all[kaggle_mask]

    Xk_ctrl_raw, yk_ctrl_raw, gk_ctrl_raw = load_controlled_kaggle(args.controlled_csv)
    Xk_ctrl, yk_ctrl, gk_ctrl = dedupe(Xk_ctrl_raw, yk_ctrl_raw, gk_ctrl_raw)
    print(f"Controlled Kaggle extraction: {len(Xk_ctrl_raw)} raw rows -> {len(Xk_ctrl)} after dedup "
          f"(original 0.5-confidence extraction: {len(Xk_orig)} rows)")

    shared_orig = sorted(c for c in SHARED_CLASSES_ORIGINAL if (yk_orig == c).sum() > 0 and (yw == c).sum() > 0)
    shared_ctrl = sorted(c for c in set(yk_ctrl) if (yk_ctrl == c).sum() > 0 and (yw == c).sum() > 0)

    out = {}
    out["original_0.5conf"] = run_four_settings(Xk_orig, yk_orig, gk_orig, Xw, yw, gw, shared_orig, "ORIGINAL (0.5 confidence)")
    out["controlled_0.7conf"] = run_four_settings(Xk_ctrl, yk_ctrl, gk_ctrl, Xw, yw, gw, shared_ctrl, "CONTROLLED (0.7 confidence, matched to live app)")

    print("\n" + "=" * 70)
    print("COMPARISON: does the gap survive after matching the confidence threshold?")
    print("=" * 70)
    comparison = {}
    for key in ["in_domain_kaggle", "in_domain_webcam", "kaggle_to_webcam", "webcam_to_kaggle"]:
        o, c = out["original_0.5conf"][key], out["controlled_0.7conf"][key]
        delta_acc = c["accuracy"] - o["accuracy"]
        delta_f1 = c["macro_f1"] - o["macro_f1"]
        print(f"  {key:20s} original acc={o['accuracy']*100:.2f}% -> controlled acc={c['accuracy']*100:.2f}% "
              f"(delta {delta_acc*100:+.2f}pp)   original F1={o['macro_f1']:.3f} -> controlled F1={c['macro_f1']:.3f} "
              f"(delta {delta_f1:+.3f})")
        comparison[key] = {"original_accuracy": o["accuracy"], "controlled_accuracy": c["accuracy"],
                            "delta_accuracy_pp": delta_acc * 100,
                            "original_macro_f1": o["macro_f1"], "controlled_macro_f1": c["macro_f1"],
                            "delta_macro_f1": delta_f1}
    out["comparison"] = comparison
    out["limitation_one_euro_filtering_not_controllable"] = (
        "One-Euro temporal filtering (webcam) vs. none (Kaggle, static photos) could NOT be "
        "controlled for -- raw unfiltered webcam video frames were never saved, only the final "
        "filtered feature vectors. Retroactively removing the filter's effect, or fabricating "
        "an 'unfiltered webcam' dataset, is not possible without new data collection. This "
        "asymmetry is partly inherent to comparing a photo dataset against a video-capture "
        "dataset and is reported as an unresolved limitation, not a fixed confound."
    )

    with open("results/controlled_cross_source_comparison.json", "w") as f:
        json.dump(out, f, indent=2)

    settings = [("in_domain_kaggle", "In-domain\nKaggle"), ("in_domain_webcam", "In-domain\nWebcam"),
                ("kaggle_to_webcam", "Kaggle\n→ Webcam"), ("webcam_to_kaggle", "Webcam\n→ Kaggle")]
    x = np.arange(len(settings)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    orig_acc = [out["original_0.5conf"][k]["accuracy"] for k, _ in settings]
    ctrl_acc = [out["controlled_0.7conf"][k]["accuracy"] for k, _ in settings]
    orig_err = np.array([[out["original_0.5conf"][k]["accuracy"] - out["original_0.5conf"][k]["accuracy_95ci"][0],
                           out["original_0.5conf"][k]["accuracy_95ci"][1] - out["original_0.5conf"][k]["accuracy"]]
                          for k, _ in settings]).T
    ctrl_err = np.array([[out["controlled_0.7conf"][k]["accuracy"] - out["controlled_0.7conf"][k]["accuracy_95ci"][0],
                           out["controlled_0.7conf"][k]["accuracy_95ci"][1] - out["controlled_0.7conf"][k]["accuracy"]]
                          for k, _ in settings]).T
    ax.bar(x - w / 2, orig_acc, width=w, color="#2a78d6", label="Original (conf=0.5)",
           yerr=orig_err, capsize=3, ecolor="#0b0b0b")
    ax.bar(x + w / 2, ctrl_acc, width=w, color="#eb6834", label="Controlled (conf=0.7, matched)",
           yerr=ctrl_err, capsize=3, ecolor="#0b0b0b")
    ax.set_xticks(x); ax.set_xticklabels([s[1] for s in settings])
    ax.set_ylim(0, 1.15); ax.set_ylabel("Accuracy")
    ax.set_title("Cross-source accuracy: original vs. confidence-matched (controlled) Kaggle extraction")
    ax.legend(frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/fig_controlled_vs_original.png", dpi=150)
    plt.close(fig)
    print("\nWritten: results/controlled_cross_source_comparison.json, results/fig_controlled_vs_original.png")


if __name__ == "__main__":
    main()
