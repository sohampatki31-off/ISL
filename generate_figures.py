#!/usr/bin/env python3
"""
generate_figures.py — Phase 8: publication figures, generated ONLY from
numbers already computed and saved by evaluate.py, analyze_domain_shift.py,
and build_invariant_features.py (results/all_results.json,
results/domain_shift_forensics.json, results/invariant_features_test.json).

No figure here re-derives or approximates a number independently of those
files -- if a number isn't in one of them, it doesn't appear in a figure.
Run evaluate.py -> analyze_domain_shift.py -> build_invariant_features.py
first (in that order) if any of the three JSON inputs is missing.

Color use follows the project's validated categorical palette (blue/orange/
aqua, colorblind-safe adjacent pairs) -- see the dataviz-skill palette
reference. Kaggle is always blue, Webcam is always orange, wherever both
appear, for visual consistency across figures.

Outputs -> results/:
  fig_pipeline_architecture.png
  fig_dataset_composition.png
  fig_leakage_demo.png
  fig_cross_source_key_finding.png
  fig_problematic_classes.png
  fig_feature_ablation.png
  fig_invariant_features_fix.png
  fig_handedness.png
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#c3c2b7"
plt.rcParams["axes.labelcolor"] = "#0b0b0b"
plt.rcParams["text.color"] = "#0b0b0b"
plt.rcParams["xtick.color"] = "#52514e"
plt.rcParams["ytick.color"] = "#52514e"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#e1e0d9"
plt.rcParams["grid.linewidth"] = 0.8

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK, MUTED = "#0b0b0b", "#898781"

with open("results/all_results.json") as f:
    R = json.load(f)
with open("results/domain_shift_forensics.json") as f:
    F = json.load(f)
with open("results/invariant_features_test.json") as f:
    IV = json.load(f)


def _no_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _bar_labels(ax, bars, fmt="{:.1f}%"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", fontsize=8, color=INK)


# ── 1. Pipeline architecture (schematic, not data-driven) ───────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, text, color, textsize=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                     linewidth=1.4, edgecolor=color, facecolor="#fcfcfb"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=textsize, color=INK)

    def arrow(x0, y0, x1, y1, color=MUTED):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                      mutation_scale=14, linewidth=1.3, color=color))

    box(0.3, 3.1, 2.0, 1.0, "Kaggle images\n(prathumarikeri ISL,\nCC BY-SA 4.0)", BLUE)
    box(0.3, 0.9, 2.0, 1.0, "Live webcam\n(Trainer UI,\nisl_main.py)", ORANGE)
    box(3.0, 3.1, 2.2, 1.0, "import_kaggle_dataset.py\n(MediaPipe, conf=0.5,\nno temporal filter)", BLUE)
    box(3.0, 0.9, 2.2, 1.0, "isl_main.py capture\n(MediaPipe, conf=0.7,\nOne-Euro filtered)", ORANGE)
    box(5.9, 2.0, 2.1, 1.0, "132-d feature\nextraction\n(wrist/scale-normalized\nlandmarks)", INK)
    box(8.5, 2.0, 2.1, 1.0, "isl_dataset.csv\n31,688 rows (dedup)\n31 classes", INK)
    box(8.5, 0.2, 2.1, 1.0, "RandomForest\n(train_model.py)\nisl_model.pkl", INK)

    arrow(2.3, 3.6, 3.0, 3.6); arrow(2.3, 1.4, 3.0, 1.4)
    arrow(5.2, 3.6, 5.9, 2.7); arrow(5.2, 1.4, 5.9, 2.3)
    arrow(8.0, 2.5, 8.5, 2.5)
    arrow(9.55, 2.0, 9.55, 1.2)

    ax.text(5.5, 4.6, "Two source pipelines merge into one feature schema, but with different\n"
                       "detection thresholds and temporal filtering (see docs/reproducibility_checklist.md)",
            ha="center", va="top", fontsize=8.5, color=MUTED, style="italic")
    fig.suptitle("Data + model pipeline architecture", fontsize=12, color=INK, y=0.98)
    fig.tight_layout()
    fig.savefig("results/fig_pipeline_architecture.png", dpi=150)
    plt.close(fig)


# ── 2. Dataset/source structure ─────────────────────────────────────────
def fig_dataset_composition():
    d = R["descriptive"]
    classes = sorted(d["class_counts_deduped"].keys())
    # domain split per class isn't stored per-class in all_results.json, so
    # this figure uses what IS stored: overall per-class counts (real) plus
    # the overall domain totals (real) as a companion panel -- no per-class
    # domain breakdown is fabricated.
    counts = [d["class_counts_deduped"][c] for c in classes]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    axes[0].bar(classes, counts, color=BLUE, width=0.7)
    axes[0].set_ylabel("Samples"); axes[0].set_title("Per-class sample count (deduplicated)")
    axes[0].tick_params(axis="x", rotation=60, labelsize=8)
    _no_top_right(axes[0])

    dom = d["domain_counts_deduped"]
    dom_order = sorted(dom.keys(), key=lambda k: 0 if k == "kaggle" else 1)  # kaggle=blue, webcam=orange, always
    dom_colors = {"kaggle": BLUE, "webcam": ORANGE}
    bars = axes[1].bar(dom_order, [dom[k] for k in dom_order], color=[dom_colors[k] for k in dom_order])
    axes[1].set_title("Total by source domain")
    _bar_labels(axes[1], bars, fmt="{:.0f}")
    _no_top_right(axes[1])
    fig.tight_layout()
    fig.savefig("results/fig_dataset_composition.png", dpi=150)
    plt.close(fig)


# ── 3. Leakage / burst-disjoint demonstration ───────────────────────────
def fig_leakage():
    lk = R["leakage_demo"]
    metrics = ["Accuracy", "Macro-F1"]
    naive = [lk["naive_random_split"]["accuracy"], lk["naive_random_split"]["macro_f1"]]
    group = [lk["burst_disjoint_split"]["accuracy"], lk["burst_disjoint_split"]["macro_f1"]]
    x = range(len(metrics)); w = 0.32
    fig, ax = plt.subplots(figsize=(6, 4.5))
    b1 = ax.bar([i - w / 2 for i in x], naive, width=w, color=BLUE, label="Naive random split")
    b2 = ax.bar([i + w / 2 for i in x], group, width=w, color=ORANGE,
                label=f"Burst-disjoint split ({lk['burst_disjoint_split']['n_classes_evaluated']} classes)")
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.3); ax.set_ylabel("Score (0-1)")
    ax.set_title("burst_id as a leakage control (post-dedup)")
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    _bar_labels(ax, b1, fmt="{:.3f}"); _bar_labels(ax, b2, fmt="{:.3f}")
    _no_top_right(ax)
    fig.tight_layout()
    fig.savefig("results/fig_leakage_demo.png", dpi=150)
    plt.close(fig)


# ── 4. Cross-source performance -- THE key finding ──────────────────────
def fig_cross_source():
    cs = R["cross_source_evaluation"]
    settings = [("in_domain_kaggle", "In-domain\nKaggle"), ("in_domain_webcam", "In-domain\nWebcam"),
                ("kaggle_to_webcam", "Kaggle\n→ Webcam"), ("webcam_to_kaggle", "Webcam\n→ Kaggle")]
    labels = [s[1] for s in settings]
    acc = [cs[s[0]]["accuracy"] for s in settings]
    f1 = [cs[s[0]]["macro_f1"] for s in settings]
    acc_err = [[cs[s[0]]["accuracy"] - cs[s[0]]["accuracy_95ci"][0],
                cs[s[0]]["accuracy_95ci"][1] - cs[s[0]]["accuracy"]] for s in settings]
    f1_err = [[cs[s[0]]["macro_f1"] - cs[s[0]]["macro_f1_95ci"][0],
               cs[s[0]]["macro_f1_95ci"][1] - cs[s[0]]["macro_f1"]] for s in settings]
    acc_err = np.array(acc_err).T; f1_err = np.array(f1_err).T
    x = range(len(settings)); w = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar([i - w / 2 for i in x], acc, width=w, color=BLUE, label="Accuracy",
                yerr=acc_err, capsize=3, ecolor=INK, error_kw={"linewidth": 1})
    b2 = ax.bar([i + w / 2 for i in x], f1, width=w, color=ORANGE, label="Macro-F1",
                yerr=f1_err, capsize=3, ecolor=INK, error_kw={"linewidth": 1})
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score (0-1)")
    ax.set_title(f"Cross-source degradation ({len(cs['shared_classes'])} shared letter classes) — key finding")
    ax.legend(frameon=False, fontsize=9)
    _bar_labels(ax, b1, fmt="{:.2f}"); _bar_labels(ax, b2, fmt="{:.2f}")
    _no_top_right(ax)
    ns = "  ".join(f"{s[1].replace(chr(10),' ')}: n_test={cs[s[0]]['n_test']}" for s in settings)
    ax.text(0.5, -0.28, ns, transform=ax.transAxes, ha="center", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    fig.savefig("results/fig_cross_source_key_finding.png", dpi=150)
    plt.close(fig)


# ── 5. Problematic-class forensic analysis ──────────────────────────────
def fig_problematic_classes():
    per_class = F["h4_kaggle_to_webcam_per_class"]
    items = sorted(per_class.items(), key=lambda kv: kv[1]["recall"])
    labels = [k for k, v in items]
    recalls = [v["recall"] * 100 for k, v in items]
    ns = [v["n"] for k, v in items]
    colors = [RED if r == 0 else BLUE for r in recalls]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, recalls, color=colors)
    for b, r, n in zip(bars, recalls, ns):
        ax.annotate(f"{r:.0f}%\n(n={n})", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center", va="bottom",
                    fontsize=7.5, color=INK, linespacing=1.3)
    ax.set_ylabel("Recall on webcam test set (%)")
    ax.set_title("Kaggle-trained model, per-class recall on webcam (shared classes)")
    ax.set_ylim(0, 122)
    handles = [plt.Rectangle((0, 0), 1, 1, color=RED), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax.legend(handles, ["0% recall (deterministic misclassification)", "Partial/full recall"],
              frameon=False, fontsize=8, loc="upper center")
    _no_top_right(ax)
    fig.tight_layout()
    fig.savefig("results/fig_problematic_classes.png", dpi=150)
    plt.close(fig)


# ── 6. Feature ablation ─────────────────────────────────────────────────
def fig_feature_ablation():
    fa = R["feature_ablation"]
    names = list(fa.keys())
    acc = [fa[n]["accuracy"] for n in names]
    f1 = [fa[n]["macro_f1"] for n in names]
    x = range(len(names)); w = 0.32
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar([i - w / 2 for i in x], acc, width=w, color=BLUE, label="Accuracy")
    b2 = ax.bar([i + w / 2 for i in x], f1, width=w, color=ORANGE, label="Macro-F1")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{n}\n({fa[n]['n_features']} feat)" for n in names], fontsize=8.5)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Held-out score (0-1)")
    ax.set_title("Feature ablation (held-out, mixed-domain split)")
    ax.legend(frameon=False, fontsize=9)
    _bar_labels(ax, b1, fmt="{:.3f}"); _bar_labels(ax, b2, fmt="{:.3f}")
    _no_top_right(ax)
    fig.tight_layout()
    fig.savefig("results/fig_feature_ablation.png", dpi=150)
    plt.close(fig)


# ── 7. Invariant-features candidate fix (Phase 5) ───────────────────────
def fig_invariant_fix():
    reps = [("raw_132", "Raw (132-d)"), ("invariant_41", "Rotation-invariant"),
            ("handslot_agnostic_41", "Handslot-agnostic")]
    directions = [("kaggle_to_webcam_macro_f1", "kaggle_to_webcam_ci", "Kaggle→Webcam"),
                  ("webcam_to_kaggle_macro_f1", "webcam_to_kaggle_ci", "Webcam→Kaggle")]
    x = range(len(directions)); w = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = [BLUE, ORANGE, AQUA]
    for i, (rep_key, rep_label) in enumerate(reps):
        vals = [IV[rep_key][d_key] for d_key, _, _ in directions]
        err = np.array([[IV[rep_key][d_key] - IV[rep_key][ci_key]["macro_f1_95ci"][0],
                          IV[rep_key][ci_key]["macro_f1_95ci"][1] - IV[rep_key][d_key]]
                         for d_key, ci_key, _ in directions]).T
        offset = (i - 1) * w
        bars = ax.bar([xi + offset for xi in x], vals, width=w, color=colors[i], label=rep_label,
                       yerr=err, capsize=3, ecolor=INK, error_kw={"linewidth": 1})
        _bar_labels(ax, bars, fmt="{:.3f}")
    ax.set_xticks(list(x)); ax.set_xticklabels([d[2] for d in directions])
    ax.set_ylim(0, max(IV[r][d] for r, _ in reps for d, _, _ in directions) + 0.2)
    ax.set_ylabel("Macro-F1 (cross-domain)")
    ax.set_title("Candidate fix: hand-slot-agnostic features (Phase 5, 95% CI bootstrapped)")
    ax.legend(frameon=False, fontsize=8.5)
    _no_top_right(ax)
    fig.tight_layout()
    fig.savefig("results/fig_invariant_features_fix.png", dpi=150)
    plt.close(fig)


# ── 8. Handedness robustness ────────────────────────────────────────────
def fig_handedness():
    ha = R["handedness_analysis"]["held_out_breakdown_by_test_subset_handedness"]
    order = [k for k in ["both", "right_only", "left_only"] if k in ha]
    acc = [ha[k]["accuracy"] * 100 for k in order]
    ns = [ha[k]["n_test"] for k in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(order, acc, color=[BLUE, ORANGE, AQUA][:len(order)])
    for b, n in zip(bars, ns):
        ax.annotate(f"n={n}", (b.get_x() + b.get_width() / 2, 4), ha="center", fontsize=8, color="white")
    _bar_labels(ax, bars, fmt="{:.2f}%")
    ax.set_ylim(0, 108); ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title("Accuracy by detected handedness\n(single mixed-training model, held-out test subsets)", fontsize=11)
    _no_top_right(ax)
    fig.tight_layout()
    fig.savefig("results/fig_handedness.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_dataset_composition()
    fig_leakage()
    fig_cross_source()
    fig_problematic_classes()
    fig_feature_ablation()
    fig_invariant_fix()
    fig_handedness()
    print("Wrote 8 figures to results/")
