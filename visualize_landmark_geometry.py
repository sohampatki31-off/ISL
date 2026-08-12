#!/usr/bin/env python3
"""
visualize_landmark_geometry.py — CVIP2026-v2 final-validation addition.

Every forensic test so far (H1-H9 in analyze_domain_shift.py, the
confidence-controlled re-extraction, the hand-swap test) is purely
quantitative. This script adds the one qualitative check the audit was
missing: what does the actual landmark GEOMETRY look like for the classes
that fail completely (A, E, F, K) vs. a class that works (G), Kaggle vs.
webcam, side by side?

This is reconstructed directly from isl_dataset.csv's stored feature
vectors (wrist-relative, scale-normalized per hand) -- no original images
are needed or used, so this IS reproducible from the repo alone, unlike
the manual image-resolution inspection (H7). Only x/y are plotted (z drives
point color as a depth cue); this is exactly the same normalized space the
classifier itself sees, so if Kaggle and webcam samples of the same class
look visually similar here but the model still fails cross-domain, that's
evidence AGAINST a simple geometric mismatch and FOR something the
landmark representation itself doesn't capture (framing/resolution/
pre-processing artifacts upstream of the landmarks). If they look
different, that IS visual evidence of genuine landmark-geometry shift.

Outputs -> results/fig_landmark_geometry_comparison.png
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import load_dataset, dedupe_exact

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

CLASSES_TO_SHOW = ["A", "E", "F", "K", "G"]  # first 4 = unexplained failures, G = working control
N_SAMPLES = 3


def plot_hand(ax, block63, color, label=None):
    """block63: (63,) wrist-relative, scale-normalized right- or left-hand
    columns from FEATURE_COLUMNS. All-zero means hand absent."""
    if np.abs(block63).sum() < 1e-9:
        ax.text(0.5, 0.5, "(hand absent)", ha="center", va="center", fontsize=7, color="#898781",
                transform=ax.transAxes)
        return
    pts = block63.reshape(21, 3)
    for s, e in HAND_CONNECTIONS:
        ax.plot([pts[s, 0], pts[e, 0]], [-pts[s, 1], -pts[e, 1]], color=color, linewidth=1.2, alpha=0.85)
    ax.scatter(pts[:, 0], -pts[:, 1], c=color, s=10, zorder=3)
    if label:
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=6, va="top", color="#52514e")


def main():
    X_raw, y_raw, groups_raw, signers_raw, sessions_raw, domains_raw = load_dataset()
    X, y, groups, signers, sessions, domains, _ = dedupe_exact(
        X_raw, y_raw, groups_raw, signers_raw, sessions_raw, domains_raw)

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(len(CLASSES_TO_SHOW), 2 * N_SAMPLES, figsize=(2.1 * 2 * N_SAMPLES, 2.3 * len(CLASSES_TO_SHOW)))

    for row_i, cls in enumerate(CLASSES_TO_SHOW):
        for dom_i, dom in enumerate(["kaggle", "webcam"]):
            idx = np.where((y == cls) & (domains == dom))[0]
            chosen = rng.choice(idx, size=min(N_SAMPLES, len(idx)), replace=False) if len(idx) > 0 else []
            for s_i in range(N_SAMPLES):
                ax = axes[row_i, dom_i * N_SAMPLES + s_i]
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if s_i < len(chosen):
                    row = X[chosen[s_i]]
                    plot_hand(ax, row[0:63], "#2a78d6")   # right hand, blue
                    plot_hand(ax, row[63:126], "#eb6834")  # left hand, orange
                    ax.set_aspect("equal")
                else:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7, color="#c3c2b7",
                            transform=ax.transAxes)
                if s_i == 0:
                    ax.set_ylabel(f"{cls}\n{dom}", fontsize=9, rotation=0, labelpad=28, va="center")
                if row_i == 0:
                    ax.set_title(f"sample {s_i+1}", fontsize=8)

    fig.suptitle("Reconstructed hand landmark geometry (blue=right, orange=left), Kaggle vs. Webcam\n"
                  "A/E/F/K = unexplained 0%-recall cross-domain failures; G = working control class",
                  fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])
    fig.savefig("results/fig_landmark_geometry_comparison.png", dpi=150)
    print("Written: results/fig_landmark_geometry_comparison.png")


if __name__ == "__main__":
    main()
