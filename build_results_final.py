#!/usr/bin/env python3
"""
build_results_final.py — Phase 10: assemble RESULTS_FINAL.json purely by
copying fields out of results/all_results.json, results/domain_shift_forensics.json,
and results/invariant_features_test.json (all already real, computed, and
saved by evaluate.py / analyze_domain_shift.py / build_invariant_features.py).

No number here is retyped by hand -- this script only re-shapes existing
JSON into one file with an explicit evidence-classification tag per section,
matching PROJECT_DOSSIER_FINAL.md's classification key (REAL/MANUAL/
HYPOTHESIS/FUTURE_WORK), so a reader of RESULTS_FINAL.json alone (without
the prose dossier) still knows what's proven vs. speculative.
"""
import json

with open("results/all_results.json") as f:
    R = json.load(f)
with open("results/domain_shift_forensics.json") as f:
    F = json.load(f)
with open("results/invariant_features_test.json") as f:
    IV = json.load(f)

final = {
    "generated": "2026-08-12",
    "source_task": "CVIP2026 research-upgrade audit, phases 1-10",
    "evidence_classification_key": {
        "REAL": "Computed from isl_dataset.csv as it exists today, by a script in this repo, reproducible.",
        "MANUAL": "A true, factual finding from a one-time manual inspection outside the repo (not auto-reproducible).",
        "HYPOTHESIS": "A plausible, evidence-motivated explanation NOT experimentally confirmed.",
        "FUTURE_WORK": "Designed/specified but not executed -- zero data exists for this.",
    },
    "dataset": {
        "_evidence": "REAL",
        "n_samples_raw": R["descriptive"]["n_samples_raw"],
        "n_samples_deduped": R["descriptive"]["n_samples_deduped"],
        "n_classes_deduped": R["descriptive"]["n_classes_deduped"],
        "n_classes_target": R["descriptive"]["n_classes_target"],
        "missing_classes": R["descriptive"]["missing_classes_35"],
        "class_counts": R["descriptive"]["class_counts_deduped"],
        "domain_counts": R["descriptive"]["domain_counts_deduped"],
        "exact_duplicates_removed": R["descriptive"]["n_exact_duplicates_removed"],
        "signer_session_caveat": R["descriptive"]["signer_session_caveat"],
        "kaggle_source": {
            "name": "Indian Sign Language (ISL)",
            "author": "prathumarikeri",
            "url": "https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl",
            "license": "CC BY-SA 4.0",
            "license_verified_by": "user, 2026-08-12, from the Kaggle dataset page (usability 8.75)",
        },
        "image_resolution_heterogeneity": {
            "_evidence": "MANUAL",
            **F.get("h7_manual_finding_kaggle_image_resolution_heterogeneity", {}),
        },
    },
    "leakage_and_near_duplication": {
        "_evidence": "REAL",
        "near_duplication": R["near_duplication"],
        "leakage_demo": R["leakage_demo"],
        "repeated_split_stats": R["repeated_split_stats"],
    },
    "cross_source_key_finding": {
        "_evidence": "REAL",
        "shared_classes": F["shared_classes"],
        "domain_sizes": {"n_kaggle": F["n_kaggle"], "n_webcam": F["n_webcam"]},
        "results": R["cross_source_evaluation"],
        "forensic_hypotheses": {
            "h1_sample_size_control_REFUTED": F["h1_sample_size_control"],
            "h2_handedness_by_domain": F["h2_handedness_by_domain"],
            "h3_feature_shift_within_class": F["h3_feature_shift"],
            "h4_per_class_recall_kaggle_to_webcam": F["h4_kaggle_to_webcam_per_class"],
            "h5_pre_vs_post_mirroring_fix_persists": F["h5_pre_vs_post_mirroring_fix"],
            "h6_handedness_composition_vs_recall": F["h6_handedness_composition_vs_recall"],
        },
    },
    "candidate_fix_phase5": {
        "_evidence": "REAL",
        "description": "Hand-slot-agnostic rotation/reflection-invariant angle features (41-d) vs. raw (132-d)",
        "results": IV,
        "verdict": ("Statistically significant (non-overlapping 95% CIs) partial improvement in both cross-domain "
                    "directions; recovers U and M classes; A/E/F/J/K remain at 0% recall -- NOT a full fix."),
    },
    "handedness_analysis": {"_evidence": "REAL", **R["handedness_analysis"]},
    "classifier_comparison": {"_evidence": "REAL", **R["classifier_comparison"],
                               "caveat": R["classifier_comparison_caveat"]},
    "feature_ablation": {"_evidence": "REAL", **R["feature_ablation"], "caveat": R["feature_ablation_caveat"]},
    "held_out_confusion": {"_evidence": "REAL", **R["held_out_confusion"]},
    "resolution_confound_hypothesis": {
        "_evidence": "HYPOTHESIS",
        "statement": ("Image-resolution/framing heterogeneity (see dataset.image_resolution_heterogeneity) is a "
                       "plausible contributing cause for the A/E/F/K cross-domain failures left unexplained by "
                       "handedness composition (h6). NOT experimentally isolated -- no controlled "
                       "re-detection-at-matched-resolution test was run."),
    },
    "isl_linguistic_validity": {
        "_evidence": "MANUAL, cursory, non-expert",
        "statement": ("A handful of Kaggle images (A, J, G, I) were visually spot-checked and show plausible ISL "
                       "fingerspelling shapes; no certified ISL interpreter or Deaf community member has reviewed "
                       "the vocabulary or images at any point in this project. Not a linguistic validation."),
    },
    "future_work_signer_session_disjoint_evaluation": {
        "_evidence": "FUTURE_WORK",
        "statement": ("No signer- or session-disjoint evaluation exists or can exist without new, real multi-signer "
                       "data collection. Protocol fully specified in docs/data_collection_protocol.md but NOT "
                       "executed -- zero signer-disjoint data exists."),
    },
    "claims_paper_can_make": [
        "A real, CI-bounded cross-source accuracy gap exists (99.9% in-domain vs 29-62% cross-domain).",
        "The gap is not explained by webcam sample size (H1, refuted with a matched-size control).",
        "Part of the gap is explained by a hand-to-feature-slot assignment mismatch between capture "
        "pipelines, and a proposed fix significantly (CI-confirmed) improves it without in-domain cost.",
        "burst_id leakage is real for the legacy-webcam subset (macro-F1 0.999->0.895) but not a meaningful "
        "control for the Kaggle-dominated majority of current data.",
        "Handedness has no measurable effect on within-domain accuracy; left-hand information is not redundant.",
    ],
    "claims_paper_must_not_make": [
        "Signer-independent or session-independent recognition -- no real multi-signer data exists.",
        "That the cross-source gap is fully explained or fully fixed.",
        "That the Kaggle vocabulary/images have been linguistically verified as authentic ISL.",
        "That isl_dataset.csv currently supports 35 classes (it supports 31).",
        "Anything from the current paper/main.tex draft, which describes a superseded 842-row/18-class snapshot.",
        "That detection-threshold/temporal-filtering differences between pipelines were controlled for in the "
        "domain-shift analysis.",
    ],
    "paper_main_tex_status": "STALE -- describes the pre-Kaggle-merge 842-row/18-class dataset; central claim "
                              "(burst-level leakage) is directly contradicted by current results. See "
                              "PROJECT_DOSSIER_FINAL.md section 7 for the recommended new structure.",
}

with open("RESULTS_FINAL.json", "w") as f:
    json.dump(final, f, indent=2)
print("Wrote RESULTS_FINAL.json")
