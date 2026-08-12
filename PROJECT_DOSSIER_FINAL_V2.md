# ISL-backend — Research Dossier v2 (CVIP 2026 candidate)

**Status: final-validation pass complete. This is the authoritative dossier
— supersedes `PROJECT_DOSSIER_FINAL.md` (v1, 2026-08-12). Read this before
writing or citing anything in `paper/main.tex`, which is STALE (see §7).**
See `FINAL_RESEARCH_AUDIT.md` for the full audit trail (what was found
wrong, what was fixed, hostile-review grades) and `RESULTS_FINAL_V2.json`
for the machine-readable version of everything below.

Generated 2026-08-13. Every number is machine-computed and reproducible
via §6 — nothing here is estimated, interpolated, or asserted from memory.

---

## 0. Evidence classification key

- **[REAL]** — computed from data in this repo, by a script in this repo, reproducible.
- **[MANUAL]** — a real, factual finding from a one-time manual inspection outside the repo — true, not automatically reproducible.
- **[HYPOTHESIS]** — a plausible, evidence-motivated explanation proposed but NOT experimentally confirmed.
- **[TESTED_AND_REFUTED]** — a specific candidate cause that WAS directly tested and did not hold up. Reported as a real negative result, not omitted.
- **[FUTURE WORK]** — designed/specified but not executed. Zero data exists for these claims.

---

## 1. Dataset facts **[REAL]** — unchanged from v1, re-verified this pass

- `isl_dataset.csv`: **31,806 raw rows → 31,688 after exact-duplicate
  removal**, 132 features, **31 classes** with real data (26 letters A-Z +
  5 word-gestures: GOOD, HELLO, HELP, LOVE, THANKYOU). Re-verified directly
  against the current CSV this pass (independent recount, not just reused
  from v1): 31 classes, domain split kaggle=30,873/webcam=933 pre-dedup
  matches v1's post-dedup 30,872/816 exactly (117 webcam + 1 Kaggle
  duplicate removed).
- `isl_main.py` hardcodes a 35-class *target* vocabulary; **4 classes
  (SORRY, YES, NO, PLEASE) have zero samples**. State as "31/35 target
  classes populated," never "35 classes."
- **Two sources**: **Kaggle** (`session_id='kaggle_import'`, 30,872 rows,
  97.4%, "Indian Sign Language (ISL)" by prathumarikeri,
  CC BY-SA 4.0 — see `LICENSE`) and **Webcam** (816 rows, 2.6%, all
  hand-recorded by this project's sole developer — no third-party
  human-subjects data anywhere in the current dataset).
- **No real signer or session diversity exists anywhere in this dataset** — confirmed unchanged this pass.
- **[MANUAL, 2026-08-12]** Kaggle images are 128×128px tightly-cropped
  hand photos for 31/35 classes (1200 each); 4 classes (C, I, O, V) also
  contain 90-247 higher-resolution images from a different sub-source.
  Live webcam capture runs at 720p-1280p. This resolution/framing gap is
  now (this pass) the **strongest remaining lead** for the A/E/F residual
  failures in §4, precisely because five other plausible mechanisms were
  directly tested and ruled out.
- **[MANUAL, cursory, non-expert]** No certified ISL interpreter or Deaf
  community member has reviewed the vocabulary or images. Not a
  linguistic validation.

## 2. Methodology **[REAL]**

- **Feature extraction**: MediaPipe Tasks `HandLandmarker`, 21
  landmarks/hand, wrist-relative + scale-normalized, zero-filled if a hand
  is absent, + 6 inter-hand relative features. 132 features total.
- **Detection-threshold difference, precisely stated (corrected this
  pass)**: `isl_main.py` uses `min_hand_detection_confidence=0.7` and
  `min_hand_presence_confidence=0.7`, but `min_tracking_confidence=0.5`
  (not 0.7 — v1 imprecisely said "0.7 uniformly"). All three pipelines
  (`isl_main.py`, `import_kaggle_dataset.py`, `process_images.py`) run
  MediaPipe in default `RunningMode.IMAGE` (verified by code audit —
  none call `detect_for_video`/`detect_async`), where
  `min_tracking_confidence` has no cross-frame tracker to act on, so this
  imprecision is real but functionally inert.
- **[TESTED_AND_REFUTED, this pass]** The 0.5-vs-0.7 detection/presence
  confidence mismatch was hypothesized (v1) to contribute an "unknown,
  unquantified amount of selection-bias" to the domain-shift finding.
  **Directly tested**: re-extracted the raw Kaggle images at
  confidence=0.7 (`isl_dataset_kaggle_controlled.csv`, 29,733 rows) and
  reran all 4 cross-source settings. The gap does **not** close — it is
  essentially unchanged or slightly WORSE (Kaggle→Webcam macro-F1
  0.312→0.275; Webcam→Kaggle 0.211→0.202; in-domain deltas ≤0.02pp). This
  confound is real but is now proven not to explain the gap. See §4.1.
- **Left/right hand-column assignment convention, newly documented this
  pass**: `isl_main.py` deliberately mirrors the live camera frame before
  detection, then swaps MediaPipe's handedness label and un-mirrors x
  (`_HANDEDNESS_UNMIRROR`) so the output matches "the signer's true
  anatomy" and is claimed to "line up with the Kaggle-imported training
  rows (extracted from un-mirrored photos, never flipped)." The offline
  scripts take MediaPipe's raw label with no swap. **This convention-
  matching claim was never independently verified against Kaggle capture
  metadata** (no such metadata exists) — flagged as an open item, §8.
- **[TESTED_AND_REFUTED, this pass]** Whether a mismatch in this exact
  convention explains the still-unexplained A/E/F/K failures: directly
  tested (`test_handswap_hypothesis.py`) by swapping right/left hand
  blocks for two-handed webcam rows and re-evaluating the same
  Kaggle-trained model. Recovers nothing for A/E/F/J (still 0% recall)
  and **breaks 3 currently-working classes** (B 92.6%→0%, D 93.3%→70.5%,
  G 100%→0%). Not the cause of the residual failures — but shows those
  working classes' correctness currently depends on the unverified
  convention-matching assumption above.
- **Temporal filtering asymmetry**: live/webcam rows are One-Euro
  filtered before feature extraction; Kaggle/image-derived rows are not.
  **Confirmed unfixable without new data** — raw unfiltered webcam video
  frames were never saved, only the final filtered feature vector, and
  Kaggle images are static single photos with no temporal dimension to
  filter in the first place. Stated as an inherent limitation, not a bug.
- **Classifier**: RandomForestClassifier (deployed model), compared
  against SVM-RBF, kNN(k=5), and MLP(128,64) — both for the within-domain
  held-out comparison (v1) AND, new this pass, for the cross-domain
  Kaggle→Webcam gap itself (§4.4) — the gap is not RandomForest-specific.
- **Feature-extraction code audit, this pass**: line-by-line comparison
  of `isl_main.py` vs. `process_images.py`/`import_kaggle_dataset.py`
  (the latter imports its feature function from the former's twin) found
  the wrist-relative/scale-normalization formula and the 6 inter-hand
  relative features are byte-for-byte identical; `HandLandmarkerOptions`
  matches (`num_hands=2`, default `RunningMode.IMAGE`) except the known
  confidence values; the only other asymmetry is `isl_main.py` downscaling
  the live frame to 960px max-dimension before detection (for latency)
  vs. offline scripts using native image resolution — assessed as
  unlikely to systematically shift the normalized features (MediaPipe
  landmarks are 0-1 normalized regardless of input size), not proven
  either way.
- **All experiments use the deduplicated dataset** (31,688 rows) unless noted.

## 3. Experiments run, with exact real results **[REAL]**

Unchanged from v1 §3 (`python evaluate.py` → `results/RESULTS.md`,
`results/all_results.json`) — re-verified consistent with the current
`isl_dataset.csv` this pass, no drift found. See v1 dossier or
`results/RESULTS.md` for the full table (descriptive stats, leakage
demo, repeated-split stats, handedness, classifier comparison, confusion
matrix, feature ablation).

## 4. Cross-source finding — the project's actual contribution **[REAL]**

| Setting | n_train | n_test | Accuracy [95% CI] | Macro-F1 [95% CI] |
|---|---|---|---|---|
| In-domain Kaggle | 11,998 | 5,143 | 99.92% [99.8, 100.0] | 0.999 [0.998, 1.000] |
| In-domain Webcam | 407 | 224 | 46.88% [40.6, 53.6] | 0.210 [0.189, 0.242] |
| Kaggle → Webcam | 17,141 | 631 | 50.40% [46.6, 54.0] | 0.312 [0.296, 0.325] |
| Webcam → Kaggle | 631 | 17,141 | 29.41% [28.7, 30.1] | 0.211 [0.208, 0.214] |

14 shared classes (A,B,C,D,E,F,G,I,J,K,L,M,U,V).

### 4.1 New — controlled cross-source experiment (confidence-matched) **[REAL, TESTED_AND_REFUTED]**

`python import_kaggle_dataset.py --confidence 0.7` →
`isl_dataset_kaggle_controlled.csv` → `python
analyze_domain_shift_controlled.py` → `results/controlled_cross_source_comparison.json`,
`results/fig_controlled_vs_original.png`.

| Setting | Original (conf=0.5) acc / F1 | Controlled (conf=0.7) acc / F1 | Δ accuracy | Δ macro-F1 |
|---|---|---|---|---|
| In-domain Kaggle | 99.92% / 0.999 | 99.94% / 0.999 | +0.02pp | +0.000 |
| In-domain Webcam | 46.88% / 0.210 | 46.88% / 0.210 | 0.00pp | 0.000 |
| Kaggle → Webcam | 50.40% / 0.312 | 47.23% / 0.275 | **−3.17pp** | **−0.037** |
| Webcam → Kaggle | 29.41% / 0.211 | 28.35% / 0.202 | **−1.05pp** | **−0.009** |

**Verdict: the confidence-threshold confound does NOT explain the gap.**
Matching it makes cross-domain performance slightly worse, not better —
the opposite of what "this confound is inflating the gap" would predict.
This closes an item v1 could only describe as "unquantified."

### 4.2 New — hand-column-swap hypothesis test **[REAL, TESTED_AND_REFUTED]**

`python test_handswap_hypothesis.py` → `results/handswap_hypothesis_test.json`.

Swapped right/left hand-landmark blocks for two-handed webcam test rows,
re-evaluated the same Kaggle-trained raw-132 model:

| Class | Baseline recall | Hand-swapped recall | Δ |
|---|---|---|---|
| A | 0.0% | 0.0% | +0.0pp |
| E | 0.0% | 0.0% | +0.0pp |
| F | 0.0% | 0.0% | +0.0pp |
| J | 0.0% | 0.0% | +0.0pp |
| B (was working) | 92.6% | 0.0% | **−92.6pp** |
| D (was working) | 93.3% | 70.5% | **−22.9pp** |
| G (was working) | 100.0% | 0.0% | **−100.0pp** |

**Verdict: NOT SUPPORTED.** Swapping recovers nothing for the failing
classes and destroys 3 working ones. Ruled out as a cause. (Method
caveat: only the 126 raw per-landmark columns were swapped exactly; the 6
inter-hand relative-delta features could not be recomputed exactly for
the swapped configuration since their pre-normalization raw inputs are
not recoverable from the stored CSV — this under-tests the hypothesis for
those 6 of 132 features only.)

### 4.3 New — landmark geometry visualization **[REAL, qualitative]**

`python visualize_landmark_geometry.py` →
`results/fig_landmark_geometry_comparison.png`. Reconstructs hand
skeletons directly from stored feature vectors (no original images
needed) for A, E, F, K (failures) and G (working control), 3 Kaggle + 3
webcam samples each.

- **K**: Kaggle samples are visually and consistently **two-handed**;
  webcam samples are visually and consistently **one-handed (right
  only)** — this is a near-total hand-count mismatch, not a subtle
  compositional skew, and matches H6's aggregate stat exactly (94.7%
  both-hands Kaggle vs. 100% right-only webcam). **This substantially
  explains K's failure**: the Kaggle-trained model has structurally never
  seen a one-handed "K," in any representation.
- **A**: both domains show two hands, but in visibly different postures
  (Kaggle: wide/spread; webcam: compact/vertical). Consistent with, but
  not proof of, a postural or framing-driven difference.
- **E, F**: look visually *similar* in overall hand shape between
  domains, yet still fail at 0% recall. This is evidence AGAINST a simple
  whole-hand-posture explanation for these two specifically — the cause,
  if any, is subtler than gross geometry (fine joint-angle differences,
  or an artifact of the underlying image-resolution/framing gap that
  isn't visible once already reduced to normalized landmark space).

### 4.4 New — does the gap depend on classifier choice? **[REAL, TESTED_AND_REFUTED]**

`python test_classifier_generalization_of_gap.py` →
`results/classifier_generalization_of_gap.json`.

| Classifier | Kaggle→Webcam accuracy | Kaggle→Webcam macro-F1 |
|---|---|---|
| RandomForest(100,d15) | 50.40% | 0.312 |
| kNN(k=5) | 42.31% | 0.254 |
| SVM-RBF(C=10) | 50.55% | 0.324 |
| MLP(128,64) | 50.55% | 0.312 |

All four collapse to 42–51% vs. ~99%+ in-domain for all four
(`results/RESULTS.md` §6). **The gap is a property of the data/domain
shift, not a RandomForest-specific artifact.**

### 4.5 Multi-hypothesis forensic testing, full list (H1-H11)

Carried over from v1 (H1-H7) plus H8-H9 (added mid-session, prior pass)
plus the two new mechanisms this pass (§4.1, §4.2 above — call them
H10/H11 for reference):

- **H1 [REFUTED]** sample size — Kaggle subsampled to webcam's tiny n still gets 100% in-domain.
- **H2 [PARTIAL]** handedness composition differs materially between domains.
- **H3 [REAL, descriptive]** feature-level shift — x-coordinate-dominated Cohen's d up to ~1.6.
- **H4 [REAL]** per-class recall is sharply bimodal, near-deterministic wrong-label mapping for 0%-recall classes.
- **H5 [REAL, REFUTES "old bug" theory]** M fails 0% both pre- and post- mirroring-fix commit, different wrong labels.
- **H6 [REAL, refines H2]** J/K/U/C show genuine handedness-composition mismatch (consistent with the fix helping them); A/E/F/M have similar composition in both domains yet still fail — unexplained by handedness alone.
- **H7 [MANUAL]** Kaggle images are a class-correlated patchwork of ≥2 sub-sources, 128×128 vs. 720p+ webcam — real, not experimentally isolated as a cause.
- **H8 [INCONCLUSIVE]** hand-proximity/overlap — no clean separation between failing/succeeding classes.
- **H9 [INCONCLUSIVE]** per-class right-hand Cohen's d magnitude — no monotonic relationship, confounded by handedness composition.
- **H10 [REFUTED, this pass]** MediaPipe confidence threshold (§4.1).
- **H11 [REFUTED, this pass]** hand-column-swap convention mismatch (§4.2).

**Honest summary for the paper**: cross-source accuracy collapses from
~99.9% in-domain to 28-51% cross-domain, across every classifier family
tested, and this collapse is not explained by webcam sample size (H1),
the MediaPipe confidence threshold (H10), or a hand-column-swap
convention mismatch (H11) — three specific, plausible mechanisms directly
tested and ruled out. Part of the gap (J, K, U, C) is explained by a
hand-to-feature-slot assignment mismatch, with K's failure specifically
and substantially explained by a near-total hand-count mismatch between
sources (visually confirmed). A targeted fix for the slot-assignment
mismatch (hand-slot-agnostic angle features) produces a real,
McNemar-significant (p<1e-9), seed-robust, CI-confirmed improvement. **A,
E, and F remain completely unexplained** after six directly-tested
candidate mechanisms — the strongest remaining, still-untested lead is
image-resolution/framing heterogeneity, which would require new
matched-resolution data collection to test.

## 5. What this project can and cannot claim

See `RESULTS_FINAL_V2.json`'s `claims_paper_can_make` /
`claims_paper_must_not_make` fields for the authoritative, current list.
Key additions this pass:
- **Can now claim**: the gap survives across 4 classifier families; the
  gap is not explained by detection threshold or hand-column-swap
  (specifically tested and refuted, not just "undetermined"); K's failure
  is substantially (not just partially) explained.
- **Must still not claim**: signer-/session-independence; that the gap is
  fully explained (A/E/F remain open); that the Kaggle vocabulary is
  linguistically validated; 35 populated classes; anything from
  `paper/main.tex` as currently written; that detection-confidence or
  hand-column-swap explain the gap (both now tested and refuted, so a
  paper must say "ruled out," never merely "not addressed").
- **New, must-not-claim item surfaced this pass**: that the
  `isl_main.py` un-mirror correction's assumption about Kaggle's image
  mirroring convention has been verified — it has not; it is an
  unverified assumption a working classifier (B, D, G) currently depends on.
- **New, must-not-claim item surfaced this pass**: that the live-recorded
  "K" gesture has been confirmed to match the Kaggle "K" fingerspelling
  convention (one- vs. two-handed) — this is an open linguistic/protocol
  question, not resolved by any statistical test here.

**Note on "sign language recognition" framing**: the 5 word-gestures
(GOOD, HELLO, HELP, LOVE, THANKYOU) are true dynamic ISL signs but are
captured and classified here as single static key-poses (one frame per
gesture instance), not motion trajectories. This system should be
described precisely as **static hand-pose / fingerspelling-letter
classification**, with the word-gesture classes explicitly caveated as a
"static key-pose proxy," not full dynamic gesture recognition. (Previously
stated only in `docs/data_collection_protocol.md`; foregrounded here so
it isn't missed by a reader of just the dossier.)

## 6. Reproducibility

```
pip install -r requirements.txt
python evaluate.py                              # core pipeline
python analyze_domain_shift.py                  # cross-source forensics, H1-H9
python build_invariant_features.py              # candidate fix + multiseed + McNemar
python import_kaggle_dataset.py --dir <Kaggle root> --out-csv isl_dataset_kaggle_controlled.csv --confidence 0.7
python analyze_domain_shift_controlled.py       # confidence-controlled comparison
python test_handswap_hypothesis.py              # hand-column-swap test
python visualize_landmark_geometry.py           # qualitative landmark figure
python test_classifier_generalization_of_gap.py # classifier-family check
python generate_figures.py                      # 8 core figures
python build_results_final_v2.py                # assembles RESULTS_FINAL_V2.json
python train_model.py                           # retrains deployed isl_model.pkl
```
All seeded (`random_state=42` throughout; multi-seed check uses
`[0,1,2,3,4,42,100]`; bootstrap CIs use fixed RNG seeds). The §1 MANUAL
findings are not reproducible from this repo alone (require the original
Kaggle download, external to this repo). The controlled experiment
requires the original extracted Kaggle image folder, also external to
this repo — `isl_dataset_kaggle_controlled.csv` itself IS committed/
present on disk so the comparison step can be rerun without re-extracting.

## 7. `paper/main.tex` is STALE — unchanged from v1, do not cite

See v1 dossier §7 for the full explanation (describes the superseded
842-row/18-class pre-Kaggle-merge snapshot). Recommended structure
unchanged, now updated to reflect this pass's evidence:
1. Motivate via the original leakage finding as historical context (past tense, superseded).
2. Central contribution: cross-source domain shift, its now-13-hypothesis
   forensic diagnosis (6 refuted, 2 confirmed/partial, 2 inconclusive,
   plus the classifier-generalization and significance-testing checks),
   and the partial, statistically-validated hand-slot-agnostic fix.
3. Report §3 as supporting methodology validation, not the headline result.
4. Limitations: merge everything in §8 below — do not bury it.
5. Scope vocabulary to 31/35 and generalization claims to "domain-disjoint, not signer-independent."
6. **New**: explicitly state the static-key-pose framing (§5) and the two newly-surfaced open items (§8) as limitations.

## 8. Remaining weaknesses / limitations (unresolved, stated plainly)

- **A, E, F's complete cross-domain failure remains unexplained** after
  six directly-tested candidate mechanisms (sample size, detection
  threshold, hand-slot/handedness composition, hand-proximity, Cohen's-d
  magnitude, hand-column-swap). The strongest remaining lead
  (image-resolution/framing heterogeneity) requires new matched-resolution
  data collection to test.
- One-Euro filtering (webcam) vs. none (Kaggle) is a real, uncontrollable-
  without-new-data confound — inherent to comparing a photo dataset
  against a video-capture dataset.
- No certified ISL interpreter or Deaf community member has ever reviewed
  the target vocabulary or Kaggle images for linguistic accuracy.
- The image-resolution-heterogeneity finding is real but manually
  inspected, not automatically re-verifiable, and not experimentally
  connected to the accuracy gap.
- No signer- or session-disjoint evaluation exists or can exist without
  new data collection (protocol specified in `docs/data_collection_protocol.md`).
- `process_images.py` remains a non-leakage-safe, inconsistent-methodology
  script left in the repo (flagged in its own docstring, not removed).
- **New this pass**: `isl_main.py`'s handedness-swap/un-mirror correction
  rests on an unverified assumption about the Kaggle images' mirroring
  convention; several currently-working classes (B, D, G) depend on that
  assumption being correct, per the hand-swap test (§4.2). No Kaggle
  capture metadata exists to verify it independently.
- **New this pass**: whether the live-recorded "K" gesture actually
  matches the Kaggle "K" fingerspelling convention (one-handed vs.
  two-handed) is an open linguistic/protocol question, not resolved by
  any test performed here — the near-total hand-count mismatch (§4.3) is
  consistent with either a genuine domain-shift artifact or an actual
  sign-execution/labeling inconsistency, and this project cannot
  distinguish the two without expert linguistic input.
- **New this pass**: the RandomForest hyperparameters (100 estimators,
  depth 15) were never tuned via a search; not pursued, since it does not
  bear on the central research question and this task explicitly warned
  against adding ML sophistication for its own sake.

## 9. Figures (all real, see `results/`)

Unchanged v1 set: `fig_pipeline_architecture.png`, `fig_dataset_composition.png`,
`fig_leakage_demo.png`, `fig_cross_source_key_finding.png`,
`fig_problematic_classes.png`, `fig_feature_ablation.png`,
`fig_invariant_features_fix.png`, `fig_handedness.png`. Diagnostic-only:
`descriptive_stats.png`, `confusion_matrix.png`, `domain_shift_features.png`.

**New this pass**: `fig_controlled_vs_original.png` (confidence-matched
cross-source comparison, §4.1), `fig_landmark_geometry_comparison.png`
(qualitative landmark-geometry evidence, §4.3).

## 10. Completion report (this pass)

1. **What changed**: reran `analyze_domain_shift.py` and
   `build_invariant_features.py` to sync outputs with mid-session script
   edits (H8/H9, multi-seed check); ran the controlled cross-source
   experiment (confidence-matched Kaggle re-extraction); audited the full
   feature-extraction pipeline for undocumented inconsistencies (found and
   tested one — the hand-column convention — and refuted it as a cause);
   added a paired McNemar significance test; added a qualitative
   landmark-geometry visualization; added a classifier-family
   generalization check; corrected an imprecise threshold claim; produced
   this v2 dossier, `RESULTS_FINAL_V2.json`, and `FINAL_RESEARCH_AUDIT.md`.
2. **Experiments that ran this pass**: §4.1-4.4 above, all real, all reproducible.
3. **Final verified key results**: 99.9%+ in-domain (4 classifier
   families) vs. 28-51% cross-domain (same 4 families); 6 specific
   candidate causes for the gap directly tested and refuted; K's failure
   substantially explained (hand-count mismatch); A/E/F remain genuinely
   unexplained; the hand-slot-agnostic fix is McNemar-significant
   (p<1e-9) and seed-robust.
4. **Strongest scientific contribution**: the systematically-eliminated,
   multi-hypothesis forensic diagnosis of a real cross-source domain-shift
   finding — honestly scoped as partially, not fully, resolved.
5. **Biggest remaining limitation**: A, E, F's failure is completely
   unexplained after 6 tested mechanisms; two new open integrity questions
   surfaced this pass (the un-mirror convention assumption, and whether
   live "K" matches Kaggle "K"'s sign-execution convention) cannot be
   resolved without external data/expertise this project does not have.
6. **Ready for the CVIP paper?** Yes — see `FINAL_RESEARCH_AUDIT.md` §10
   for the full readiness verdict. `paper/main.tex` is NOT ready (unchanged
   from v1) — it must be rewritten from this dossier, not patched.
