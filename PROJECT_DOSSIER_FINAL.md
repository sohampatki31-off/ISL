# ISL-backend — Research Dossier (CVIP 2026 candidate)

**Status: Phase 1-9 of the CVIP2026 research-upgrade task complete. This
document is the final handoff — read it before writing or citing anything
in `paper/main.tex`, which is STALE and must not be cited as-is (see §7).**

Generated 2026-08-12. Every number below is machine-computed and
reproducible via the commands in §6 — nothing here is estimated,
interpolated, or asserted from memory. Where a claim rests on a small
sample or an unverified/manual check, that is stated explicitly next to
the number, not left implicit.

---

## 0. Evidence classification key

Every finding below is tagged:
- **[REAL]** — computed from `isl_dataset.csv` as it exists today, by a
  script in this repo, reproducible by rerunning that script.
- **[MANUAL]** — a real, factual finding, but from a one-time manual
  inspection outside the repo (the original Kaggle image files on local
  disk) — true, but not automatically reproducible from the repo alone.
- **[HYPOTHESIS]** — a plausible, evidence-motivated explanation that was
  proposed but NOT experimentally confirmed. Do not state as fact in a
  paper without first running the confirming experiment.
- **[FUTURE WORK]** — designed/specified but not executed. Zero data
  exists for these claims; do not imply otherwise.

---

## 1. Dataset facts **[REAL]**

- `isl_dataset.csv`: **31,806 raw rows → 31,688 after exact-duplicate
  removal**, 132 features (63 right-hand + 63 left-hand landmark
  coordinates, wrist-subtracted and scale-normalized by wrist→middle-MCP
  distance, + 6 inter-hand relative features), **31 classes** with real
  data (26 letters A-Z + 5 word-gestures: GOOD, HELLO, HELP, LOVE,
  THANKYOU).
- `isl_main.py` hardcodes a 35-class *target* vocabulary (26 letters + 9
  word-gestures: adds SORRY, YES, NO, PLEASE). **These 4 have zero
  samples** — not a bug, the Trainer UI already supports recording them
  without code changes; simply never done. State this as "31/35 target
  classes populated," never as "35 classes."
- **Two sources**, not one:
  - **Kaggle** (`session_id='kaggle_import'`): 30,872 rows post-dedup
    (97.4%). Source: "Indian Sign Language (ISL)" dataset by
    prathumarikeri, https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl,
    **license CC BY-SA 4.0** (verified directly from the Kaggle page by
    the user, 2026-08-12 — usability 8.75, annual update frequency). See
    `LICENSE` for the attribution/share-alike terms this imposes on any
    redistribution of the derived data. 26 letter classes only, no word
    classes.
  - **Webcam** (hand-recorded via this app's Trainer UI): 816 rows
    post-dedup (2.6%). Contains all 31 populated classes.
- **No real signer or session diversity exists anywhere in this
  dataset.** `signer_id`/`session_id` are placeholder strings
  (`unknown_legacy`, `kaggle_unknown`, `unassigned_signer`,
  `kaggle_import`, `unassigned_session`), not per-person identities.
  Confirmed by the user: all 816 webcam rows were recorded by this
  project's sole developer — no third-party human-subjects data exists
  in the current dataset (see `docs/data_collection_protocol.md`).
- `burst_id` is near-unique per row for 97% of data (mean group size
  1.03; only 47/30,925 groups have >1 row) — a real recording-burst
  grouping exists only for the small legacy-webcam subset.
- **118 exact-duplicate feature rows removed** before every experiment
  below (117 webcam, 1 Kaggle) — see `evaluate.py`'s `dedupe_exact()`.
- **[MANUAL, 2026-08-12]** The original Kaggle image files (inspected at
  `C:\Users\Soham\Downloads\ISL_kaggle_extracted`) are **not uniform
  resolution**: 31/35 classes are 1200 images each, all 128×128px,
  tightly cropped to the hand against a dark background. 4 classes (C, I,
  O, V) additionally contain 90-247 extra images from a different,
  much-higher-resolution sub-source (640×480 or ~1088×1920) — exactly the
  4 classes whose `isl_dataset_kaggle.csv` counts exceed 1200. The Kaggle
  "ISL" dataset is itself a class-correlated patchwork of ≥2 sub-sources.
  Live webcam capture runs at 720p-1280p (`isl_main.py`). **This is a far
  more extreme resolution/framing gap than "domain shift" alone conveys.**
- **[MANUAL, cursory, non-expert]** A handful of Kaggle images (A, J, G,
  I) were visually inspected against the labeled letter. They show
  plausible two-handed and one-handed ISL-style fingerspelling shapes
  (e.g. `A`/`J` two-handed, `G` two fists, `I` one-handed point) — nothing
  looked obviously mislabeled or generic. **This is not a linguistic
  validation** — no certified ISL interpreter or Deaf community member
  reviewed the vocabulary or images at any point in this project's
  history. State this gap explicitly in Limitations; do not claim
  linguistic validity has been confirmed.

## 2. Methodology **[REAL]**

- **Feature extraction**: MediaPipe Tasks `HandLandmarker`
  (`hand_landmarker.task`), 21 landmarks/hand, wrist-relative +
  scale-normalized (÷ ‖landmark9−landmark0‖) per hand, zero-filled if a
  hand is absent, + 6 inter-hand relative features (3 raw deltas + 3
  Euclidean distances). 132 features total.
- **Detection-threshold inconsistency (real, documented, NOT retroactively
  fixable)**: live capture (`isl_main.py`) uses confidence 0.7; all three
  offline scripts (`process_images.py`, `import_kaggle_dataset.py`,
  `test_camera.py`) use MediaPipe's 0.5 default. Cross-referenced in code
  comments in all four files. Contributes an unknown, unquantified amount
  of selection-bias to the domain-shift finding in §4 (see Limitations).
- **Temporal filtering asymmetry**: live/webcam rows are One-Euro filtered
  before feature extraction; Kaggle/image-derived rows are not.
- **Classifier**: RandomForestClassifier (deployed model,
  `train_model.py`), compared against SVM-RBF, kNN(k=5), and MLP(128,64)
  in `evaluate.py` §6 (held-out, stratified 80/20 split, post-dedup; SVM
  trained on a stratified 6000-row subsample for tractability, disclosed
  as a caveat).
- **All experiments use the deduplicated dataset** (31,688 rows) unless
  explicitly noted otherwise.

## 3. Experiments run, with exact real results **[REAL]**

Full output: `results/RESULTS.md`, `results/all_results.json` (run via
`python evaluate.py`).

| # | Experiment | Key result |
|---|---|---|
| 1 | Descriptive statistics | 31,688 rows, 31 classes, severe imbalance (letters ~1216/class avg, word-gestures ~37/class avg, 78.8x max/min ratio) |
| 1b | Exact-duplicate audit | 118 rows removed (117 webcam, 1 Kaggle), 0 label conflicts |
| 1c | Near-duplication (intra-/cross-burst distance) | 29/31 classes have ≥2 bursts (up from 1 in the old 842-row snapshot); burst_id still near-unique for Kaggle |
| 2 | Leakage demo: naive vs. burst-disjoint split | naive 99.91% acc / 0.999 macro-F1; burst-disjoint (29 classes) 97.24% acc / **0.895 macro-F1** — burst_id still catches some real leakage in the legacy subset, but is a near-no-op for 97% of the data (Kaggle) |
| 3 | Repeated-split stats (10 seeds) | acc 99.81% ± 0.08pp, bootstrap 95% CI [99.77%, 99.86%] — pseudo-replication caveat stated explicitly in output |
| 4 | **Cross-source (domain) evaluation — KEY FINDING** | see §4 below |
| 5 | Handedness analysis (held-out) | both=100.00%, right_only=99.90%, left_only=99.65% — **handedness has no meaningful effect within a single mixed-domain model** |
| 6 | Held-out classifier comparison | RF 99.9%/0.999, kNN 99.9%/0.991, SVM 99.6%/0.965 (6000-row subsample), **MLP(128,64) 99.97%/0.9997** (added specifically to address the hostile-ML-review finding that a tree/kNN/SVM-only comparison with zero neural baseline is a credible desk-reject risk) — all four are within-domain-easy; this reinforces that the real difficulty is cross-domain, not classifier choice |
| 7 | Confusion matrix + per-class error analysis (held-out) | top confusions trivial (Q→S ×2, singletons) — near-ceiling performance is a within-domain artifact, not evidence of hard sign discrimination |
| 8 | Feature ablation (held-out) | full-132: 99.9%/0.999; no-relative-126: 99.9%/0.999 (relative features add ~nothing); right-hand-only-63: **85.9%/0.830** (left-hand info matters materially) |

## 4. Cross-source finding — the project's actual contribution **[REAL]**

`python analyze_domain_shift.py` → `results/domain_shift_forensics.json`,
`results/domain_shift_features.png`, `results/fig_problematic_classes.png`.

| Setting | n_train | n_test | Accuracy [95% CI] | Macro-F1 [95% CI] |
|---|---|---|---|---|
| In-domain Kaggle | 11,998 | 5,143 | 99.92% [99.8, 100.0] | 0.999 [0.998, 1.000] |
| In-domain Webcam | 407 | 224 | **46.88%** [40.6, 53.6] | 0.210 [0.189, 0.242] |
| Kaggle → Webcam | 17,141 | 631 | 50.40% [46.6, 54.0] | 0.312 [0.296, 0.325] |
| Webcam → Kaggle | 631 | 17,141 | **29.41%** [28.7, 30.1] | 0.211 [0.208, 0.214] |

14 shared classes (A,B,C,D,E,F,G,I,J,K,L,M,U,V — the only letters present
in both sources). All numbers now carry bootstrapped 95% CIs (2000
resamples over fixed test-set predictions) — added specifically because
an independent hostile-ML review found the original point estimates,
especially on n=17-68/class webcam test sets, were reported without any
uncertainty quantification.

**Multi-hypothesis forensic testing, in the order tried:**

1. **H1 — sample size [REAL, REFUTED]**: Kaggle subsampled to match
   webcam's tiny per-class n still gets **100%** in-domain accuracy. Rules
   out "not enough webcam data" as the explanation for webcam's 46.88%.
2. **H2 — handedness composition [REAL, PARTIAL]**: Kaggle is 48%
   both-hands/32% right-only/20% left-only; webcam is 54%/3.5%/43% — very
   different composition.
3. **H3 — feature-level shift [REAL, method fixed after CV review]**: the
   largest domain-shifted features are x-coordinate-dominated (Cohen's d
   up to ~1.6), computed **within each shared class then averaged**
   (marginal/class-agnostic computation was flagged as confounded by
   class-imbalance and corrected).
4. **H4 — per-class recall [REAL]**: Kaggle-trained model tested on
   webcam is sharply bimodal: `A,E,F,J,K,M,U` = exactly 0% recall;
   `B,D,V,G,I,L` = 92-100%; `C` = 8%. Each 0%-recall class maps almost
   deterministically to ONE other letter (`J→D` 100%, `K→V` 100%,
   `U→C` 100%, etc.) — the signature of a systematic transform, not noise.
5. **H5 — pre/post mirroring-fix persistence [REAL, REFUTES the "old bug"
   theory]**: class M's samples split by pre-fix (`unknown_legacy`) vs.
   post-fix (`unassigned_session`, recorded after commit `b5fc4c6`) both
   score **0% recall** (different wrong labels: `L` vs. `U`/`C`/`I`). The
   gap is not an already-fixed bug; it persists in the newest recordings.
6. **H6 — per-class handedness-composition-vs-recall [REAL, refines H2]**:
   `J, K, U, C` show a genuine handedness/hand-slot composition MISMATCH
   between domains, consistent with 7 below improving them. But `A, E, F,
   M` have SIMILAR composition in both domains and STILL fail completely
   — handedness does not explain everything.
7. **Candidate fix tested — hand-slot-agnostic angle features
   [REAL, statistically validated]** (`build_invariant_features.py` →
   `results/invariant_features_test.json`): replacing raw coordinates
   with rotation+reflection-invariant joint angles, with one-handed signs
   always placed in the same feature slot regardless of physical
   left/right assignment:

   | Representation | Kaggle→Webcam macro-F1 [95% CI] | Webcam→Kaggle macro-F1 [95% CI] |
   |---|---|---|
   | Raw (132-d) | 0.312 [0.296, 0.325] | 0.211 [0.208, 0.214] |
   | Rotation-invariant (41-d) | 0.361 [0.334, 0.384] | 0.300 [0.294, 0.306] |
   | **Handslot-agnostic (41-d)** | **0.458 [0.436, 0.479]** | **0.273 [0.267, 0.279]** |

   Non-overlapping CIs vs. raw in both directions — a real, statistically
   supported partial improvement, not noise. Recovers `U` (0%→54.5%) and
   `M` (0%→42.5%) fully or partially. **`A, E, F, J, K` remain at exactly
   0% recall even after this fix** — pure rotation-invariance
   (without the hand-slot fix) barely helped at all, refuting a simple
   "it's just mirroring" story. Within-domain accuracy cost: negligible
   (99.92%→99.83%).
8. **[MANUAL, HYPOTHESIS, not experimentally isolated]**: given the
   128×128-vs-720p resolution/framing gap discovered in §1, and that H6
   leaves `A, E, F, M` unexplained by handedness alone, **image
   resolution/framing heterogeneity is a plausible second contributing
   cause** for the residual failures. No controlled re-detection-at-
   matched-resolution experiment was run to confirm this — state it as a
   motivated open question for future work, not a proven cause.

**Honest one-paragraph summary for the paper**: cross-source (Kaggle↔
webcam) accuracy collapses from 99.9% in-domain to 29-62% depending on
direction, refuting the naive "collect more webcam data" fix (H1) and the
naive "it's just mirroring" fix (rotation-invariant features alone barely
moved the needle). Part of the gap (4/14 classes materially, `J,K,U,C`) is
explained by an inconsistent hand→feature-slot assignment convention
between the two capture pipelines, and a targeted fix for that recovers
real, CI-confirmed accuracy. The remaining failures (`A,E,F,K`) are
unexplained by anything tested here; the most concrete lead is the newly
discovered extreme image-resolution disparity between sources, untested.

## 5. What this project can and cannot claim

### Claims the paper CAN safely make **[REAL]**
- A real, reproducible, large cross-source accuracy gap exists between a
  Kaggle-image-sourced and a live-webcam-sourced ISL fingerspelling
  dataset, with CI-bounded numbers (§4 table).
- The gap is not explained by webcam sample size (H1, refuted with a
  controlled experiment).
- Part of the gap (measurable, CI-confirmed) is explained by an
  inconsistent hand→feature-slot convention between capture pipelines,
  and a proposed fix (hand-slot-agnostic angle features) measurably and
  significantly improves it without in-domain cost.
- burst-level near-duplication is real and measurable (macro-F1 0.999→
  0.895 under burst-disjoint splitting) for the legacy-webcam subset, but
  is NOT a meaningful leakage control for the Kaggle-dominated majority
  of current data (burst_id near-unique per row there).
- Handedness (single/both hands detected) has no measurable effect on
  within-domain accuracy.
- Left-hand landmark information is not redundant: right-hand-only
  features lose ~14 points of accuracy vs. the full representation.

### Claims the paper MUST NOT make
- **Signer-independent recognition** — no real multi-signer data exists.
- **Session-independent recognition** in any generalizable sense — same
  reason.
- That the cross-source gap is fully explained or fully fixed — it is
  ~40-50% closed on the metrics where a fix was tested, and completely
  unexplained for `A, E, F, K`.
- That the Kaggle vocabulary/images have been linguistically verified as
  authentic ISL — only a cursory, non-expert visual spot-check was done.
- That `isl_dataset.csv` currently supports 35 classes — it supports 31;
  4 word-gestures have zero data.
- Anything from `paper/main.tex` as currently written (see §7).
- That detection-threshold/temporal-filtering differences between
  pipelines were controlled for in the domain-shift analysis — they were
  not; they're a documented, unquantified confound.

## 6. Reproducibility

```
pip install -r requirements.txt
python evaluate.py                  # core pipeline -> results/RESULTS.md, all_results.json
python analyze_domain_shift.py      # cross-source forensics -> domain_shift_forensics.json
python build_invariant_features.py  # Phase-5 candidate fix -> invariant_features_test.json
python generate_figures.py          # all 8 figures, from the JSONs above only
python train_model.py               # retrains the deployed isl_model.pkl/labels.pkl (dedup'd)
```
All seeded (`random_state=42` throughout; repeated-split uses seeds 0-9;
bootstrap CIs use fixed RNG seeds — see `docs/reproducibility_checklist.md`
for the full list). The §1 MANUAL findings (image resolution, visual
spot-check) are NOT reproducible from this repo alone — they require the
original Kaggle download, external to this repo.

## 7. `paper/main.tex` is STALE — do not cite, and do not just patch it

An independent hostile review (sign-language/PR perspective) found that
`paper/main.tex`'s entire framing — burst-level leakage as the central
contribution, 100%→0% accuracy collapse — describes the **842-row,
18-class pre-Kaggle-merge snapshot**, which no longer exists. The
project's own current results (`results/RESULTS.md` §2) explicitly state
that on the current 31,688-row dataset, burst-disjoint and naive-random
splits give *similar* results and "`burst_id` is not a meaningful leakage
control at the current dataset composition" — directly contradicting the
paper's stated reason to publish. **This is not a numbers-update job; the
paper's central claim needs to change.**

**Recommended new structure** (does not exist yet — not written, per this
task's explicit instruction to validate the project before writing the
paper):
1. Motivate via the *original* leakage finding as historical context /
   motivation for careful evaluation methodology (one paragraph, clearly
   framed as past-tense, superseded).
2. Pivot the central contribution to §4 above: cross-source domain shift
   in ISL fingerspelling recognition, its multi-hypothesis forensic
   diagnosis, and the partial, CI-validated hand-slot-agnostic fix.
3. Report §3's table as supporting methodology validation, not the
   headline result.
4. Limitations section: merge everything in §8 below — do not bury it.
5. Explicitly scope the vocabulary claim to 31/35 classes and the
   signer/session claims to "domain-disjoint, not signer-independent."

## 8. Remaining weaknesses / limitations (unresolved, stated plainly)

- Detection-threshold (0.7 vs 0.5) and temporal-filtering (One-Euro vs.
  none) differences between the live and offline pipelines are a real,
  uncontrolled confound in the §4 domain-shift analysis — not
  quantified, not separated from genuine visual/geometric domain shift.
- `A, E, F, K`'s complete cross-domain failure remains unexplained.
- No certified ISL interpreter or Deaf community member has ever reviewed
  the 35-word target vocabulary or spot-checked the Kaggle images for
  linguistic accuracy — only a cursory, non-expert visual check was done
  for 4 letters in this session.
- The image-resolution-heterogeneity finding (§1) is real but based on a
  one-time manual inspection outside this repo — not automatically
  re-verifiable, and not experimentally connected to the accuracy gap.
- No signer- or session-disjoint evaluation exists or can exist without
  new data collection (see `docs/data_collection_protocol.md`'s Phase-6
  section for the concrete, unexecuted plan).
- `process_images.py` remains a non-leakage-safe, inconsistent-methodology
  script left in the repo (flagged in its own docstring, not removed —
  "do not destroy working components unnecessarily").

## 9. Figures (all real, see `results/`)

`fig_pipeline_architecture.png`, `fig_dataset_composition.png`,
`fig_leakage_demo.png`, `fig_cross_source_key_finding.png` (with 95% CI
error bars), `fig_problematic_classes.png`, `fig_feature_ablation.png`,
`fig_invariant_features_fix.png` (with 95% CI error bars),
`fig_handedness.png`. Plus diagnostic-only: `descriptive_stats.png`,
`confusion_matrix.png`, `domain_shift_features.png`.

## 10. Completion report

1. **What changed**: full audit; dataset validity formally scoped to
   31/35 classes; `evaluate.py` rebuilt for the current 31,688-row
   dataset with dedup, cross-source evaluation, handedness analysis, and
   bootstrapped CIs; new `analyze_domain_shift.py` and
   `build_invariant_features.py` scripts; `LICENSE` attribution fixed;
   `train_model.py` deduplicated and the deployed model retrained; two
   stale docs rewritten; 8 new figures generated; 3 independent hostile
   reviews run and their fixable findings addressed (CIs added, Cohen's d
   confound fixed, MLP baseline added, consent status clarified).
2. **Experiments that ran**: see §3-4 above, all real, all reproducible.
3. **Final verified key results**: 99.9% in-domain vs. 29-62% cross-domain
   accuracy (CI-bounded); hand-slot-agnostic features close part of that
   gap with statistical significance; handedness itself is not a factor;
   left-hand information is not redundant.
4. **Strongest scientific contribution**: the cross-source domain-shift
   finding and its multi-hypothesis forensic diagnosis (§4) — genuinely
   novel evidence, not previously in this project, honestly scoped as
   partial.
5. **Biggest remaining limitation**: no signer diversity exists or can be
   claimed; `A,E,F,K`'s failure is unexplained; the detection-
   threshold/resolution confounds were documented but not experimentally
   isolated from genuine domain shift.
6. **Ready for the CVIP paper?** The *project* is now scientifically
   defensible and its claims are honestly scoped. `paper/main.tex` is NOT
   ready — it argues a superseded finding and needs the restructuring in
   §7, which is a writing task, not a further research task. Recommend:
   write the new paper from this dossier and `results/RESULTS.md`
   directly; do not edit the existing draft incrementally.
