# Data Collection Protocol for Signer/Session-Generalization Evaluation

## Why this document exists

`python evaluate.py` (see `results/RESULTS.md`) demonstrates, on the data that
exists today, that:

- `isl_dataset.csv` has 842 rows across 18/35 target classes, essentially
  **one continuous recording burst per class** (19 bursts / 18 classes).
- A naive random 80/20 split reports **100.00% accuracy** because it puts
  near-duplicate frames from the same press-and-hold recording on both sides
  of the split (measured: 100% of test samples' nearest same-label neighbor
  comes from their own recording burst, across 20 random seeds).
- A burst-disjoint split — the minimum correction — collapses to **0%**,
  not because the model is bad, but because holding out the only burst for a
  class removes that class from training entirely.

**Conclusion: with the current data, no split can produce a real
generalization estimate, for *any* classifier.** This is not fixable by
better modeling or better splitting code — it requires collecting the same
signs under genuinely independent conditions. This document specifies
exactly how to do that with the app as it now stands.

## What "independent" means here, precisely

Three levels of generalization, from weakest claim to strongest:

1. **Session-disjoint, same signer**: the same person signs the same class on
   two different occasions (different sitting, ideally different day,
   lighting, and camera distance). Tests whether the model generalizes past
   one recording session's specific hand pose/lighting/framing.
2. **Signer-disjoint**: a person who contributed zero training data for a
   class is evaluated on that class. Tests whether the model has learned the
   *sign*, not one person's hand geometry.
3. **Cross-session AND cross-signer**: both at once — the real target for any
   "generalizes to new users" claim.

The app now records `signer_id` and `session_id` per sample (Trainer tab, two
new fields above "Target Label" — persisted in the browser via
`localStorage` so you don't retype them every burst). Every row in
`isl_dataset.csv` also gets a `burst_id` (auto-incremented per
press-and-hold action) and a `ts` (Unix timestamp).

## Minimum viable collection plan

Goal: produce at least ONE real signer-disjoint number for the paper,
honestly caveated as a pilot/preliminary result (not a strong claim).

- **Signers**: 3 people minimum (you + 2 others willing to help — classmates,
  family). Record each person's `signer_id` as a short pseudonym (`s1`,
  `s2`, `s3`) — do not use real names in the dataset; keep the name↔id
  mapping in a separate, private file if you need it, not in the CSV.
- **Sessions per signer**: at least 2 per signer per class, on different days
  if possible (minimum: different sittings — stand up, move the camera,
  change something about lighting/framing between "sessions" even on the
  same day). Set a new `session_id` each time you sit down to record
  (e.g. `2026-08-10-morning`).
- **Classes**: cover all 35 target classes (26 letters + 9 gestures) — 17 are
  currently missing entirely (H, N, O, P, Q, R, S, T, W, X, Y, Z, HELP, NO,
  PLEASE, SORRY, YES). Prioritize these first; they contribute nothing to
  the model at all right now.
- **Samples per burst**: 60-100 frames (the existing 2-3 second press-and-hold
  UI flow already produces this).
- **Total new recording effort**: 3 signers x 2 sessions x 35 classes x ~1
  burst each ≈ 210 recording actions, a few seconds each. Realistic in a
  few hours split across signers.

This gets you: `n_groups_per_class >= 2` for most classes (fixing the
blocking issue in `train_model_bg`/`evaluate.py`), plus enough signer
diversity for a first, honestly-scoped leave-one-signer-out number.

## Stronger collection plan (if time/participants allow)

- **Signers**: 8-10+, spanning different hand sizes, skin tones, and (ideally)
  at least one left-handed signer, to let the two-hand relative-feature
  design actually be tested for handedness robustness rather than assumed.
- **Sessions per signer**: 3+, spread across different days, lighting
  conditions (daylight vs indoor lamp), and camera distances (30cm vs 60cm,
  as the existing docs already recommend for data quality).
- **Environmental variation checklist** (vary deliberately, don't hold
  constant): background (plain wall vs cluttered room), sleeve
  length/forearm visibility, camera angle (a few degrees off-axis).
- **Dynamic signs**: if you want to honestly claim "gesture recognition" for
  HELLO/LOVE/THANKYOU/etc. rather than "static key-pose proxy recognition"
  (see the technical audit — the current system classifies a single static
  frame per gesture, not a motion trajectory), that requires a different
  data representation entirely: recording short landmark *sequences*
  (e.g. 15-30 frames at fixed stride) per gesture instance and a
  sequence classifier (a small GRU/1D-CNN over the landmark sequence, or
  DTW-based matching) instead of per-frame RandomForest. This is a
  separate, larger effort — out of scope unless you want to pursue it
  explicitly.

## Ethics / consent

If any of this data is going into a paper, treat these as requirements, not
suggestions — a reviewer will ask, and "we asked verbally" is not an
adequate answer once results are published.

- **Get written consent before recording** — a signed form or even a saved
  email/text message agreeing to participate is the minimum bar for
  publishable human-subjects data; verbal-only agreement is not enough once
  this leaves an internal project.
- Tell every participant, in that consent step: what's recorded (hand
  landmark coordinates only — no raw video or images are saved to
  `isl_dataset.csv`, only the derived 132-number feature vector, per frame),
  why (a student CV project on ISL recognition, possibly published), how
  long it will be kept, and that they can withdraw at any time.
- Do not record real names as `signer_id`; use pseudonyms, and keep the
  name↔pseudonym mapping in a separate private file, not in the repo.
- **Per-signer deletion is not implemented yet.** `/clear_class` deletes by
  class label only, not by signer — a participant asking to withdraw their
  data cannot currently be honored via the UI. Add a small filtered-delete
  script (filter `isl_dataset.csv` rows by `signer_id`, rewrite the file)
  before collecting real participant data, not after.
- **Signer/session ID collisions are a real, currently-unmitigated risk.**
  The Trainer UI's ID fields are free text with no uniqueness check — two
  different people who both type `s1` are silently merged into one signer
  in the data, quietly corrupting a planned signer-disjoint evaluation.
  Mitigate manually for now: keep a shared roster (a plain text file or
  spreadsheet, not committed to the repo) mapping each real participant to
  their exact `signer_id` string, and check it before every new recording
  session rather than trusting recall.
- If this project is affiliated with a university or company with an ethics
  review process, go through it before recording rather than after —
  retroactive approval is usually not possible.

## Evaluation protocol once the data exists

Run (or extend) `evaluate.py`:

1. **Leave-one-signer-out (LOSO)**: for each signer, train on all other
   signers' data, test on the held-out signer. Report per-signer accuracy
   and macro-F1, then mean ± std across signers (this std is your real
   variance estimate — do not just report the mean).
2. **Session-disjoint, same-signer**: `GroupShuffleSplit` grouped by
   `session_id` within each signer, same idea as the current burst-disjoint
   check but at the coarser session granularity.
3. **Statistics**: with only 3-10 signers, do not compute a p-value that
   implies more precision than the sample size supports. Report the
   per-signer accuracy distribution (mean, std, min, max) and a
   bootstrap confidence interval over signers (resample signers with
   replacement, 1000x) rather than a parametric test. If you have a
   genuine paired comparison (e.g. RF vs SVM on the *same* LOSO folds),
   a paired sign test or Wilcoxon signed-rank test across signers/folds is
   appropriate — a McNemar's test on pooled predictions is NOT valid once
   folds are grouped by signer (it assumes independent trials).
4. **Expected tables/figures**: per-signer accuracy bar chart, LOSO summary
   table (signer, n_train, n_test, accuracy, macro-F1), confusion matrix
   pooled across LOSO folds.

## Code that already supports this (no further changes needed to collect)

- Trainer tab: signer/session ID fields (`templates/index.html`,
  `static/js/app.js`).
- `/start_recording` accepts `signer_id`/`session_id` and mints a
  `burst_id` (`isl_main.py`).
- `isl_dataset.csv` schema: `<132 features>, signer_id, session_id,
  burst_id, ts, label` (`isl_main.py: DATASET_COLUMNS`).
- `_load_dataset()` / `evaluate.py`'s loader already read by column name and
  will pick up real signer/session values automatically once recorded —
  no code changes needed to start using multi-signer data, only new
  recordings.

## What would need to change for LOSO evaluation specifically

`train_model_bg` currently groups by `burst_id` only. Once you have >=2
signers with usable per-class coverage, add a `GroupShuffleSplit(...,
groups=signers)` variant (or a full leave-one-group-out loop) alongside the
existing burst-disjoint check — this is a small, mechanical extension of the
existing `evaluate.py` structure, not a redesign.
