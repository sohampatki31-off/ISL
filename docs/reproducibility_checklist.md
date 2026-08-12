# Reproducibility Checklist

## Environment
- Python 3.14.0, scikit-learn 1.8.0, NumPy 2.4.2, SciPy 1.17.1, Matplotlib 3.10.8, statsmodels 0.14.6.
- Windows 11 (10.0.26200), Intel Core Ultra 5 125H, CPU only — no GPU used anywhere.
- Exact package list: `requirements.txt` (note: versions are not currently pinned there — pin them, e.g. via `pip freeze`, before archiving for a submission).

## Data provenance
- `isl_dataset.csv`: **31,806 raw rows (31,688 after exact-duplicate removal), 31/35 target
  classes**, updated 2026-08-12 during the CVIP2026 research audit. Two sources:
  - ~97% (`session_id='kaggle_import'`): imported from the Kaggle "Indian Sign Language (ISL)"
    dataset by prathumarikeri (CC BY-SA 4.0 — see `LICENSE` for the attribution/share-alike terms
    this imposes) via `import_kaggle_dataset.py`. No real signer/session tracking exists for these
    rows (`signer_id='kaggle_unknown'` for all of them, by construction).
  - ~3% (`session_id` in `unknown_legacy`/`unassigned_session`): hand-recorded via the live webcam
    Trainer UI. `signer_id` is a fixed placeholder for all of these too — **no real multi-signer
    data exists anywhere in this file**, despite the per-sample signer/session fields the app
    exposes (see `docs/data_collection_protocol.md`'s Phase-6 generalization-claims section).
  - The original pre-Kaggle-merge snapshot (842 rows, the numbers this checklist used to describe)
    is preserved in git history at commit `c93321d`'s parent, not in a `.bak` file on disk.
- `isl_dataset.csv.pre_schema_migration.bak` — pre-dates even that: the only record of the data
  before `signer_id`/`session_id`/`burst_id`/`ts` columns existed at all. Do not delete.
- `burst_id` for legacy rows was inferred (not recorded at collection time) via contiguous
  same-label runs in file order — see `_migrate_dataset_schema_if_needed()` in `isl_main.py`.
  Kaggle-imported rows get a synthetic per-image `burst_id` (`kaggle_N`) that is NOT a real
  recording burst — see `import_kaggle_dataset.py`.
- Every number in `paper/main.tex` and every number this checklist described before 2026-08-12
  refers to the OLD 842-row/18-class snapshot and is **stale — do not cite it**. The paper has not
  yet been rewritten against the current dataset; that is intentional (the research-upgrade task
  explicitly defers paper-writing until the project itself is validated) but means `paper/main.tex`
  must not be treated as reproducible or current until it is rewritten from `results/RESULTS.md`
  and `results/all_results.json` as they exist now.

## Random seeds
- `train_test_split(..., random_state=42)` and `RandomForestClassifier(..., random_state=42)`
  throughout `evaluate.py`, `train_model.py`, `analyze_domain_shift.py`,
  `build_invariant_features.py`, and `isl_main.py`'s `train_model_bg`.
- The repeated-split experiment in `evaluate.py` uses `random_state=0..9` explicitly (reduced from
  20 to 10 seeds when the dataset grew to 31k+ rows, to keep runtime reasonable — this only affects
  the stability of the CI width, not the methodology).
- Bootstrap CIs use `np.random.default_rng(0)` (fixed seed, see `_bootstrap_ci` in `evaluate.py`).
- Cross-burst distance sampling and the stratified-subsample cap for SVM fitting both use
  `np.random.default_rng(42)`.

## How to reproduce every number in results/RESULTS.md
```
pip install -r requirements.txt
python evaluate.py
```
This regenerates `results/all_results.json`, `results/RESULTS.md`, `results/descriptive_stats.png`,
and `results/confusion_matrix.png` from `isl_dataset.csv` as it currently exists (31,688 rows after
dedup, 31 classes). These are the current, real, non-stale numbers — **`paper/main.tex` has not
been rewritten to match them yet** (see Data provenance above); do not cross-check the paper
against this output until it is.

For the cross-source forensic analysis (the project's key finding):
```
python analyze_domain_shift.py      # -> results/domain_shift_forensics.json, domain_shift_features.png
python build_invariant_features.py  # -> results/invariant_features_test.json
```
Wall-clock latency figures will vary a few ms run-to-run; accuracy/F1/collision figures are
deterministic given the fixed seeds above.

To re-verify the live Trainer path (not just the standalone `evaluate.py` script):
```
python -c "
import time, isl_main as m
with m.app.test_client() as c:
    c.post('/train')
    time.sleep(2)
    print(c.get('/training_status').get_json())
"
```
Expect `status: success`, `eval_note` explaining no held-out accuracy is available (only one class,
E, currently has more than one recording burst).

## What is NOT reproducible in this environment, and why
- **Live camera / end-to-end frame-rate benchmarking**: this was drafted in an environment with no
  webcam attached. `python isl_main.py` will start and serve the UI, but the video feed will show
  the "waiting for camera" placeholder. Anyone with a webcam can run it directly — see `start.bat`.
- **Signer-disjoint (LOSO) accuracy**: not obtainable from any code change alone; requires the data
  collection described in `docs/data_collection_protocol.md` to actually be carried out. (Source-
  disjoint Kaggle-vs-webcam evaluation IS now available and reproducible — see above — but must not
  be described as signer-independence.)
- **LaTeX compilation of `paper/main.tex`**: no LaTeX toolchain (`pdflatex`/`latexmk`) was available
  in this environment. The source is written for the standard Springer `llncs.cls` (e.g. via
  Overleaf's "Springer LNCS" template), has not been compiled to PDF, and (see Data provenance
  above) still describes the old 842-row dataset — it needs a full rewrite against current numbers
  before either compiling or submitting it.

## Known non-determinism / caveats to keep in mind when re-running
- The repeated-split experiment in `evaluate.py` resamples the same deduplicated rows N times per
  run — it is not N independent datasets (pseudo-replication), so its spread is a lower bound on
  true variance, not a substitute for real repeated data collection.
- The intra-/cross-burst distance statistic (`evaluate.py` section 1c) now draws from 29/31 classes
  with >=2 recording bursts (up from just class E in the old 842-row snapshot) and is meaningful
  for the legacy-webcam subset, but `burst_id` is still near-unique per row for the Kaggle-sourced
  majority of the data — see the caveat printed alongside that number.
- The Kaggle-vs-webcam cross-domain numbers (`analyze_domain_shift.py`) are computed on genuinely
  small webcam-side samples for some classes (as few as 17-68 rows/class) — real signal, but report
  the exact `n` alongside any percentage, don't imply more precision than that `n` supports.
- `isl_dataset_one_hand.csv` / `isl_model_one_hand.pkl` / `labels_one_hand.pkl` are a separate,
  apparently-vestigial 63-feature one-handed dataset/model not referenced anywhere in `isl_main.py`
  — not used in any experiment here, provenance/purpose still undocumented.
- `isl_dataset_kaggle.csv` / `isl_model_kaggle.pkl` / `labels_kaggle.pkl` are the raw Kaggle-only
  extract, kept (gitignored) purely as a provenance record of exactly what was imported before the
  merge into `isl_dataset.csv` — not read by `isl_main.py`, not needed to reproduce anything above.
