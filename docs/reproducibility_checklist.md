# Reproducibility Checklist

## Environment
- Python 3.14.0, scikit-learn 1.8.0, NumPy 2.4.2, SciPy 1.17.1, Matplotlib 3.10.8, statsmodels 0.14.6.
- Windows 11 (10.0.26200), Intel Core Ultra 5 125H, CPU only — no GPU used anywhere.
- Exact package list: `requirements.txt` (note: versions are not currently pinned there — pin them, e.g. via `pip freeze`, before archiving for a submission).

## Data provenance
- `isl_dataset.csv`: 842 rows, 18/35 classes, migrated in-place from a pre-existing legacy schema.
  Pre-migration copy preserved at `isl_dataset.csv.pre_schema_migration.bak` — do not delete; it is
  the only record of the data before `signer_id`/`session_id`/`burst_id`/`ts` columns were added.
- `burst_id` for legacy rows was inferred (not recorded at collection time) via contiguous
  same-label runs in file order — see `_migrate_dataset_schema_if_needed()` in `isl_main.py`
  and "Reconstructing bursts from an unlabeled legacy file" in `paper/main.tex`, Section 4.
- All new recordings (post-migration) carry real `signer_id`/`session_id`/`burst_id`/`ts` values
  captured at collection time by the Trainer UI — see `docs/data_collection_protocol.md`.

## Random seeds
- `train_test_split(..., random_state=42)` and `RandomForestClassifier(..., random_state=42)`
  throughout `evaluate.py`, `train_model.py`, and `isl_main.py`'s `train_model_bg`.
- The 20-seed repeated-split experiment in `evaluate.py` uses `random_state=0..19` explicitly.
- Bootstrap CIs use `np.random.default_rng(0)` (fixed seed, see `_bootstrap_ci` in `evaluate.py`).
- Cross-burst distance sampling uses `np.random.default_rng(42)`.

## How to reproduce every number in the paper
```
pip install -r requirements.txt
python evaluate.py
```
This regenerates `results/all_results.json`, `results/RESULTS.md`, `results/descriptive_stats.png`,
and `results/confusion_matrix.png` from `isl_dataset.csv` as it currently exists. Every number in
`paper/main.tex` traces to this output — cross-check by grepping the specific figure (e.g.
`grep "37.0" results/RESULTS.md` for the Random Forest predict latency in Table 2). Wall-clock
latency figures will vary a few ms run-to-run (see the caveat in the paper); accuracy/F1/collision
figures are deterministic given the fixed seeds above.

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
- **Signer/session-disjoint accuracy**: not obtainable from any code change alone; requires the
  data collection described in `docs/data_collection_protocol.md` to actually be carried out.
- **LaTeX compilation of `paper/main.tex`**: no LaTeX toolchain (`pdflatex`/`latexmk`) was available
  in this environment. The source is written for the standard Springer `llncs.cls` (e.g. via
  Overleaf's "Springer LNCS" template) but has not been compiled to PDF or proofread for page
  breaks/float placement — do this before submission.

## Known non-determinism / caveats to keep in mind when re-running
- The 20-seed "repeated split" experiment resamples the same 842 rows 20 times — it is not 20
  independent datasets (pseudo-replication), so its spread is a lower bound on true variance, not
  a substitute for real repeated data collection.
- The cross-burst distance statistic (Section 4 of the paper) is computed almost entirely from one
  class (E), the only one with more than one recording burst — treat it as illustrative, not precise.
- `isl_dataset_one_hand.csv` / `isl_model_one_hand.pkl` / `labels_one_hand.pkl` are a separate,
  apparently-vestigial 63-feature one-handed dataset/model not referenced anywhere in `isl_main.py`
  — they were not used in any experiment in this paper and their provenance/purpose is undocumented.
