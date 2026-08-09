# Evaluation Results (real, reproducible via `python evaluate.py`)
Environment: Python 3.14.0, Windows-11-10.0.26200-SP0, Intel(R) Core(TM) Ultra 5 125H, scikit-learn 1.8.0, numpy 2.4.2, scipy 1.17.1.
## 1. Dataset
- 842 samples across 18/35 target classes (missing: H, HELP, N, NO, O, P, PLEASE, Q, R, S, SORRY, T, W, X, Y, YES, Z).
- signer_id values present: ['unknown_legacy'] -- no real signer diversity recorded.
- Classes with >=2 independent recording bursts: ['E'].
![per-class counts](descriptive_stats.png)
## 1b. Near-duplication (intra- vs cross-burst distance)
- Intra-burst consecutive-frame L2 distance: mean 0.3538, median 0.1371 (n=823 pairs).
- Cross-burst same-label L2 distance: mean 3.0318, median 2.1243 (n=165 pairs), ratio (cross/intra) 8.57x.
- Exact/near-exact duplicate fraction among consecutive intra-burst pairs (L2 < 1e-6): 14.2%.
> cross-burst pairs come almost entirely from the single class (E) with >1 burst -- this estimate is not class-balanced and should not be read as precise; it is directionally indicative only, given n_pairs above.
## 2. Leakage demonstration
- Naive random 80/20 split: **100.00%** accuracy, macro-F1 1.000.
- Burst-disjoint split: **0.00%** accuracy, macro-F1 0.000 (classes absent from train: ['B', 'C', 'D', 'U']).
> The near-100% naive-split accuracy is an artifact of near-duplicate frames from the same continuous recording burst appearing on both sides of the split. The burst-disjoint collapse is not a measure of poor model quality either -- most classes have only ONE recording burst, so holding a burst out removes that class from training entirely. Neither number estimates real-world generalization; see docs/data_collection_protocol.md.
## 3. Repeated-split statistics (20 seeds) + 1-NN same-burst collision rate
- Naive split accuracy: 99.94% +/- 0.26pp, bootstrap 95% CI [99.82%, 100.00%] (macro-F1 1.000 +/- 0.002).
- Same-burst 1-NN collision rate: 100.0% +/- 0.0pp, bootstrap 95% CI [100.0%, 100.0%].
> A same-burst 1-NN collision rate this high means the reported accuracy is dominated by near-duplicate matching within a burst, not sign discrimination.
> **Caveat:** The 20 seeds are overlapping 80/20 resamples of the same 842 rows, not 20 independent datasets -- this measures sensitivity to split randomness, not true sampling variability, and likely understates real-world variance.
## 4. In-sample classifier comparison (not a generalization estimate)
| Classifier | In-sample acc | Macro-F1 | Train time (s) | Predict latency (ms/sample) |
|---|---|---|---|---|
| RandomForest(100,d15) | 100.0% | 1.000 | 0.227 | 37.04 |
| SVM-RBF(C=10) | 100.0% | 1.000 | 0.061 | 0.13 |
| kNN(k=5) | 99.8% | 0.998 | 0.002 | 12.51 |

> Trained and evaluated on the SAME data (all of it) -- this measures whether each classifier's capacity suffices to fit the observed classes with this feature representation, not whether it generalizes to unseen signers/sessions. Latency is single-sample CPU inference on the development machine (see environment.processor), not the deployment device, and excludes MediaPipe detection time.
## 5. Confusion matrix
![confusion matrix](confusion_matrix.png)
Top in-sample confusions (true, pred, count): []
## 6. Feature ablation (in-sample fit)
| Feature set | # features | In-sample acc | Macro-F1 |
|---|---|---|---|
| full_132 | 132 | 100.0% | 1.000 |
| no_relative_126 | 126 | 100.0% | 1.000 |
| right_hand_only_63 | 63 | 95.4% | 0.936 |

> In-sample fit only, same caveat as section 4.
