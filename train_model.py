import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import csv

# ── Load dataset by column name ──────────────────────────
# Reads isl_dataset.csv via header names rather than fixed positions, so it
# works whether or not the signer_id/session_id/burst_id/ts metadata columns
# are present (see isl_main.py's DATASET_COLUMNS / _migrate_dataset_schema).
print("Loading dataset...")


def _build_feature_columns():
    cols = []
    for i in range(21):
        cols.extend([f"rx{i}", f"ry{i}", f"rz{i}"])
    for i in range(21):
        cols.extend([f"lx{i}", f"ly{i}", f"lz{i}"])
    cols.extend(["rel_dx", "rel_dy", "rel_dz", "rel_d_wrist", "rel_d_index", "rel_d_thumb"])
    return cols


FEATURE_COLUMNS = _build_feature_columns()

X, y, groups = [], [], []
with open("isl_dataset.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row or row.get("label") is None:
            continue
        X.append([float(row[c]) for c in FEATURE_COLUMNS])
        y.append(row["label"].strip())
        groups.append(row.get("burst_id") or "unknown_legacy")

X = np.array(X)
y = np.array(y, dtype=object)          # object dtype: keeps plain python str,
groups = np.array(groups, dtype=object)  # avoids "np.str_('...')" in printed output

print(f"Total samples loaded: {len(X)}")
unique, counts = np.unique(y, return_counts=True)
print(f"Classes found: {list(unique)}")
print("Samples per class (and independent recording bursts):")
for label, count in zip(unique, counts):
    n_bursts = len(set(groups[y == label]))
    print(f"  {label}: {count} samples, {n_bursts} burst(s){' -- cannot be held out, only session available' if n_bursts < 2 else ''}")

# ── Held-out evaluation: burst-disjoint only ─────────────────────────
# A plain random split puts near-duplicate frames from the same
# press-and-hold recording action on both sides, which reports near-100%
# accuracy regardless of whether the model generalizes at all. See
# docs/data_collection_protocol.md and results/RESULTS.md for a
# quantified demonstration (run `python evaluate.py`).
n_groups_per_class = {c: len(set(groups[y == c])) for c in unique}
evaluable = [c for c, n in n_groups_per_class.items() if n >= 2]

if len(evaluable) >= 2:
    mask = np.isin(y, evaluable)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X[mask], y[mask], groups=groups[mask]))
    Xs, ys = X[mask], y[mask]
    probe = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, random_state=42, n_jobs=-1)
    probe.fit(Xs[tr_idx], ys[tr_idx])
    pred = probe.predict(Xs[te_idx])
    acc = accuracy_score(ys[te_idx], pred)
    f1m = f1_score(ys[te_idx], pred, average="macro")
    skipped = sorted(set(unique) - set(evaluable))
    print(f"\n{'='*40}")
    print(f"BURST-DISJOINT HELD-OUT ACCURACY: {acc*100:.2f}%  (macro-F1 {f1m:.3f})")
    print(f"Evaluated over {len(evaluable)}/{len(unique)} classes"
          f"{f' (skipped, only 1 session each: {skipped})' if skipped else ''}")
    print(f"{'='*40}")
    print("\nPer-class report (evaluated classes only):")
    print(classification_report(ys[te_idx], pred))
else:
    print(f"\n{'='*40}")
    print("NO HELD-OUT ACCURACY AVAILABLE")
    print(f"{'='*40}")
    print("Fewer than 2 classes have more than one independent recording "
          "session, so no split can measure generalization rather than "
          "memorization of the same session. Record each class in at least "
          "2 separate sessions (see docs/data_collection_protocol.md) to "
          "get a real held-out number.")

# ── Deploy: train the final model on ALL data ────────────────────────
# The split above is only to obtain an honest accuracy estimate where
# possible; the model that actually ships is trained on every sample.
print("\nTraining final RandomForest on all available data...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
)
model.fit(X, y)
print("Training complete.")

with open("isl_model.pkl", "wb") as f:
    pickle.dump(model, f)

labels = sorted(set(y))
with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("\nModel saved as isl_model.pkl")
print("Labels saved as labels.pkl")
print("Run isl_main.py for live recognition.")
