import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import csv

# ── Load dataset manually ────────────────────────────────
print("Loading dataset...")

X = []
y = []

with open("isl_dataset.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if row and len(row) > 1:
            features = [float(val) for val in row[:-1]]
            label = row[-1].strip()
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Total samples loaded: {len(X)}")

# Count per letter
unique, counts = np.unique(y, return_counts=True)
print(f"Letters found: {list(unique)}")
print("Samples per letter:")
for letter, count in zip(unique, counts):
    print(f"  {letter}: {count}")

# ── Split data ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── Train model ──────────────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Training complete.")

# ── Evaluate ─────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
print(f"{'='*40}")
print("\nPer-letter accuracy:")
print(classification_report(y_test, y_pred))

# ── Save model ───────────────────────────────────────────
with open("isl_model.pkl", "wb") as f:
    pickle.dump(model, f)

labels = sorted(set(y))
with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("\n✅ Model saved as isl_model.pkl")
print("✅ Labels saved as labels.pkl")
print("\nRun isl_main.py for live recognition!")