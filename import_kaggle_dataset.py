"""Bulk-import a Kaggle ISL image dataset into a standalone dataset/model,
without touching isl_dataset.csv / isl_model.pkl.

Why separate: isl_dataset.csv is hand-recorded from the live webcam pipeline
(One-Euro filtered) and is the basis of the signer/burst-disjoint evaluation
in docs/data_collection_protocol.md. The Kaggle images are static photos
from unknown contributors with no signer/session tracking — mixing them into
the main dataset would muddy that story. This script writes to
isl_dataset_kaggle.csv / isl_model_kaggle.pkl / labels_kaggle.pkl instead.

Usage:
    # 1) See what would be imported without processing anything (fast):
    python import_kaggle_dataset.py --dir <path-to-extracted-dataset> --list-only

    # 2) Extract features + write the CSV:
    python import_kaggle_dataset.py --dir <path-to-extracted-dataset>

    # 3) Same, and also train isl_model_kaggle.pkl / labels_kaggle.pkl:
    python import_kaggle_dataset.py --dir <path-to-extracted-dataset> --train

<path-to-extracted-dataset> is whatever kagglehub.dataset_download(...)
printed, or wherever you unzipped the manual download — the folder that
directly contains one subfolder per sign (A, B, C, ..., 1, 2, ...).

Numeric-only folders in 1-10 (however named: "1", "01", "10", ...) are
skipped per the request to drop digit classes and keep everything else
(letters, word signs) that the dataset provides.
"""
import os
import csv
import pickle
import argparse

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

from process_images import extract_features

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LANDMARKER_MODEL_PATH = "hand_landmarker.task"

FEATURE_COLUMNS = []
for _i in range(21):
    FEATURE_COLUMNS.extend([f"rx{_i}", f"ry{_i}", f"rz{_i}"])
for _i in range(21):
    FEATURE_COLUMNS.extend([f"lx{_i}", f"ly{_i}", f"lz{_i}"])
FEATURE_COLUMNS.extend(["rel_dx", "rel_dy", "rel_dz", "rel_d_wrist", "rel_d_index", "rel_d_thumb"])
META_COLUMNS = ["signer_id", "session_id", "burst_id", "ts"]
DATASET_COLUMNS = FEATURE_COLUMNS + META_COLUMNS + ["label"]


def is_excluded_numeric_folder(name):
    stripped = name.strip()
    return stripped.isdigit() and 1 <= int(stripped) <= 10


def discover_classes(root_dir):
    """{label: [image_paths]} for every immediate subfolder of root_dir, skipping numeric 1-10 folders."""
    classes = {}
    skipped = []
    for entry in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, entry)
        if not os.path.isdir(full):
            continue
        if is_excluded_numeric_folder(entry):
            skipped.append(entry)
            continue
        images = []
        for dirpath, _, files in os.walk(full):
            for f in files:
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                    images.append(os.path.join(dirpath, f))
        if images:
            classes[entry.strip().upper()] = images
    return classes, skipped


def build_detector(confidence=0.5):
    if not os.path.exists(LANDMARKER_MODEL_PATH):
        raise FileNotFoundError(f"'{LANDMARKER_MODEL_PATH}' not found next to this script.")
    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        # Default 0.5 preserves the original isl_dataset_kaggle.csv exactly
        # (reproducibility). Pass confidence=0.7 to match isl_main.py's live
        # threshold -- this is the CVIP2026-research-upgrade-v2 controlled
        # cross-source experiment (see analyze_domain_shift_controlled.py):
        # re-detecting the SAME images at the SAME threshold the live app
        # uses, to test whether the 0.5-vs-0.7 confound explains any of the
        # cross-source accuracy gap.
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_dataset(classes, out_csv, confidence=0.5):
    print(f"\nInitializing MediaPipe Hand Landmarker (confidence={confidence})...")
    detector = build_detector(confidence)

    rows = []
    burst_counter = 0
    no_hand = 0
    failed = 0

    for label, images in sorted(classes.items()):
        usable = 0
        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None:
                failed += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            if not result.hand_landmarks:
                no_hand += 1
                continue
            features = extract_features(result)
            burst_counter += 1
            # Each image is an independent sample from an unknown contributor,
            # so it gets its own burst_id (never grouped with another image) —
            # that keeps burst-disjoint evaluation meaningful on this data.
            rows.append(features + ["kaggle_unknown", "kaggle_import", f"kaggle_{burst_counter}", "", label])
            usable += 1
        print(f"  {label}: {usable}/{len(images)} usable")

    print(f"\nTotal usable samples: {len(rows)}  (no-hand-detected: {no_hand}, failed to load: {failed})")
    if not rows:
        return None

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(DATASET_COLUMNS)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    return out_csv


def train_from_csv(csv_path, model_path, labels_path):
    print(f"\nLoading {csv_path} for training...")
    X, y, groups = [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[c]) for c in FEATURE_COLUMNS])
            y.append(row["label"].strip())
            groups.append(row.get("burst_id") or "unknown")

    X = np.array(X)
    y = np.array(y, dtype=object)
    groups = np.array(groups, dtype=object)

    unique, counts = np.unique(y, return_counts=True)
    print(f"Classes: {len(unique)}, total samples: {len(X)}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    probe = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, random_state=42, n_jobs=-1)
    probe.fit(X[tr_idx], y[tr_idx])
    pred = probe.predict(X[te_idx])
    acc = accuracy_score(y[te_idx], pred)
    f1m = f1_score(y[te_idx], pred, average="macro")
    print(f"\nHeld-out accuracy (sample-disjoint): {acc*100:.2f}%  (macro-F1 {f1m:.3f})")
    print(classification_report(y[te_idx], pred))

    print("\nTraining final model on all data...")
    model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, random_state=42, n_jobs=-1)
    model.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    labels = sorted(set(y))
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)
    print(f"Saved {model_path} and {labels_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="Path to the extracted Kaggle dataset root.")
    parser.add_argument("--out-csv", default="isl_dataset_kaggle.csv")
    parser.add_argument("--out-model", default="isl_model_kaggle.pkl")
    parser.add_argument("--out-labels", default="labels_kaggle.pkl")
    parser.add_argument("--train", action="store_true", help="Also train + save a model from the resulting CSV.")
    parser.add_argument("--list-only", action="store_true", help="Only print discovered classes/skips; don't process images.")
    parser.add_argument("--confidence", type=float, default=0.5,
                         help="MediaPipe min_hand_detection/presence/tracking confidence (default 0.5, matching "
                              "the original import). Pass 0.7 to match isl_main.py's live threshold for the "
                              "controlled cross-source experiment.")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: '{args.dir}' is not a directory.")
        return

    classes, skipped = discover_classes(args.dir)

    if skipped:
        print(f"Skipping numeric folders (1-10): {sorted(skipped)}")
    if not classes:
        print("No non-numeric label folders with images found — nothing to do.")
        return

    total_images = sum(len(v) for v in classes.values())
    print(f"Found {len(classes)} classes, {total_images} images total:")
    for label, images in sorted(classes.items()):
        print(f"  {label}: {len(images)} images")

    if args.list_only:
        print("\n--list-only set: stopping before feature extraction.")
        return

    out_csv = extract_dataset(classes, args.out_csv, confidence=args.confidence)
    if out_csv and args.train:
        train_from_csv(out_csv, args.out_model, args.out_labels)


if __name__ == "__main__":
    main()
