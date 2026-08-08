import os
import cv2
import csv
import pickle
import argparse
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Feature Extraction logic (must match main app) ─────────────────
def extract_features(result):
    right_landmarks = left_landmarks = None
    if result.hand_landmarks and result.handedness:
        for landmarks, handedness_list in zip(result.hand_landmarks, result.handedness):
            t = handedness_list[0].category_name
            if t == 'Right': right_landmarks = landmarks
            elif t == 'Left': left_landmarks = landmarks
    r_coords = l_coords = None
    r_scale = l_scale = 1.0
    if right_landmarks:
        r_coords = np.array([[lm.x, lm.y, lm.z] for lm in right_landmarks])
        r_scale  = max(np.linalg.norm(r_coords[9] - r_coords[0]), 1e-6)
        r_norm   = ((r_coords - r_coords[0]) / r_scale).flatten().tolist()
    else:
        r_norm = [0.0] * 63
    if left_landmarks:
        l_coords = np.array([[lm.x, lm.y, lm.z] for lm in left_landmarks])
        l_scale  = max(np.linalg.norm(l_coords[9] - l_coords[0]), 1e-6)
        l_norm   = ((l_coords - l_coords[0]) / l_scale).flatten().tolist()
    else:
        l_norm = [0.0] * 63
    if r_coords is not None and l_coords is not None:
        rw, lw = r_coords[0], l_coords[0]
        dx = (lw[0]-rw[0])/r_scale; dy = (lw[1]-rw[1])/r_scale; dz = (lw[2]-rw[2])/r_scale
        d_wrist = np.linalg.norm(lw - rw) / r_scale
        d_index = np.linalg.norm(l_coords[8] - r_coords[8]) / r_scale
        d_thumb = np.linalg.norm(l_coords[4] - r_coords[4]) / r_scale
    else:
        dx = dy = dz = d_wrist = d_index = d_thumb = 0.0
    return r_norm + l_norm + [dx, dy, dz, d_wrist, d_index, d_thumb]

def main():
    parser = argparse.ArgumentParser(description="Process a folder of sign images and retrain the model.")
    parser.add_argument("--dir", type=str, default="images_dataset", help="Path to directory containing images.")
    args = parser.parse_args()

    base_dir = args.dir
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        print("Please create this directory and place your images inside subfolders named after their labels (e.g. 'images_dataset/A', 'images_dataset/HELLO').")
        return

    # Initialize MediaPipe detector
    print("Initializing MediaPipe Hand Landmarker...")
    MODEL_PATH = "hand_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        print(f"Error: MediaPipe model task file '{MODEL_PATH}' not found in the current directory.")
        return

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    mp_options   = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(mp_options)

    # Initialize dataset file with header (Overwrite mode since user is starting fresh)
    print("Writing fresh headers to isl_dataset.csv...")
    with open("isl_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header.extend([f"rx{i}", f"ry{i}", f"rz{i}"])
        for i in range(21):
            header.extend([f"lx{i}", f"ly{i}", f"lz{i}"])
        header.extend(["rel_dx","rel_dy","rel_dz","rel_d_wrist","rel_d_index","rel_d_thumb"])
        header.append("label")
        writer.writerow(header)

    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    processed_count = 0
    failed_count = 0
    no_hand_count = 0

    print(f"Scanning directory: {base_dir}")
    csv_rows = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in supported_extensions:
                continue

            img_path = os.path.join(root, file)

            # Determine label
            rel_path = os.path.relpath(root, base_dir)
            if rel_path != "." and rel_path != "":
                label = rel_path.split(os.sep)[0].strip().upper()
            else:
                filename_without_ext = os.path.splitext(file)[0]
                parts = filename_without_ext.replace("-", "_").split("_")
                label = parts[0].strip().upper() if parts else None

            if not label:
                print(f"Skipping {file}: Could not determine label.")
                continue

            # Load and process image
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Skipping {file}: Failed to load image.")
                failed_count += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                features = extract_features(result)
                csv_rows.append(features + [label])
                processed_count += 1
                print(f"✅ Processed {file} (Label: {label})")
            else:
                print(f"⚠️ No hands detected in {file} (Label: {label}) — skipping")
                no_hand_count += 1

    # Save to CSV
    if csv_rows:
        with open("isl_dataset.csv", "a", newline="") as f:
            csv.writer(f).writerows(csv_rows)
        print(f"\nProcessing Complete: {processed_count} samples saved to isl_dataset.csv.")
    else:
        print("\n❌ No hand landmarks were detected in any of the scanned images. No features saved.")
        return

    if failed_count:
        print(f"❌ Failed to load {failed_count} files.")
    if no_hand_count:
        print(f"⚠️ MediaPipe failed to find hands in {no_hand_count} images.")

    # ── Model Training ───────────────────────────────────────────────
    print("\nTraining classifier on the new image features...")
    X = [row[:-1] for row in csv_rows]
    y = [row[-1] for row in csv_rows]

    X = np.array(X)
    y = np.array(y)

    unique, counts = np.unique(y, return_counts=True)
    print("\nSamples per class:")
    for lbl, cnt in zip(unique, counts):
        print(f"  {lbl}: {cnt}")

    if len(X) < 10:
        print("Error: Need at least 10 samples in total to train.")
        return
    if len(unique) < 2:
        print("Error: Need at least 2 classes/labels to train.")
        return

    # Train Random Forest classifier
    stratify_y = y if np.min(counts) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_y
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=2, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Test Accuracy: {accuracy * 100:.2f}%")
    print("\nPer-class Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and labels
    with open("isl_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    labels = sorted(set(y))
    with open("labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    print("\n✅ New model saved successfully as 'isl_model.pkl'!")
    print("✅ Labels saved successfully as 'labels.pkl'!")
    print("Restart your Flask app and launch the dashboard to try it out.")

if __name__ == "__main__":
    main()
