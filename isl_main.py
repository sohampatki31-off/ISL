#!/usr/bin/env python3
"""
ISL Recognition System — FINAL (Windows)
Install:  pip install flask flask-socketio mediapipe opencv-python scikit-learn numpy pyttsx3
Run:      python final.py
Browser:  http://localhost:5000

Also place 'hand_landmarker.task' in this same folder for hand detection:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import csv
import threading
import time
from flask import Flask, Response, jsonify, request
from flask_socketio import SocketIO, emit
import webbrowser
import os
import queue

# ── TTS (server-side fallback; browser handles primary TTS) ────────
_engine     = None
TTS_AVAILABLE = False
_tts_lock   = threading.Lock()

def _init_tts():
    global _engine, TTS_AVAILABLE
    try:
        import pyttsx3
        e = pyttsx3.init()
        e.setProperty('rate', 150)
        with _tts_lock:
            _engine = e
            TTS_AVAILABLE = True
    except Exception:
        pass

threading.Thread(target=_init_tts, daemon=True, name="tts-init").start()

def speak(text):
    if not TTS_AVAILABLE:
        return
    def _run():
        with _tts_lock:
            _engine.say(text)
            _engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

# ── Flask ──────────────────────────────────────────────────────────
app = Flask(__name__)
sio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ── Load model ─────────────────────────────────────────────────────
print("Loading model...")
model = None
labels = []
model_features = 132

try:
    if os.path.exists("isl_model.pkl"):
        with open("isl_model.pkl", "rb") as f:
            model = pickle.load(f)
        if hasattr(model, "n_features_in_"):
            model_features = model.n_features_in_
            print(f"Model loaded — {model_features} features.")
    if os.path.exists("labels.pkl"):
        with open("labels.pkl", "rb") as f:
            labels = pickle.load(f)
        print(f"Labels loaded — {len(labels)} classes.")
except Exception as e:
    print(f"Warning: Could not load model: {e}")

# ── Initialize dataset ─────────────────────────────────────────────
def init_dataset_file():
    if not os.path.exists("isl_dataset.csv") or os.path.getsize("isl_dataset.csv") == 0:
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
init_dataset_file()

# ── Shared state ───────────────────────────────────────────────────
app_state = {
    "letter": "",
    "confidence": 0.0,
    "stable_count": 0,
    "current_word": "",
    "full_text": "",
    "suggestions": [],
    "hand_detected": False,
    "left_hand_detected": False,
    "right_hand_detected": False,
    "letter_flash": False,
    "model_features_loaded": model_features,
    "history": [],
    "auto_speak": False,
    "total_letters": 0,
    "total_words": 0,
    "tts_id": 0,        # increments each time something should be spoken
    "tts_text": "",     # the text for the latest tts_id
    "camera_status": "starting",
}
state_lock = threading.Lock()
output_frame = None
output_lock = threading.Lock()
camera_status = "starting"   # "starting" | "ok" | error message

def queue_tts(text):
    """Record text to be spoken by the browser on its next /state poll,
    and prepend it to the spoken-history list."""
    text = (text or "").strip()
    if not text:
        return
    with state_lock:
        app_state["tts_id"] += 1
        app_state["tts_text"] = text
        hist = app_state["history"]
        if not hist or hist[0] != text:
            app_state["history"] = ([text] + hist)[:10]

# ── Recording / Training state ─────────────────────────────────────
recording_state = {"is_recording": False, "label": "", "sample_count": 0}
recording_lock = threading.Lock()
training_state  = {"status": "idle", "accuracy": 0.0, "error": ""}
training_lock   = threading.Lock()

# ── Camera-thread internal state ───────────────────────────────────
cam = {
    "current_letter": "",
    "current_word": "",
    "full_text": "",
    "stable_count": 0,
    "last_letter": "",
    "STABILITY_THRESHOLD": 8,
}
cam_lock = threading.Lock()

# ── Runtime resolution control ─────────────────────────────────────
_res = {"width": 640, "height": 480, "pending": False}

FULL_WORD_GESTURES = {
    'LOVE','HELLO','THANKYOU','HELP','SORRY','YES','NO','GOOD','PLEASE'
}

# ── Word suggestions ───────────────────────────────────────────────
ISL_WORDS = [
    "ABLE","ABOUT","ABOVE","AFTER","AGAIN","AGO","AGREE","ALL","ALSO","AND",
    "ANY","ARE","ASK","BAD","BE","BECAUSE","BEFORE","BEST","BETTER","BIG",
    "BOY","BUT","BYE","CALL","CAN","COME","COULD","DAY","DO","DONE","DOWN",
    "EACH","EAT","FEEL","FIND","FINE","FOR","FROM","GET","GIVE","GO","GOOD",
    "GREAT","HAPPY","HAVE","HELLO","HELP","HER","HERE","HIM","HIS","HOW",
    "IF","IN","IS","IT","KNOW","LEARN","LIKE","LOOK","LOVE","MAKE","MAY",
    "ME","MEET","MORE","MY","NAME","NEED","NEW","NEXT","NO","NOT","NOW",
    "OF","OK","OLD","ON","ONE","OR","OUR","OUT","PAIN","PLEASE","PUT",
    "QUICK","QUIET","READY","RIGHT","RUN","SAD","SAY","SEE","SICK","SINCE",
    "SO","SOME","SORRY","STOP","TAKE","TALK","THANK","THANKS","THAT","THE",
    "THEIR","THEM","THEN","THERE","THEY","THIS","TIME","TO","TODAY",
    "TOGETHER","TOO","TRY","UNDERSTAND","UP","US","VERY","WAIT","WANT",
    "WE","WELL","WHAT","WHEN","WHERE","WHO","WHY","WILL","WITH","WORK",
    "WOULD","YES","YOU","YOUR",
]

def get_suggestions(word):
    if not word:
        return []
    return [w for w in ISL_WORDS if w.startswith(word.upper())][:3]

# ── Hand connections ───────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ── MediaPipe detector (created lazily inside detector_loop) ───────
MODEL_PATH = "hand_landmarker.task"
smoothed_landmarks = {}

def draw_landmarks(frame, result):
    global smoothed_landmarks
    if not result.hand_landmarks or not getattr(result, 'handedness', None):
        smoothed_landmarks.clear()
        return frame
    h, w = frame.shape[:2]
    alpha = 0.5
    current_hands = set()
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        hand_type = handedness[0].category_name
        current_hands.add(hand_type)
        pts = [[lm.x * w, lm.y * h] for lm in hand_landmarks]
        if hand_type in smoothed_landmarks:
            for i in range(len(pts)):
                pts[i][0] = alpha * pts[i][0] + (1-alpha) * smoothed_landmarks[hand_type][i][0]
                pts[i][1] = alpha * pts[i][1] + (1-alpha) * smoothed_landmarks[hand_type][i][1]
        smoothed_landmarks[hand_type] = pts
        draw_pts = [(int(p[0]), int(p[1])) for p in pts]
        if hand_type == 'Right':
            line_col=(180,140,20); node_col=(220,165,30); tip_col=(80,255,160); edge_col=(80,60,0)
        else:
            line_col=(20,165,140); node_col=(30,200,170); tip_col=(255,200,80); edge_col=(0,80,70)
        for s, e in HAND_CONNECTIONS:
            cv2.line(frame, draw_pts[s], draw_pts[e], line_col, 2)
        for i, pt in enumerate(draw_pts):
            r = 7 if i in [4,8,12,16,20] else 5
            col = tip_col if i in [4,8,12,16,20] else node_col
            cv2.circle(frame, pt, r, col, -1)
            cv2.circle(frame, pt, r+1, edge_col, 1)
    for k in list(smoothed_landmarks.keys()):
        if k not in current_hands:
            del smoothed_landmarks[k]
    return frame

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

# ── Queues ─────────────────────────────────────────────────────────
raw_q    = queue.Queue(maxsize=1)
encode_q = queue.Queue(maxsize=1)

def _put_drop(q, item):
    if q.full():
        try: q.get_nowait()
        except queue.Empty: pass
    q.put_nowait(item)

# ── Thread 1: Frame grabber ────────────────────────────────────────
def frame_grabber():
    global camera_status
    # Windows backend order: DirectShow, then Media Foundation, then the
    # default — each tried on camera indices 0 and 1.
    attempts = [
        (0, cv2.CAP_DSHOW, "DSHOW"), (1, cv2.CAP_DSHOW, "DSHOW"),
        (0, cv2.CAP_MSMF,  "MSMF"),  (1, cv2.CAP_MSMF,  "MSMF"),
        (0, 0, "DEFAULT"),           (1, 0, "DEFAULT"),
    ]

    cap = None
    for src, flag, name in attempts:
        try:
            test = cv2.VideoCapture(src, flag) if flag else cv2.VideoCapture(src)
            if test.isOpened():
                ok, _ = test.read()        # confirm it actually yields a frame
                if ok:
                    cap = test
                    print(f"Camera opened: index {src} via {name} backend.")
                    break
            test.release()
        except Exception as e:
            print(f"Camera attempt (index {src}, {name}) failed: {e}")

    if cap is None or not cap.isOpened():
        camera_status = ("Could not open camera. Close other apps using it "
                         "(Zoom, Teams, Camera app, browser tabs) and check "
                         "Settings > Privacy > Camera > Let desktop apps use camera.")
        print("ERROR: " + camera_status)
        return

    camera_status = "ok"
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _res["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _res["height"])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        if _res["pending"]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  _res["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _res["height"])
            _res["pending"] = False
            print(f"Camera resolution changed to {_res['width']}x{_res['height']}")
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue
        _put_drop(raw_q, cv2.flip(frame, 1))
    cap.release()

# ── Thread 2: Detector ─────────────────────────────────────────────
def detector_loop():
    global model, labels, model_features, camera_status
    detector = None
    if not os.path.exists(MODEL_PATH):
        print("=" * 60)
        print(f"WARNING: '{MODEL_PATH}' not found in this folder.")
        print("Hand detection is DISABLED — the camera feed will still")
        print("show, but no letters will be recognized. Download it from:")
        print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        print("and place it next to this script.")
        print("=" * 60)
    else:
        print("Loading MediaPipe hand landmarker...")
        try:
            base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            mp_options   = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.7,
            )
            detector = vision.HandLandmarker.create_from_options(mp_options)
            print("MediaPipe ready.")
        except Exception as e:
            detector = None
            print(f"WARNING: Could not initialize MediaPipe ({e}).")
            print("Camera feed will still show, but detection is disabled.")

    while True:
        try:
            frame = raw_q.get(timeout=1.0)
        except queue.Empty:
            continue

        # No detector → just pass the raw camera frame straight to the
        # encoder so the live feed is always visible.
        if detector is None:
            _put_drop(encode_q, frame)
            continue

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        det_letter   = ""
        det_conf     = 0.0
        hand_det     = bool(result.hand_landmarks)
        letter_flash = False
        right_det = left_det = False
        auto_speak_text = None

        if result.hand_landmarks and result.handedness:
            for hl in result.handedness:
                t = hl[0].category_name
                if t == 'Right': right_det = True
                elif t == 'Left': left_det = True

        if result.hand_landmarks:
            draw_landmarks(frame, result)
            feat_132 = extract_features(result)

            with recording_lock:
                is_rec    = recording_state["is_recording"]
                rec_label = recording_state["label"]
            if is_rec:
                try:
                    with open("isl_dataset.csv", "a", newline="") as f:
                        csv.writer(f).writerow(feat_132 + [rec_label])
                    with recording_lock:
                        recording_state["sample_count"] += 1
                        count = recording_state["sample_count"]
                    sio.emit("sample_recorded", {"label": rec_label, "count": count})
                except Exception as e:
                    print(f"Error saving sample: {e}")

            if model is not None:
                try:
                    if model_features == 132:
                        feat = np.array(feat_132).reshape(1, -1)
                    else:
                        coords = np.array([[lm.x,lm.y,lm.z] for lm in result.hand_landmarks[0]])
                        coords -= coords[0]
                        s = np.linalg.norm(coords[9])
                        if s > 0: coords /= s
                        feat = coords.flatten().reshape(1, -1)
                    pred    = model.predict(feat)[0]
                    prob    = model.predict_proba(feat)[0]
                    det_letter = str(pred)
                    det_conf   = float(np.max(prob))
                except Exception:
                    det_letter = "Err"; det_conf = 0.0

            with cam_lock:
                if det_letter == cam["last_letter"] and det_conf > 0.5:
                    cam["stable_count"] = min(cam["stable_count"]+1, cam["STABILITY_THRESHOLD"])
                else:
                    cam["stable_count"] = max(cam["stable_count"]-1, 0)
                    cam["last_letter"]  = det_letter

                if cam["stable_count"] >= cam["STABILITY_THRESHOLD"]:
                    if det_letter != cam["current_letter"]:
                        cam["current_letter"] = det_letter
                        letter_flash = True
                        if det_letter in FULL_WORD_GESTURES:
                            if cam["current_word"]:
                                cam["full_text"] += cam["current_word"] + " "
                            cam["full_text"]    += det_letter + " "
                            cam["current_word"]  = ""
                            cam["stable_count"]  = 0
                            with state_lock:
                                if app_state["auto_speak"]:
                                    auto_speak_text = det_letter
                        else:
                            cam["current_word"] += det_letter
                            cam["stable_count"]  = 0

                word = cam["current_word"]; full = cam["full_text"]; stab = cam["stable_count"]
        else:
            with cam_lock:
                cam["stable_count"] = max(cam["stable_count"]-1, 0)
                word = cam["current_word"]; full = cam["full_text"]; stab = cam["stable_count"]

        sugs = get_suggestions(word)

        with state_lock:
            app_state.update({
                "letter": det_letter, "confidence": det_conf, "stable_count": stab,
                "current_word": word, "full_text": full, "suggestions": sugs,
                "hand_detected": hand_det, "left_hand_detected": left_det,
                "right_hand_detected": right_det,
                "model_features_loaded": model_features,
                "camera_status": camera_status,
            })
            if letter_flash:
                # Leave it True; the /state route resets it once the client
                # has read it, so polling never misses a flash.
                app_state["letter_flash"] = True
            s_copy = app_state.copy()

        sio.emit('state_update', s_copy)   # harmless if no socket client

        if auto_speak_text:
            queue_tts(auto_speak_text)

        _put_drop(encode_q, frame)

# ── Thread 3: Encoder ──────────────────────────────────────────────
def camera_loop():
    global output_frame
    _first_frame = True
    while True:
        try:
            frame = encode_q.get(timeout=1.0)
        except queue.Empty:
            continue
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if ok:
            jpg_bytes = buf.tobytes()
            with output_lock:
                output_frame = jpg_bytes
            # The browser pulls video by polling /frame.jpg, so we don't
            # push frames over Socket.IO (that flooded the channel and
            # stalled text updates). We only signal first-frame readiness.
            if _first_frame:
                _first_frame = False
                sio.emit('camera_ready', {})

# ── Model training ─────────────────────────────────────────────────
def train_model_bg(silent=False):
    global model, labels, model_features
    # Silent (auto) retraining: bail out quietly unless there's enough data.
    if silent:
        try:
            if not os.path.exists("isl_dataset.csv"):
                return
            X, y = [], []
            with open("isl_dataset.csv", "r") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row:
                        X.append([float(v) for v in row[:-1]])
                        y.append(row[-1].strip())
            if len(X) < 10 or len(set(y)) < 2:
                return
        except Exception:
            return

    with training_lock:
        if training_state["status"] == "training":
            return
        training_state["status"] = "training"; training_state["error"] = ""
    sio.emit("training_status", training_state)
    try:
        X, y = [], []
        if not os.path.exists("isl_dataset.csv"):
            raise Exception("Dataset file not found.")
        with open("isl_dataset.csv", "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header: raise Exception("Dataset is empty.")
            for row in reader:
                if row:
                    X.append([float(v) for v in row[:-1]])
                    y.append(row[-1].strip())
        if len(X) < 10:
            raise Exception("Need at least 10 samples to train.")
        X = np.array(X); y = np.array(y)
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2: raise Exception("Need at least 2 classes.")
        strat = y if np.min(counts) > 1 else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
        clf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=2, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        acc = float(accuracy_score(y_te, clf.predict(X_te)))
        with open("isl_model.pkl", "wb") as f: pickle.dump(clf, f)
        new_labels = sorted(set(y))
        with open("labels.pkl", "wb") as f: pickle.dump(new_labels, f)
        with cam_lock:
            model = clf; labels = new_labels
            model_features = clf.n_features_in_ if hasattr(clf, "n_features_in_") else len(X[0])
        with training_lock:
            training_state["status"] = "success"; training_state["accuracy"] = acc
        sio.emit("training_status", training_state)
    except Exception as e:
        print(f"Training error: {e}")
        with training_lock:
            training_state["status"] = "error"; training_state["error"] = str(e)
        sio.emit("training_status", training_state)

# ── Dataset utilities ──────────────────────────────────────────────
def get_dataset_counts():
    counts = {l: 0 for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    for g in FULL_WORD_GESTURES: counts[g] = 0
    if os.path.exists("isl_dataset.csv"):
        try:
            with open("isl_dataset.csv", "r") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row:
                        lbl = row[-1].strip().upper()
                        counts[lbl] = counts.get(lbl, 0) + 1
        except Exception as e:
            print(f"Error reading dataset: {e}")
    return counts

# ── Flask Routes ───────────────────────────────────────────────────
@app.route('/state')
def get_state():
    with state_lock:
        s = app_state.copy()
        if s["letter_flash"]: app_state["letter_flash"] = False
    return jsonify(s)

@app.route('/dataset_info')
def dataset_info():
    return jsonify(get_dataset_counts())

@app.route('/start_recording', methods=['POST'])
def start_recording():
    data  = request.get_json(silent=True) or {}
    label = data.get("label", "").strip().upper()
    if not label: return jsonify({"error": "No label"}), 400
    counts = get_dataset_counts()
    with recording_lock:
        recording_state.update({"is_recording": True, "label": label, "sample_count": counts.get(label, 0)})
    return jsonify({"status": "ok", "current_count": counts.get(label, 0)})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    with recording_lock:
        recording_state["is_recording"] = False
        lbl, count = recording_state["label"], recording_state["sample_count"]
    threading.Thread(target=train_model_bg, args=(True,), daemon=True).start()
    return jsonify({"status": "ok", "label": lbl, "count": count, "counts": get_dataset_counts()})

@app.route('/delete_latest', methods=['POST'])
def delete_latest():
    data  = request.get_json(silent=True) or {}
    label = data.get("label", "").strip().upper()
    if not label: return jsonify({"error": "No label"}), 400
    rows_other, rows_letter, header = [], [], None
    if os.path.exists("isl_dataset.csv"):
        with open("isl_dataset.csv", "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    (rows_letter if row[-1].strip().upper() == label else rows_other).append(row)
    if rows_letter: rows_letter.pop()
    with open("isl_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        if header: writer.writerow(header)
        writer.writerows(rows_other + rows_letter)
    threading.Thread(target=train_model_bg, args=(True,), daemon=True).start()
    return jsonify({"status": "ok", "count": len(rows_letter), "counts": get_dataset_counts()})

@app.route('/clear_class', methods=['POST'])
def clear_class():
    data  = request.get_json(silent=True) or {}
    label = data.get("label", "").strip().upper()
    if not label: return jsonify({"error": "No label"}), 400
    rows_other, header = [], None
    if os.path.exists("isl_dataset.csv"):
        with open("isl_dataset.csv", "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and row[-1].strip().upper() != label:
                    rows_other.append(row)
    with open("isl_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        if header: writer.writerow(header)
        writer.writerows(rows_other)
    threading.Thread(target=train_model_bg, args=(True,), daemon=True).start()
    return jsonify({"status": "ok", "count": 0, "counts": get_dataset_counts()})

@app.route('/train', methods=['POST'])
def trigger_train():
    threading.Thread(target=train_model_bg, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/training_status')
def get_training_status():
    with training_lock: return jsonify(training_state)

@app.route('/action', methods=['POST'])
def action():
    data     = request.get_json(silent=True) or {}
    act_name = data.get('action', '')
    tts_text = None

    if act_name == 'auto_speak':
        with state_lock:
            app_state["auto_speak"] = bool(data.get("value", False))
        return jsonify({"status": "ok"})

    if act_name == 'clear_history':
        with state_lock:
            app_state["history"] = []
        sio.emit("history_update", {"history": []})
        return jsonify({"status": "ok"})

    with cam_lock:
        if act_name == 'space':
            if cam["current_word"]:
                committed = cam["current_word"]
                cam["full_text"]    += committed + " "
                cam["current_word"]  = ""
                cam["current_letter"]= ""
                cam["stable_count"]  = 0
                with state_lock:
                    app_state["total_words"]   += 1
                    app_state["total_letters"] += len(committed)
                    if app_state["auto_speak"]:
                        tts_text = committed
        elif act_name == 'enter':
            tts_text = (cam["full_text"] + " " + cam["current_word"]).strip()
        elif act_name == 'backspace':
            if cam["current_word"]:
                cam["current_word"]   = cam["current_word"][:-1]
                cam["current_letter"] = ""
            elif cam["full_text"]:
                cam["full_text"] = cam["full_text"].rstrip()
                cam["full_text"] = (cam["full_text"][:cam["full_text"].rfind(" ")+1]
                                    if " " in cam["full_text"] else "")
        elif act_name == 'backspace_word':
            if cam["current_word"]:
                cam["current_word"]   = ""
                cam["current_letter"] = ""
                cam["stable_count"]   = 0
            elif cam["full_text"]:
                cam["full_text"] = cam["full_text"].rstrip()
                cam["full_text"] = (cam["full_text"][:cam["full_text"].rfind(" ")+1]
                                    if " " in cam["full_text"] else "")
        elif act_name == 'clear':
            cam["current_word"] = cam["full_text"] = cam["current_letter"] = ""
        elif act_name == 'punct':
            c = data.get('char', '')
            if c in '.!?,;:…':
                if cam["full_text"]:
                    cam["full_text"] = cam["full_text"].rstrip() + c + " "
                elif cam["current_word"]:
                    cam["current_word"] += c
        elif act_name == 'quick_phrase':
            phrase = data.get('text', '').strip().upper()
            if phrase:
                if cam["current_word"]:
                    cam["full_text"] += cam["current_word"] + " "
                    cam["current_word"] = ""
                cam["full_text"] += phrase + " "
                cam["current_letter"] = ""
                cam["stable_count"] = 0
                with state_lock:
                    app_state["total_words"] += len(phrase.split())
                tts_text = phrase
        elif act_name == 'suggest':
            idx  = data.get('index', 0)
            sugs = get_suggestions(cam["current_word"])
            if 0 <= idx < len(sugs):
                chosen = sugs[idx]
                cam["full_text"]    += chosen + " "
                cam["current_word"]  = ""
                cam["current_letter"]= ""
                cam["stable_count"]  = 0
                with state_lock:
                    if app_state["auto_speak"]:
                        tts_text = chosen

    if tts_text:
        queue_tts(tts_text)

    return jsonify({"status": "ok", "tts": (tts_text or "")})

@app.route('/set_resolution', methods=['POST'])
def set_resolution():
    data = request.get_json(silent=True) or {}
    w, h = int(data.get('width', 1280)), int(data.get('height', 720))
    if (w, h) not in [(1280, 720), (640, 480)]:
        return jsonify({"error": "Unsupported resolution"}), 400
    _res["width"] = w; _res["height"] = h; _res["pending"] = True
    return jsonify({"status": "ok", "width": w, "height": h})

@app.route('/download_history')
def download_history():
    with state_lock:
        hist = list(app_state.get("history", []))
    lines = ["ISL Recognition System — Session Transcript", "=" * 44, ""]
    lines += list(reversed(hist)) if hist else ["(no sentences spoken in this session)"]
    return Response("\n".join(lines), mimetype='text/plain',
                    headers={"Content-Disposition": "attachment;filename=isl_session.txt"})

BUILD_TAG = "final-poll-2-nosocket"   # bump to confirm the running code is current

@app.route('/')
def index():
    resp = Response(HTML, mimetype='text/html')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp

@app.route('/version')
def version():
    # Open http://localhost:5000/version to confirm you're running THIS file.
    return jsonify({"build": BUILD_TAG, "camera_status": camera_status,
                    "has_frame": output_frame is not None})

# Cached placeholder frames so the <img> always renders something,
# making it obvious whether the stream works vs. the camera feeding it.
_placeholder_cache = {}
def _placeholder_jpeg(msg):
    if msg in _placeholder_cache:
        return _placeholder_cache[msg]
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (10, 8, 6)  # match the dark UI background (BGR)
    # Wrap the message across lines so it fits.
    words, lines, cur = msg.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= 34:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    y = 240 - (len(lines) - 1) * 16
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(img, line, (320 - tw // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (90, 162, 232), 1, cv2.LINE_AA)
        y += 32
    ok, buf = cv2.imencode('.jpg', img)
    data = buf.tobytes() if ok else b""
    _placeholder_cache[msg] = data
    return data

@app.route('/frame.jpg')
def frame_jpg():
    # Single JPEG per request. The browser polls this many times a second.
    # This is the most reliable delivery method for the Flask-SocketIO
    # threading server (no streaming, no WebSocket). When the camera has
    # no frame yet, a visible placeholder is returned so the panel never
    # shows a broken image and instead tells you what's happening.
    try:
        with output_lock:
            jpg = output_frame
        if jpg is None:
            if camera_status in ("ok", "starting"):
                jpg = _placeholder_jpeg("Waiting for camera...")
            else:
                jpg = _placeholder_jpeg(camera_status)
    except Exception as e:
        jpg = _placeholder_jpeg("Server error: " + str(e))
    resp = Response(jpg, mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp


# ══════════════════════════════════════════════════════════════════
# HTML DASHBOARD — Production Build v2.0
# Design: Syne + DM Sans, amber/saffron accent on deep navy
# Features: Browser TTS, Voice Selector, History, Copy, Auto-Speak
# ══════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ISL Recognition System</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<!-- Socket.IO removed: the UI now uses plain HTTP polling, which avoids
     Socket.IO/Engine.IO client-vs-server version mismatches entirely. -->
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --ground:#07080D;--surface:rgba(11,13,22,0.9);--surface2:rgba(16,19,32,0.95);
  --border:rgba(255,255,255,0.06);--border-hi:rgba(255,255,255,0.12);
  --text:#EDE8DA;--text2:rgba(237,232,218,0.55);--text3:rgba(237,232,218,0.2);
  --amber:#E8A225;--amberd:rgba(232,162,37,0.12);--amberg:rgba(232,162,37,0.28);--amberm:rgba(232,162,37,0.5);
  --teal:#00C4A7;--teald:rgba(0,196,167,0.1);--tealg:rgba(0,196,167,0.22);
  --emerald:#22C55E;--emg:rgba(34,197,94,0.2);
  --rose:#F43F5E;--roseg:rgba(244,63,94,0.2);
  --purple:#7C6FF7;--purpleg:rgba(124,111,247,0.2);
  --shadow:0 8px 40px rgba(0,0,0,0.55);
}
html,body{height:100%;background:var(--ground);color:var(--text);font-family:'DM Sans',sans-serif;overflow:hidden}
body{
  display:flex;flex-direction:column;
  background-image:
    radial-gradient(ellipse at 15% 0%,rgba(232,162,37,0.04) 0,transparent 55%),
    radial-gradient(ellipse at 85% 100%,rgba(0,196,167,0.04) 0,transparent 55%);
}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.08);border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,0.14)}
*{scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.08) transparent}

/* ── LOADING ─────────────────────────────────────────────────── */
.loading-screen{
  position:fixed;inset:0;z-index:9999;background:var(--ground);
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;
  transition:opacity .55s ease,visibility .55s ease;
}
.loading-screen.hidden{opacity:0;visibility:hidden;pointer-events:none}
.ld-logo{font-family:'Syne',sans-serif;font-weight:800;font-size:5rem;color:var(--amber);
  letter-spacing:-0.02em;animation:ld-pulse 2s ease-in-out infinite;}
@keyframes ld-pulse{0%,100%{text-shadow:0 0 40px var(--amberg)}50%{text-shadow:0 0 80px rgba(232,162,37,.5),0 0 120px rgba(232,162,37,.2)}}
.ld-sub{font-size:.72rem;color:var(--text3);letter-spacing:.12em;text-transform:uppercase}
.ld-bar{width:200px;height:2px;background:rgba(255,255,255,.05);border-radius:1px;overflow:hidden}
.ld-fill{height:100%;background:var(--amber);border-radius:1px;animation:ld-sweep 1.4s ease-in-out infinite}
@keyframes ld-sweep{0%{transform:translateX(-100%)}100%{transform:translateX(200px)}}

/* ── HEADER ──────────────────────────────────────────────────── */
header{
  height:58px;min-height:58px;background:rgba(7,8,13,.92);
  border-bottom:1px solid var(--border);backdrop-filter:blur(20px);
  display:flex;align-items:center;padding:0 18px;gap:10px;z-index:100;flex-shrink:0;
}
.logo{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;letter-spacing:.1em;color:var(--amber);white-space:nowrap;flex-shrink:0}
.logo em{font-style:normal;color:var(--text3);font-weight:400;font-size:.85em;letter-spacing:.05em}
.nav-tabs{display:flex;gap:3px;margin-left:16px;flex-shrink:0}
.tab-btn{
  background:transparent;border:1px solid transparent;color:var(--text3);
  font-family:'DM Sans',sans-serif;font-size:.72rem;font-weight:600;letter-spacing:.08em;
  padding:6px 13px;border-radius:7px;cursor:pointer;transition:all .22s;
}
.tab-btn:hover{color:var(--text);background:rgba(255,255,255,.03)}
.tab-btn.active{color:var(--amber);background:var(--amberd);border-color:rgba(232,162,37,.2)}
.hdr-end{display:flex;align-items:center;gap:9px;margin-left:auto;flex-shrink:0}
.hdr-div{width:1px;height:18px;background:var(--border);flex-shrink:0}
.voice-sel{
  background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:7px;
  color:var(--text2);font-family:'DM Sans',sans-serif;font-size:.7rem;
  padding:5px 26px 5px 9px;outline:none;cursor:pointer;appearance:none;max-width:150px;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%236E6A62'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 8px center;
}
.voice-sel:focus{border-color:var(--amber);color:var(--text)}
.voice-sel option,.voice-sel optgroup{background:#111318}
.rate-wrap{display:flex;align-items:center;gap:5px;font-size:.68rem;color:var(--text3);white-space:nowrap}
.rate-wrap input[type=range]{width:62px;accent-color:var(--amber);cursor:pointer}
.toggle-wrap{display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none}
.toggle-wrap span{font-size:.68rem;font-weight:600;letter-spacing:.06em;color:var(--text3);white-space:nowrap}
.toggle-wrap input{display:none}
.tgl-track{width:32px;height:17px;background:rgba(255,255,255,.05);border:1px solid var(--border);border-radius:9px;position:relative;transition:all .22s}
.tgl-thumb{width:11px;height:11px;background:var(--text3);border-radius:50%;position:absolute;top:2px;left:2px;transition:all .22s}
.toggle-wrap input:checked + .tgl-track{background:var(--amberd);border-color:rgba(232,162,37,.3)}
.toggle-wrap input:checked + .tgl-track .tgl-thumb{transform:translateX(15px);background:var(--amber)}
.stat-pill{
  background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:6px;
  padding:3px 9px;font-size:.66rem;color:var(--text3);white-space:nowrap;
}
.stat-pill b{color:var(--teal)}

/* ── LAYOUT ──────────────────────────────────────────────────── */
.tab-content{flex:1;display:none;min-height:0;padding:12px;gap:12px}
.tab-content.active{display:grid}
#recognize-tab{grid-template-columns:42fr 28fr 30fr}
#train-tab{grid-template-columns:45fr 55fr}

/* ── GLASS PANEL ─────────────────────────────────────────────── */
.glass-panel{
  background:var(--surface);border:1px solid var(--border);border-radius:13px;
  backdrop-filter:blur(16px);box-shadow:var(--shadow);
  display:flex;flex-direction:column;min-height:0;overflow:hidden;
}
.panel-hdr{
  padding:10px 16px;border-bottom:1px solid var(--border);
  font-size:.62rem;font-weight:600;letter-spacing:.14em;color:var(--text3);
  display:flex;align-items:center;gap:8px;flex-shrink:0;
}
.panel-hdr::before{content:'';width:3px;height:10px;background:var(--amber);border-radius:2px;box-shadow:0 0 6px var(--amberg)}

/* ── VIDEO ───────────────────────────────────────────────────── */
.video-wrap{flex:1;position:relative;background:#03040A;min-height:0}
.video-wrap canvas{display:block;width:100%;height:100%}
.video-wrap img.feed{display:block;width:100%;height:100%;object-fit:cover}
.hud{position:absolute;inset:12px;pointer-events:none;display:flex;flex-direction:column;justify-content:space-between}
.hud-row{display:flex;justify-content:space-between;align-items:flex-start}
.hud-row.btm{align-items:flex-end}
.hand-card{
  background:rgba(7,8,13,.88);backdrop-filter:blur(10px);
  border:1px solid var(--border);border-radius:9px;
  padding:6px 12px;display:flex;gap:12px;
}
.hand-ind{display:flex;align-items:center;gap:5px;font-size:.62rem;font-weight:700;letter-spacing:.05em;color:var(--text3);transition:color .3s}
.hand-ind .dot{width:6px;height:6px;border-radius:50%;background:var(--text3);transition:all .3s}
.hand-ind.on{color:var(--teal)}.hand-ind.on .dot{background:var(--teal);box-shadow:0 0 7px var(--teal)}
.warn-badge{background:rgba(244,63,94,.08);border:1px solid rgba(244,63,94,.35);border-radius:9px;padding:6px 12px;color:var(--rose);font-size:.68rem;font-weight:600;backdrop-filter:blur(6px);animation:pulse-rose 2s infinite}
@keyframes pulse-rose{0%,100%{border-color:rgba(244,63,94,.2)}50%{border-color:rgba(244,63,94,.6)}}
.rec-badge{background:rgba(244,63,94,.12);border:1px solid var(--rose);border-radius:20px;padding:4px 11px;color:var(--rose);font-size:.62rem;font-weight:700;letter-spacing:.06em;display:flex;align-items:center;gap:5px}
.pulse-dot{width:6px;height:6px;border-radius:50%;background:var(--rose);animation:blink 1.2s infinite}
@keyframes blink{0%{transform:scale(.9);opacity:.5}50%{transform:scale(1.2);opacity:1;box-shadow:0 0 7px var(--rose)}100%{transform:scale(.9);opacity:.5}}

/* ── RECOGNITION PANEL ───────────────────────────────────────── */
.letter-box{
  height:158px;flex-shrink:0;display:flex;align-items:center;justify-content:center;
  position:relative;border-bottom:1px solid var(--border);
  background:radial-gradient(circle at 50% 55%,rgba(232,162,37,.04) 0,transparent 65%);
}
.letter-char{font-family:'Syne',sans-serif;font-size:7.2rem;font-weight:800;line-height:1;color:var(--amber);text-shadow:0 0 40px var(--amberg);user-select:none;transition:color .15s}
.letter-char.empty{color:var(--text3);text-shadow:none}
.letter-char.flash{animation:lflash .45s cubic-bezier(.1,.8,.3,1)}
@keyframes lflash{0%{transform:scale(.76);color:#fff;text-shadow:0 0 50px #fff,0 0 90px var(--amber)}55%{transform:scale(1.04)}100%{transform:scale(1);color:var(--amber)}}

/* scrollable rec content */
.rec-scroll{flex:1;overflow-y:auto;min-height:0;display:flex;flex-direction:column}

.metrics{padding:12px 16px;border-bottom:1px solid var(--border);display:flex;flex-direction:column;gap:9px;flex-shrink:0}
.mrow{display:flex;align-items:center;gap:9px}
.mlabel{font-size:.59rem;font-weight:700;letter-spacing:.1em;color:var(--text3);width:82px;flex-shrink:0}
.mbar{flex:1;height:4px;background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:2px;overflow:hidden}
.mfill{height:100%;border-radius:2px;width:0%;transition:width .14s ease,background-color .3s}
.mfill.amber{background:var(--amber);box-shadow:0 0 5px var(--amberg)}
.mfill.teal{background:var(--teal);box-shadow:0 0 5px var(--tealg)}
.mfill.em{background:var(--emerald);box-shadow:0 0 5px var(--emg)}
.mfill.rose{background:var(--rose);box-shadow:0 0 5px var(--roseg)}
.mval{font-family:'JetBrains Mono',monospace;font-size:.69rem;font-weight:600;color:var(--text3);width:38px;text-align:right;flex-shrink:0}

.sparkline-wrap{padding:7px 16px 5px;border-bottom:1px solid var(--border);flex-shrink:0}
.sp-lbl{font-size:.57rem;font-weight:700;letter-spacing:.14em;color:var(--text3);margin-bottom:4px}
#sparkline-canvas{width:100%;height:28px;border-radius:3px;background:rgba(0,0,0,.1);display:block}

.word-sec{padding:11px 16px;border-bottom:1px solid var(--border);flex-shrink:0}
.sec-lbl{font-size:.58rem;font-weight:700;letter-spacing:.14em;color:var(--text3);margin-bottom:4px}
.word-disp{font-family:'JetBrains Mono',monospace;font-size:1.45rem;font-weight:500;color:var(--text);letter-spacing:.04em;min-height:1.8rem;display:flex;align-items:center}
.word-disp.empty{color:var(--text3)}
.text-sec .sec-lbl{display:flex;align-items:center;justify-content:space-between}
.mini-copy{background:var(--amber-dim,rgba(232,162,37,.12));border:1px solid rgba(232,162,37,.3);color:var(--amber);font-family:'DM Sans',sans-serif;font-size:.62rem;font-weight:700;letter-spacing:.06em;padding:3px 9px;border-radius:6px;cursor:pointer;transition:all .15s}
.mini-copy:hover{background:rgba(232,162,37,.22)}
.text-box{
  margin-top:4px;min-height:64px;max-height:140px;overflow-y:auto;
  background:rgba(0,0,0,.28);border:1px solid rgba(232,162,37,.18);border-radius:10px;
  padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:1.05rem;line-height:1.5;
  color:var(--text);white-space:pre-wrap;word-break:break-word;letter-spacing:.02em;
}
.text-box.empty{color:var(--text3);font-style:italic;font-family:'DM Sans',sans-serif;font-size:.82rem}
.text-box .pending{color:var(--amber);border-bottom:2px solid var(--amber);padding-bottom:1px}

.sugs-sec{padding:9px 16px;border-bottom:1px solid var(--border);flex-shrink:0}
.sugs-list{display:flex;gap:6px;flex-wrap:wrap;min-height:24px}
.sug-pill{background:var(--amberd);border:1px solid rgba(232,162,37,.18);border-radius:20px;color:var(--amber);font-family:'JetBrains Mono',monospace;font-size:.67rem;font-weight:500;padding:3px 11px;cursor:pointer;transition:all .18s}
.sug-pill:hover{background:rgba(232,162,37,.18);border-color:var(--amberm);transform:translateY(-1px);box-shadow:0 3px 10px var(--amberg)}
.sug-pill .hint{opacity:.4;font-size:.65em;margin-right:3px}

.quick-sec{padding:9px 16px;flex-shrink:0}
.quick-pills{display:flex;gap:5px;flex-wrap:wrap;margin-top:5px}
.qpill{background:var(--teald);border:1px solid rgba(0,196,167,.15);border-radius:20px;color:var(--teal);font-family:'DM Sans',sans-serif;font-size:.65rem;font-weight:600;padding:3px 10px;cursor:pointer;transition:all .18s}
.qpill:hover{background:rgba(0,196,167,.16);border-color:var(--teal);transform:translateY(-1px);box-shadow:0 3px 9px var(--tealg)}

/* fixed bottom controls */
.punct-row{display:flex;gap:5px;padding:7px 12px;flex-shrink:0;border-top:1px solid var(--border)}
.punct-btn{flex:1;background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:6px;color:var(--text3);font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:500;padding:5px 3px;cursor:pointer;transition:all .16s}
.punct-btn:hover{background:rgba(255,255,255,.06);color:var(--text);border-color:var(--border-hi)}

.action-controls{padding:0 12px 12px;display:flex;flex-direction:column;gap:6px;flex-shrink:0}
.act-row{display:grid;gap:6px}
.act-row-2{grid-template-columns:1fr 1fr}
.act-row-3{grid-template-columns:1fr 1fr 1fr}
.abtn{background:rgba(255,255,255,.025);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:'DM Sans',sans-serif;font-weight:600;font-size:.72rem;padding:8px 10px;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:6px;transition:all .18s}
.abtn:hover{background:rgba(255,255,255,.05);border-color:var(--border-hi)}
.abtn.amber-btn{background:var(--amberd);border-color:rgba(232,162,37,.22);color:var(--amber)}
.abtn.amber-btn:hover{background:rgba(232,162,37,.18);border-color:var(--amberm);box-shadow:0 0 12px var(--amberg)}
.abtn.danger:hover{background:var(--roseg);border-color:var(--rose);color:var(--rose)}
.abtn.teal-btn{background:var(--teald);border-color:rgba(0,196,167,.2);color:var(--teal)}
.abtn.teal-btn:hover{background:rgba(0,196,167,.16);border-color:var(--teal);box-shadow:0 0 10px var(--tealg)}
.kbd-hint{font-family:'JetBrains Mono',monospace;font-size:.57rem;opacity:.35;background:rgba(255,255,255,.06);padding:1px 4px;border-radius:3px}

/* ── REFERENCE PANEL ─────────────────────────────────────────── */
.ref-body{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:12px}
.ref-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(88px,1fr));gap:6px}
.ref-card{background:rgba(255,255,255,.01);border:1px solid var(--border);border-radius:8px;padding:9px 6px;display:flex;flex-direction:column;align-items:center;text-align:center;transition:all .25s;cursor:default}
.ref-card:hover{background:rgba(255,255,255,.025);border-color:var(--border-hi);transform:translateY(-2px)}
.ref-card.active{background:var(--amberd);border-color:rgba(232,162,37,.3);box-shadow:0 0 12px var(--amberg)}
.ref-card .char{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:700;color:var(--teal);margin-bottom:2px}
.ref-card.active .char{color:var(--amber);text-shadow:0 0 10px var(--amberg)}
.ref-card .emoji{font-size:.9rem;margin-bottom:4px}
.ref-card .desc{font-size:.55rem;color:var(--text3);line-height:1.3}
.ref-card.active .desc{color:var(--text2)}
.gest-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.gest-card{background:rgba(255,255,255,.01);border:1px solid var(--border);border-radius:8px;padding:8px 6px;display:flex;flex-direction:column;align-items:center;gap:3px;text-align:center;transition:all .25s}
.gest-card:hover{background:rgba(255,255,255,.025);border-color:var(--border-hi)}
.gest-card.active{background:rgba(232,162,37,.05);border-color:rgba(232,162,37,.28);box-shadow:0 0 12px var(--amberg)}
.gest-card .emoji{font-size:1.1rem}
.gest-card .label{font-size:.6rem;font-weight:700;color:var(--amber)}
.gest-card .desc{font-size:.53rem;color:var(--text3);line-height:1.2}

/* ── TRAINER ─────────────────────────────────────────────────── */
.train-right{display:flex;flex-direction:column;gap:10px;min-height:0;overflow-y:auto}

/* Resolution */
.res-sec{padding:11px 16px;border-bottom:1px solid var(--border);flex-shrink:0}
.res-btns{display:flex;gap:6px;margin:5px 0 4px}
.res-btn{flex:1;background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:8px;color:var(--text3);font-family:'DM Sans',sans-serif;font-size:.72rem;font-weight:600;padding:7px 8px;cursor:pointer;transition:all .2s;text-align:center;line-height:1.35}
.res-btn .rtag{display:block;font-family:'JetBrains Mono',monospace;font-size:.57rem;letter-spacing:.1em;margin-bottom:1px;color:var(--text3)}
.res-btn:hover{background:rgba(255,255,255,.05);border-color:var(--border-hi);color:var(--text)}
.res-btn.active{background:var(--amberd);border-color:rgba(232,162,37,.28);color:var(--amber)}
.res-btn.active .rtag{color:var(--amber)}
.res-btn.tr-mode{background:var(--teald);border-color:rgba(0,196,167,.2);color:var(--teal)}
.res-btn.tr-mode .rtag{color:var(--teal)}
.res-note{font-size:.62rem;color:var(--text3);line-height:1.4}
.res-note b{color:var(--teal)}

/* Class picker */
.class-pick{padding:11px 16px;border-bottom:1px solid var(--border);flex-shrink:0}
.pick-wrap{position:relative;margin-top:5px}
.pick-sel{width:100%;background:#090B12;border:1px solid var(--border);border-radius:7px;padding:8px 11px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:.85rem;font-weight:600;cursor:pointer;outline:none;appearance:none}
.pick-sel:focus{border-color:rgba(232,162,37,.35)}
.pick-wrap::after{content:'▼';font-size:.52rem;color:var(--text3);position:absolute;right:11px;top:50%;transform:translateY(-50%);pointer-events:none}

/* Countdown Recorder */
.recorder-body{padding:14px 16px;display:flex;flex-direction:column;align-items:center;gap:12px;flex-shrink:0}
.sample-ctr{font-size:.75rem;font-weight:600;color:var(--text3);text-align:center}
.sample-ctr span{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:var(--amber);display:block;margin-top:1px;text-shadow:0 0 18px var(--amberg)}

.rec-btn-wrap{position:relative;width:116px;height:116px;flex-shrink:0}
.record-btn{
  width:100%;height:100%;border-radius:50%;
  background:radial-gradient(circle,var(--rose) 0%,#9f1239 100%);
  border:3px solid rgba(255,255,255,.1);
  box-shadow:0 6px 22px rgba(159,18,57,.45);
  cursor:pointer;display:flex;flex-direction:column;align-items:center;justify-content:center;
  color:#fff;font-family:'DM Sans',sans-serif;transition:all .2s;user-select:none;
}
.record-btn:hover{transform:scale(1.03);box-shadow:0 9px 26px rgba(159,18,57,.6)}
.record-btn.recording{transform:scale(.96);background:radial-gradient(circle,#ef4444 0%,#7f1d1d 100%);border-color:rgba(255,255,255,.2)}
.record-btn .rbig{font-weight:700;font-size:.85rem;letter-spacing:.04em}
.record-btn .rsub{font-size:.57rem;opacity:.55;margin-top:2px;letter-spacing:.03em}

.countdown-ring{
  position:absolute;inset:0;border-radius:50%;
  background:rgba(7,8,13,.94);backdrop-filter:blur(4px);
  border:3px solid var(--amber);box-shadow:0 0 22px var(--amberg);
  display:none;align-items:center;justify-content:center;
}
.countdown-ring.show{display:flex}
.countdown-num{font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;color:var(--amber);animation:cd-pop .85s ease-out}
@keyframes cd-pop{0%{transform:scale(1.5);opacity:.4}100%{transform:scale(1);opacity:1}}

.rec-progress{width:100%;display:none;flex-direction:column;gap:5px}
.rec-progress.show{display:flex}
.rec-prog-bar{height:18px;background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:5px;overflow:hidden;position:relative}
.rec-prog-fill{height:100%;background:var(--rose);width:0%;transition:width .1s linear;box-shadow:0 0 7px var(--roseg)}
.rec-prog-lbl{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;font-size:.62rem;font-weight:600;color:#fff}
.rec-prog-text{font-size:.65rem;color:var(--text3);text-align:center}

.rec-actions{display:flex;gap:7px;width:100%}
.rec-actions button{flex:1}

/* Model trainer */
.trainer-body{padding:14px 16px;display:flex;flex-direction:column;gap:12px;flex-shrink:0}
.status-box{background:rgba(0,0,0,.12);border:1px solid var(--border);border-radius:9px;padding:12px;display:flex;flex-direction:column;gap:7px}
.status-row{display:flex;justify-content:space-between;font-size:.78rem}
.status-row .lbl{color:var(--text3)}
.status-row .val{font-weight:600}
.status-row .val b{color:var(--teal)}
.train-btn{
  background:linear-gradient(135deg,var(--amber) 0%,var(--purple) 100%);
  border:none;border-radius:8px;color:#fff;font-family:'DM Sans',sans-serif;
  font-weight:700;font-size:.85rem;padding:10px;cursor:pointer;
  box-shadow:0 4px 16px var(--amberg);transition:all .2s;
  display:flex;align-items:center;justify-content:center;gap:7px;
}
.train-btn:hover{transform:translateY(-1px);box-shadow:0 6px 20px var(--amberg)}
.train-btn:disabled{background:var(--text3);color:rgba(255,255,255,.3);box-shadow:none;cursor:not-allowed;transform:none}
.spinner{width:14px;height:14px;border:2px solid rgba(255,255,255,.25);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* Dataset grid */
.stats-scroll{flex:1;overflow-y:auto;padding:10px}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(66px,1fr));gap:4px}
.sbadge{background:rgba(255,255,255,.01);border:1px solid var(--border);border-radius:6px;padding:6px 3px;display:flex;flex-direction:column;align-items:center;gap:2px}
.sbadge.has{background:rgba(232,162,37,.02);border-color:rgba(232,162,37,.1)}
.sbadge .sl{font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:700;color:var(--text3)}
.sbadge.has .sl{color:var(--text)}
.sbadge .sc{font-family:'JetBrains Mono',monospace;font-size:.67rem;font-weight:600;color:var(--amber)}

/* ── HISTORY PANEL ───────────────────────────────────────────── */
.history-panel{
  position:fixed;bottom:68px;left:0;right:0;
  background:rgba(10,11,18,.98);border-top:1px solid var(--border);
  backdrop-filter:blur(20px);z-index:90;max-height:210px;
  transform:translateY(100%);transition:transform .28s cubic-bezier(.4,0,.2,1);
  display:flex;flex-direction:column;
}
.history-panel.open{transform:translateY(0)}
.hist-hdr{display:flex;align-items:center;justify-content:space-between;padding:9px 18px;border-bottom:1px solid var(--border);font-size:.62rem;font-weight:700;letter-spacing:.12em;color:var(--text3);flex-shrink:0}
.hist-clr{background:transparent;border:none;color:var(--text3);font-size:.68rem;cursor:pointer;padding:2px 6px;border-radius:4px;transition:color .18s}
.hist-clr:hover{color:var(--rose)}
.hist-list{flex:1;overflow-y:auto;padding:7px 12px;display:flex;flex-direction:column;gap:3px}
.hist-empty{color:var(--text3);font-size:.72rem;padding:10px 6px;text-align:center}
.hist-item{display:flex;align-items:center;gap:9px;padding:6px 9px;background:rgba(255,255,255,.015);border:1px solid var(--border);border-radius:7px;transition:background .18s}
.hist-item:hover{background:rgba(255,255,255,.03)}
.hist-n{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--text3);width:13px;flex-shrink:0}
.hist-t{flex:1;font-size:.82rem;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hist-play{background:var(--amberd);border:1px solid rgba(232,162,37,.18);border-radius:5px;color:var(--amber);font-size:.68rem;padding:2px 8px;cursor:pointer;flex-shrink:0;transition:all .18s}
.hist-play:hover{background:rgba(232,162,37,.2);border-color:var(--amberm)}

/* ── SENTENCE BAR ────────────────────────────────────────────── */
.sent-bar{
  height:68px;min-height:68px;background:rgba(7,8,13,.94);
  border-top:1px solid var(--border);display:flex;align-items:center;
  padding:0 16px;gap:10px;z-index:10;flex-shrink:0;backdrop-filter:blur(16px);
}
.sent-lbl{font-size:.6rem;font-weight:700;letter-spacing:.14em;color:var(--text3);flex-shrink:0}
.sent-box{
  flex:1;background:rgba(0,0,0,.2);border:1px solid var(--border);border-radius:8px;
  height:40px;display:flex;align-items:center;padding:0 12px;
  font-family:'JetBrains Mono',monospace;font-size:.9rem;color:var(--text);
  overflow:hidden;white-space:nowrap;text-overflow:ellipsis;transition:border-color .3s;
}
.sent-box.empty{color:var(--text3);font-style:italic;font-family:'DM Sans',sans-serif;font-size:.78rem}
.sent-box.has{border-color:rgba(232,162,37,.14)}
.wave-wrap{display:flex;align-items:center;gap:3px;flex-shrink:0;padding:0 3px}
.wave-bar{width:3px;background:var(--amber);border-radius:2px;animation:wave 1.1s ease-in-out infinite;transform-origin:bottom}
.wave-bar:nth-child(1){height:5px;animation-delay:0s}
.wave-bar:nth-child(2){height:13px;animation-delay:.14s}
.wave-bar:nth-child(3){height:9px;animation-delay:.28s}
.wave-bar:nth-child(4){height:15px;animation-delay:.42s}
.wave-bar:nth-child(5){height:7px;animation-delay:.56s}
@keyframes wave{0%,100%{transform:scaleY(.4);opacity:.5}50%{transform:scaleY(1);opacity:1;box-shadow:0 0 5px var(--amberg)}}
@media(prefers-reduced-motion:reduce){.wave-bar{animation:none}}
.sent-btns{display:flex;gap:6px;flex-shrink:0}
.sbtn{background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:.72rem;font-weight:600;padding:8px 14px;cursor:pointer;transition:all .18s;white-space:nowrap;display:flex;align-items:center;gap:5px}
.sbtn:hover{background:rgba(255,255,255,.05);border-color:var(--border-hi)}
.sbtn.speak{background:var(--amberd);border-color:rgba(232,162,37,.24);color:var(--amber)}
.sbtn.speak:hover{background:rgba(232,162,37,.18);border-color:var(--amberm);box-shadow:0 0 11px var(--amberg)}
.sbtn.speak.speaking{background:rgba(232,162,37,.08);border-color:var(--amber)}
.sbtn.copy:hover{background:var(--teald);border-color:var(--teal);color:var(--teal)}
.sbtn.hist-btn.active{background:var(--teald);border-color:var(--teal);color:var(--teal)}
.sbtn.clear-btn:hover{background:var(--roseg);border-color:var(--rose);color:var(--rose)}

/* ── SHORTCUTS MODAL ─────────────────────────────────────────── */
.modal-bg{position:fixed;inset:0;z-index:8000;background:rgba(0,0,0,.7);backdrop-filter:blur(6px);display:flex;align-items:center;justify-content:center;opacity:0;visibility:hidden;transition:opacity .22s,visibility .22s}
.modal-bg.open{opacity:1;visibility:visible}
.shortcuts-box{background:#0E1020;border:1px solid var(--border-hi);border-radius:13px;padding:22px 26px;max-width:420px;width:90%;box-shadow:0 24px 60px rgba(0,0,0,.6)}
.sc-title{font-family:'Syne',sans-serif;font-weight:800;font-size:1.05rem;color:var(--amber);letter-spacing:.05em;margin-bottom:16px}
.sc-table{width:100%;border-collapse:collapse}
.sc-table tr{border-bottom:1px solid rgba(255,255,255,.04)}
.sc-table tr:last-child{border-bottom:none}
.sc-table td{padding:7px 5px;font-size:.8rem}
.sc-table td:first-child{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--amber);font-weight:600;width:120px}
.sc-table td:last-child{color:var(--text2)}
.sc-close{margin-top:16px;width:100%;background:var(--amberd);border:1px solid rgba(232,162,37,.2);border-radius:7px;color:var(--amber);font-family:'DM Sans',sans-serif;font-size:.78rem;font-weight:700;padding:7px;cursor:pointer;transition:all .18s}
.sc-close:hover{background:rgba(232,162,37,.2)}

/* ── CONFETTI ────────────────────────────────────────────────── */
#conf-cv{position:fixed;inset:0;z-index:7000;pointer-events:none}

/* ── TOAST ───────────────────────────────────────────────────── */
.toast{position:fixed;bottom:82px;left:50%;transform:translateX(-50%) translateY(18px);background:rgba(34,197,94,.12);border:1px solid var(--emerald);border-radius:7px;color:var(--emerald);font-size:.72rem;font-weight:600;padding:7px 16px;z-index:999;opacity:0;transition:all .28s;pointer-events:none}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
</style>
</head>
<body>

<!-- ═══ LOADING ═══ -->
<div class="loading-screen" id="ls">
  <div class="ld-logo">ISL</div>
  <div class="ld-sub">Recognition System</div>
  <div class="ld-bar"><div class="ld-fill"></div></div>
  <div class="ld-sub" style="font-size:.62rem">Initializing MediaPipe &amp; Camera...</div>
</div>

<!-- ═══ HEADER ═══ -->
<header>
  <div class="logo">&#9670; ISL <em>RECOGNITION</em></div>
  <nav class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('recognize')">RECOGNITION</button>
    <button class="tab-btn"        onclick="switchTab('train')">TRAINER</button>
  </nav>
  <div class="hdr-end">
    <div class="hdr-div"></div>
    <select id="voice-sel" class="voice-sel" title="TTS Voice"><option value="-1">System Voice</option></select>
    <div class="rate-wrap">
      <span>SPEED</span>
      <input type="range" id="tts-rate" min="0.5" max="2" step="0.1" value="1" oninput="onRateChange(this.value)">
      <span id="rate-val">1.0×</span>
    </div>
    <div class="hdr-div"></div>
    <label class="toggle-wrap" title="Speak each word as committed">
      <span>AUTO&#8209;SPEAK</span>
      <input type="checkbox" id="auto-spk" onchange="setAutoSpeak(this.checked)">
      <div class="tgl-track"><div class="tgl-thumb"></div></div>
    </label>
    <div class="hdr-div"></div>
    <div class="stat-pill">&#9670; <b id="sess-timer">00:00</b></div>
    <div class="stat-pill">LETTERS <b id="stat-ltrs">0</b></div>
    <div class="stat-pill">Model: <b id="class-count">—</b></div>
    <button class="tab-btn" onclick="toggleSC()" title="Keyboard shortcuts">?</button>
  </div>
</header>

<!-- ═══ RECOGNITION TAB ═══ -->
<div class="tab-content active" id="recognize-tab">

  <!-- Camera -->
  <div class="glass-panel">
    <div class="panel-hdr">LIVE SENSOR FEED</div>
    <div class="video-wrap">
      <img id="cv-rec" class="feed" alt="live feed">
      <div class="hud">
        <div class="hud-row">
          <div class="hand-card">
            <div class="hand-ind" id="lh-r"><div class="dot"></div>L HAND</div>
            <div class="hand-ind" id="rh-r"><div class="dot"></div>R HAND</div>
          </div>
        </div>
        <div class="hud-row btm" id="warn-r" style="display:none">
          <div class="warn-badge" id="warn-txt-r"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Recognition panel -->
  <div class="glass-panel">
    <div class="panel-hdr">RECOGNITION</div>

    <div class="letter-box">
      <div class="letter-char empty" id="letter">—</div>
    </div>

    <div class="rec-scroll">
      <div class="metrics">
        <div class="mrow">
          <div class="mlabel">CONFIDENCE</div>
          <div class="mbar"><div class="mfill amber" id="conf-bar"></div></div>
          <div class="mval" id="conf-val">0%</div>
        </div>
        <div class="mrow">
          <div class="mlabel">STABILITY</div>
          <div class="mbar"><div class="mfill teal" id="stab-bar"></div></div>
          <div class="mval" id="stab-val">0/8</div>
        </div>
      </div>

      <div class="sparkline-wrap">
        <div class="sp-lbl">CONFIDENCE HISTORY</div>
        <canvas id="sparkline-canvas" height="28"></canvas>
      </div>

      <div class="word-sec">
        <div class="sec-lbl">CURRENT WORD</div>
        <div class="word-disp empty" id="word-disp">_</div>
      </div>

      <div class="text-sec">
        <div class="sec-lbl">
          YOUR SENTENCE
          <button class="mini-copy" onclick="copyClip()" title="Copy sentence">⎘ COPY</button>
        </div>
        <div class="text-box empty" id="full-text-box">Sign letters to build your sentence — it appears here.</div>
      </div>

      <div class="sugs-sec">
        <div class="sec-lbl">SUGGESTIONS <span style="opacity:.4;font-size:.55em">(1 / 2 / 3)</span></div>
        <div class="sugs-list" id="sugs"></div>
      </div>

      <div class="quick-sec">
        <div class="sec-lbl">QUICK PHRASES</div>
        <div class="quick-pills">
          <button class="qpill" onclick="quickPhrase('HELLO')">Hello</button>
          <button class="qpill" onclick="quickPhrase('THANK YOU')">Thank You</button>
          <button class="qpill" onclick="quickPhrase('HELP ME')">Help Me</button>
          <button class="qpill" onclick="quickPhrase('I AM SORRY')">I Am Sorry</button>
          <button class="qpill" onclick="quickPhrase('YES PLEASE')">Yes Please</button>
          <button class="qpill" onclick="quickPhrase('NO THANK YOU')">No Thanks</button>
          <button class="qpill" onclick="quickPhrase('GOOD')">Good</button>
          <button class="qpill" onclick="quickPhrase('PLEASE WAIT')">Please Wait</button>
        </div>
      </div>
    </div><!-- /rec-scroll -->

    <div class="punct-row">
      <button class="punct-btn" onclick="addPunct('.')">.</button>
      <button class="punct-btn" onclick="addPunct(',')"><b>,</b></button>
      <button class="punct-btn" onclick="addPunct('?')">?</button>
      <button class="punct-btn" onclick="addPunct('!')">!</button>
      <button class="punct-btn" onclick="addPunct('...')">…</button>
    </div>

    <div class="action-controls">
      <div class="act-row act-row-2">
        <button class="abtn" onclick="act('space')">⎵ SPACE <span class="kbd-hint">SPC</span></button>
        <button class="abtn amber-btn" onclick="speakAndCommit()">&#9658; SPEAK <span class="kbd-hint">↵</span></button>
      </div>
      <div class="act-row act-row-3">
        <button class="abtn" onclick="act('backspace')">⌫ LETTER <span class="kbd-hint">BSP</span></button>
        <button class="abtn teal-btn" onclick="act('backspace_word')">⌫ WORD</button>
        <button class="abtn danger" onclick="act('clear')">✕ CLEAR</button>
      </div>
    </div>
  </div>

  <!-- Reference panel -->
  <div class="glass-panel">
    <div class="panel-hdr">ISL POSTURE REFERENCE</div>
    <div class="ref-body" id="ref-body"></div>
  </div>

</div><!-- /recognize-tab -->

<!-- ═══ TRAINER TAB ═══ -->
<div class="tab-content" id="train-tab">

  <!-- Camera -->
  <div class="glass-panel">
    <div class="panel-hdr">TRAINER VIEWER</div>
    <div class="video-wrap">
      <img id="cv-train" class="feed" alt="trainer feed">
      <div class="hud">
        <div class="hud-row">
          <div class="hand-card">
            <div class="hand-ind" id="lh-t"><div class="dot"></div>L HAND</div>
            <div class="hand-ind" id="rh-t"><div class="dot"></div>R HAND</div>
          </div>
          <div class="rec-badge" id="rec-ind" style="display:none"><div class="pulse-dot"></div>REC</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Right column (scrollable) -->
  <div class="train-right">

    <!-- Resolution -->
    <div class="glass-panel" style="flex-shrink:0">
      <div class="panel-hdr">CAMERA RESOLUTION</div>
      <div class="res-sec">
        <div class="res-btns">
          <button class="res-btn" id="res-hd" onclick="setResolution(1280,720)">
            <span class="rtag">HD PREVIEW</span>1280 &times; 720
          </button>
          <button class="res-btn active tr-mode" id="res-tr" onclick="setResolution(640,480)">
            <span class="rtag">FAST &#10003;</span>640 &times; 480
          </button>
        </div>
        <div class="res-note" id="res-note">Current: <b>640&times;480</b> — Fast &amp; recommended (HD is heavier)</div>
      </div>
    </div>

    <!-- Sign Recorder -->
    <div class="glass-panel" style="flex-shrink:0">
      <div class="panel-hdr">SIGN RECORDER</div>

      <div class="class-pick">
        <div class="sec-lbl">TARGET LABEL</div>
        <div class="pick-wrap" style="margin-top:5px">
          <select class="pick-sel" id="class-sel" onchange="onClassChange()"></select>
        </div>
      </div>

      <div class="recorder-body">
        <div class="sample-ctr">SAMPLES FOR CLASS<span id="target-cnt">0</span></div>

        <div class="rec-btn-wrap">
          <button class="record-btn" id="rec-btn" onclick="triggerAutoRecord()">
            <span class="rbig">RECORD</span>
            <span class="rsub">CLICK TO START</span>
          </button>
          <div class="countdown-ring" id="cd-ring">
            <div class="countdown-num" id="cd-num">3</div>
          </div>
        </div>

        <div class="rec-progress" id="rec-prog">
          <div class="rec-prog-bar">
            <div class="rec-prog-fill" id="rec-prog-fill"></div>
            <div class="rec-prog-lbl" id="rec-prog-lbl">0.0s / 5.0s</div>
          </div>
          <div class="rec-prog-text" id="rec-prog-txt">Recording...</div>
        </div>

        <div class="rec-actions">
          <button class="abtn danger" onclick="deleteLatestSample()">DELETE LATEST</button>
          <button class="abtn danger" onclick="clearClassSamples()">CLEAR ALL</button>
        </div>
      </div>
    </div>

    <!-- Model Trainer -->
    <div class="glass-panel" style="flex-shrink:0">
      <div class="panel-hdr">MODEL TRAINER</div>
      <div class="trainer-body">
        <div class="status-box">
          <div class="status-row"><span class="lbl">Status:</span><span class="val" id="train-stat">Idle</span></div>
          <div class="status-row" id="acc-row" style="display:none"><span class="lbl">Test Accuracy:</span><span class="val"><b id="train-acc">—</b></span></div>
          <div class="status-row"><span class="lbl">Model Features:</span><span class="val" id="model-feat">—</span></div>
        </div>
        <button class="train-btn" id="train-btn" onclick="trainModel()">TRAIN MODEL</button>
      </div>
    </div>

    <!-- Dataset Matrix -->
    <div class="glass-panel" style="flex:1;min-height:200px">
      <div class="panel-hdr">DATASET MATRIX</div>
      <div class="stats-scroll">
        <div class="stats-grid" id="stats-grid"></div>
      </div>
    </div>

  </div><!-- /train-right -->
</div><!-- /train-tab -->

<!-- ═══ HISTORY PANEL ═══ -->
<div class="history-panel" id="hist-panel">
  <div class="hist-hdr">
    <span>SPOKEN HISTORY</span>
    <button class="hist-clr" onclick="clearHistory()">✕ CLEAR</button>
  </div>
  <div class="hist-list" id="hist-list"><div class="hist-empty">No sentences spoken yet.</div></div>
</div>

<!-- ═══ SENTENCE BAR ═══ -->
<div class="sent-bar">
  <div class="sent-lbl">SENTENCE</div>
  <div class="sent-box empty" id="sent-txt">Sign letters to build a sentence...</div>
  <div class="wave-wrap" id="wave-ind" style="display:none">
    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
    <div class="wave-bar"></div><div class="wave-bar"></div>
  </div>
  <div class="sent-btns">
    <button class="sbtn copy"     onclick="copyClip()">⎘ COPY</button>
    <button class="sbtn"          onclick="exportSess()">&#8595; EXPORT</button>
    <button class="sbtn hist-btn" id="hist-btn" onclick="toggleHist()">▤ HISTORY</button>
    <button class="sbtn speak"    id="speak-btn" onclick="speakAndCommit()">&#9658; SPEAK ALL</button>
    <button class="sbtn clear-btn" onclick="act('clear')">✕ CLEAR</button>
  </div>
</div>

<!-- ═══ SHORTCUTS MODAL ═══ -->
<div class="modal-bg" id="sc-modal" onclick="toggleSC()">
  <div class="shortcuts-box" onclick="event.stopPropagation()">
    <div class="sc-title">&#9670; Keyboard Shortcuts</div>
    <table class="sc-table">
      <tr><td>Space</td><td>Commit current word to sentence</td></tr>
      <tr><td>Enter</td><td>Speak the full sentence</td></tr>
      <tr><td>Backspace</td><td>Delete last letter (or last word)</td></tr>
      <tr><td>1 / 2 / 3</td><td>Accept word suggestion</td></tr>
      <tr><td>R (hold)</td><td>Manual record in Trainer (fallback)</td></tr>
      <tr><td>?</td><td>Toggle this help overlay</td></tr>
    </table>
    <button class="sc-close" onclick="toggleSC()">&#215; CLOSE</button>
  </div>
</div>

<!-- Confetti canvas -->
<canvas id="conf-cv"></canvas>

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
// ── Reference data ────────────────────────────────────────────
const ALPHA = [
  {l:'A',e:'✊✊',d:'Tips of both index fingers touching'},
  {l:'B',e:'🖐🖐',d:'Both hands curved, double circle'},
  {l:'C',e:'🤏',d:'Right hand curved into C'},
  {l:'D',e:'☝️🤏',d:'Left index up; right curves to touch'},
  {l:'E',e:'☝️👉',d:'Left index up; right touches tip'},
  {l:'F',e:'✌️✌️',d:'Left index/middle; right crosses over'},
  {l:'G',e:'✊✊',d:'Both fists, one on top of other'},
  {l:'H',e:'🖐🖐',d:'Left flat; right palm rubs across'},
  {l:'I',e:'☝️',d:'Right index points to cheek'},
  {l:'J',e:'☝️🖐',d:'Right index traces J on left palm'},
  {l:'K',e:'☝️👉',d:'Left index up; right bent touches it'},
  {l:'L',e:'🤙',d:'Left index and thumb form L'},
  {l:'M',e:'🖐🖐',d:'Left palm flat; three fingertips touch'},
  {l:'N',e:'🖐🖐',d:'Left palm flat; two fingertips touch'},
  {l:'O',e:'👌',d:'Index and thumb form circle O'},
  {l:'P',e:'☝️👌',d:'Left index up; right circle touches tip'},
  {l:'Q',e:'👌👉',d:'Left circle; right hooks inside'},
  {l:'R',e:'🖐👉',d:'Left palm flat; right index crosses'},
  {l:'S',e:'☝️✊',d:'Left index up; right wraps around wrist'},
  {l:'T',e:'☝️👉',d:'Left index up; right touches side'},
  {l:'U',e:'✌️👉',d:'Left index/middle up; right touches tips'},
  {l:'V',e:'✌️👉',d:'Left V; right index touches cleft'},
  {l:'W',e:'🖐👉',d:'Three fingers up; right touches tips'},
  {l:'X',e:'👉👉',d:'Both index fingers crossed'},
  {l:'Y',e:'☝️👉',d:'Left index up; right points to cleft'},
  {l:'Z',e:'🖐👉',d:'Left index up; right points to palm'},
];
const GESTS = [
  {l:'HELLO',    e:'👋',d:'Open palm wave'},
  {l:'LOVE',     e:'🤟',d:'ILY hand shape'},
  {l:'THANKYOU', e:'🙏',d:'Flat hand moves from chin'},
  {l:'HELP',     e:'🆘',d:'Fist lifts on palm'},
  {l:'SORRY',    e:'✊',d:'Fist circles heart'},
  {l:'YES',      e:'✅',d:'Fist nods up and down'},
  {l:'NO',       e:'❌',d:'Index/middle open and close'},
  {l:'GOOD',     e:'👍',d:'Thumbs up gesture'},
  {l:'PLEASE',   e:'🙏',d:'Flat hand circles chest'},
];
const ALL_CLASSES = [...ALPHA.map(a=>a.l), ...GESTS.map(g=>g.l)];
const WORD_GESTS = new Set(GESTS.map(g=>g.l));

// ── Build reference panel ─────────────────────────────────────
(function(){
  const rb = document.getElementById('ref-body');
  let h = '<div class="sec-lbl">A – Z FINGERSPELLING</div><div class="ref-grid">';
  ALPHA.forEach(a=>{
    h+=`<div class="ref-card" id="card-${a.l}" title="${a.d}"><div class="char">${a.l}</div><div class="emoji">${a.e}</div><div class="desc">${a.d}</div></div>`;
  });
  h+='</div><div class="sec-lbl" style="margin-top:12px">SPECIAL GESTURES</div><div class="gest-grid">';
  GESTS.forEach(g=>{
    h+=`<div class="gest-card" id="card-${g.l}" title="${g.d}"><div class="emoji">${g.e}</div><div class="label">${g.l}</div><div class="desc">${g.d}</div></div>`;
  });
  h+='</div>';
  rb.innerHTML=h;
})();

// ── Build trainer class dropdown ──────────────────────────────
(function(){
  const sel = document.getElementById('class-sel');
  ALL_CLASSES.forEach(c=>{
    const o=document.createElement('option');
    o.value=c; o.textContent=c; sel.appendChild(o);
  });
})();

// ── Tab switching ─────────────────────────────────────────────
let activeTab = 'recognize';
function switchTab(id){
  activeTab = id;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  document.querySelector(`.tab-btn[onclick="switchTab('${id}')"]`).classList.add('active');
  document.getElementById(id+'-tab').classList.add('active');
  setActiveFeed(id);
  if(id==='train') fetchStats();
}

// ── Loading screen helpers (registered BEFORE socket init so they always fire) ──
function _hideLS(){const e=document.getElementById('ls');if(e&&!e.classList.contains('hidden'))e.classList.add('hidden');}
setTimeout(_hideLS,5000);

// ── Video feed + state, all via plain HTTP polling ────────────
// No Socket.IO: we poll /frame.jpg for video and /state for everything
// else. This sidesteps the Socket.IO/Engine.IO version mismatch that
// was blocking the connection on this machine.
const cvRec = document.getElementById('cv-rec');
const cvTrn = document.getElementById('cv-train');

let _feedImg = cvRec;          // the <img> currently being driven
let _feedStop = false;

function _pumpFeed(){
  if(_feedStop){ return; }
  const img = _feedImg;
  const probe = new Image();
  probe.onload = ()=>{
    img.src = probe.src;       // swap in the freshly loaded frame
    _hideLS();
    setTimeout(_pumpFeed, 40); // ~25 fps
  };
  probe.onerror = ()=>{ setTimeout(_pumpFeed, 300); };
  probe.src = '/frame.jpg?t=' + Date.now();
}

function setActiveFeed(id){
  _feedImg = (id==='train') ? cvTrn : cvRec;
  if(_feedImg.src && _feedImg.src.indexOf('/frame.jpg')===-1){ _feedImg.removeAttribute('src'); }
}

setActiveFeed(activeTab || 'recognize');
_pumpFeed();

// ── Apply a /state snapshot to the UI ─────────────────────────
let prevLetter='', flashTmr, prevCard=null, _lastTtsId=0, _ttsInit=false;

let _prevState = {};   // cache for change-detection to avoid per-poll DOM thrash
let currentFullText = '';

function applyState(d){
  // Speak any new utterance (works for manual actions and auto-speak).
  if(d.tts_id !== undefined){
    if(!_ttsInit){ _lastTtsId = d.tts_id; _ttsInit = true; }  // don't replay backlog on load
    else if(d.tts_id > _lastTtsId){ _lastTtsId = d.tts_id; if(d.tts_text) speakText(d.tts_text); }
  }

  // Model badge (only on change)
  if(d.model_features_loaded!==_prevState.model_features_loaded){
    const is2h=d.model_features_loaded===132;
    document.getElementById('class-count').textContent=is2h?'2-Handed':'1-Handed';
    document.getElementById('model-feat').innerHTML=`<b>${d.model_features_loaded}</b> feat`;
    const wb=document.getElementById('warn-r');
    if(!is2h){wb.style.display='flex';document.getElementById('warn-txt-r').textContent='⚠ Old model — retrain with 2-handed data';}
    else wb.style.display='none';
  }

  // Hand indicators (only on change)
  if(d.left_hand_detected!==_prevState.left_hand_detected || d.right_hand_detected!==_prevState.right_hand_detected){
    [['lh-r','lh-t','left_hand_detected'],['rh-r','rh-t','right_hand_detected']].forEach(([a,b,k])=>{
      [a,b].forEach(id=>{const el=document.getElementById(id);if(el)el.classList.toggle('on',d[k]);});
    });
  }

  // Letter
  const le=document.getElementById('letter');
  if(d.letter!==prevLetter){le.textContent=d.letter||'—';le.className='letter-char'+(d.letter?'':' empty');prevLetter=d.letter;}
  if(d.letter_flash){le.classList.remove('flash');void le.offsetWidth;le.classList.add('flash');clearTimeout(flashTmr);flashTmr=setTimeout(()=>le.classList.remove('flash'),460);}

  // Bars (only on change)
  const cf=Math.round(d.confidence*100);
  if(cf!==_prevState._cf){
    const cb=document.getElementById('conf-bar');
    cb.style.width=cf+'%'; cb.className='mfill '+(cf>70?'em':cf>40?'amber':'rose');
    document.getElementById('conf-val').textContent=cf+'%';
    _prevState._cf=cf;
  }
  if(d.stable_count!==_prevState.stable_count){
    document.getElementById('stab-bar').style.width=(d.stable_count/8*100)+'%';
    document.getElementById('stab-val').textContent=d.stable_count+'/8';
  }

  // Sparkline
  drawSpark(d.confidence||0);

  // Current word (only on change)
  if(d.current_word!==_prevState.current_word){
    const wd=document.getElementById('word-disp');
    wd.textContent=d.current_word||'_'; wd.className='word-disp'+(d.current_word?'':' empty');
  }

  // Suggestions (only on change)
  const sugKey=(d.suggestions||[]).join('|');
  if(sugKey!==_prevState._sugKey){
    document.getElementById('sugs').innerHTML=(d.suggestions||[]).map((s,i)=>
      `<button class="sug-pill" onclick="act('suggest',${i})"><span class="hint">${i+1}</span>${s}</button>`
    ).join('');
    _prevState._sugKey=sugKey;
  }

  // ── Live sentence (full text + the word being signed) ──
  if(d.full_text!==_prevState.full_text || d.current_word!==_prevState.current_word){
    currentFullText=((d.full_text||'')+(d.current_word||'')).trim();
    const box=document.getElementById('full-text-box');
    if(!d.full_text && !d.current_word){
      box.className='text-box empty';
      box.textContent='Sign letters to build your sentence — it appears here.';
    } else {
      box.className='text-box';
      const safe=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      box.innerHTML=safe(d.full_text||'')+(d.current_word?`<span class="pending">${safe(d.current_word)}</span>`:'');
      box.scrollTop=box.scrollHeight;   // keep latest text in view
    }
    const st=document.getElementById('sent-txt');
    if(st){ st.textContent=currentFullText||'Sign letters to build a sentence...'; st.className='sent-box'+(currentFullText?' has':' empty'); }
  }

  // Active ref card (only on letter change)
  if(d.letter!==_prevState.letter){
    if(prevCard){const old=document.getElementById('card-'+prevCard);if(old)old.classList.remove('active');}
    if(d.letter){const card=document.getElementById('card-'+d.letter);if(card){card.classList.add('active');prevCard=d.letter;const rb=document.getElementById('ref-body');const cr=card.getBoundingClientRect(),rr=rb.getBoundingClientRect();if(cr.top<rr.top||cr.bottom>rr.bottom)card.scrollIntoView({behavior:'smooth',block:'nearest'});}}
  }

  // Session stats (only on change)
  if(d.total_letters!==undefined && d.total_letters!==_prevState.total_letters){const el=document.getElementById('stat-ltrs');if(el)el.textContent=d.total_letters;}

  // History (only on change)
  if(d.history){
    const hk=d.history.join('|');
    if(hk!==_prevState._hk){ renderHist(d.history); _prevState._hk=hk; }
  }

  // Confetti on word gesture
  if(d.letter_flash&&WORD_GESTS.has(d.letter)) launchConfetti();

  // remember the keys we diff on
  _prevState.model_features_loaded=d.model_features_loaded;
  _prevState.left_hand_detected=d.left_hand_detected;
  _prevState.right_hand_detected=d.right_hand_detected;
  _prevState.stable_count=d.stable_count;
  _prevState.current_word=d.current_word;
  _prevState.full_text=d.full_text;
  _prevState.letter=d.letter;
  _prevState.total_letters=d.total_letters;
}

async function pollState(){
  try{
    const r=await fetch('/state',{cache:'no-store'});
    applyState(await r.json());
  }catch(e){ /* transient; keep polling */ }
  setTimeout(pollState, 120);   // ~8 polls/sec
}
pollState();

// ── Training status polling ───────────────────────────────────
function applyTraining(data){
  const se=document.getElementById('train-stat');
  const ai=document.getElementById('acc-row');
  const tb=document.getElementById('train-btn');
  if(!se||!tb) return;
  if(data.status==='training'){
    se.innerHTML='<span style="color:var(--amber)">Training...</span>';
    tb.disabled=true; tb.innerHTML='<div class="spinner"></div> TRAINING';
  } else if(data.status==='success'){
    se.innerHTML='<span style="color:var(--emerald)">Success — model loaded</span>';
    if(ai) ai.style.display='flex';
    document.getElementById('train-acc').textContent=(data.accuracy*100).toFixed(1)+'%';
    if(tb.disabled){ tb.disabled=false; tb.textContent='TRAIN MODEL'; fetchStats(); }
    else { tb.disabled=false; tb.textContent='TRAIN MODEL'; }
  } else if(data.status==='error'){
    se.innerHTML=`<span style="color:var(--rose)" title="${data.error}">Error: ${(data.error||'').substring(0,20)}…</span>`;
    tb.disabled=false; tb.textContent='TRAIN MODEL';
  } else { se.textContent='Idle'; tb.disabled=false; tb.textContent='TRAIN MODEL'; }
}

async function pollTraining(){
  try{
    const r=await fetch('/training_status',{cache:'no-store'});
    applyTraining(await r.json());
  }catch(e){}
  setTimeout(pollTraining, 900);
}
pollTraining();

// ── Session timer ─────────────────────────────────────────────
const _t0=Date.now();
setInterval(()=>{
  const s=Math.floor((Date.now()-_t0)/1000);
  const el=document.getElementById('sess-timer');
  if(el) el.textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');
},1000);

// ── Sparkline ─────────────────────────────────────────────────
const _spData=[];
const _spCv=document.getElementById('sparkline-canvas');
const _spCtx=_spCv?_spCv.getContext('2d'):null;
function drawSpark(v){
  if(!_spCtx)return;
  _spData.push(v); if(_spData.length>50)_spData.shift();
  const W=_spCv.offsetWidth||200,H=28;
  _spCv.width=W; _spCv.height=H;
  _spCtx.clearRect(0,0,W,H);
  const col=v>0.7?'#22C55E':v>0.4?'#E8A225':'#F43F5E';
  const fill=v>0.7?'rgba(34,197,94,.1)':v>0.4?'rgba(232,162,37,.1)':'rgba(244,63,94,.1)';
  const n=_spData.length;
  _spCtx.beginPath();
  _spData.forEach((p,i)=>{const x=i/(n-1||1)*W,y=H-p*(H-3);i===0?_spCtx.moveTo(x,y):_spCtx.lineTo(x,y);});
  _spCtx.lineTo(W,H); _spCtx.lineTo(0,H); _spCtx.closePath();
  _spCtx.fillStyle=fill; _spCtx.fill();
  _spCtx.beginPath();
  _spData.forEach((p,i)=>{const x=i/(n-1||1)*W,y=H-p*(H-3);i===0?_spCtx.moveTo(x,y):_spCtx.lineTo(x,y);});
  _spCtx.strokeStyle=col; _spCtx.lineWidth=1.5; _spCtx.stroke();
  _spCtx.setLineDash([2,3]); _spCtx.strokeStyle='rgba(255,255,255,.1)'; _spCtx.lineWidth=.7;
  _spCtx.beginPath(); _spCtx.moveTo(0,H*.5); _spCtx.lineTo(W,H*.5); _spCtx.stroke();
  _spCtx.setLineDash([]);
}

// ── Web Speech API ────────────────────────────────────────────
let ttsVoices=[], ttsRate=1.0;
function loadVoices(){
  ttsVoices=window.speechSynthesis.getVoices();
  const sel=document.getElementById('voice-sel');
  if(!sel||!ttsVoices.length)return;
  const en=ttsVoices.filter(v=>v.lang.startsWith('en'));
  const ot=ttsVoices.filter(v=>!v.lang.startsWith('en'));
  let h='<option value="-1">System Default</option>';
  if(en.length){h+='<optgroup label="English">';en.forEach(v=>{h+=`<option value="${ttsVoices.indexOf(v)}">${v.name}</option>`;});h+='</optgroup>';}
  if(ot.length){h+='<optgroup label="Other">';ot.forEach(v=>{h+=`<option value="${ttsVoices.indexOf(v)}">${v.name} (${v.lang})</option>`;});h+='</optgroup>';}
  sel.innerHTML=h;
}
if('speechSynthesis' in window){window.speechSynthesis.onvoiceschanged=loadVoices; loadVoices();}

function onRateChange(v){ttsRate=parseFloat(v); document.getElementById('rate-val').textContent=parseFloat(v).toFixed(1)+'×';}

function speakText(text){
  if(!text||!('speechSynthesis' in window))return;
  window.speechSynthesis.cancel();
  const utt=new SpeechSynthesisUtterance(text);
  const vi=parseInt((document.getElementById('voice-sel')||{}).value||-1);
  if(vi>=0&&ttsVoices[vi])utt.voice=ttsVoices[vi];
  utt.rate=ttsRate;
  utt.onstart=()=>showSpeaking(true);
  utt.onend=utt.onerror=()=>showSpeaking(false);
  window.speechSynthesis.speak(utt);
}
function showSpeaking(on){
  const w=document.getElementById('wave-ind'), b=document.getElementById('speak-btn');
  if(w)w.style.display=on?'flex':'none';
  if(b){b.innerHTML=on?'&#9658; SPEAKING...':'&#9658; SPEAK ALL'; b.classList.toggle('speaking',on);}
}
async function speakAndCommit(){await act('enter');}

// ── History ───────────────────────────────────────────────────
let histItems=[];
function addHistLocal(t){if(!t)return; if(!histItems.length||histItems[0]!==t){histItems=[t,...histItems].slice(0,10);renderHist(histItems);}}
function renderHist(items){
  histItems=items;
  const l=document.getElementById('hist-list');
  if(!items||!items.length){l.innerHTML='<div class="hist-empty">No sentences spoken yet.</div>';return;}
  l.innerHTML=items.map((t,i)=>`<div class="hist-item"><span class="hist-n">${i+1}</span><span class="hist-t">${t}</span><button class="hist-play" onclick="speakText(histItems[${i}])">&#9658;</button></div>`).join('');
}
function toggleHist(){
  const p=document.getElementById('hist-panel'),b=document.getElementById('hist-btn');
  p.classList.toggle('open'); b.classList.toggle('active');
}
async function clearHistory(){await act('clear_history'); histItems=[]; renderHist([]);}

// ── Copy / Export ─────────────────────────────────────────────
async function copyClip(){
  const el=document.getElementById('sent-txt');
  if(el.classList.contains('empty'))return;
  try{await navigator.clipboard.writeText(el.textContent.trim()); showToast('Copied to clipboard');}
  catch{showToast('Copy failed — try Ctrl+C');}
}
function exportSess(){const a=document.createElement('a');a.href='/download_history';a.download='isl_session.txt';a.click();showToast('Session exported');}

// ── Toast ─────────────────────────────────────────────────────
function showToast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2200);}

// ── Auto-speak ────────────────────────────────────────────────
async function setAutoSpeak(v){await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'auto_speak',value:v})});}

// ── Generic action ────────────────────────────────────────────
async function act(action,index){
  const body={action}; if(index!==undefined)body.index=index;
  await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
}

// ── Quick phrases / Punct ─────────────────────────────────────
async function quickPhrase(t){await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'quick_phrase',text:t})});}
async function addPunct(c){await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'punct',char:c})});}

// ── Shortcuts modal ───────────────────────────────────────────
function toggleSC(){document.getElementById('sc-modal').classList.toggle('open');}

// ── Keyboard shortcuts ────────────────────────────────────────
let _recHK=false;
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(e.key===' '){e.preventDefault();act('space');}
  else if(e.key==='Enter'){speakAndCommit();}
  else if(e.key==='Backspace'){act('backspace');}
  else if(e.key==='1'){act('suggest',0);}
  else if(e.key==='2'){act('suggest',1);}
  else if(e.key==='3'){act('suggest',2);}
  else if(e.key==='?'){toggleSC();}
  else if((e.key==='r'||e.key==='R')&&activeTab==='train'&&!e.repeat&&!_recHK){_recHK=true;_manualStart();}
});
document.addEventListener('keyup',e=>{if((e.key==='r'||e.key==='R')&&_recHK){_recHK=false;_manualStop();}});

// ── Countdown Auto-Record ─────────────────────────────────────
let _cdTimer=null, _arTimer=null, _recActive=false;

async function triggerAutoRecord(){
  if(_recActive){
    // Stop mid-recording
    clearInterval(_arTimer); _arTimer=null; _recActive=false;
    await _stopRec(); resetRecUI();
    return;
  }
  if(_cdTimer){
    // Cancel countdown
    clearInterval(_cdTimer); _cdTimer=null; resetRecUI();
    return;
  }
  // Start 3s countdown
  const ring=document.getElementById('cd-ring');
  const numEl=document.getElementById('cd-num');
  const btn=document.getElementById('rec-btn');
  ring.classList.add('show');
  btn.querySelector('.rbig').textContent='CANCEL';
  btn.querySelector('.rsub').textContent='CLICK TO CANCEL';
  let count=3;
  numEl.textContent=count;

  _cdTimer=setInterval(async()=>{
    count--;
    if(count>0){
      numEl.style.animation='none'; void numEl.offsetWidth; numEl.style.animation='cd-pop .85s ease-out';
      numEl.textContent=count;
    } else {
      clearInterval(_cdTimer); _cdTimer=null;
      ring.classList.remove('show');
      // Launch recording
      await _startRec();
      _recActive=true;
      btn.querySelector('.rbig').textContent='RECORDING';
      btn.querySelector('.rsub').textContent='CLICK TO STOP';
      btn.classList.add('recording');
      const prog=document.getElementById('rec-prog');
      const fill=document.getElementById('rec-prog-fill');
      const lbl=document.getElementById('rec-prog-lbl');
      prog.classList.add('show');
      let elapsed=0; const dur=5000;
      _arTimer=setInterval(async()=>{
        elapsed+=100;
        fill.style.width=Math.min(elapsed/dur*100,100)+'%';
        lbl.textContent=(elapsed/1000).toFixed(1)+'s / 5.0s';
        if(elapsed>=dur){
          clearInterval(_arTimer); _arTimer=null; _recActive=false;
          await _stopRec(); resetRecUI();
          showToast('Recording complete ✓');
        }
      },100);
    }
  },1000);
}

function resetRecUI(){
  const btn=document.getElementById('rec-btn');
  const ring=document.getElementById('cd-ring');
  const prog=document.getElementById('rec-prog');
  btn.classList.remove('recording');
  btn.querySelector('.rbig').textContent='RECORD';
  btn.querySelector('.rsub').textContent='CLICK TO START';
  ring.classList.remove('show');
  prog.classList.remove('show');
  document.getElementById('rec-prog-fill').style.width='0%';
  document.getElementById('rec-ind').style.display='none';
}

let _recActive2=false;
async function _startRec(){
  if(_recActive2)return; _recActive2=true;
  const cur=document.getElementById('class-sel').value;
  document.getElementById('rec-ind').style.display='flex';
  await fetch('/start_recording',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:cur})});
}
async function _stopRec(){
  if(!_recActive2)return; _recActive2=false;
  await fetch('/stop_recording',{method:'POST'});
  fetchStats();
}

// Manual hold-to-record fallback (R key)
async function _manualStart(){
  if(_recActive||_cdTimer)return;
  _recActive2=true;
  const cur=document.getElementById('class-sel').value;
  document.getElementById('rec-btn').classList.add('recording');
  document.getElementById('rec-ind').style.display='flex';
  await fetch('/start_recording',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:cur})});
}
async function _manualStop(){
  if(!_recActive2)return; _recActive2=false;
  document.getElementById('rec-btn').classList.remove('recording');
  document.getElementById('rec-ind').style.display='none';
  await fetch('/stop_recording',{method:'POST'});
  fetchStats();
}

// ── Dataset stats ─────────────────────────────────────────────
async function fetchStats(){
  try{
    const res=await fetch('/dataset_info');
    const counts=await res.json();
    const cur=document.getElementById('class-sel').value;
    document.getElementById('target-cnt').textContent=counts[cur]||0;
    const grid=document.getElementById('stats-grid');
    grid.innerHTML=ALL_CLASSES.map(c=>{
      const n=counts[c]||0;
      return `<div class="sbadge ${n>0?'has':''}"><div class="sl">${c}</div><div class="sc">${n}</div></div>`;
    }).join('');
  }catch(e){console.error(e);}
}
function onClassChange(){fetchStats();}
async function deleteLatestSample(){
  const cur=document.getElementById('class-sel').value;
  if(!confirm(`Delete latest sample for '${cur}'?`))return;
  const r=await fetch('/delete_latest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:cur})});
  const d=await r.json(); document.getElementById('target-cnt').textContent=d.count; fetchStats();
}
async function clearClassSamples(){
  const cur=document.getElementById('class-sel').value;
  if(!confirm(`Clear ALL samples for '${cur}'?`))return;
  await fetch('/clear_class',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:cur})});
  document.getElementById('target-cnt').textContent=0; fetchStats();
}
async function trainModel(){await fetch('/train',{method:'POST'});}

// ── Resolution switcher ───────────────────────────────────────
async function setResolution(w,h){
  const r=await fetch('/set_resolution',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({width:w,height:h})});
  const d=await r.json(); if(d.status!=='ok'){showToast('Resolution change failed');return;}
  document.getElementById('res-hd').className='res-btn'+(w===1280?' active':'');
  document.getElementById('res-tr').className='res-btn'+(w===640?' active tr-mode':'');
  document.getElementById('res-note').innerHTML=w===640?'Current: <b>640×480</b> — Recommended for training accuracy':'Current: <b>1280×720</b> (HD Preview)';
  showToast(`Resolution → ${w}×${h}`);
}

// ── Confetti ──────────────────────────────────────────────────
const _ccv=document.getElementById('conf-cv');
if(_ccv){_ccv.width=innerWidth;_ccv.height=innerHeight;addEventListener('resize',()=>{_ccv.width=innerWidth;_ccv.height=innerHeight;});}
let _cpieces=[];
function launchConfetti(){
  if(!_ccv)return;
  const ctx=_ccv.getContext('2d');
  const cols=['#E8A225','#00C4A7','#7C6FF7','#F43F5E','#22C55E','#F5C05A'];
  for(let i=0;i<90;i++){_cpieces.push({x:Math.random()*_ccv.width,y:-10,vx:(Math.random()-.5)*5,vy:Math.random()*3+2,w:Math.random()*9+4,h:Math.random()*6+3,color:cols[Math.floor(Math.random()*cols.length)],a:Math.random()*Math.PI*2,va:(Math.random()-.5)*.2,alpha:1});}
  (function step(){
    ctx.clearRect(0,0,_ccv.width,_ccv.height);
    _cpieces=_cpieces.filter(p=>p.alpha>.04);
    _cpieces.forEach(p=>{
      p.x+=p.vx;p.y+=p.vy;p.vy+=.09;p.a+=p.va;
      if(p.y>_ccv.height*.75)p.alpha-=.025;
      ctx.save();ctx.translate(p.x,p.y);ctx.rotate(p.a);ctx.globalAlpha=p.alpha;ctx.fillStyle=p.color;ctx.fillRect(-p.w/2,-p.h/2,p.w,p.h);ctx.restore();
    });
    if(_cpieces.length>0)requestAnimationFrame(step);
    else ctx.clearRect(0,0,_ccv.width,_ccv.height);
  })();
}

// Initial load
fetchStats();
</script>
</body>
</html>"""

# ── Entry point ────────────────────────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=frame_grabber, daemon=True, name="grabber").start()
    threading.Thread(target=detector_loop, daemon=True, name="detector").start()
    threading.Thread(target=camera_loop,   daemon=True, name="encoder").start()

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║    ISL RECOGNITION SYSTEM — FINAL (Windows)  ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  BUILD: {BUILD_TAG:<36} ║")
    print("║  Video: polled /frame.jpg (no WebSocket)     ║")
    print("║  Open:  http://localhost:5000                ║")
    print("║  Test:  http://localhost:5000/frame.jpg      ║")
    print("║  Stop:  Ctrl+C                               ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    def _open():
        time.sleep(0.5)
        webbrowser.open('http://localhost:5000')
    threading.Thread(target=_open, daemon=True).start()

    sio.run(app, host='0.0.0.0', port=5000, debug=False,
            use_reloader=False, allow_unsafe_werkzeug=True)
