#!/usr/bin/env python3
"""
ISL Recognition System — FINAL (Windows)
Install:  pip install flask flask-socketio mediapipe opencv-python scikit-learn numpy pyttsx3
Run:      python final.py
Browser:  http://localhost:5000

Also place 'hand_landmarker.task' in this same folder for hand detection:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import csv
import threading
import time
from flask import Flask, Response, jsonify, request, render_template, make_response
from flask_socketio import SocketIO, emit
import webbrowser
import os
import queue

# Windows consoles/pipes often default to cp1252, which can't encode the
# box-drawing characters in the startup banner below — force UTF-8 so
# `python isl_main.py` doesn't crash regardless of how it's launched.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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

# ── Dataset schema ──────────────────────────────────────────────────
# FEATURE_COLUMNS are the 132 model inputs (unchanged). META_COLUMNS record
# WHO recorded a sample, in WHICH session, and in WHICH continuous recording
# burst — without this, train/test splits can only be random over rows, which
# silently leaks near-duplicate frames from one press-and-hold recording
# action across the split (see docs/data_collection_protocol.md).
DATASET_PATH = "isl_dataset.csv"

def _build_feature_columns():
    cols = []
    for i in range(21):
        cols.extend([f"rx{i}", f"ry{i}", f"rz{i}"])
    for i in range(21):
        cols.extend([f"lx{i}", f"ly{i}", f"lz{i}"])
    cols.extend(["rel_dx", "rel_dy", "rel_dz", "rel_d_wrist", "rel_d_index", "rel_d_thumb"])
    return cols

FEATURE_COLUMNS = _build_feature_columns()
META_COLUMNS    = ["signer_id", "session_id", "burst_id", "ts"]
DATASET_COLUMNS = FEATURE_COLUMNS + META_COLUMNS + ["label"]

def init_dataset_file():
    if not os.path.exists(DATASET_PATH) or os.path.getsize(DATASET_PATH) == 0:
        with open(DATASET_PATH, "w", newline="") as f:
            csv.writer(f).writerow(DATASET_COLUMNS)
init_dataset_file()

BURST_SPLIT_DISTANCE = 2.0  # Euclidean distance, over the 132 raw feature columns

def _migrate_dataset_schema_if_needed():
    """One-time, idempotent upgrade of a pre-existing isl_dataset.csv that
    predates signer_id/session_id/burst_id/ts. Legacy rows get honest
    placeholders (never a fabricated signer/session identity) and a burst_id
    inferred from contiguous same-label runs, since recordings were appended
    to the CSV one continuous press-and-hold action at a time. A run also
    splits mid-label if consecutive feature vectors jump by more than
    BURST_SPLIT_DISTANCE, since a same-label gap that large means the hand
    left and re-entered frame between two separate recording sessions rather
    than staying in one continuous press-and-hold action."""
    if not os.path.exists(DATASET_PATH):
        return
    with open(DATASET_PATH, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or "signer_id" in header:
            return  # empty, or already migrated
        rows = list(reader)

    print(f"Migrating {DATASET_PATH} to signer/session/burst schema "
          f"({len(rows)} legacy rows; provenance unknown, marked 'unknown_legacy')...")
    n_feats = len(FEATURE_COLUMNS)
    burst_id, prev_label, prev_feats, out_rows = -1, None, None, []
    for row in rows:
        if not row:
            continue
        label = row[-1].strip()
        feats = row[:-1]
        feat_vec = np.array(feats[:n_feats], dtype=float)
        jumped = (
            prev_feats is not None
            and np.linalg.norm(feat_vec - prev_feats) > BURST_SPLIT_DISTANCE
        )
        if label != prev_label or jumped:
            burst_id += 1
            prev_label = label
        prev_feats = feat_vec
        out_rows.append(feats + ["unknown_legacy", "unknown_legacy", f"legacy_{burst_id}", "", label])

    backup_path = DATASET_PATH + ".pre_schema_migration.bak"
    if not os.path.exists(backup_path):
        with open(backup_path, "w", newline="") as f:
            csv.writer(f).writerow(header)
            csv.writer(f).writerows(rows)
    with open(DATASET_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(DATASET_COLUMNS)
        writer.writerows(out_rows)
    print(f"Migration complete. Pre-migration copy saved to {backup_path}.")

_migrate_dataset_schema_if_needed()

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
recording_state = {
    "is_recording": False, "label": "", "sample_count": 0,
    "signer_id": "", "session_id": "", "burst_id": "",
}
recording_lock = threading.Lock()
_burst_counter = 0
training_state  = {"status": "idle", "accuracy": 0.0, "macro_f1": None, "eval_note": "", "error": ""}
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

# ── Landmark smoothing (One Euro Filter, Casiez et al. 2012) ───────
# Low lag when the hand moves fast, strong smoothing when it's nearly
# still. Applied once, per hand, to the normalized (x,y,z) landmarks right
# after detection — the SAME smoothed signal then feeds the classifier,
# the recorded training rows, AND the render overlay below, so whatever
# isl_model.pkl gets retrained on next matches exactly what it sees live.
# Tune by eye against the live feed: lower MIN_CUTOFF = smoother at rest;
# higher BETA = less lag on fast moves (e.g. folding into a fist).
# BETA pushed further (6.0 -> 12.0) and D_CUTOFF (1.6 -> 2.2) after live
# testing showed fist-close motion still felt a bit behind — a fist-close
# is a big, fast landmark motion, so it needs the filter to open up (stop
# smoothing) much more aggressively than a small idle wobble does. D_CUTOFF
# governs how quickly the filter's own velocity *estimate* reacts to a
# sudden speed change — too low, and there's a warm-up lag before BETA
# even kicks in at the start of the fast motion.
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA       = 12.0
ONE_EURO_D_CUTOFF   = 2.2

class OneEuroFilter:
    """Vectorized One Euro Filter over one hand's (21, 3) landmark array."""
    def __init__(self, shape, min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA, d_cutoff=ONE_EURO_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev     = None
        self.dx_prev    = np.zeros(shape)
        self.t_prev     = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.t_prev = x.copy(), t
            return x
        dt = max(t - self.t_prev, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * ((x - self.x_prev) / dt) + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

_landmark_filters = {}   # hand_type ('Left'/'Right') -> OneEuroFilter

# Detection runs on the horizontally-flipped (mirror/selfie-view) frame from
# camera_loop, not the raw sensor frame. Mirroring a hand image is exactly
# what turns a right hand into a left hand visually, so MediaPipe's
# handedness call comes back inverted relative to the signer's actual hand
# — and every landmark's x is mirrored (1 - true_x) too. Both are corrected
# here: swap the label back, and un-mirror x, so hand_type and coordinates
# match the signer's true anatomy and line up with the Kaggle-imported
# training rows (extracted from un-mirrored photos, never flipped).
_HANDEDNESS_UNMIRROR = {'Left': 'Right', 'Right': 'Left'}

def _raw_hand_coords(result):
    """hand_type -> raw (21,3) normalized landmark array for this frame,
    corrected to the signer's true handedness/orientation (see comment
    above _HANDEDNESS_UNMIRROR)."""
    out = {}
    if result.hand_landmarks and result.handedness:
        for landmarks, handedness_list in zip(result.hand_landmarks, result.handedness):
            hand_type = _HANDEDNESS_UNMIRROR[handedness_list[0].category_name]
            out[hand_type] = np.array([[1.0 - lm.x, lm.y, lm.z] for lm in landmarks])
    return out

def smooth_hands(raw_hands, t):
    """One-Euro-filter every hand present this frame. Filter state for a
    hand not seen this frame is dropped, so a later reappearance starts
    fresh instead of extrapolating across the gap."""
    smoothed = {}
    for hand_type, coords in raw_hands.items():
        f = _landmark_filters.get(hand_type)
        if f is None:
            f = _landmark_filters[hand_type] = OneEuroFilter(coords.shape)
        smoothed[hand_type] = f(coords, t)
    for hand_type in list(_landmark_filters.keys()):
        if hand_type not in raw_hands:
            del _landmark_filters[hand_type]
    return smoothed

# ── Render-only extra smoothing ─────────────────────────────────────
# On top of the One Euro filter above, the on-screen skeleton gets one more
# delta-time-normalized EMA pass — display can tolerate more lag than the
# classifier, so this stays separate rather than slowing the shared filter
# down for everyone.
smoothed_landmarks = {}
_hand_loss_grace    = {}
_last_draw_ts       = {"t": None}

# alpha = 1 - exp(-dt/tau): smoothing strength stays constant in wall-clock
# time regardless of render framerate dips (a fixed per-call alpha would
# smooth more aggressively whenever frames render slower). This stage is
# NOT speed-adaptive like the One Euro filter feeding it is — it applies
# the same lag whether the hand is still or snapping into a fist. Lowered
# further (0.05 -> 0.035) after live testing still showed some residual
# lag on fast folds — still visibly smoother than raw at rest, but adds
# even less of its own fixed delay on top of the adaptive filter.
DISPLAY_SMOOTH_TAU = 0.035
# Consecutive miss-frames to hold the last overlay pose before clearing it,
# so a brief tracking drop (a frame or two) doesn't flicker the skeleton
# off and back on. Visual only.
HAND_LOSS_GRACE_FRAMES = 5

def draw_landmarks(frame, hands):
    """hands: dict of hand_type -> (21,3) normalized landmark array, already
    One-Euro-filtered by smooth_hands()."""
    global smoothed_landmarks, _hand_loss_grace
    h, w = frame.shape[:2]

    now = time.perf_counter()
    last_t = _last_draw_ts["t"]
    dt = (now - last_t) if last_t is not None else (1 / 30)
    _last_draw_ts["t"] = now
    alpha = 1.0 - np.exp(-dt / DISPLAY_SMOOTH_TAU)

    current_hands = set(hands.keys())
    for hand_type, coords in hands.items():
        _hand_loss_grace[hand_type] = HAND_LOSS_GRACE_FRAMES
        # coords.x is true-orientation (see _HANDEDNESS_UNMIRROR); frame is
        # the mirrored display feed, so mirror x back just for drawing.
        pts = [[(1.0 - coords[i][0]) * w, coords[i][1] * h] for i in range(len(coords))]
        if hand_type in smoothed_landmarks:
            for i in range(len(pts)):
                pts[i][0] = alpha * pts[i][0] + (1-alpha) * smoothed_landmarks[hand_type][i][0]
                pts[i][1] = alpha * pts[i][1] + (1-alpha) * smoothed_landmarks[hand_type][i][1]
        smoothed_landmarks[hand_type] = pts

    # Hand briefly missing this frame: hold its last drawn pose for a few
    # frames instead of snapping the overlay away immediately, then clear
    # once the grace period actually runs out.
    for hand_type in list(smoothed_landmarks.keys()):
        if hand_type not in current_hands:
            _hand_loss_grace[hand_type] = _hand_loss_grace.get(hand_type, 0) - 1
            if _hand_loss_grace[hand_type] <= 0:
                del smoothed_landmarks[hand_type]
                del _hand_loss_grace[hand_type]

    for hand_type, pts in smoothed_landmarks.items():
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
    return frame

def extract_features(hands):
    """hands: dict of hand_type -> (21,3) normalized landmark array (see
    draw_landmarks) — the same smoothed landmarks used for display."""
    r_coords = hands.get('Right')
    l_coords = hands.get('Left')
    r_scale = 1.0
    if r_coords is not None:
        r_scale  = max(np.linalg.norm(r_coords[9] - r_coords[0]), 1e-6)
        r_norm   = ((r_coords - r_coords[0]) / r_scale).flatten().tolist()
    else:
        r_norm = [0.0] * 63
    if l_coords is not None:
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
# raw_q feeds the detector (downscaled for speed at high capture res),
# render_q feeds the renderer with full-res frames so display framerate
# never depends on how long detection takes.
raw_q    = queue.Queue(maxsize=1)
render_q = queue.Queue(maxsize=1)
encode_q = queue.Queue(maxsize=1)

def _put_drop(q, item):
    if q.full():
        try: q.get_nowait()
        except queue.Empty: pass
    q.put_nowait(item)

# Detection only needs to see hands well enough to place normalized
# landmarks (0..1 regardless of input size) — shrinking the frame we feed
# MediaPipe cuts detector latency a lot at 1280x720 without touching the
# resolution actually recorded/displayed. Left alone at/under 640 wide.
DETECT_MAX_DIM = 640
def _downscale_for_detect(frame):
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= DETECT_MAX_DIM:
        return frame.copy()   # independent array — the renderer draws on the render_q copy
    scale = DETECT_MAX_DIM / longest
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

# Most recent detector result, shared with the renderer so hand overlays
# can be drawn on every displayed frame without the renderer waiting on
# the (much slower) detection+ML pipeline.
latest_result = {"data": None}
latest_result_lock = threading.Lock()

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
        flipped = cv2.flip(frame, 1)
        _put_drop(render_q, flipped)
        _put_drop(raw_q, _downscale_for_detect(flipped))
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
                # Detection/presence stay strict (0.7, above MediaPipe's 0.5
                # default) — these gate whether a hand is "real" at all, and
                # loosening them would let weaker detections feed noisier
                # landmarks into both training data and live classification.
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                # Tracking confidence only decides whether the pipeline keeps
                # following an already-detected hand vs. re-running the full
                # (heavier) palm detector — it doesn't loosen what counts as
                # a valid detection. At 0.7 it was re-detecting (and briefly
                # losing landmarks) on ordinary fast hand motion or partial
                # occlusion; lowered to reduce that flicker. Revert to 0.7 if
                # you see it holding onto a stale/wrong hand region too long.
                min_tracking_confidence=0.5,
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

        # No detector → the renderer already shows the raw camera feed on
        # its own (it just won't have a result to draw an overlay from).
        if detector is None:
            continue

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        # Filter once, here, so classification/recording/rendering all see
        # the identical smoothed landmarks (see smooth_hands() above).
        raw_hands      = _raw_hand_coords(result)
        smoothed_hands = smooth_hands(raw_hands, time.perf_counter())

        # Hand off to the renderer immediately so overlay drawing never
        # waits on the feature-extraction / prediction work below.
        with latest_result_lock:
            latest_result["data"] = smoothed_hands

        det_letter   = ""
        det_conf     = 0.0
        hand_det     = bool(raw_hands)
        letter_flash = False
        right_det    = 'Right' in raw_hands
        left_det     = 'Left' in raw_hands
        auto_speak_text = None

        if raw_hands:
            feat_132 = extract_features(smoothed_hands)

            with recording_lock:
                is_rec     = recording_state["is_recording"]
                rec_label  = recording_state["label"]
                rec_signer = recording_state["signer_id"]
                rec_sess   = recording_state["session_id"]
                rec_burst  = recording_state["burst_id"]
            if is_rec:
                try:
                    with open(DATASET_PATH, "a", newline="") as f:
                        csv.writer(f).writerow(
                            feat_132 + [rec_signer, rec_sess, rec_burst, time.time(), rec_label])
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
                        coords = next(iter(smoothed_hands.values())).copy()
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

# ── Thread 3: Renderer ──────────────────────────────────────────────
# Draws the latest available hand-landmark overlay onto every captured
# frame and hands it to the encoder immediately. Decoupled from detection
# so the displayed video framerate never drops to the detector's rate —
# the overlay is just whatever result was most recently computed.
def render_loop():
    while True:
        try:
            frame = render_q.get(timeout=1.0)
        except queue.Empty:
            continue
        with latest_result_lock:
            result = latest_result["data"]
        if result is not None:
            draw_landmarks(frame, result)
        _put_drop(encode_q, frame)

# ── Thread 4: Encoder ──────────────────────────────────────────────
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

def _load_dataset():
    """Read isl_dataset.csv by column name (not position), so it works
    whether or not the signer/session/burst_id/ts metadata columns are
    present, and never silently mis-parses them as numeric features."""
    X, y, groups = [], [], []
    if not os.path.exists(DATASET_PATH):
        return np.empty((0, len(FEATURE_COLUMNS))), np.array([]), np.array([])
    with open(DATASET_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or row.get("label") is None:
                continue
            X.append([float(row[c]) for c in FEATURE_COLUMNS])
            y.append(row["label"].strip())
            groups.append(row.get("burst_id") or "unknown_legacy")
    # dtype=object keeps elements as plain python str, so lists of labels
    # printed/serialized later (e.g. training_state["eval_note"]) don't show
    # numpy's "np.str_('X')" repr instead of "X".
    return np.array(X), np.array(y, dtype=object), np.array(groups, dtype=object)

# ── Model training ─────────────────────────────────────────────────
def train_model_bg(silent=False):
    global model, labels, model_features
    # Silent (auto) retraining: bail out quietly unless there's enough data.
    if silent:
        try:
            X, y, _ = _load_dataset()
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
        X, y, groups = _load_dataset()
        if len(X) < 10:
            raise Exception("Need at least 10 samples to train.")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import GroupShuffleSplit
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2: raise Exception("Need at least 2 classes.")

        # Held-out estimate MUST be burst-disjoint: a random row split leaks
        # near-duplicate frames from the same press-and-hold recording into
        # both sides and reports near-100% accuracy regardless of whether the
        # model generalizes at all (see docs/data_collection_protocol.md).
        acc, f1m, eval_note = None, None, ""
        n_groups_per_class = {c: len(set(groups[y == c])) for c in unique}
        evaluable = [c for c, ng in n_groups_per_class.items() if ng >= 2]
        if len(evaluable) >= 2:
            mask = np.isin(y, evaluable)
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_idx, te_idx = next(gss.split(X[mask], y[mask], groups=groups[mask]))
            Xs, ys = X[mask], y[mask]
            probe = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=2, random_state=42, n_jobs=-1)
            probe.fit(Xs[tr_idx], ys[tr_idx])
            te_pred = probe.predict(Xs[te_idx])
            acc = float(accuracy_score(ys[te_idx], te_pred))
            f1m = float(f1_score(ys[te_idx], te_pred, average="macro"))
            skipped = sorted(set(unique) - set(evaluable))
            if skipped:
                eval_note = (f"Burst-disjoint eval over {len(evaluable)}/{len(unique)} classes "
                             f"(skipped, only 1 recording session each: {', '.join(skipped)})")
            else:
                eval_note = f"Burst-disjoint eval over all {len(evaluable)} classes"
        else:
            eval_note = ("No held-out accuracy available: fewer than 2 classes have more than "
                          "one independent recording session, so no split can measure "
                          "generalization rather than memorization of the same session.")

        # Deploy a model trained on ALL data — the split above is only to
        # obtain an honest accuracy estimate, not to withhold data from the
        # model that actually serves live predictions.
        clf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=2, random_state=42, n_jobs=-1)
        clf.fit(X, y)
        with open("isl_model.pkl", "wb") as f: pickle.dump(clf, f)
        new_labels = sorted(set(y))
        with open("labels.pkl", "wb") as f: pickle.dump(new_labels, f)
        with cam_lock:
            model = clf; labels = new_labels
            model_features = clf.n_features_in_ if hasattr(clf, "n_features_in_") else len(X[0])
        with training_lock:
            training_state["status"] = "success"
            training_state["accuracy"] = acc if acc is not None else 0.0
            training_state["macro_f1"] = f1m
            training_state["eval_note"] = eval_note
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

@app.route('/known_signers')
def known_signers():
    """Distinct signer_id values already recorded, so the Trainer UI can
    suggest them and reduce the chance of two different contributors
    accidentally typing the same id (which would silently merge them into
    one signer group and corrupt a signer-disjoint evaluation)."""
    ids = set()
    if os.path.exists(DATASET_PATH):
        try:
            with open(DATASET_PATH, "r", newline="") as f:
                for row in csv.DictReader(f):
                    sid = (row.get("signer_id") or "").strip()
                    if sid and sid != "unknown_legacy":
                        ids.add(sid)
        except Exception as e:
            print(f"Error reading known signers: {e}")
    return jsonify(sorted(ids))

@app.route('/dataset_info')
def dataset_info():
    return jsonify(get_dataset_counts())

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global _burst_counter
    data      = request.get_json(silent=True) or {}
    label     = data.get("label", "").strip().upper()
    signer_id = (data.get("signer_id") or "unassigned_signer").strip()
    session_id= (data.get("session_id") or "unassigned_session").strip()
    if not label: return jsonify({"error": "No label"}), 400
    counts = get_dataset_counts()
    with recording_lock:
        _burst_counter += 1
        recording_state.update({
            "is_recording": True, "label": label, "sample_count": counts.get(label, 0),
            "signer_id": signer_id, "session_id": session_id,
            "burst_id": f"{signer_id}_{session_id}_{_burst_counter}",
        })
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

BUILD_TAG = "final-mjpeg-3-decoupled"   # bump to confirm the running code is current

@app.route('/')
def index():
    resp = make_response(render_template('index.html'))
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

def _current_jpeg():
    try:
        with output_lock:
            jpg = output_frame
        if jpg is None:
            jpg = _placeholder_jpeg("Waiting for camera..." if camera_status in ("ok", "starting") else camera_status)
        return jpg
    except Exception as e:
        return _placeholder_jpeg("Server error: " + str(e))

@app.route('/frame.jpg')
def frame_jpg():
    # Single JPEG per request — kept for compatibility/debugging
    # (see /video_feed for the streamed feed the UI actually uses).
    resp = Response(_current_jpeg(), mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp

def _gen_mjpeg():
    # multipart/x-mixed-replace: one persistent connection, browser repaints
    # the <img> as each part arrives. No per-frame HTTP handshake like
    # /frame.jpg polling, so the feed rides the encoder's own framerate.
    boundary = b'--frame'
    last_sent = None
    while True:
        jpg = _current_jpeg()
        if jpg is not last_sent:
            last_sent = jpg
            yield (boundary + b'\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                   str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(_gen_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ── Entry point ────────────────────────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=frame_grabber, daemon=True, name="grabber").start()
    threading.Thread(target=detector_loop, daemon=True, name="detector").start()
    threading.Thread(target=render_loop,   daemon=True, name="renderer").start()
    threading.Thread(target=camera_loop,   daemon=True, name="encoder").start()

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║    ISL RECOGNITION SYSTEM — FINAL (Windows)  ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  BUILD: {BUILD_TAG:<36} ║")
    print("║  Video: streamed /video_feed (MJPEG)         ║")
    print("║  Open:  http://localhost:5000                ║")
    print("║  Test:  http://localhost:5000/video_feed     ║")
    print("║  Stop:  Ctrl+C                               ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    def _open():
        time.sleep(0.5)
        webbrowser.open('http://localhost:5000')
    threading.Thread(target=_open, daemon=True).start()

    sio.run(app, host='0.0.0.0', port=5000, debug=False,
            use_reloader=False, allow_unsafe_werkzeug=True)
