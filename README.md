# ISL Recognition System

A single Flask process: Python backend (camera + hand tracking + ML) serving
a plain HTML/CSS/JS frontend directly — no build step, no separate frontend
server, no API base URL to configure. The browser just talks to whatever this
Flask process is running on (default `http://localhost:5000`).

## Architecture

```
Camera → MediaPipe HandLandmarker → One-Euro filter → classifier + overlay
                                                            │
                                                     Flask routes (isl_main.py)
                                                            │
                                              templates/index.html + static/js/app.js
                                              (polls /state every ~120ms, reads
                                               /video_feed as an <img> stream)
```

## File ownership — read this before changing anything

**Default is deny, not allow.** Only the frontend row below is editable.
Everything else in the repo is backend/data/tooling and is off-limits, even
files not explicitly named in the backend row — that list is examples, not
the full set.

| Owns | Files | Rule |
|---|---|---|
| **Backend** | `isl_main.py`, `train_model.py`, `evaluate.py`, `fix_dataset.py`, `inspect_models.py`, `process_images.py`, `test_camera.py`, `isl_model.pkl`, `labels.pkl`, `isl_dataset.csv`, `isl_model_one_hand.pkl`, `labels_one_hand.pkl`, `isl_dataset_one_hand.csv`, `hand_landmarker.task`, `start.bat`, `sp_asep.pdf`, `docs/`, `paper/`, `results/`, `requirements.txt`, `.gitignore`, plus this README and `AGENTS.md` — **and anything else not listed in the Frontend row** | Do not modify. This is tuned, working hand-tracking/ML code plus supporting data/docs. If a frontend feature needs new data from the backend, add a new field to an existing response — never rename/remove an existing one — and flag it rather than editing `isl_main.py` directly. |
| **Frontend** | `templates/index.html`, `static/css/style.css`, `static/js/app.js` (and new frontend files added alongside them) | Safe to redesign freely — layout, styling, UX, all of it — as long as it keeps calling the API contract below with the same URLs, methods, and field names. |

If you (or an agent) want to introduce a frontend build step or framework:
it still has to end up served by, or proxied to, this same Flask process —
the backend has no CORS setup and isn't a separate API server on its own
origin. The simplest path is to keep editing the existing plain JS/CSS/HTML
in place.

## API contract

All responses are JSON unless noted. This is what the frontend currently
depends on — preserve it.

### Page / assets
- `GET /` — renders `templates/index.html`.
- `GET /version` — `{ build, camera_status, has_frame }`. Confirms the running build.

### Video
- `GET /video_feed` — MJPEG stream (`multipart/x-mixed-replace`). The `<img>` tag's `src` points here; the browser repaints it itself.
- `GET /frame.jpg` — single JPEG snapshot (fallback/debugging).

### Live state (poll every ~120ms)
- `GET /state` →
  ```json
  {
    "letter": "A", "confidence": 0.0, "stable_count": 0,
    "current_word": "", "full_text": "", "suggestions": [],
    "hand_detected": false, "left_hand_detected": false, "right_hand_detected": false,
    "letter_flash": false, "model_features_loaded": 132,
    "history": [], "auto_speak": false,
    "total_letters": 0, "total_words": 0,
    "tts_id": 0, "tts_text": "", "camera_status": "ok"
  }
  ```
  `tts_id` increments each time something new should be spoken; the frontend
  speaks `tts_text` when `tts_id` increases. `letter_flash` is true for one
  poll after a new letter locks in, then auto-resets server-side.

- `GET /training_status` → `{ status, accuracy, macro_f1, eval_note, error }`. `status` is one of `idle | training | success | error`.
- `GET /dataset_info` → `{ "A": 12, "B": 40, ... }` — sample counts per class.
- `GET /known_signers` → `["s1", "s2", ...]` — distinct signer ids recorded so far.

### Actions
- `POST /action` — body `{ "action": "...", ...extra }`. `action` is one of:
  `space | enter | backspace | backspace_word | clear | punct (needs "char") | quick_phrase (needs "text") | suggest (needs "index") | auto_speak (needs "value") | clear_history`.
  Returns `{ status: "ok", tts: "..." }`.

### Recording / training (Trainer tab)
- `POST /start_recording` — body `{ label, signer_id, session_id }` → `{ status, current_count }`.
- `POST /stop_recording` → `{ status, label, count, counts }`.
- `POST /delete_latest` — body `{ label }` → `{ status, count, counts }`.
- `POST /clear_class` — body `{ label }` → `{ status, count, counts }`.
- `POST /train` → `{ status: "started" }` (kicks off background training; poll `/training_status` for progress).

### Misc
- `POST /set_resolution` — body `{ width, height }`, only `(1280,720)` or `(640,480)` accepted → `{ status, width, height }`.
- `GET /download_history` — plain-text file download of the session transcript.

## Hand-tracking pipeline notes

`isl_main.py` applies a One-Euro filter (per hand, per landmark) immediately
after MediaPipe detection. The *same* filtered landmarks feed the classifier,
the recorded training CSV rows, and the render overlay — this is deliberate,
so whatever the model gets retrained on next matches what it sees live.
`ONE_EURO_MIN_CUTOFF` / `ONE_EURO_BETA` / `ONE_EURO_D_CUTOFF` and
`DISPLAY_SMOOTH_TAU` were tuned by hand against a live camera to balance
smoothness against lag on fast motion (e.g. folding into a fist). Don't
re-tune or remove this without a reason — it's not arbitrary.

## Running it

```
pip install -r requirements.txt
# place hand_landmarker.task next to isl_main.py (see the download URL in isl_main.py's docstring)
python isl_main.py
```
Opens `http://localhost:5000` automatically.

## Retraining

The classifier (`isl_model.pkl` / `labels.pkl`) is retrained from
`isl_dataset.csv` via the Trainer tab's "TRAIN MODEL" button (or `POST
/train`). Plan is to re-record a larger dataset (~200 samples/class) on this
smoothed pipeline before the next retrain — see `docs/data_collection_protocol.md`
for the signer/session/burst methodology already in place.
