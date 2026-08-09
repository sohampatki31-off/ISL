# Agent instructions

Read `README.md` in full before making any change, especially the "File
ownership" and "API contract" sections. This project is a single Flask
process: a Python backend (camera + hand tracking + ML) serving a plain
HTML/CSS/JS frontend directly, with no build step and no separate API
server.

## Rules

1. **Default is deny, not allow.** The only files you may edit are
   `templates/index.html`, `static/css/style.css`, `static/js/app.js`, and
   new frontend files you add alongside them. If a file isn't one of those,
   don't modify, move, rename, or delete it — even if it isn't explicitly
   named below. Don't treat rule 2's list as exhaustive; treat it as
   examples of what rule 1 already covers.

2. This currently includes (non-exhaustive): `isl_main.py`, `train_model.py`,
   `evaluate.py`, `fix_dataset.py`, `inspect_models.py`, `process_images.py`,
   `test_camera.py`, `isl_model.pkl`, `labels.pkl`, `isl_dataset.csv`,
   `isl_model_one_hand.pkl`, `labels_one_hand.pkl`, `isl_dataset_one_hand.csv`,
   `hand_landmarker.task`, `start.bat`, `sp_asep.pdf`, everything under
   `docs/`, `paper/`, and `results/`, plus `requirements.txt`, `.gitignore`,
   `README.md`, and this file. These are backend/data/tooling files,
   including a deliberately tuned hand-tracking pipeline (One-Euro filter,
   MediaPipe confidence thresholds) — off limits.

3. Every network call the frontend makes must keep using the exact endpoint
   paths, HTTP methods, and JSON field names documented in README.md's API
   contract. You may *add* a new field to a request/response if a new UI
   feature genuinely needs backend data that doesn't exist yet — but never
   rename or remove an existing field, and call out explicitly when you do
   this so the backend can be checked to actually support it.

4. If you introduce a build step or frontend framework, it must still end
   up served by (or proxied to) this same Flask process on port 5000 —
   there is no separate API server and no CORS setup.

5. Before any change that touches how the frontend talks to the backend,
   re-read the API contract section of README.md and confirm you're still
   consistent with it.

6. If a task seems to require breaking any of the above, stop and ask
   rather than proceeding.
