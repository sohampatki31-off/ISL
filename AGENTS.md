# Agent instructions

Read `README.md` in full before making any change, especially the "File
ownership" and "API contract" sections. This project is a single Flask
process: a Python backend (camera + hand tracking + ML) serving a plain
HTML/CSS/JS frontend directly, with no build step and no separate API
server.

## Rules

1. Never modify `isl_main.py`, `train_model.py`, `evaluate.py`,
   `isl_model.pkl`, `labels.pkl`, `isl_dataset.csv`, or `hand_landmarker.task`.
   These are backend files with a deliberately tuned hand-tracking pipeline
   (One-Euro filter, MediaPipe confidence thresholds) — off limits.

2. Only edit `templates/index.html`, `static/css/style.css`,
   `static/js/app.js`, or new frontend files you add alongside them.

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
