# SecureLens

SecureLens is a Django-based image authenticity studio for comparing AI-generated and real-looking images. It combines a polished web UI with research-style signals like entropy, FFT, edge density, metadata, image quality, batch analysis, and side-by-side comparison.

## What it does

- Single-image AI vs real analysis
- Batch analysis with a card-based matrix layout
- Two-image comparison mode
- Gesture-to-emoji webcam demo
- Image metadata, compression cues, and quality metrics
- FFT and heatmap visual evidence
- Saved account history and stats
- PDF report export
- Optional manual labeling for model performance tracking

## Project layout

```text
SecureLens/
├── core/                 # Django project settings and routing
├── dashboard/            # App with views, models, forms, templates
├── media/                # Uploaded images in local development
├── db.sqlite3            # Local SQLite database
├── manage.py             # Django entry point
├── requirements.txt     # Python dependencies
└── render.yaml          # Deployment config
```

## Data storage

- Users are stored in Django auth tables inside `db.sqlite3` during local development.
- Analysis records are stored in `dashboard_imageanalysis`.
- Uploaded images are stored in `media/uploads/`.
- `analysis_meta` stores FFT, heatmap, quality, metadata, and model evidence.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Then open `http://127.0.0.1:8000/`.

## Main pages

- `/` Home
- `/analyze/` Single image analysis
- `/batch/` Batch analysis
- `/compare/` Compare two images
- `/research/` Research overview
- `/stats/` Dataset and performance dashboard
- `/history/` Saved analyses

## Research signals

- Resolution
- File size
- Color channels
- Entropy
- FFT spectrum
- Edge density
- Compression score
- Sharpness, blur, brightness, and contrast

## Notes

- SecureLens is cautious by design and can return `UNCERTAIN` when signals are too close.
- The app uses multiple detector views and heuristic image cues together instead of one weak label.
- If you deploy to Render or another host, the database can switch to PostgreSQL through `DATABASE_URL`.

## Dependencies

Core packages include Django, OpenCV, Pillow, NumPy, Transformers, Torch, TensorFlow, Matplotlib, and ReportLab.
