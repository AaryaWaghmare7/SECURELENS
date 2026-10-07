# Repository review and Streamlit integration

## Existing analysis

- `src/image_analyzer.py`: CLI ELA (JPEG quality 90, gain 20), FFT
  (`20 * log(abs(fftshift(fft2(gray))) + 1)`), Canny edges (100/200),
  Laplacian variance, 16-pixel texture patches, RGB statistics, EXIF, and a
  legacy heuristic AI/real scorer. ELA and FFT now delegate to shared modules.
  A main guard makes imports safe without scanning datasets or exiting Python.
- `dashboard/views.py`: Django ensemble inference plus ELA (gain 15), entropy,
  FFT previews and radial magnitude features, JPEG quality experiments,
  metadata, image quality, comparison, reports, and account-specific history.
  Its existing behavior is preserved; it is not imported by Streamlit.
- `src/dashboard_analysis.py` and `src/image_analysis_graphs.py`: overlapping
  dataset statistics and Matplotlib plots with pandas. Left intact as research
  scripts, alongside `src/visualize_features.py` and notebook checkpoints.
- `models/train_model.py`: existing PyTorch MobileNet training, left intact.
  Local model weights and data remain on disk and outside version control.
- `download_images.py`, `download_ai_images.py`, `test_models.py`: existing
  download and model experiments, left intact. The AI-style download script
  labels filtered genuine photographs as AI; these labels are not valid
  evidence for evaluating AI detection. Existing scripts also disable TLS
  certificate verification; the new app does not execute them.

## New application boundary

`streamlit_app.py` calls only `src/analyzer.py`, `src/image_utils.py`,
`src/ela.py`, and `src/fft.py`. There are no Django imports, database writes,
model downloads, or fixed machine paths in this flow. Recompression uses
memory buffers, so uploads create no repository files. Results stay in the
browser's server session and can be downloaded as JSON. No global upload
cache is shared across users.

The old scorer's ELA < 8 and FFT mean > 120 thresholds and weights (2 and 1)
are retained, but only these two signals form the Streamlit summary. Zero
points is LOW, one or two is MODERATE, three is HIGH. Category cutoffs are
explicit unvalidated heuristics in `HEURISTIC_THRESHOLDS`, not learned
boundaries or probabilities. ELA values use gain 20, and raw recompression
errors are reported separately. FFT means depend on resolution and content.
Other original CLI scoring rules remain available only in the legacy CLI.

## Dependencies and preservation

Root `requirements.txt` contains the four direct dependencies for Streamlit.
Matplotlib is unnecessary for this entry point because Pillow images display
the ELA/FFT outputs directly. `requirements-django.txt` preserves the existing
Django/model dependencies and adds pandas/torchvision used by research scripts.
OpenCV uses the headless package. The Render build script explicitly installs
the Django dependencies before running migrations and collecting static files.

The pre-existing deletion of `dashboard/templates/dashboard/result.html` and
untracked `src/src/` directory were left untouched. Dataset assets, the SQLite
database, and notebook files were not edited.
