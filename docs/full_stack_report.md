# SecureLens Full-Stack Implementation Report

Local upgrade, 2026-10-07. No commit, push or deployment was performed.
See the root README for installation and configuration.

This is the initial implementation snapshot. The subsequent production
preparation changes, current test counts and exact hosting settings are in
[production readiness report](production_readiness_report.md) and
[deployment guide](deployment.md). Those supersede this snapshot's descriptions
of cleanup scheduling, fixed cookie settings and missing deployment templates.

## 1. Architecture

React 19, Vite and Tailwind CSS form the primary browser application. FastAPI
handles cookie-authenticated requests, image validation, analysis, saved history,
preferences and reports. SQLAlchemy persists accounts/results in PostgreSQL;
Alembic manages schema changes. A storage protocol separates private local files
from a future cloud adapter. The API calls the existing pure `src/` analysis
engine instead of replacing its scientifically meaningful algorithms.

Django remains a preserved legacy/research application; Streamlit remains a
research interface. Neither is the recommended production backend. No dataset,
model, research module or Django database was deleted or automatically migrated.

## 2. Frontend Files

```text
frontend/
  package.json, pnpm-lock.yaml, .prettierrc.json, .env.example
  index.html, vite.config.js
  src/App.jsx, src/main.jsx, src/test-setup.js
  src/components/UI.jsx
  src/components/UI.test.jsx
  src/components/Providers.jsx
  src/components/CookieBanner.jsx
  src/layouts/Layouts.jsx
  src/hooks/useResource.js
  src/hooks/useCamera.js, useCamera.test.jsx
  src/hooks/useRetention.js
  src/services/api.js, api.test.js
  src/styles/app.css
  src/pages/Landing.jsx
  src/pages/Auth.jsx
  src/pages/Dashboard.jsx
  src/pages/Analyze.jsx
  src/pages/Live.jsx
  src/pages/Batch.jsx
  src/pages/Compare.jsx
  src/pages/History.jsx
  src/pages/Settings.jsx
  src/pages/Information.jsx
```

Reusable components include Navbar, Sidebar/application layout, Footer,
AuthForm, UploadZone, ImagePreview, ResultCard, AnalysisCard, MetricCard,
ForensicViewer, ProgressIndicator, HistoryTable, BatchTable, ComparePanel,
CookieBanner, skeletons, errors, empty states and notifications.

The soft-pink design uses gradients, translucent cards, rose accents, expressive
typography, an illustrative scanner and restrained animation. Main assessments
precede supporting evidence; exact technical values stay collapsed by default.
Reduced-motion preferences and desktop/mobile layouts are implemented.

## 3. Backend Files

```text
backend/
  requirements.txt, requirements-dev.txt, .env.example, pytest.ini
  alembic.ini
  app/main.py, app/config.py, app/__init__.py
  app/api/auth.py, analysis.py, history.py
  app/analysis/engine.py
  app/auth/security.py
  app/database/session.py
  app/models/records.py
  app/schemas/requests.py
  app/services/storage.py, records.py
  app/utils/limits.py
  migrations/env.py
  migrations/versions/0001_workspace.py
  scripts/setup_local.py
  scripts/cleanup_storage.py
  tests/conftest.py
  tests/test_workspace.py
```

Package directories also have Python `__init__.py` files. Additional setup files:
`compose.yaml`, `tools/dev.py`, `tools/create_test_images.py`, and
`tools/local-postgres/{package.json,pnpm-lock.yaml,pnpm-workspace.yaml,start.mjs}`.
README and `.gitignore` were updated. Generated credentials, dependencies,
database files, reports, previews and test images are ignored.

## 4. Database Models

- `users`: UUID, name, unique normalized email, Argon2 password hash, creation
  timestamp and session revocation version.
- `analyses`: UUID, owner UUID, type (single/live/batch/compare), filename,
  status, timestamp, result JSON, report key, retained-preview keys and expiry.
- `user_preferences`: owner key and preferred preview-retention setting.
- `cookie_consents`: owner key, essential/authentication/analytics flags and
  update timestamp.

Batch items and comparison details remain JSON within `analyses`. This avoids
unnecessary job tables while retaining each image's status, metrics and result.
Progress is streamed during processing; this is not a durable background queue.

Real PostgreSQL 18.4 was started privately on `127.0.0.1:55432` using the portable
development helper. Alembic created the four tables and `alembic_version`.
The original root `db.sqlite3` remains the legacy Django database.

## 5. Authentication

Passwords use Argon2 via pwdlib. Signed HS256 JWTs are stored in an HttpOnly,
SameSite=Lax cookie; their secret is generated into ignored `backend/.env`.
Authenticated writes require a matching CSRF header. Allowed origins are
explicit, and unsafe requests from other origins are rejected. Session expiry
and revocation are checked server-side. Logout revokes that account's existing
sessions, including other signed-in browsers. Frontend session expiry returns
the user to sign-in. No token is stored in browser local storage.

Missing-account login uses a dummy password hash to avoid a fast timing path.
Sign-in/register requests have a per-process, per-IP rate limit. Use a shared
limiter before running multiple public API workers. Email recovery is clearly
marked as not implemented; no fake recovery emails are sent.

## 6. Storage

The `Storage` protocol defines write/read/delete; `LocalStorage` implements
private directory/file permissions, bounded paths and atomic file writes.
UUID storage keys prevent filename collisions. All reads are owner-checked API
routes, not public static-media URLs.

- PostgreSQL: account records, saved measurements, preferences and consent.
- `backend/.local/postgres/`: local portable PostgreSQL data directory.
- `backend/.local/storage/reports/`: saved JSON reports.
- `backend/.local/storage/images/`: optional downscaled original/ELA/FFT previews.

Original upload bytes are not permanently retained. Multipart parsing can spool
large files to temporary disk; upload files are closed after validation/reading.
Analysis works from decoded bytes. Preview retention defaults off, is configurable
per analysis, and expires after seven days by default. Metrics/reports persist
until their analysis is deleted. Settings can pre-fill upload retention choices;
live saving still requires explicit action.

Expired previews are removed on owner history/detail/preview requests. A provided
maintenance command must be scheduled for idle-user retention enforcement.
Cloud storage is not installed or falsely configured; a private provider adapter
can implement the protocol later.

## 7. Cookies And Privacy

The banner offers Accept, Reject Non-Essential and Manage Preferences. Essential
and authentication categories remain enabled; analytics remains disabled because
there is no analytics/advertising integration. Preferences are kept in local
storage and synced to the signed-in account. Local storage contains preference
flags, not authentication credentials.

The privacy page describes actual saved data, preview expiry, deletion and
limitations. It is not a legal compliance claim. Google Fonts requests are
disclosed; self-host fonts when third-party asset requests are unacceptable.

## 8. Preserved Features

Shared ELA, FFT, brightness, pixel deviation, entropy, edges, texture, source
metadata and JPEG quality 90/50/20 experiments remain intact. Existing forensic
and Streamlit tests are unchanged. Django's account/history/batch/compare ideas
were adapted without deleting its code or touching its saved login database.
Research files, training scripts, datasets and model artifacts remain separate.

A deleted `dashboard/templates/dashboard/result.html` was already present in the
working-tree status before this upgrade and was not restored or otherwise
changed. The existing Django tests pass, but they do not fully exercise every
legacy result-page template. This pre-existing deletion may affect that old page.

## 9. Live Analysis

The live page requests video only, not audio. Start enables camera access and
analyzes one PNG frame at a time with a pause between completed requests. Stop
aborts pending client requests and closes tracks; unmount also stops the camera.
Late permission grants are closed if the user has already stopped. Captures are
not automatically written to permanent storage.

Users can explicitly save a frame's metrics or capture up to ten frames into a
session-only batch. Preview retention and batch saving are separate choices.
Camera origin does not force an authentic label. Gesture/emotion/reaction
features were not reintroduced.

Camera control and privacy behavior pass mocked-browser hook tests; camera-frame
analysis passes API tests. A physical laptop webcam was not enabled during this
verification, so device permission and hardware capture still need a user check
in their browser. Camera APIs require localhost or HTTPS.

## 10. Batch Analysis

Multi-image uploads and captured batches share the analyzer. Real per-item SSE
events update completed/total counts. Each item has a thumbnail, filename, status,
classification and ELA/FFT values; selecting it opens detailed evidence. Invalid
items produce explicit errors without discarding valid results. Up to ten files
and a 64 MB request are allowed. JSON and CSV work without saving a batch; saved
batches additionally support PDF reports and reopen from private history.

The two-image real API smoke test finished successfully and persisted its results
to PostgreSQL. The automated suite also checks partial failures and progress.

## 11. Compare

Two uploads produce side-by-side classification and original/ELA/FFT views.
Tables compare original/processing dimensions, brightness, ELA/FFT metrics,
format, channels, file size, EXIF presence and JPEG quality 90/50/20 pixel loss.
Histogram, edge and pixel similarity are shown with an explicit non-confidence
label. An identical-image API test yields 100/100; the real two-image smoke test
also succeeds. This simple similarity is not identity or authenticity proof.

## 12. Current Result Calculation

The new API reuses the existing thresholds exactly:

1. Gain-20 ELA mean below 8 adds two points.
2. FFT log-magnitude mean above 120 adds one point.
3. Zero points: LOW, Likely Authentic. One/two: MODERATE, Inconclusive.
   Three: HIGH, Potential AI-Generated.

Brightness, entropy, metadata and compression experiments add context, not
classification points. Explanations refer only to the two triggered/not-triggered
rules. Frequency statistics are size/scene dependent; smooth real photographs
can trigger ELA. The result's probability is null, and no percentage is shown.

Manipulation-specific indicators are NOT_ESTABLISHED: no validated decision or
localization method exists here. Inventing a LOW/HIGH manipulation assessment
would misrepresent the evidence. JPEG loss describes newly introduced
recompression changes, not original compression history.

## 13. Trained Classifier Status

The primary app has **no validated trained AI-vs-real classifier integrated**.
Existing experiments/model artifacts were not promoted into an accuracy claim.
Adding a classifier requires verified real/generated labels, source-disjoint
splits, saved class mapping/preprocessing, an independent held-out evaluation,
false-positive measurement and calibration before showing model probabilities.
Filtered camera photos are not valid synthetic-image labels. See README's ML
integration section for the next research steps.

## 14. Verification

- 18 new API tests pass using isolated SQLite fixtures.
- The same 18 API tests pass using isolated real PostgreSQL databases.
- 11 frontend unit/hook/API-client tests pass, covering truthful result wording, uploads,
  progress, camera privacy/start/stop, late permission grants, CSV escaping,
  streamed chunk parsing and session-expiry handling.
- Report-export tests verify authenticated fetching, file naming, object-URL
  cleanup and local CSV export without saving an analysis. Browser download-event
  capture was unavailable; report endpoints and export logic were checked instead.
- 19 preserved forensic/Streamlit tests pass.
- 9 preserved Django tests pass; Django system checks report no issues.
- React production build succeeds.
- Real local API health, login, single, batch, comparison and JSON/CSV/PDF requests
  succeed against PostgreSQL; retained previews and saved history reopen.
- Browser signup, cookie choices, file selection, saved-result rendering and
  expandable ELA/FFT/compression values were verified with generated test images.
- Mobile landing/history/navigation were checked at 390 pixels, without page
  horizontal overflow. Desktop screenshots and a full mobile capture are in
  `docs/screenshots/`.

The test run emits an upstream FastAPI/Starlette TestClient deprecation warning;
it does not cause failures. Streamlit AppTest's bare-mode context warnings are
expected during its existing tests.

## 15. Remaining Limitations

There is no measured AI-detection accuracy or calibrated model probability.
Manipulation localization is not implemented. WEBP is intentionally rejected
because the shared engine currently supports JPG/PNG. Batches are request-bound
and do not resume as durable jobs after a browser/server interruption. PDF export
is a text/metrics report, not a visual report containing every graph. Non-Latin
PDF filenames use replacement characters with the built-in font.

Password recovery, secure password changes, account deletion, shared public
rate limiting, production object storage/backups and scheduled cleanup need
implementation/configuration before a public launch. Production requires HTTPS,
secure cookies, correct origins and SPA/API routing. A physical webcam check is
still needed. New PostgreSQL accounts are separate from old Django accounts.

## 16. Exact Local Launch

On this workstation dependencies and the ignored local environment are ready.
An existing server occupies 8000, so the new API uses 8001. The frontend is 5173.
The portable PostgreSQL development instance uses 55432.

```bash
cd /Users/aaryawaghmare7/SecureLens
python3 tools/dev.py --api-port 8001
```

Open `http://127.0.0.1:5173`. API docs are `http://127.0.0.1:8001/docs`.
If these new services are already running, open the URL instead of starting a
duplicate. Use Ctrl+C in their launcher terminal before a later relaunch.
For a first setup on another machine, run `python3 tools/dev.py --setup` with
Node.js installed; see README for separate-terminal/manual database setup.
