# SecureLens Production Preparation Report

2026-10-07. Local changes only. No staging, commit, push, deployment, account
provisioning or purchase was performed.

## 1. Existing Architecture Found

The repository already contained the React/Vite/Tailwind frontend, FastAPI,
SQLAlchemy/PostgreSQL/Alembic, private local storage, cookie authentication and
single/live/batch/compare/history/report flows from the previous upgrade.
Streamlit, Django (`core/`, `dashboard/`), model scripts/weights, datasets and
research modules were also present. The earlier audit is in
`repository_review.md`; the app inventory is in `full_stack_report.md`.

The root `render.yaml` and `build.sh` were still Django deployment files.
Cookies were fixed to Lax, and retention for idle users required separately
scheduled maintenance. Those were deployment gaps, not reasons to rebuild.

## 2. Architecture Prepared

React -> Vercel same-origin `/api` proxy -> Render FastAPI -> managed PostgreSQL
and a private durable disk -> unchanged `src/` ELA/FFT/statistics engine.
Vercel/Render templates and safety checks are prepared, not publicly verified.
Legacy code remains at its original paths to preserve imports and tooling.

## 3. Frontend Files

Created `frontend/vercel.mjs`, `frontend/deployment/config.mjs`,
`frontend/src/services/configuration.js` and its test file.
Updated API configuration, `App.jsx`, `Landing.jsx`, `Information.jsx`,
`Layouts.jsx`, `package.json` and `.env.example`. Pink palette/CSS and all
analysis workflows remain. Added current desktop/mobile screenshots in `docs/screenshots/`.

The hero names SecureLens prominently and no longer claims "AI-Powered"
inference. Supporting method cards explain the actual measurements. Added a
dedicated cookie-preferences route and kept the existing working dialog.

## 4. Backend Files

Created `app/services/maintenance.py`, `tests/test_deployment.py` and the
`0002_user_updated_at` Alembic migration. Updated configuration, session cookie
issuance/removal, account model, database engine settings and API middleware.
Root `render.fastapi.yaml` is a separate new backend deployment template.

## 5. Database Models

Preserved four tables: `users`, `analyses`, `user_preferences`, `cookie_consents`.
Added `users.updated_at` and backfilled existing values from `created_at`.
ORM updates refresh it. Accounts, hashes and existing analysis rows were not
replaced. Batch/comparison items remain JSON in `analyses`; separate job/result
tables would not add durable processing by themselves and were not invented.

## 6. Analysis Preserved

No analysis formula, dataset, model artifact, original Django database or
research module was rewritten by this request. The shared engine still performs
bounded decoding, EXIF orientation, white transparency composition, ELA JPEG
quality 90/gain 20, FFT, pixel/channel statistics, Canny edges, entropy,
Laplacian/texture metrics and JPEG 90/50/20 loss experiments.

## 7. Actual AI/Real Result Calculation

Gain-enhanced ELA mean below 8 adds two heuristic points. FFT spectrum mean
above 120 adds one. Zero points -> LOW / Likely Authentic. One or two points ->
MODERATE / Inconclusive. Three -> HIGH / Potential AI-Generated.
FFT remains `20 * log(abs(fftshift(fft2(gray))) + 1)`.
`probability` remains null. Compression/pixel statistics provide context,
not extra invented classification rules. Comparison similarity is not AI confidence.

## 8. Trained Classifier Status

Experimental training code and artifacts exist, but no **validated** classifier
is integrated into the primary API. These rules are unvalidated and can flag
real photographs. Manipulation indicators stay NOT_ESTABLISHED; there is no
validated manipulation-specific verdict/localizer. "Potentially Manipulated"
is not emitted merely to fill a requested label list.

A proper integration needs verified real/generated labels, independent source-
disjoint splits, documented preprocessing/class mapping/model version, held-out
unseen camera/generator/compression tests, false-positive/precision/recall metrics
and calibrated probabilities. Existing filtered real-photo "AI" labels are not
valid training/evaluation ground truth.

## 9. Live Analysis

Preserved browser `getUserMedia`, serial frame requests, start/stop, frame capture,
session-only default behavior and explicit saving/captured batches. No server
webcam, emotion or gesture dependency. Mocked camera lifecycle tests pass.
Physical camera/permission behavior on production HTTPS still needs testing.

## 10. Batch Analysis

Preserved 1-10 images, actual per-item SSE progress, thumbnail/result matrix,
partial/all failure states and JSON/CSV reports. Jobs remain request-bound and
do not resume after interruption. Production proxy time/body limits must be tested.

## 11. Comparison

Preserved two images, side-by-side original/ELA/FFT, measured metadata,
brightness/compression/pixel values and explicitly non-authenticity similarity.
API regression tests pass. No similarity percentage is relabeled AI confidence.

## 12. Authentication

Argon2 hashes, expiring signed JWT in an HttpOnly cookie, CSRF headers, exact
Origin/CORS checks, owner authorization and account-wide logout revocation remain.
SameSite is now configurable; None requires Secure. Production requires HTTPS.
Logout uses matching cookie attributes. Validation errors omit submitted inputs,
SQL errors hide parameters and unexpected-error logs omit sensitive exception bodies.

Host allowlisting and additional response headers were added. Rate limiting
remains per process; proxy/IP handling and shared limiting need public-load review.
No verified email recovery/password reset or account deletion is falsely claimed.

## 13. Database Status

The timestamp migration was applied to local PostgreSQL, and `alembic check`
reports no schema drift. All 32 backend tests pass against separately created,
randomly named PostgreSQL test databases, as well as isolated SQLite fixtures.
Managed connection URLs normalize to the installed psycopg driver.
The original Django `db.sqlite3` remains separate and untouched.

## 14. Cookies, Privacy And Storage

Cookie preferences remain browser-local and sync to signed-in accounts. Essential/
authentication categories are required; analytics stays off because none exists.
Added `/cookie-preferences`, retained the dialog and updated the privacy notice.
Raw uploads are transient; retained previews are explicit opt-in and downscaled.
The storage interface remains private local `write/read/delete`, with Render
durable disk configuration and a documented seam for future cloud adapters.

The API cleanup worker runs at startup and periodically while the API is alive;
owner reads/manual cleanup also work. Expired previews are removed without
removing saved metrics/reports. Cleanup resumes after downtime. Formal privacy
review, backup policies and self-hosted fonts remain deployment considerations.

## 15. Test Results

| Check | Result |
| --- | --- |
| Backend API/security/deployment tests, isolated SQLite | 32 passed |
| Same tests, isolated real PostgreSQL | 32 passed |
| Frontend component/hook/API/deployment configuration tests | 14 passed |
| Preserved forensic and Streamlit tests | 19 passed |
| Preserved Django tests | 9 passed; no system-check issues |
| Alembic schema comparison | No new upgrade operations |
| Frontend formatting check | Passed |
| Python compilation and Git whitespace checks | Passed |
| Current checkout Git candidate safety scan | No common-secret/large-file findings |

The upstream Starlette/httpx TestClient deprecation and Streamlit bare-context
warnings are non-failing. No ML accuracy was measured.
Browser checks verified the revised landing, cookie page/dialog, expired-session
redirect and 390 px mobile navigation with no page horizontal overflow. No
console errors were captured in that preview. This turn's browser file-picker
test was interrupted, so fresh browser upload/analysis/download completion is
not claimed. The API and frontend upload/report unit tests pass.

## 16. Frontend Build

Vite production build passes, approximately 331.48 kB JavaScript (103.22 kB gzip)
and 34.57 kB CSS (8.85 kB gzip). Commands used the existing bundled Node runtime
on this workstation; they run the same Vite/Vitest/Prettier programs as the
documented npm/pnpm scripts. No new dependency installation was needed.

## 17. Exact Local Startup

```bash
cd /Users/aaryawaghmare7/SecureLens
python3 tools/dev.py --api-port 8001
```

Open `http://127.0.0.1:5173`; API docs: `http://127.0.0.1:8001/docs`.
Both currently return HTTP 200. If already running, just open the link. The
launcher now explains occupied ports without telling you to reuse an occupied
port. For first setup use `python3 tools/dev.py --setup --api-port 8001`.
Separate-terminal virtualenv/PostgreSQL/frontend commands remain in README.

## 18. Exact Environment Variables

Backend required: `DATABASE_URL`, `JWT_SECRET`. Production additionally requires
`ENVIRONMENT=production`, `COOKIE_SECURE=true`, exact HTTPS `FRONTEND_ORIGINS`,
explicit `TRUSTED_HOSTS`, absolute durable `STORAGE_ROOT`; use
`COOKIE_SAMESITE=lax` with the recommended proxy. Configurable:
`SESSION_MINUTES`, `IMAGE_RETENTION_DAYS`, `AUTH_RATE_LIMIT`,
`STORAGE_CLEANUP_INTERVAL_SECONDS`. Render supplies `PORT`; the template pins
`PYTHON_VERSION=3.14.3`. Local-only: `LOCAL_POSTGRES_PASSWORD`.

Vercel: `VITE_API_BASE_URL=https://<your-real-Render-host>` and
`VITE_API_PROXY=true`. Local-only: `VITE_PROXY_TARGET` if overriding the API port.
See `deployment.md` for the full exact examples; none contains real credentials.

## 19. Exact Render Settings

Repository root (not `backend`), Python runtime, build:
`pip install -r backend/requirements.txt`. Pre-deploy:
`cd backend && python -m alembic upgrade head`. Start:
`cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --no-access-log`.
Health `/api/health`. Private disk `/var/data/securelens`. Blueprint
`render.fastapi.yaml` describes paid resources; no cost has been incurred here.

## 20. Exact Vercel Settings

Root `frontend`, Vite preset, Node 24.x, install via lockfile or
`pnpm install --frozen-lockfile`, build `npm run build`, output `dist`.
`vercel.mjs` builds API/SPA rewrites from public environment variables and sets
privacy/security headers. Direct unrelated-domain API mode is supported through
SameSite=None/Secure, but subject to third-party cookie blocking. Prefer proxy.
Actual provider config evaluation, cookies, streaming and public URLs await deployment testing.

## 21. Remaining Limitations And Git Inventory

No validated classifier or manipulation localization, no durable job queue,
no connected cloud storage, no automated email recovery/account deletion,
no physical-webcam production test and no complete public-load/proxy test.
PDF is metrics/text rather than a fully visual report; WEBP remains unsupported.
The CI workflow is authored but has not run on GitHub. Hosting templates were
locally inspected/unit-tested, not validated by an authenticated provider deploy.
The Git scanner covers current candidates, not all historical secrets.

Tracked changes already present on entry were `.gitignore`, `README.md`,
`build.sh`, `requirements.txt`, `src/image_analyzer.py` and the deleted Django
result template. Backend/frontend/Streamlit/shared-engine files were already
untracked from earlier work. This request adds production configuration, tests,
maintenance, CI/safety tools, documentation and screenshots, and updates files
listed above. **No file was deleted by this request.** No unrelated changes were
reverted or staged. Known secrets/data/runtime paths remain ignored.

## 22. What To Do Next

Review the current app and changes first. Complete the missing public-launch
controls and read `deployment.md`. Choose your database/storage budget, then
authorize a normal Git commit/push and deploy Render followed by Vercel. Test
the actual URLs and full workflow before sharing publicly. Independently build
a provenance-verified evaluation set before claiming better AI detection.
