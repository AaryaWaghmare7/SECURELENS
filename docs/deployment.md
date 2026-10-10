# SecureLens Deployment Guide

## Current Production

- Frontend: https://securelens-ten.vercel.app
- API: https://securelens-api-fo41.onrender.com/api/health
- Browser requests use relative `/api/...`; Vercel rewrites them to Render.
- Vercel Production: `VITE_API_PROXY=true` and
  `VITE_API_BASE_URL=https://securelens-api-fo41.onrender.com`.
- Render: `ENVIRONMENT=production`, `COOKIE_SECURE=true`,
  `COOKIE_SAMESITE=none`, `STORAGE_ROOT=/tmp/securelens`,
  `FRONTEND_ORIGINS=["https://securelens-ten.vercel.app"]`,
  `TRUSTED_HOSTS=["securelens-api-fo41.onrender.com"]`.
- Preserve the existing private `DATABASE_URL` and `JWT_SECRET` unchanged.
- The current service uses Root Directory `backend`, build
  `pip install -r requirements.txt`, and start
  `python -m alembic upgrade head && python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
  The repository-level `src/` engine must remain available. Do not replace this
  service with the paid Blueprint below without reviewing its cost and settings.

### Reliability And Free Hosting

The frontend shares one readiness check across concurrent requests. It polls
unauthenticated `/api/health` for up to 180 seconds, with 65-second attempts and
2/3/5/8/10-second backoff, then continues automatically. Successful readiness
is cached for 10 seconds; there is no background keep-alive. A real loading
notice appears during a slow wake. Only network/timeouts and 502/503/504 health
responses and temporary HTML hosting pages are retried. A healthy JSON response
is still required before any account or analysis request is submitted. Other
HTTP failures retain their own error categories.

No submitted mutation (signup, login, upload, batch, compare, settings or logout)
is replayed automatically. An interrupted POST may already have succeeded;
check the account/history before manually submitting again. A failed safe GET
can retry once after a fresh readiness check. Temporary session-check errors
do not silently log the user out.

Health executes `SELECT 1`, returns 503 if the database is unavailable, and uses
bounded PostgreSQL connection/pool waits. Alembic upgrades run before Uvicorn
starts; no database recreation is required.

Render Free spins down after inactivity and can take about a minute to start.
Software can handle the wait, not guarantee an instant cold start. An always-on
paid Render instance removes idle spin-down. Vercel external rewrites have a
120-second proxy timeout; very large batches still need a durable job queue.
See [Render Free](https://render.com/docs/free) and
[Vercel limits](https://vercel.com/docs/limits).

Account, preferences, consent, analysis metrics and report contents live in
PostgreSQL. JSON/CSV/PDF downloads are generated from the owned database record,
so loss of temporary files does not break report downloads. Optional image
previews remain ephemeral: `/tmp` can disappear on sleep/redeploy, earlier than
the configured retention limit. Durable previews require private cloud storage
or an approved paid disk. Do not promise permanent storage on the free database.

The sections below describe provisioning alternatives and hardening, not
additional services that have already been provisioned.

## Recommended Hosting Layout

```text
Browser -> Vercel React frontend
           /api reverse proxy -> Render FastAPI -> managed PostgreSQL
                                            -> private persistent storage
                                            -> preserved src/ analysis engine
```

The Vercel API proxy keeps the browser's session cookie first-party. Merely
allowing CORS does not make a SameSite=Lax cookie work between unrelated
`vercel.app` and `onrender.com` sites. SameSite=None requires Secure and can
still be blocked by browsers. Prefer the proxy or same-site custom domains.
See [Vercel rewrites](https://vercel.com/docs/routing/rewrites) and
[MDN third-party cookies](https://developer.mozilla.org/en-US/docs/Web/Privacy/Guides/Third-party_cookies).

The frontend always reads `VITE_API_BASE_URL`; no Render hostname is embedded
in a component. With `VITE_API_PROXY=true`, browser calls use relative `/api`
paths and `frontend/vercel.mjs` builds an external rewrite to that origin.
The SPA fallback excludes `/api` and `/assets`, so an API failure is not
silently turned into an HTML page.

## 1. Review Git Before Publishing

From the root:

```bash
git status --short
git diff --stat
git ls-files --others --exclude-standard
python3 tools/check_git_safety.py
```

Review the full changes before staging or committing. `.env`, local database
files, uploads, datasets, model weights, dependencies and generated builds must
remain ignored. The read-only scanner checks common patterns and large files,
not every possible secret or historical commit. Deleting a file in a later
commit does not remove it from earlier commits. Review old history before
public sharing and rotate any exposed secret; do not force-push or rewrite
history as part of this upgrade.

The missing legacy `dashboard/templates/dashboard/result.html` was already
deleted before this request. It was not deleted by this upgrade. Review that
separately if you still need the legacy Django result page.

When you explicitly authorize publishing, stage only the reviewed files, use a
normal commit and a normal push. Publishing source code is not a public deployment.

## 2. Production PostgreSQL

Choose Render PostgreSQL in the same region as the API, or Supabase PostgreSQL.
Create a fresh production database; the local database is not silently uploaded.
Use backups and a plan appropriate for retained account/report data.

On Render, use the internal connection string for a Render database. The API
normalizes `postgres://` or `postgresql://` to `postgresql+psycopg://` while
preserving connection options. Never put this URL into frontend variables.

For Supabase, use the dashboard-provided direct or **session-mode** pooler URL
appropriate for your network and add `sslmode=require`. Avoid transaction-mode
pooling unless psycopg prepared-statement behavior is configured and tested.
Percent-encode special characters in credentials; do not print the URL in logs.
The supplied Blueprint assumes Render PostgreSQL; configure an external
database manually if choosing Supabase.
See [Supabase connection guidance](https://supabase.com/docs/guides/database/connecting-to-postgres)
for direct/session pooler options and SSL settings.

## 3. Render FastAPI Settings

Use a new Web Service, not the old Django service configuration.

| Setting | Exact value |
| --- | --- |
| Runtime | Python 3 |
| Root Directory | Leave empty (repository root; shared `src/` is required) |
| Build Command | `pip install -r backend/requirements.txt` |
| Pre-Deploy Command | `cd backend && python -m alembic upgrade head` |
| Start Command | `cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --no-access-log` |
| Health Check | `/api/health` |
| Python | `PYTHON_VERSION=3.14.3` |
| Storage disk mount | `/var/data/securelens` |

For a new service, the repository-root layout above makes the shared `src/`
dependency explicit. Do not change the current working service's root just to
match this alternative. The analysis adapter imports repository-level `src/`.
See [Render monorepo roots](https://render.com/docs/monorepo-support).

Alternatively, choose `render.fastapi.yaml` as the Blueprint file. It describes
a Starter API, a Basic PostgreSQL database and a 1 GB disk in Frankfurt.
**These are paid resources; do not create them until you approve the cost.**
This Blueprint has not been provisioned. Automatic Git-triggered deploys are off
in this template. The old `render.yaml` and `build.sh` remain the Django setup.

Render's [FastAPI guide](https://render.com/docs/deploy-fastapi) documents the
Uvicorn binding. The [Blueprint reference](https://render.com/docs/blueprint-spec)
documents migration and health-check settings. Use the paid pre-deploy command
for migrations; the database must be reachable from the deploy environment.
Migration imports do not initialize the runtime storage disk.

### Required Render Environment

```dotenv
ENVIRONMENT=production
PYTHON_VERSION=3.14.3
DATABASE_URL=<private managed PostgreSQL connection URL>
JWT_SECRET=<random secret, at least 32 characters>
FRONTEND_ORIGINS=[]
TRUSTED_HOSTS=["securelens-api.onrender.com"]
COOKIE_SECURE=true
COOKIE_SAMESITE=lax
STORAGE_ROOT=/var/data/securelens
IMAGE_RETENTION_DAYS=7
STORAGE_CLEANUP_INTERVAL_SECONDS=3600
SESSION_MINUTES=120
AUTH_RATE_LIMIT=20
```

Generate the JWT secret using the provider's secret generator or a password
manager for a new service. For an existing service, preserve `DATABASE_URL` and
`JWT_SECRET` unchanged. Do not paste real values into this document. `LOCAL_POSTGRES_PASSWORD`
is only for the local helper and is not needed on Render. `$PORT` is provided
by Render. Do not enable development reload in production.

Use your service's actual public hostname in `TRUSTED_HOSTS`; the value above
applies only if Render assigned `securelens-api.onrender.com`. If you configure a
custom API domain, also allowlist it for Render's health checks. Host checking
stays enabled. [Render health checks](https://render.com/docs/health-checks)
use a verified custom domain or the service's `onrender.com` hostname.

Deploy Render first with `FRONTEND_ORIGINS=[]`. An explicitly empty/blank list
allows the API and unauthenticated `/api/health` to start, but does **not** allow
browser cross-origin access. Omitting this variable in production also defaults
to an empty list, not development origins. Explicitly configured local origins
still fail production validation. After Vercel deployment, change only
`FRONTEND_ORIGINS` to `["https://<actual-vercel-domain>"]` and redeploy the API
(with the proxy/cookie mode already chosen). No placeholder origin is needed.

Both list variables accept JSON arrays (recommended) or comma-separated strings,
for example `TRUSTED_HOSTS=securelens-api.onrender.com,127.0.0.1`. Malformed JSON,
non-string/empty entries, schemes or ports in hosts, and wildcards are rejected
with field-specific errors. An empty `TRUSTED_HOSTS` is never accepted.
This uses Pydantic Settings' field-scoped
[NoDecode/custom validation](https://pydantic.dev/docs/validation/dev/concepts/pydantic_settings/#disabling-json-parsing),
not a global parsing or security bypass.

Set exact origins, without trailing slashes, paths or wildcards. Add a specific
preview origin only when needed; never blanket-allow `*.vercel.app`. Hostnames
in `TRUSTED_HOSTS` have no scheme/path. Add the frontend hostname if your chosen
proxy preserves it as the request Host, then test the actual proxy behavior.
Production startup rejects insecure cookies, SQLite, wildcard origins and a
relative storage root.

Python remains at the already tested 3.14 series (`PYTHON_VERSION=3.14.3` in the
Blueprint). The reported build succeeds and the backend regression suite runs
on Python 3.14; this startup failure is configuration validation, not a Python
compatibility error. No downgrade is needed for this fix. Render supports both
`PYTHON_VERSION` and `.python-version`; see its
[Python version guide](https://render.com/docs/python-version).

For the existing direct Vercel-to-Render setup, use `COOKIE_SAMESITE=none` and
`COOKIE_SECURE=true` from the start. `none` without Secure is rejected. Session
cookies remain HttpOnly, host-only and CSRF-protected. Browsers may still block
third-party cookies; the recommended Vercel proxy uses `COOKIE_SAMESITE=lax`
instead. These are two supported modes, not interchangeable CORS fixes.

The current free service uses writable `STORAGE_ROOT=/tmp/securelens`.
It is **ephemeral**: retained previews and redundant local report copies can
disappear on redeploy/restart/sleep. Report downloads are rebuilt from PostgreSQL
and remain available while their owned record exists. Use private cloud storage
or an approved paid disk before promising durable image previews.

### Private Storage And Retention

Local development uses `backend/.local/storage`. Production uses the attached
private disk, never your laptop's path. Reports and optional downscaled previews
must survive a restart/redeploy. The current `Storage` protocol exposes
`write/read/delete`; implement a private S3, Supabase Storage or Cloudinary
adapter later and inject it through the API and maintenance command. No cloud
adapter is falsely claimed as connected.

Render disks are available only to the owning runtime, not pre-deploy jobs or
cron services. They require paid services and prevent multi-instance scaling
and zero-downtime deployment. See [persistent disks](https://render.com/docs/disks).
Do not run a separate cron service against a disk it cannot access.

The API now removes expired previews at startup and periodically (hourly by
default), plus on owner reads. Cleanup pauses while the service is stopped.
Metrics and reports remain until deleted. Manual maintenance, in the **owning
service's** shell:

```bash
cd backend
python scripts/cleanup_storage.py
```

Verify retention settings and disk/database backups before accepting sensitive
uploads. Render's free ephemeral filesystem is not sufficient for durable
previews. Do not promise permanent retention on a free database.

## 4. Test The Render API

Open `https://your-api.onrender.com/api/health` and confirm `status: ok`.
The endpoint also accurately reports `classifier_validated: false`.
It requires no authentication or frontend origin and executes `SELECT 1` against
the configured database. An unreachable database correctly fails the health
check; this is not hidden by returning a static success response.
Run migrations before registration. Test login and private history through the
frontend after its domain is allowlisted. Health alone does not verify uploads,
cookie delivery, ownership or report persistence.

For an existing production database, apply the committed Alembic migrations,
never delete/recreate it. With the repository root setting above, the pre-deploy
command is `cd backend && python -m alembic upgrade head`. If your shell is
already inside `backend`, run `python -m alembic upgrade head`. The command is
idempotent when the database is already at head. Changing only these settings
does not require a new schema migration.

## 5. Vercel Frontend Settings

| Setting | Exact value |
| --- | --- |
| Root Directory | `frontend` |
| Framework Preset | Vite |
| Node.js | 24.x |
| Install | Automatic lockfile detection, or `pnpm install --frozen-lockfile` |
| Build | `npm run build` |
| Output | `dist` |
| Configuration | `frontend/vercel.mjs` |

Add these **public** production build variables before deploying:

```dotenv
VITE_API_BASE_URL=https://your-api.onrender.com
VITE_API_PROXY=true
```

Use the API **origin**, not `/api` or `/docs`. The dynamic configuration rejects
insecure/local/credential-bearing destinations and fails on Vercel if the URL
is missing. Do not put database passwords or signing secrets in any `VITE_`
variable. `VITE_PROXY_TARGET` is only for the local Vite server.

Vercel supports build-time [programmatic configuration](https://vercel.com/docs/project-configuration/vercel-ts),
including `vercel.mjs`. Its SPA routing guidance is in
[Vite on Vercel](https://vercel.com/docs/frameworks/frontend/vite).
Public URLs, actual cookie/proxy behavior and upload/streaming limits still need
production testing; local config tests do not replace it. Vercel documents a
120-second external proxy timeout in its [limits](https://vercel.com/docs/limits).
Large/slow batches may need durable queued jobs later. Do not deploy the local
Vite server or portable PostgreSQL to the public internet.

## Alternative: Direct API Or Custom Domain

For direct calls between `vercel.app` and `onrender.com` set
`VITE_API_PROXY=false`, keep `VITE_API_BASE_URL`, and use
`COOKIE_SAMESITE=none` with `COOKIE_SECURE=true`. CSRF and exact Origin/CORS
checks remain enabled. This mode depends on third-party cookie availability;
the proxy is recommended for dependable public sign-in.

Later you can use `www.securelens.example` on Vercel and
`api.securelens.example` on Render. Add exact HTTPS origins and hostnames, update
`VITE_API_BASE_URL`, redeploy the frontend and test DNS/TLS/session behavior.
No custom domain is required now. Cookies stay host-only, not broadly scoped to
the parent domain. Keep Lax for same-site HTTPS domains or continue using the proxy.

## Deployment Order

1. Review Git files, ignored data and old history, then explicitly authorize a normal commit/push.
2. Create a production PostgreSQL database and backups.
3. Create the Render API with private durable storage and production variables.
4. Run Alembic migrations and test the API health response.
5. Configure the Vercel frontend variables and build the React app.
6. Allowlist its exact production origin/required hostnames in Render.
7. Test the complete browser workflow, including proxy cookies and report downloads.
8. Optionally connect custom domains and repeat security/workflow tests.

## Public Launch Checklist

- Registration, duplicate account handling, incorrect passwords, login and logout.
- Session expiry, CSRF rejection, unknown Origin/Host rejection and owner-only history/reports/previews.
- Single JPG/PNG upload, corrupt/disguised/oversized files and unsupported WEBP rejection.
- Live camera permission on HTTPS, start/stop/navigation cleanup, no automatic image retention.
- Batch progress, partial/all failures, report CSV/JSON, and side-by-side comparison.
- JSON/PDF report downloads, history reopening/deletion and account preview preferences.
- Data survives redeploy; preview expiration works without user visits; backups are restorable.
- Mobile navigation, deep-link refresh, cookie preferences and privacy notice.
- Proxy time/body limits and concurrent analyses tested with representative public workloads.
- Trusted proxy/IP handling verified. The limiter is per API process and may see a proxy IP;
  use a shared limiter/queue for scale and never blindly trust caller-supplied forwarding headers.
- Add verified password recovery/email delivery, abuse monitoring and account deletion before
  broad public signup. Self-host fonts if third-party font requests are unacceptable.
- Validate a classifier on genuinely labeled held-out data before claiming detection accuracy.

Use the complete checklist for each release; successful health alone is not
proof that authentication, uploads and reports work.
