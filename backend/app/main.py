import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from sqlalchemy import text

from .config import Settings
from .database.session import make_database
from .services.storage import LocalStorage
from .services.maintenance import retention_worker
from .utils.limits import AuthRateLimit, RequestSizeLimit
from .api import auth, analysis, history


def create_app(settings=None):
    settings = settings or Settings()
    engine, factory = make_database(settings.database_url)

    @asynccontextmanager
    async def lifespan(app):
        stopped = asyncio.Event()
        maintenance = asyncio.create_task(retention_worker(app, stopped))
        try:
            yield
        finally:
            stopped.set()
            await maintenance
            engine.dispose()

    app = FastAPI(title="SecureLens API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = factory
    app.state.storage = LocalStorage(settings.storage_path)
    app.state.analysis_slots = asyncio.Semaphore(2)
    auth_limiter = AuthRateLimit(settings.auth_rate_limit)
    app.add_middleware(RequestSizeLimit, settings=settings)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts, www_redirect=False)
    app.add_middleware(CORSMiddleware, allow_origins=settings.frontend_origins,
                       allow_credentials=True, allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
                       allow_headers=["Content-Type", "X-CSRF-Token"])

    @app.middleware("http")
    async def protect_requests(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin not in settings.frontend_origins:
            return JSONResponse({"detail": "Request origin is not allowed."}, status_code=403)
        try:
            if int(request.headers.get("content-length", "0")) > settings.max_request_bytes:
                return JSONResponse({"detail": "The request exceeds the 64 MB upload limit."}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid request size."}, status_code=400)
        if request.method == "POST" and request.url.path in ("/api/auth/login", "/api/auth/register"):
            address = request.client.host if request.client else "unknown"
            if not auth_limiter.allows(address):
                return JSONResponse({"detail": "Too many sign-in attempts. Please wait a minute."},
                                    status_code=429, headers={"Retry-After": "60"})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if settings.environment == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def unexpected_error(request, error):
        logging.getLogger(__name__).error("SecureLens request failed (%s).", type(error).__name__)
        return JSONResponse({"detail": "The request could not finish. Check the backend logs or try again."}, status_code=500)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        # Keep useful field messages, but never echo a password/token in validation input.
        return JSONResponse({"detail": [{"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
                                       for item in error.errors()]}, status_code=422)

    @app.get("/api/health")
    def health():
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok", "engine": "forensic-heuristics", "classifier_validated": False,
                "image_retention_days": settings.image_retention_days}

    app.include_router(auth.router)
    app.include_router(analysis.router)
    app.include_router(history.router)
    return app


app = create_app()
