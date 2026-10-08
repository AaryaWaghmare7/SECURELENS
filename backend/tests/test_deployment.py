from datetime import datetime, timedelta, timezone
from pathlib import Path
import secrets

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

from app.config import Settings
from app.models import User, Analysis
from app.services.maintenance import cleanup_once
from app.database.session import Base
from test_workspace import analyze


def settings(**overrides):
    return Settings(_env_file=None, database_url="sqlite://", jwt_secret=secrets.token_urlsafe(48), **overrides)


def test_managed_postgres_urls_select_the_installed_driver():
    for scheme in ("postgres", "postgresql"):
        value = Settings(_env_file=None, database_url=f"{scheme}://user:example@database.invalid/db?sslmode=require",
                         jwt_secret=secrets.token_urlsafe(48))
        assert value.database_url.startswith("postgresql+psycopg://")
        assert value.database_url.endswith("?sslmode=require")


@pytest.mark.parametrize("origins", [["*"], ["https://*.vercel.app"], ["https://web.example/path"],
                                     ["https://user:password@web.example"], ["https://web.example/"]])
def test_cors_rejects_wildcards_and_non_origins(origins):
    with pytest.raises(ValidationError):
        settings(frontend_origins=origins)


def test_cross_site_cookie_requires_https():
    with pytest.raises(ValidationError, match="SameSite=None"):
        settings(cookie_samesite="none")


def test_production_configuration_fails_closed(tmp_path):
    good = dict(database_url="postgresql://user:example@database.invalid/db", jwt_secret=secrets.token_urlsafe(48),
                environment="production", cookie_secure=True, frontend_origins=["https://web.example"],
                trusted_hosts=["api.example", "127.0.0.1"], storage_root=tmp_path)
    assert Settings(_env_file=None, **good).database_url.startswith("postgresql+psycopg:")
    for invalid in ({"cookie_secure": False}, {"database_url": "sqlite://"},
                    {"frontend_origins": ["http://web.example"]}, {"frontend_origins": ["https://localhost"]},
                    {"trusted_hosts": ["*"]}, {"storage_root": ".local/storage"}):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **{**good, **invalid})


def test_validation_errors_do_not_echo_password_input(client):
    response = client.post("/api/auth/register", json={"name": "Test", "email": "a@example.com", "password": "private"})
    assert response.status_code == 422
    assert "private" not in response.text
    assert all("input" not in item for item in response.json()["detail"])


def test_trusted_host_and_api_security_headers(client):
    assert client.get("/api/health", headers={"host": "evil.example"}).status_code == 400
    response = client.get("/api/health")
    assert response.json()["status"] == "ok"
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_database_health_failure_is_unavailable_not_false_success(client, monkeypatch):
    def unavailable():
        raise OperationalError("SELECT 1", {}, Exception("private database details"))
    monkeypatch.setattr(client.app.state.engine, "connect", unavailable)
    response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert response.headers["Retry-After"] == "5"
    assert "private" not in response.text


def test_migrations_build_all_required_tables_and_are_idempotent(client, monkeypatch):
    from alembic import command
    from alembic.config import Config
    engine = client.app.state.engine
    # This fixture owns a disposable database, never the application database.
    assert engine.url.database.endswith("test.db") or engine.url.database.startswith("securelens_test_")
    Base.metadata.drop_all(engine)
    monkeypatch.setenv("DATABASE_URL", engine.url.render_as_string(hide_password=False))
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    command.upgrade(config, "head")
    command.upgrade(config, "head")
    inspector = inspect(engine)
    assert {"users", "analyses", "user_preferences", "cookie_consents", "alembic_version"} <= set(inspector.get_table_names())
    assert "updated_at" in {column["name"] for column in inspector.get_columns("users")}
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == "0002_user_updated_at"


def test_cookie_attributes_match_cross_site_logout(client):
    client.app.state.settings.cookie_secure = True
    client.app.state.settings.cookie_samesite = "none"
    response = client.post("/api/auth/register", json={"name": "Other", "email": "secure@example.com", "password": "a-safe-test-password"})
    cookie = response.headers["set-cookie"]
    assert "SameSite=none" in cookie and "Secure" in cookie and "HttpOnly" in cookie
    # The HTTP TestClient correctly does not send Secure cookies; supply the fixture cookie explicitly.
    response = client.post("/api/auth/logout", headers={"X-CSRF-Token": response.json()["csrf_token"],
                           "Cookie": cookie.split(";", 1)[0]})
    assert response.status_code == 200
    assert "SameSite=none" in response.headers["set-cookie"]
    assert "Secure" in response.headers["set-cookie"]


def test_account_update_timestamp(client, signed_in):
    with client.app.state.session_factory() as db:
        user = db.scalar(select(User))
        identifier = user.id
        before = user.updated_at
    assert client.patch("/api/settings", json={"name": "New name"}, headers=signed_in).status_code == 200
    with client.app.state.session_factory() as db:
        assert db.get(User, identifier).updated_at >= before


def test_background_cleanup_preserves_reports_and_metrics(client, signed_in):
    record = analyze(client, signed_in, retain_images="true").json()
    with client.app.state.session_factory() as db:
        row = db.get(Analysis, record["id"])
        keys = list(row.image_keys)
        report_key = row.report_key
        row.images_expire_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
    cleanup_once(client.app)
    for key in keys:
        with pytest.raises(FileNotFoundError):
            client.app.state.storage.read(key)
    assert client.app.state.storage.read(report_key)
    with client.app.state.session_factory() as db:
        row = db.get(Analysis, record["id"])
        assert row.result["items"][0]["metrics"]["width"] == 64
        assert row.result["items"][0]["visualizations"] == {}


def test_json_report_survives_ephemeral_storage_loss(client, signed_in):
    record = analyze(client, signed_in, retain_images="true").json()
    with client.app.state.session_factory() as db:
        row = db.get(Analysis, record["id"])
        keys = list(row.image_keys) + [row.report_key]
    for key in keys:
        client.app.state.storage.delete(key)
    report = client.get(f'/api/reports/{record["id"]}')
    assert report.status_code == 200
    assert report.json()["id"] == record["id"]
    assert report.json()["result"]["items"][0]["metrics"]["width"] == 64
    assert "visualizations" not in report.json()["result"]["items"][0]
    assert client.get(f'/api/reports/{record["id"]}?format=pdf').content.startswith(b"%PDF")
    assert client.get(f'/api/reports/{record["id"]}?format=csv').status_code == 200
    assert client.get(f'/api/history/{record["id"]}/images/0/original').status_code == 404
