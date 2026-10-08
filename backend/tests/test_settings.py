import json
import secrets

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError
from sqlalchemy import text

from app.config import Settings
from app.main import create_app


@pytest.fixture
def production_env(monkeypatch, tmp_path):
    values = {
        "ENVIRONMENT": "production",
        "DATABASE_URL": "postgresql://test:example@database.invalid/securelens",
        "JWT_SECRET": secrets.token_urlsafe(48),
        "COOKIE_SECURE": "true",
        "COOKIE_SAMESITE": "none",
        "STORAGE_ROOT": str(tmp_path / "storage"),
        "FRONTEND_ORIGINS": "[]",
        "TRUSTED_HOSTS": '["securelens-api.onrender.com"]',
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


@pytest.mark.parametrize("origins", ["[]", "", "https://securelens.example.com",
                                     '["https://securelens.example.com"]'])
def test_production_reads_render_environment(production_env, monkeypatch, origins):
    monkeypatch.setenv("FRONTEND_ORIGINS", origins)
    config = Settings(_env_file=None)
    assert config.frontend_origins == ([] if origins in ("[]", "") else ["https://securelens.example.com"])
    assert config.trusted_hosts == ["securelens-api.onrender.com"]
    assert config.database_url.startswith("postgresql+psycopg://")
    app = create_app(config)
    assert any(route.path == "/api/health" for route in app.routes)
    app.state.engine.dispose()


def test_production_defaults_to_no_browser_origins(production_env, monkeypatch):
    monkeypatch.delenv("FRONTEND_ORIGINS")
    assert Settings(_env_file=None).frontend_origins == []


@pytest.mark.parametrize("origin", ["http://localhost:5173", "https://localhost", "https://127.0.0.1",
                                    "https://[::1]", "https://192.168.1.2", "https://10.0.0.1",
                                    "https://studio.localhost", "https://studio.local", "https://api.internal",
                                    "https://singlelabel", "http://securelens.example.com"])
def test_production_rejects_nonpublic_or_insecure_origins(production_env, monkeypatch, origin):
    monkeypatch.setenv("FRONTEND_ORIGINS", json.dumps([origin]))
    with pytest.raises(ValidationError, match="Production frontend origins"):
        Settings(_env_file=None)


def test_development_keeps_local_origins(production_env, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("COOKIE_SAMESITE", "lax")
    monkeypatch.setenv("FRONTEND_ORIGINS", "http://localhost:5173, http://127.0.0.1:5173")
    assert Settings(_env_file=None).frontend_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]
    monkeypatch.delenv("FRONTEND_ORIGINS")
    assert Settings(_env_file=None).frontend_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]


@pytest.mark.parametrize("name,value,expected", [
    ("FRONTEND_ORIGINS", '["https://a.example", "https://b.example"]', ["https://a.example", "https://b.example"]),
    ("FRONTEND_ORIGINS", "https://a.example, https://b.example", ["https://a.example", "https://b.example"]),
    ("TRUSTED_HOSTS", '["securelens-api.onrender.com", "127.0.0.1"]', ["securelens-api.onrender.com", "127.0.0.1"]),
    ("TRUSTED_HOSTS", "securelens-api.onrender.com, 127.0.0.1", ["securelens-api.onrender.com", "127.0.0.1"]),
])
def test_list_formats(production_env, monkeypatch, name, value, expected):
    monkeypatch.setenv(name, value)
    assert getattr(Settings(_env_file=None), name.lower()) == expected


@pytest.mark.parametrize("name", ["FRONTEND_ORIGINS", "TRUSTED_HOSTS"])
@pytest.mark.parametrize("value", ['["broken"', "['broken']", "{}", "null", '"scalar"', "true", "42",
                                    '["ok", 42]', '[""]', "ok,,other", "ok,", "ok, \"other\""])
def test_list_parser_rejects_malformed_values(production_env, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match=name):
        Settings(_env_file=None)


@pytest.mark.parametrize("host", ["*", "*.onrender.com", "https://securelens-api.onrender.com",
                                  "securelens-api.onrender.com:443", "api.example/path", "bad host",
                                  "user@api.example", "-bad.example", "api..example", "testserver"])
def test_trusted_host_validation(production_env, monkeypatch, host):
    monkeypatch.setenv("TRUSTED_HOSTS", json.dumps([host]))
    with pytest.raises(ValidationError, match="TRUSTED_HOSTS"):
        Settings(_env_file=None)


@pytest.mark.parametrize("value", ["[]", ""])
def test_empty_trusted_hosts_are_rejected(production_env, monkeypatch, value):
    monkeypatch.setenv("TRUSTED_HOSTS", value)
    with pytest.raises(ValidationError, match="TRUSTED_HOSTS"):
        Settings(_env_file=None)


@pytest.mark.parametrize("origin", ["*", "https://*.example.com", "https://api.example/", "https://user:password@api.example",
                                    "https://api.example:bad", "https://api.example:70000", "https://bad host.example",
                                    "https://api.example:", "https://api.example\\evil", "https://api..example"])
def test_invalid_origin_syntax(production_env, monkeypatch, origin):
    monkeypatch.setenv("FRONTEND_ORIGINS", json.dumps([origin]))
    with pytest.raises(ValidationError, match="FRONTEND_ORIGINS"):
        Settings(_env_file=None)


def test_same_site_none_never_allows_insecure_cookies(production_env, monkeypatch):
    monkeypatch.setenv("COOKIE_SECURE", "false")
    with pytest.raises(ValidationError, match="SameSite=None"):
        Settings(_env_file=None)


def test_dotenv_uses_the_same_safe_list_parser(production_env, monkeypatch, tmp_path):
    monkeypatch.delenv("FRONTEND_ORIGINS")
    monkeypatch.delenv("TRUSTED_HOSTS")
    env_file = tmp_path / "settings.env"
    env_file.write_text('FRONTEND_ORIGINS=https://one.example, https://two.example\n'
                        'TRUSTED_HOSTS=["securelens-api.onrender.com"]\n')
    config = Settings(_env_file=env_file)
    assert config.frontend_origins == ["https://one.example", "https://two.example"]
    assert config.trusted_hosts == ["securelens-api.onrender.com"]


@pytest.mark.parametrize("origins", [[], ["https://securelens.example.com"]])
def test_production_health_and_cors_with_database(client, production_env, monkeypatch, origins):
    monkeypatch.setenv("DATABASE_URL", client.app.state.settings.database_url)
    # The default suite isolates HTTP/middleware tests in SQLite. The PostgreSQL
    # suite uses a fresh PostgreSQL database and validates full production settings.
    if client.app.state.settings.database_url.startswith("sqlite"):
        config = Settings(_env_file=None, environment="test", cookie_secure=True,
                          cookie_samesite="none", frontend_origins=origins)
    else:
        monkeypatch.setenv("FRONTEND_ORIGINS", json.dumps(origins))
        config = Settings(_env_file=None)
    app = create_app(config)
    with TestClient(app, base_url="https://securelens-api.onrender.com") as browser:
        health = browser.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        with app.state.engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar() == 1
        assert browser.get("/api/health", headers={"Host": "evil.example"}).status_code == 400
        for origin in ("https://securelens.example.com", "https://evil.example"):
            headers = {"Origin": origin, "Access-Control-Request-Method": "POST",
                       "Access-Control-Request-Headers": "Content-Type,X-CSRF-Token"}
            preflight = browser.options("/api/auth/login", headers=headers)
            if origin in origins:
                assert preflight.status_code == 200
                assert preflight.headers["access-control-allow-origin"] == origin
                assert preflight.headers["access-control-allow-credentials"] == "true"
                result = browser.post("/api/auth/register", headers={"Origin": origin},
                                      json={"name": "Production", "email": "prod@example.com", "password": "production-test-password"})
                assert result.status_code == 201
                cookie = result.headers["set-cookie"]
                assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=none" in cookie
                assert browser.get("/api/auth/me", headers={"Origin": origin}).status_code == 200
                assert browser.post("/api/auth/logout", headers={"Origin": origin}).status_code == 403
                logout = browser.post("/api/auth/logout", headers={"Origin": origin, "X-CSRF-Token": result.json()["csrf_token"]})
                assert logout.status_code == 200
                assert "SameSite=none" in logout.headers["set-cookie"] and "Secure" in logout.headers["set-cookie"]
            else:
                assert preflight.status_code == 400
                assert "access-control-allow-origin" not in preflight.headers
                assert browser.post("/api/auth/login", headers={"Origin": origin}, json={}).status_code == 403
