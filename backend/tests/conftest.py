import os
import secrets
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("JWT_SECRET", secrets.token_urlsafe(48))
os.environ.setdefault("DATABASE_URL", "sqlite://")

import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.config import Settings
from app.database.session import Base
from sqlalchemy.engine import make_url


@pytest.fixture
def client(tmp_path):
    database_url = f'sqlite:///{tmp_path / "test.db"}'
    admin = None
    test_database = None
    if os.environ.get("SECURELENS_TEST_POSTGRES") == "1":
        import psycopg
        from psycopg import sql
        from dotenv import dotenv_values
        url = make_url(dotenv_values(Path(__file__).resolve().parents[1] / ".env")["DATABASE_URL"])
        admin = psycopg.connect(url.set(drivername="postgresql", database="postgres").render_as_string(hide_password=False), autocommit=True)
        test_database = "securelens_test_" + uuid4().hex
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)))
        database_url = url.set(database=test_database).render_as_string(hide_password=False)
    settings = Settings(database_url=database_url, jwt_secret=secrets.token_urlsafe(48),
                        storage_root=tmp_path / "storage", frontend_origins=["http://localhost:5173"])
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)
    try:
        with TestClient(app) as instance:
            yield instance
    finally:
        app.state.engine.dispose()
        if admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(test_database)))
            admin.close()


@pytest.fixture
def signed_in(client):
    response = client.post("/api/auth/register", json={"name": "Researcher", "email": "researcher@example.com", "password": "a-test-password-123"})
    assert response.status_code == 201
    return {"X-CSRF-Token": response.json()["csrf_token"]}
