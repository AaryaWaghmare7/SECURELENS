from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

BACKEND_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_ROOT / ".env", extra="ignore", hide_input_in_errors=True)
    environment: Literal["development", "test", "production"] = "development"
    database_url: str
    jwt_secret: str = Field(min_length=32)
    frontend_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    cookie_secure: bool = False
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    trusted_hosts: list[str] = ["localhost", "127.0.0.1", "testserver"]
    session_minutes: int = Field(default=120, ge=5, le=1440)
    storage_root: Path = Path(".local/storage")
    image_retention_days: int = Field(default=7, ge=1, le=365)
    max_upload_bytes: int = 16 * 1024 * 1024
    max_request_bytes: int = 64 * 1024 * 1024
    max_batch_images: int = 10
    auth_rate_limit: int = Field(default=20, ge=1, le=1000)
    storage_cleanup_interval_seconds: int = Field(default=3600, ge=60, le=86400)

    @field_validator("database_url")
    @classmethod
    def use_psycopg(cls, value):
        if value.startswith("postgres://"):
            value = "postgresql://" + value[len("postgres://"):]
        url = make_url(value)
        if url.drivername == "postgresql":
            return url.set(drivername="postgresql+psycopg").render_as_string(hide_password=False)
        return value

    @field_validator("frontend_origins")
    @classmethod
    def exact_origins(cls, values):
        if not values:
            raise ValueError("At least one explicit frontend origin is required.")
        for value in values:
            parsed = urlsplit(value)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname or
                    parsed.username or parsed.password or parsed.path or parsed.query or
                    parsed.fragment or "*" in value):
                raise ValueError("Frontend origins must be exact HTTP(S) origins, without paths or wildcards.")
        return values

    @model_validator(mode="after")
    def safe_deployment(self):
        if self.cookie_samesite == "none" and not self.cookie_secure:
            raise ValueError("SameSite=None requires COOKIE_SECURE=true and HTTPS.")
        if self.environment == "production":
            if make_url(self.database_url).drivername != "postgresql+psycopg":
                raise ValueError("Production requires PostgreSQL with the psycopg driver.")
            if not self.cookie_secure:
                raise ValueError("Production requires COOKIE_SECURE=true.")
            if any(urlsplit(origin).scheme != "https" or urlsplit(origin).hostname in
                   ("localhost", "127.0.0.1", "::1") for origin in self.frontend_origins):
                raise ValueError("Production frontend origins must use HTTPS and a public hostname.")
            if not self.trusted_hosts or any("*" in host or host == "testserver" for host in self.trusted_hosts):
                raise ValueError("Production requires explicit TRUSTED_HOSTS without wildcards.")
            if not self.storage_root.is_absolute():
                raise ValueError("Production STORAGE_ROOT must be an absolute path on persistent private storage.")
        return self

    @property
    def storage_path(self):
        return self.storage_root if self.storage_root.is_absolute() else BACKEND_ROOT / self.storage_root
