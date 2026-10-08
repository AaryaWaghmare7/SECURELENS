from ipaddress import ip_address
import json
from pathlib import Path
import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from sqlalchemy.engine import make_url

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def valid_hostname(host):
    try:
        ip_address(host)
        return True
    except ValueError:
        return len(host) <= 253 and all(
            re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label)
            for label in host.split(".")
        )


def public_hostname(host):
    try:
        address = ip_address(host)
        return address.is_global and not address.is_multicast
    except ValueError:
        return (valid_hostname(host) and "." in host and
                not host.lower().endswith((".localhost", ".local", ".internal", ".lan", ".home")))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_ROOT / ".env", extra="ignore", hide_input_in_errors=True)
    environment: Literal["development", "test", "production"] = "development"
    database_url: str
    jwt_secret: str = Field(min_length=32)
    frontend_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    cookie_secure: bool = False
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    trusted_hosts: Annotated[list[str], NoDecode] = ["localhost", "127.0.0.1", "testserver"]
    session_minutes: int = Field(default=120, ge=5, le=1440)
    storage_root: Path = Path(".local/storage")
    image_retention_days: int = Field(default=7, ge=1, le=365)
    max_upload_bytes: int = 16 * 1024 * 1024
    max_request_bytes: int = 64 * 1024 * 1024
    max_batch_images: int = 10
    auth_rate_limit: int = Field(default=20, ge=1, le=1000)
    storage_cleanup_interval_seconds: int = Field(default=3600, ge=60, le=86400)

    @model_validator(mode="before")
    @classmethod
    def backend_first_defaults(cls, values):
        # Development origins must never become implicit production permissions.
        if values.get("environment") == "production" and "frontend_origins" not in values:
            return {**values, "frontend_origins": []}
        return values

    @field_validator("frontend_origins", "trusted_hosts", mode="before")
    @classmethod
    def parse_list_setting(cls, value, info):
        name = info.field_name.upper()
        message = f"{name} must be a JSON array of non-empty strings or a comma-separated list."
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                # Broken JSON must not fall back to permissive comma splitting.
                if any(character in raw for character in "[]{}\"'"):
                    raise ValueError(message) from None
                value = raw.split(",")
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(message)
        return list(dict.fromkeys(item.strip() for item in value))

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
        for value in values:
            try:
                parsed = urlsplit(value)
                parsed.port
            except ValueError:
                raise ValueError("FRONTEND_ORIGINS contains an invalid hostname or port.") from None
            if (parsed.scheme not in ("http", "https") or not parsed.hostname or
                    parsed.username or parsed.password or parsed.path or parsed.query or
                    parsed.fragment or "*" in value or "\\" in value or parsed.netloc.endswith(":") or
                    any(character.isspace() for character in value) or not valid_hostname(parsed.hostname)):
                raise ValueError("FRONTEND_ORIGINS must contain exact HTTP(S) origins, without paths or wildcards.")
        return values

    @field_validator("trusted_hosts")
    @classmethod
    def exact_hosts(cls, values):
        if not values or any(not valid_hostname(host) for host in values):
            raise ValueError("TRUSTED_HOSTS requires explicit hostnames or IP addresses, without schemes, ports, paths or wildcards.")
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
            if any(urlsplit(origin).scheme != "https" or not public_hostname(urlsplit(origin).hostname)
                   for origin in self.frontend_origins):
                raise ValueError("Production frontend origins must use HTTPS and a public hostname.")
            if not self.trusted_hosts or any("*" in host or host == "testserver" for host in self.trusted_hosts):
                raise ValueError("Production requires explicit TRUSTED_HOSTS without wildcards.")
            if not self.storage_root.is_absolute():
                raise ValueError("Production STORAGE_ROOT must be an absolute path on persistent private storage.")
        return self

    @property
    def storage_path(self):
        return self.storage_root if self.storage_root.is_absolute() else BACKEND_ROOT / self.storage_root
