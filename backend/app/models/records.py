from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from ..database.session import Base


def now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    auth_version: Mapped[int] = mapped_column(default=0)


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    analysis_type: Mapped[str] = mapped_column(String(20))
    filename: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="complete")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    result: Mapped[dict] = mapped_column(JSON)
    report_key: Mapped[str] = mapped_column(String(200))
    image_keys: Mapped[list] = mapped_column(JSON, default=list)
    images_expire_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class UserPreference(Base):
    __tablename__ = "user_preferences"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    retain_images: Mapped[bool] = mapped_column(Boolean, default=False)


class CookieConsent(Base):
    __tablename__ = "cookie_consents"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    essential: Mapped[bool] = mapped_column(Boolean, default=True)
    authentication: Mapped[bool] = mapped_column(Boolean, default=True)
    analytics: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
