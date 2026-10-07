from datetime import datetime, timedelta, timezone
import secrets
import hmac

from fastapi import Depends, HTTPException, Request
import jwt
from pwdlib import PasswordHash
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..models import User

password_hasher = PasswordHash.recommended()
# Avoid revealing account existence through a fast missing-user password path.
DUMMY_HASH = password_hasher.hash(secrets.token_urlsafe(32))


def issue_session(response, user, settings):
    csrf = secrets.token_urlsafe(32)
    issued = datetime.now(timezone.utc)
    token = jwt.encode({"sub": user.id, "ver": user.auth_version, "csrf": csrf,
                        "iat": issued, "exp": issued + timedelta(minutes=settings.session_minutes)},
                       settings.jwt_secret, algorithm="HS256")
    response.set_cookie("securelens_session", token, httponly=True, secure=settings.cookie_secure,
                        samesite=settings.cookie_samesite, max_age=settings.session_minutes * 60, path="/")
    return csrf


def current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("securelens_session")
    try:
        claims = jwt.decode(token or "", request.app.state.settings.jwt_secret,
                            algorithms=["HS256"], options={"require": ["sub", "exp", "iat", "csrf", "ver"]})
        user = db.get(User, claims["sub"])
        if user is None or user.auth_version != claims["ver"]:
            raise ValueError("Session expired")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            supplied = request.headers.get("X-CSRF-Token", "")
            if not hmac.compare_digest(supplied, claims["csrf"]):
                raise HTTPException(403, "The security token is missing. Sign in again and retry.")
        request.state.csrf = claims["csrf"]
        return user
    except HTTPException:
        raise
    except (jwt.InvalidTokenError, ValueError, KeyError, TypeError):
        raise HTTPException(401, "Sign in to access your SecureLens workspace.")
