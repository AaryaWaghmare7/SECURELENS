from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..auth.security import current_user, issue_session, password_hasher, DUMMY_HASH
from ..database.session import get_db
from ..models import User, UserPreference, CookieConsent
from ..schemas.requests import Registration, Login, UserPublic, Preferences, Consent

router = APIRouter(prefix="/api", tags=["accounts"])


@router.post("/auth/register", status_code=201)
def register(data: Registration, request: Request, response: Response, db: Session = Depends(get_db)):
    user = User(name=data.name, email=str(data.email).lower(), password_hash=password_hasher.hash(data.password))
    try:
        db.add(user)
        db.flush()
        db.add(UserPreference(user_id=user.id))
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "An account with that email already exists.")
    return {"user": UserPublic.model_validate(user), "csrf_token": issue_session(response, user, request.app.state.settings)}


@router.post("/auth/login")
def login(data: Login, request: Request, response: Response, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == str(data.email).lower()))
    valid = password_hasher.verify(data.password, user.password_hash if user else DUMMY_HASH)
    if not user or not valid:
        raise HTTPException(401, "Email or password is incorrect.")
    return {"user": UserPublic.model_validate(user), "csrf_token": issue_session(response, user, request.app.state.settings)}


@router.get("/auth/me")
def me(request: Request, user: User = Depends(current_user)):
    return {"user": UserPublic.model_validate(user), "csrf_token": request.state.csrf}


@router.post("/auth/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db), user: User = Depends(current_user)):
    user.auth_version += 1
    db.commit()
    settings = request.app.state.settings
    response.delete_cookie("securelens_session", path="/", httponly=True,
                           secure=settings.cookie_secure, samesite=settings.cookie_samesite)
    return {"message": "Signed out. All previous sessions have been invalidated."}


@router.get("/settings")
def settings(db: Session = Depends(get_db), user: User = Depends(current_user)):
    prefs = db.get(UserPreference, user.id)
    return {"name": user.name, "email": user.email, "retain_images": prefs.retain_images if prefs else False}


@router.patch("/settings")
def update_settings(data: Preferences, db: Session = Depends(get_db), user: User = Depends(current_user)):
    if not data.name.strip():
        raise HTTPException(422, "Display name cannot be empty.")
    user.name = data.name.strip()
    prefs = db.get(UserPreference, user.id)
    if not prefs:
        prefs = UserPreference(user_id=user.id)
        db.add(prefs)
    prefs.retain_images = data.retain_images
    db.commit()
    return {"name": user.name, "email": user.email, "retain_images": prefs.retain_images}


@router.get("/cookie-consent")
def get_consent(db: Session = Depends(get_db), user: User = Depends(current_user)):
    row = db.get(CookieConsent, user.id)
    return {"essential": True, "authentication": True, "analytics": False,
            "recorded": row is not None}


@router.put("/cookie-consent")
def consent(data: Consent, db: Session = Depends(get_db), user: User = Depends(current_user)):
    row = db.get(CookieConsent, user.id)
    if not row:
        row = CookieConsent(user_id=user.id)
        db.add(row)
    row.essential, row.authentication, row.analytics = True, True, False
    from ..models.records import now
    row.updated_at = now()
    db.commit()
    return {"essential": True, "authentication": True, "analytics": False}
