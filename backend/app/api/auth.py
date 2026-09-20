"""Signup, signin, logout and "who am I".

The token is set as an httpOnly cookie and *also* returned in the body. The
cookie is what the browser uses — JavaScript cannot read it, so an injected
script cannot steal the session. The body value is for API clients and tests,
which have no cookie jar. A browser client should ignore it and never write it
to localStorage, which would hand back exactly the exposure the cookie avoids.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.schemas import SigninRequest, SignupRequest, UserOut
from app.auth.security import create_access_token, hash_password, verify_password
from app.config import get_settings
from app.database.connection import get_db
from app.database.models import User
from app.database.repositories import users as user_repo

logger = logging.getLogger("csam.auth")

settings = get_settings()

router = APIRouter(prefix="/auth", tags=["auth"])

# One message for "no such email" and "wrong password". Two messages is an
# account-enumeration oracle: an attacker learns which addresses are registered
# without ever guessing a password.
INVALID_CREDENTIALS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Incorrect email or password",
)


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
        max_age=settings.jwt_expire_minutes * 60,
        path="/",
    )


class AuthResponse(UserOut):
    access_token: str
    token_type: str = "bearer"


@router.post("/signup", response_model=AuthResponse, status_code=201)
def signup(
    payload: SignupRequest, response: Response, db: Session = Depends(get_db)
) -> AuthResponse:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name cannot be empty")

    if user_repo.get_by_email(db, payload.email) is not None:
        raise HTTPException(status_code=409, detail="Email already registered")

    try:
        user = user_repo.create_user(
            db, name=name, email=payload.email, password_hash=hash_password(payload.password)
        )
    except IntegrityError:
        # Two signups for the same address can both pass the check above and
        # race to the insert. The unique index is what actually decides, and
        # the loser is told the same thing as a plain duplicate.
        db.rollback()
        raise HTTPException(status_code=409, detail="Email already registered") from None

    token = create_access_token(user.id)
    _set_auth_cookie(response, token)
    # The password is never logged, here or anywhere.
    logger.info("signup user_id=%s", user.id)
    return AuthResponse(id=user.id, name=user.name, email=user.email, access_token=token)


@router.post("/signin", response_model=AuthResponse)
def signin(
    payload: SigninRequest, response: Response, db: Session = Depends(get_db)
) -> AuthResponse:
    user = user_repo.get_by_email(db, payload.email)
    if user is None or not verify_password(payload.password, user.password_hash):
        logger.info("failed signin attempt")
        raise INVALID_CREDENTIALS

    token = create_access_token(user.id)
    _set_auth_cookie(response, token)
    logger.info("signin user_id=%s", user.id)
    return AuthResponse(id=user.id, name=user.name, email=user.email, access_token=token)


@router.post("/logout", status_code=204)
def logout(response: Response) -> None:
    """Clears the cookie.

    The JWT itself stays valid until it expires — that is the price of
    stateless tokens. Revoking on logout needs server-side state (a denylist
    or short-lived tokens with refresh), which is the right next step if this
    ever holds anything more sensitive than support chat.
    """
    response.delete_cookie(
        key=settings.auth_cookie_name,
        path="/",
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
    )


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(current_user)
