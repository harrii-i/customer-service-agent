"""The authentication dependency.

One place turns a request into an authenticated `User`. Endpoints declare
`current_user: User = Depends(get_current_user)` and are done — none of them
verify a token themselves, so none of them can get it subtly wrong or be
forgotten when the rules change.
"""

import logging

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.auth.security import InvalidToken, decode_access_token
from app.config import get_settings
from app.database.connection import get_db
from app.database.models import User
from app.database.repositories import users as user_repo

logger = logging.getLogger("csam.auth")

settings = get_settings()

UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def extract_token(request: Request) -> str | None:
    """The cookie first, then `Authorization: Bearer`.

    The cookie is how the browser authenticates: it is httpOnly, so page
    scripts cannot read it. The header is for API clients and tests, which
    have no cookie jar and no XSS surface to protect.
    """
    cookie = request.cookies.get(settings.auth_cookie_name)
    if cookie:
        return cookie

    header = request.headers.get("Authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    return None


def get_current_user(
    request: Request, db: Session = Depends(get_db)
) -> User:
    """The authenticated account, or 401.

    Missing, malformed, expired and wrong-signature tokens all produce the
    same response. Distinguishing them tells an attacker which half of the
    attempt to keep working on.
    """
    token = extract_token(request)
    if token is None:
        raise UNAUTHENTICATED

    try:
        user_id = decode_access_token(token)
    except InvalidToken as error:
        logger.info("rejected token: %s", error)
        raise UNAUTHENTICATED from error

    user = user_repo.get_user(db, user_id)
    if user is None:
        # A validly signed token for an account that no longer exists. The
        # signature is fine; the identity is not.
        logger.info("token subject no longer exists")
        raise UNAUTHENTICATED
    return user
