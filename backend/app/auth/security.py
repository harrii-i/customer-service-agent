"""Password hashing and JWT issuing/verification.

Centralised on purpose. Every endpoint that needs to hash, verify or decode
goes through here, so the cost parameters and the signing algorithm are stated
once. Duplicating them across endpoints is how one of them ends up on weaker
settings than the rest, and nobody notices.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import jwt
from pwdlib import PasswordHash

from app.config import get_settings

logger = logging.getLogger("csam.auth")

settings = get_settings()

# Argon2id: memory-hard, so custom hardware buys an attacker far less than it
# does against a purely iterative hash.
_hasher = PasswordHash.recommended()

# Claim names. `sub` is the JWT standard for "who this token is about".
CLAIM_SUBJECT = "sub"
CLAIM_EXPIRES = "exp"
CLAIM_ISSUED = "iat"
CLAIM_TYPE = "type"
TOKEN_TYPE = "access"

# RFC 7518 §3.2 minimum for HS256.
MIN_SECRET_BYTES = 32


class InvalidToken(Exception):
    """Raised for any unusable token — bad signature, expired, malformed,
    wrong type. The caller turns every one of them into the same 401: telling
    an attacker *which* part failed is free information."""


@lru_cache
def signing_key() -> str:
    """Fails loudly at first use rather than falling back to a default.

    A hard-coded fallback secret is not a weak secret, it is a public one:
    anyone with the source can mint a token for any account.
    """
    advice = (
        "Generate one with `python -c \"import secrets; "
        "print(secrets.token_urlsafe(48))\"` and put it in backend/.env"
    )
    if not settings.jwt_secret:
        raise RuntimeError(f"JWT_SECRET is not set. {advice}")
    # RFC 7518 §3.2: an HMAC key shorter than the hash output weakens the
    # signature. PyJWT only warns; a forged-token risk deserves a refusal.
    if len(settings.jwt_secret.encode()) < MIN_SECRET_BYTES:
        raise RuntimeError(
            f"JWT_SECRET must be at least {MIN_SECRET_BYTES} bytes. {advice}"
        )
    return settings.jwt_secret


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """False rather than raising on a malformed stored hash — a corrupt row
    must read as "wrong password", not as a 500 that confirms the account
    exists."""
    try:
        return _hasher.verify(password, password_hash)
    except Exception:
        logger.exception("password hash could not be verified")
        return False


def create_access_token(user_id: uuid.UUID) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        CLAIM_SUBJECT: str(user_id),
        CLAIM_ISSUED: now,
        CLAIM_EXPIRES: now + timedelta(minutes=settings.jwt_expire_minutes),
        CLAIM_TYPE: TOKEN_TYPE,
    }
    return jwt.encode(payload, signing_key(), algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> uuid.UUID:
    """The user id this token is for, or `InvalidToken`.

    `algorithms` is pinned to the one we issue with. Accepting whatever the
    token's own header asks for is the classic JWT confusion attack — an
    attacker sets `alg: none`, or signs with the public key of an asymmetric
    pair, and walks in.
    """
    try:
        payload = jwt.decode(
            token,
            signing_key(),
            algorithms=[settings.jwt_algorithm],
            options={"require": [CLAIM_SUBJECT, CLAIM_EXPIRES]},
        )
    except jwt.PyJWTError as error:
        raise InvalidToken(str(error)) from error

    if payload.get(CLAIM_TYPE) != TOKEN_TYPE:
        raise InvalidToken("wrong token type")

    try:
        return uuid.UUID(payload[CLAIM_SUBJECT])
    except (KeyError, ValueError) as error:
        raise InvalidToken("subject is not a user id") from error
