"""Authentication and authorization.

Two different questions, tested separately. Authentication: is this a real,
signed-in account? Authorization: is this account allowed to touch *this*
resource? A system can get the first right and the second wrong, and that is
the more dangerous of the two failures.
"""

import uuid

import jwt
import pytest
from fastapi.testclient import TestClient

from app.auth import security
from app.config import get_settings
from tests.conftest import PASSWORD, signup

settings = get_settings()


def new_email() -> str:
    return f"user-{uuid.uuid4()}@example.com"


# --- signup ----------------------------------------------------------------


def test_valid_signup_returns_the_account_and_authenticates(client: TestClient):
    response = client.post(
        "/auth/signup",
        json={"name": "Rahul", "email": new_email(), "password": PASSWORD},
    )
    assert response.status_code == 201
    body = response.json()
    uuid.UUID(body["id"])
    assert body["name"] == "Rahul"
    # The cookie is set, so the client is now authenticated without doing
    # anything further with the token.
    assert settings.auth_cookie_name in response.cookies
    assert client.get("/auth/me").status_code == 200


def test_signup_never_returns_the_password_or_its_hash(client: TestClient):
    """Structural, not incidental: no response model has a field to carry it."""
    body = client.post(
        "/auth/signup",
        json={"name": "Rahul", "email": new_email(), "password": PASSWORD},
    ).json()
    serialised = str(body)
    assert "password_hash" not in body
    assert PASSWORD not in serialised
    assert "argon2" not in serialised


def test_duplicate_email_is_rejected(client: TestClient, make_client):
    email = new_email()
    signup(client, email=email)
    response = make_client().post(
        "/auth/signup", json={"name": "Impostor", "email": email, "password": PASSWORD}
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "Email already registered"


def test_email_uniqueness_is_case_insensitive(client: TestClient, make_client):
    """"Rahul@x.com" and "rahul@x.com" are one person, not two accounts."""
    signup(client, email="Mixed.Case@Example.com")
    response = make_client().post(
        "/auth/signup",
        json={"name": "Other", "email": "mixed.case@example.com", "password": PASSWORD},
    )
    assert response.status_code == 409


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "A", "email": "not-an-email", "password": PASSWORD},
        {"name": "A", "email": "a@example.com", "password": "short"},
        {"name": "", "email": "a@example.com", "password": PASSWORD},
        {"name": "   ", "email": "a@example.com", "password": PASSWORD},
    ],
    ids=["invalid email", "password too short", "empty name", "whitespace name"],
)
def test_invalid_signup_data_is_rejected(client: TestClient, payload):
    assert client.post("/auth/signup", json=payload).status_code == 422


def test_signup_normalises_email_and_name(client: TestClient):
    body = client.post(
        "/auth/signup",
        json={"name": "  Rahul  ", "email": "  Rahul@Example.com ", "password": PASSWORD},
    ).json()
    assert body["name"] == "Rahul"
    assert body["email"] == "rahul@example.com"


# --- signin ----------------------------------------------------------------


def test_valid_credentials_sign_in(client: TestClient, make_client):
    email = new_email()
    signup(client, email=email)

    fresh = make_client()
    response = fresh.post("/auth/signin", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200
    assert fresh.get("/auth/me").json()["email"] == email


def test_signin_is_case_insensitive_on_email(client: TestClient, make_client):
    signup(client, email="Person@Example.com")
    fresh = make_client()
    assert fresh.post(
        "/auth/signin", json={"email": "PERSON@example.com", "password": PASSWORD}
    ).status_code == 200


def test_wrong_password_and_unknown_email_are_indistinguishable(
    client: TestClient, make_client
):
    """Two different messages would be an account-enumeration oracle: an
    attacker learns which addresses are registered without guessing a
    password."""
    email = new_email()
    signup(client, email=email)

    wrong_password = make_client().post(
        "/auth/signin", json={"email": email, "password": "not-the-password"}
    )
    unknown_email = make_client().post(
        "/auth/signin", json={"email": new_email(), "password": PASSWORD}
    )

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()


# --- tokens ----------------------------------------------------------------


def test_valid_bearer_token_is_accepted(client: TestClient, make_client):
    account = signup(client, email=new_email())
    fresh = make_client()  # no cookie jar
    assert fresh.get("/auth/me", headers=account.headers).status_code == 200


def test_missing_token_is_rejected(make_client):
    assert make_client().get("/auth/me").status_code == 401


@pytest.mark.parametrize(
    "token",
    ["not-a-token", "a.b.c", ""],
    ids=["malformed", "wrong segments", "empty"],
)
def test_malformed_tokens_are_rejected(make_client, token):
    response = make_client().get(
        "/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401


def test_a_token_signed_with_another_key_is_rejected(make_client):
    """The whole point of the signature. Without verification, anyone could
    mint a token naming any user id."""
    forged = jwt.encode(
        {"sub": str(uuid.uuid4()), "type": "access", "exp": 9_999_999_999},
        "a-different-secret-that-is-at-least-32-bytes",
        algorithm="HS256",
    )
    assert make_client().get(
        "/auth/me", headers={"Authorization": f"Bearer {forged}"}
    ).status_code == 401


def test_expired_token_is_rejected(client: TestClient, make_client, monkeypatch):
    account = signup(client, email=new_email())
    monkeypatch.setattr(settings, "jwt_expire_minutes", -1)
    expired = security.create_access_token(uuid.UUID(account.id))

    assert make_client().get(
        "/auth/me", headers={"Authorization": f"Bearer {expired}"}
    ).status_code == 401


def test_token_for_a_deleted_account_is_rejected(client: TestClient, make_client):
    """A valid signature over an identity that no longer exists."""
    orphan = security.create_access_token(uuid.uuid4())
    assert make_client().get(
        "/auth/me", headers={"Authorization": f"Bearer {orphan}"}
    ).status_code == 401


def test_a_short_secret_is_refused(monkeypatch):
    """RFC 7518 §3.2. PyJWT only warns; a forged-token risk deserves a
    refusal."""
    monkeypatch.setattr(settings, "jwt_secret", "too-short")
    security.signing_key.cache_clear()
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        security.signing_key()
    security.signing_key.cache_clear()


def test_the_auth_cookie_is_httponly(client: TestClient):
    """httpOnly is what stops an injected script reading the session."""
    response = client.post(
        "/auth/signup",
        json={"name": "Rahul", "email": new_email(), "password": PASSWORD},
    )
    cookie_header = response.headers["set-cookie"].lower()
    assert "httponly" in cookie_header
    assert "samesite=lax" in cookie_header


# --- logout ----------------------------------------------------------------


def test_logout_clears_the_session(client: TestClient, account):
    assert client.get("/auth/me").status_code == 200
    assert client.post("/auth/logout").status_code == 204
    assert client.get("/auth/me").status_code == 401


# --- protected endpoints ---------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/auth/me"),
        ("get", "/conversations"),
        ("post", "/conversations"),
        ("post", "/chat"),
        ("get", "/memories"),
        ("delete", "/memories"),
    ],
)
def test_every_user_endpoint_requires_authentication(make_client, method, path):
    response = make_client().request(method.upper(), path, json={})
    assert response.status_code == 401, f"{method.upper()} {path} was not protected"


def test_health_stays_public(make_client):
    """Liveness checks must not need credentials."""
    assert make_client().get("/health").status_code == 200


# --- authorization ---------------------------------------------------------


def test_a_user_reads_their_own_conversation(client: TestClient, account):
    conversation_id = client.post("/conversations", json={}).json()["conversation_id"]
    assert client.get(f"/conversations/{conversation_id}").status_code == 200


def test_a_user_cannot_read_another_users_conversation(
    client: TestClient, account, other_account
):
    conversation_id = client.post("/conversations", json={}).json()["conversation_id"]
    assert (
        other_account.client.get(f"/conversations/{conversation_id}").status_code == 404
    )


def test_a_conversation_cannot_be_created_for_someone_else(
    client: TestClient, account, other_account
):
    """There is no body field naming an owner, so the attempt cannot even be
    expressed — and a stray field is ignored rather than honoured."""
    client.post("/conversations", json={"user_id": other_account.id, "title": "Mine"})
    assert other_account.client.get("/conversations").json() == []
    assert len(client.get("/conversations").json()) == 1


def test_chat_uses_the_authenticated_user_not_the_body(
    client: TestClient, account, other_account
):
    """The impersonation attempt: a valid token for one account, someone
    else's id in the body. The body must count for nothing."""
    conversation_id = client.post("/conversations", json={}).json()["conversation_id"]
    response = client.post(
        "/chat",
        json={
            "user_id": other_account.id,
            "conversation_id": conversation_id,
            "message": "Whose conversation is this?",
        },
    )
    assert response.status_code == 200
    # It landed in the authenticated account's conversation, not the other's.
    assert len(client.get("/conversations").json()) == 1
    assert other_account.client.get("/conversations").json() == []
