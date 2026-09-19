"""Transcript ordering and inbound message limits.

These are not tied to a phase. They guard two things the rest of the app
quietly assumes: that a conversation reads back in the order it was written,
and that one request cannot hand the LLM an arbitrarily large prompt.
"""

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.database.connection import SessionLocal
from app.database.models import Conversation, Message, User
from app.database.repositories import messages as message_repo


def _conversation_with_simultaneous_messages(contents: list[tuple[str, str]]):
    """Write every message inside ONE transaction, with one explicit
    timestamp. That is exactly what PostgreSQL does on its own — now() is the
    transaction clock — and it is what a batched LangGraph write will look
    like in Phase 3."""
    with SessionLocal() as db:
        user = User()
        db.add(user)
        db.commit()
        conversation = Conversation(user_id=user.id)
        db.add(conversation)
        db.commit()

        shared = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        for role, content in contents:
            db.add(
                Message(
                    conversation_id=conversation.id,
                    role=role,
                    content=content,
                    created_at=shared,
                )
            )
        db.commit()
        return conversation.id


def test_transcript_order_survives_identical_timestamps():
    """The regression this column exists for. Ordering by created_at with a
    random-UUID tie-break shuffled the conversation whenever two rows shared a
    transaction; ordering by an identity column cannot."""
    written = [("user", "first"), ("assistant", "second"), ("user", "third")]
    conversation_id = _conversation_with_simultaneous_messages(written)

    with SessionLocal() as db:
        stored = message_repo.list_messages(db, conversation_id)

        assert [m.content for m in stored] == ["first", "second", "third"]
        # All three genuinely share a timestamp, so created_at could not have
        # produced that order on its own.
        assert len({m.created_at for m in stored}) == 1
        assert [m.seq for m in stored] == sorted(m.seq for m in stored)


def test_recent_messages_takes_the_newest_window_in_order():
    written = [("user", "first"), ("assistant", "second"), ("user", "third")]
    conversation_id = _conversation_with_simultaneous_messages(written)

    with SessionLocal() as db:
        recent = message_repo.recent_messages(db, conversation_id, limit=2)

    # Newest two, still oldest-first — not reversed, not an arbitrary pair.
    assert [m.content for m in recent] == ["second", "third"]


def test_overlong_message_is_rejected(client: TestClient, user_id: str):
    """A cost and abuse control: one request must not be able to hand Gemini an
    unbounded prompt."""
    conversation_id = client.post(
        "/conversations", json={"user_id": user_id}
    ).json()["conversation_id"]
    limit = chat_api.settings.max_message_chars

    response = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "x" * (limit + 1),
        },
    )
    assert response.status_code == 400
    assert str(limit) in response.json()["detail"]

    # Nothing was persisted, so a rejected message cannot pollute the history
    # handed to the LLM on the next turn.
    detail = client.get(
        f"/conversations/{conversation_id}", params={"user_id": user_id}
    ).json()
    assert detail["messages"] == []


def test_message_at_the_limit_is_accepted(client: TestClient, user_id: str):
    """The boundary is inclusive — an off-by-one here silently truncates real
    customer messages."""
    conversation_id = client.post(
        "/conversations", json={"user_id": user_id}
    ).json()["conversation_id"]

    response = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "x" * chat_api.settings.max_message_chars,
        },
    )
    assert response.status_code == 200
