"""Phase 1 acceptance: the app shell persists conversations in PostgreSQL.

These assertions hold regardless of what the agent replies, so the LLM is
stubbed (see conftest). Gemini's own behaviour is asserted in
test_phase2_llm.py; LangGraph, ChromaDB, memory and RAG are not implemented
yet and are not asserted anywhere.
"""

import uuid

from fastapi.testclient import TestClient


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_user_returns_uuid(client: TestClient):
    response = client.post("/users")
    assert response.status_code == 201
    uuid.UUID(response.json()["user_id"])  # raises if not a UUID


def test_create_and_list_conversations(client: TestClient, user_id: str):
    created = client.post("/conversations", json={"user_id": user_id})
    assert created.status_code == 201
    conversation_id = created.json()["conversation_id"]

    listed = client.get(f"/users/{user_id}/conversations")
    assert listed.status_code == 200
    assert [c["id"] for c in listed.json()] == [conversation_id]


def test_chat_persists_both_messages_and_titles_the_conversation(
    client: TestClient, user_id: str
):
    conversation_id = client.post(
        "/conversations", json={"user_id": user_id}
    ).json()["conversation_id"]

    chat = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "My washing machine is making a loud noise.",
        },
    )
    assert chat.status_code == 200
    body = chat.json()
    assert body["message"]
    # Phase 1 must not claim memory or sources it does not have.
    assert body["memory_used"] is False
    assert body["memories"] == []
    assert body["sources"] == []

    detail = client.get(
        f"/conversations/{conversation_id}", params={"user_id": user_id}
    ).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][0]["content"] == (
        "My washing machine is making a loud noise."
    )
    # Title derives from the first user message, no LLM call.
    assert detail["title"].startswith("My washing machine")
    assert len(detail["title"]) <= 40


def test_conversation_survives_a_new_client_session(
    client: TestClient, user_id: str
):
    """Approximates the reload/restart case: data lives in PostgreSQL, not in
    any in-process state."""
    conversation_id = client.post(
        "/conversations", json={"user_id": user_id}
    ).json()["conversation_id"]
    client.post(
        "/chat",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "Hello",
        },
    )

    with TestClient(client.app) as fresh:
        detail = fresh.get(
            f"/conversations/{conversation_id}", params={"user_id": user_id}
        )
    assert detail.status_code == 200
    assert len(detail.json()["messages"]) == 2


def test_empty_message_is_rejected(client: TestClient, user_id: str):
    conversation_id = client.post(
        "/conversations", json={"user_id": user_id}
    ).json()["conversation_id"]

    response = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "   ",
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Message cannot be empty"


def test_unknown_conversation_is_404(client: TestClient, user_id: str):
    response = client.get(
        f"/conversations/{uuid.uuid4()}", params={"user_id": user_id}
    )
    assert response.status_code == 404


def test_user_cannot_read_or_post_to_another_users_conversation(
    client: TestClient, user_id: str
):
    """The isolation guarantee, at the conversation level. The equivalent test
    for long-term memory arrives with Phase 7."""
    other_user = client.post("/users").json()["user_id"]
    conversation_id = client.post(
        "/conversations", json={"user_id": user_id}
    ).json()["conversation_id"]

    assert (
        client.get(
            f"/conversations/{conversation_id}", params={"user_id": other_user}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/chat",
            json={
                "user_id": other_user,
                "conversation_id": conversation_id,
                "message": "Let me see that.",
            },
        ).status_code
        == 404
    )
    # And it does not leak into the other user's conversation list.
    assert client.get(f"/users/{other_user}/conversations").json() == []
