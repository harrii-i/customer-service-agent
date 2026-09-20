"""The app shell: conversations and messages persisted in PostgreSQL.

These assertions hold regardless of what the agent replies, so the LLM is
stubbed (see conftest). Gemini's own behaviour is asserted in
test_phase2_llm.py; the workflow in test_phase3_graph.py.

Every endpoint here is authenticated, and no request carries a user id — the
owner comes from the token.
"""

import uuid

from fastapi.testclient import TestClient


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_and_list_conversations(client: TestClient, account):
    created = client.post("/conversations", json={})
    assert created.status_code == 201
    conversation_id = created.json()["conversation_id"]

    listed = client.get("/conversations")
    assert listed.status_code == 200
    assert [c["id"] for c in listed.json()] == [conversation_id]


def test_chat_persists_both_messages_and_titles_the_conversation(
    client: TestClient, account
):
    conversation_id = client.post("/conversations", json={}).json()["conversation_id"]

    chat = client.post(
        "/chat",
        json={
            "conversation_id": conversation_id,
            "message": "My washing machine is making a loud noise.",
        },
    )
    assert chat.status_code == 200
    body = chat.json()
    assert body["message"]

    detail = client.get(f"/conversations/{conversation_id}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][0]["content"] == (
        "My washing machine is making a loud noise."
    )
    # Title derives from the first user message, no LLM call.
    assert detail["title"].startswith("My washing machine")
    assert len(detail["title"]) <= 40


def test_conversation_survives_a_new_client_session(
    client: TestClient, account, make_client
):
    """Approximates the reload/restart case: data lives in PostgreSQL, not in
    any in-process state."""
    conversation_id = client.post("/conversations", json={}).json()["conversation_id"]
    client.post(
        "/chat", json={"conversation_id": conversation_id, "message": "Hello"}
    )

    fresh = make_client()
    detail = fresh.get(
        f"/conversations/{conversation_id}", headers=account.headers
    )
    assert detail.status_code == 200
    assert len(detail.json()["messages"]) == 2


def test_empty_message_is_rejected(client: TestClient, account):
    conversation_id = client.post("/conversations", json={}).json()["conversation_id"]

    response = client.post(
        "/chat", json={"conversation_id": conversation_id, "message": "   "}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Message cannot be empty"


def test_unknown_conversation_is_404(client: TestClient, account):
    response = client.get(f"/conversations/{uuid.uuid4()}")
    assert response.status_code == 404


def test_user_cannot_read_or_post_to_another_users_conversation(
    client: TestClient, account, other_account
):
    """The isolation guarantee at the conversation level. The equivalent for
    long-term memory is in test_phase7_memory.py."""
    conversation_id = client.post("/conversations", json={}).json()["conversation_id"]

    assert (
        other_account.client.get(f"/conversations/{conversation_id}").status_code == 404
    )
    assert (
        other_account.client.post(
            "/chat",
            json={"conversation_id": conversation_id, "message": "Let me see that."},
        ).status_code
        == 404
    )
    # And it does not leak into the other account's conversation list.
    assert other_account.client.get("/conversations").json() == []
