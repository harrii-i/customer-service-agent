"""The memory endpoints, and the isolation guarantee through the HTTP API.

Elsewhere isolation is tested at the store level. Here it is tested the way an
attacker would meet it: over HTTP, with a real token, against a real second
account. A store-level guarantee that an endpoint routes around is no
guarantee at all.
"""

from fastapi.testclient import TestClient

from app.memory import store
from tests.test_phase7_memory import stub_extraction


def test_a_customer_sees_their_own_memories(client: TestClient, account):
    store.save(account.id, "Customer owns a WM-200 washing machine.", "product")

    body = client.get("/memories").json()
    assert [m["content"] for m in body] == [
        "Customer owns a WM-200 washing machine."
    ]
    assert body[0]["type"] == "product"


def test_memories_are_scoped_to_the_token_not_a_url(
    client: TestClient, account, other_account
):
    """There is no {user_id} in the path, so there is nothing to tamper with.
    The second account simply has no memories of its own."""
    store.save(account.id, "Customer owns a WM-200 washing machine.", "product")
    assert other_account.client.get("/memories").json() == []


def test_a_customer_can_delete_their_own_memory(client: TestClient, account):
    memory_id = store.save(account.id, "Customer owns a WM-200.", "product")

    assert client.delete(f"/memories/{memory_id}").status_code == 204
    assert client.get("/memories").json() == []


def test_a_customer_cannot_delete_someone_elses_memory(
    client: TestClient, account, other_account
):
    """The direct object reference attack: a valid token, someone else's id."""
    memory_id = store.save(account.id, "Customer owns a WM-200.", "product")

    response = other_account.client.delete(f"/memories/{memory_id}")
    assert response.status_code == 404
    # And it is still there.
    assert len(client.get("/memories").json()) == 1


def test_deleting_an_unknown_memory_is_404(client: TestClient, account):
    """Same response as "not yours" — a different one would confirm that some
    other customer holds that id."""
    assert client.delete("/memories/does-not-exist").status_code == 404


def test_a_customer_can_forget_everything(client: TestClient, account, other_account):
    store.save(account.id, "Customer owns a WM-200.", "product")
    store.save(account.id, "Customer's name is Rahul.", "identity")
    store.save(other_account.id, "Customer owns a DW-90.", "product")

    assert client.delete("/memories").status_code == 204
    assert client.get("/memories").json() == []
    # Someone else's memories are untouched.
    assert len(other_account.client.get("/memories").json()) == 1


def test_one_customers_memory_never_reaches_another(
    client: TestClient, account, other_account, monkeypatch
):
    """The scenario from the requirements, end to end over HTTP.

    A says they own a WM-200. B asks what washing machine they own. B must not
    be told about A's.
    """
    stub_extraction(
        monkeypatch, ("Customer owns a WM-200 washing machine.", "product")
    )

    a_conversation = client.post("/conversations", json={}).json()["conversation_id"]
    client.post(
        "/chat",
        json={"conversation_id": a_conversation, "message": "I own a WM-200 washing machine."},
    )
    assert len(client.get("/memories").json()) == 1

    b_conversation = other_account.client.post("/conversations", json={}).json()[
        "conversation_id"
    ]
    body = other_account.client.post(
        "/chat",
        json={
            "conversation_id": b_conversation,
            "message": "What washing machine do I own?",
        },
    ).json()

    assert body["memory_used"] is False
    assert body["memories"] == []
    assert "WM-200" not in str(body["memories"])


def test_the_graph_receives_the_authenticated_user_id(
    client: TestClient, account, other_account, monkeypatch
):
    """The identity ChromaDB filters on comes from the token. If a body field
    could set it, every memory-isolation guarantee above would be bypassable
    in one request."""
    seen = {}

    real_retrieve = __import__(
        "app.agent.nodes", fromlist=["x"]
    ).memory_retriever.retrieve

    def capture(user_id, query, top_k=None):
        seen["user_id"] = user_id
        return real_retrieve(user_id, query, top_k)

    import app.agent.nodes as nodes

    monkeypatch.setattr(nodes.memory_retriever, "retrieve", capture)

    conversation_id = client.post("/conversations", json={}).json()["conversation_id"]
    client.post(
        "/chat",
        json={
            "user_id": other_account.id,  # ignored
            "conversation_id": conversation_id,
            "message": "hello",
        },
    )
    assert seen["user_id"] == account.id
    assert seen["user_id"] != other_account.id
