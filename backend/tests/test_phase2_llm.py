"""Phase 2 acceptance: /chat answers with Gemini instead of a placeholder.

Two kinds of test here. The endpoint tests stub `generate_reply` and assert on
what the API does with its result. The adapter tests exercise
`app.llm.gemini` itself with a fake client, so the failure paths are covered
without a network call or an API key.

LangGraph, ChromaDB, memory and RAG are still not implemented and are not
asserted here.
"""

import pytest
from fastapi.testclient import TestClient

from app.agent import nodes as agent_nodes
from app.agent import prompts
from app.api import chat as chat_api
from app.llm import gemini
from app.llm.gemini import FALLBACK_REPLY, LlmReply
from tests.conftest import STUB_REPLY


def start_conversation(client: TestClient, user_id: str) -> str:
    return client.post("/conversations", json={"user_id": user_id}).json()[
        "conversation_id"
    ]


# --- the endpoint ----------------------------------------------------------


def test_chat_returns_and_persists_the_model_reply(
    client: TestClient, user_id: str
):
    conversation_id = start_conversation(client, user_id)

    body = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "My WM-200 is making a loud noise.",
        },
    ).json()
    assert body["message"] == STUB_REPLY

    detail = client.get(
        f"/conversations/{conversation_id}", params={"user_id": user_id}
    ).json()
    assert [m["content"] for m in detail["messages"]] == [
        "My WM-200 is making a loud noise.",
        STUB_REPLY,
    ]


def test_model_receives_history_ending_with_the_current_message(
    client: TestClient, user_id: str, stub_llm
):
    """The current user message is persisted before generation, so it must
    already be the last turn — the LLM is never asked to answer the
    second-to-last thing the customer said."""
    captured: list[list[tuple[str, str]]] = []

    def capture(turns, context=None):
        captured.append(list(turns))
        return LlmReply(STUB_REPLY, ok=True)

    stub_llm.setattr(agent_nodes, "generate_reply", capture)
    conversation_id = start_conversation(client, user_id)

    for message in ["First question", "Second question"]:
        client.post(
            "/chat",
            json={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message": message,
            },
        )

    assert captured[0] == [("user", "First question")]
    assert captured[1] == [
        ("user", "First question"),
        ("assistant", STUB_REPLY),
        ("user", "Second question"),
    ]


def test_history_handed_to_the_model_is_capped(
    client: TestClient, user_id: str, stub_llm
):
    """The LLM never receives an unbounded conversation."""
    captured: list[list[tuple[str, str]]] = []

    def capture(turns, context=None):
        captured.append(list(turns))
        return LlmReply(STUB_REPLY, ok=True)

    stub_llm.setattr(agent_nodes, "generate_reply", capture)
    stub_llm.setattr(chat_api.settings, "history_message_limit", 4)
    conversation_id = start_conversation(client, user_id)

    for index in range(5):
        client.post(
            "/chat",
            json={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message": f"Question {index}",
            },
        )

    assert all(len(turns) <= 4 for turns in captured)
    # And it is the *recent* window, oldest-first, not an arbitrary slice.
    assert captured[-1][-1] == ("user", "Question 4")
    assert captured[-1][0] == ("assistant", STUB_REPLY)


def test_llm_failure_degrades_rather_than_erroring(
    client: TestClient, user_id: str, stub_llm
):
    """A third-party outage must not cost the customer their message or return
    a 500. The fallback is shown, persisted, and reported as not-ok."""
    stub_llm.setattr(
        agent_nodes,
        "generate_reply",
        lambda turns, context=None: LlmReply(FALLBACK_REPLY, ok=False),
    )
    conversation_id = start_conversation(client, user_id)

    response = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "Is my warranty still valid?",
        },
    )
    assert response.status_code == 200
    assert response.json()["message"] == FALLBACK_REPLY

    detail = client.get(
        f"/conversations/{conversation_id}", params={"user_id": user_id}
    ).json()
    # The customer's message survives, and the transcript matches the screen.
    assert [m["content"] for m in detail["messages"]] == [
        "Is my warranty still valid?",
        FALLBACK_REPLY,
    ]


def test_phase_2_still_claims_no_memories_or_sources(
    client: TestClient, user_id: str
):
    """Memory is Phase 7 and sources are Phase 6. Until then the API must not
    imply either exists."""
    conversation_id = start_conversation(client, user_id)

    body = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "Hello",
        },
    ).json()
    assert body["memory_used"] is False
    assert body["memories"] == []
    assert body["sources"] == []


# --- the adapter -----------------------------------------------------------


def _clear_client_cache() -> None:
    # Tolerates `_client` currently being a monkeypatched stand-in, which has
    # no cache to clear: fixture teardown order is not ours to rely on.
    cache_clear = getattr(gemini._client, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


@pytest.fixture(autouse=True)
def _reset_client_cache():
    """`_client` is lru_cached, so a test that changes the key must not leak
    that client into the next test."""
    _clear_client_cache()
    yield
    _clear_client_cache()


class FakeModels:
    def __init__(self, result):
        self._result = result
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeResponse:
    def __init__(self, text, candidates=None):
        self.text = text
        self.candidates = candidates or []


def fake_client(monkeypatch, result) -> FakeModels:
    models = FakeModels(result)
    monkeypatch.setattr(
        gemini, "_client", lambda: type("C", (), {"models": models})()
    )
    return models


def test_build_contents_maps_roles_and_drops_non_turns():
    contents = gemini.build_contents(
        [
            ("user", "hello"),
            ("assistant", "hi"),
            ("system", "ignored"),  # not a conversation turn
            ("user", "   "),  # nothing to send
        ]
    )
    assert [c.role for c in contents] == ["user", "model"]
    assert [c.parts[0].text for c in contents] == ["hello", "hi"]


def test_missing_api_key_falls_back_without_calling_gemini(monkeypatch):
    monkeypatch.setattr(gemini.settings, "gemini_api_key", "")
    assert gemini.generate_reply([("user", "hello")]) == LlmReply(
        FALLBACK_REPLY, ok=False
    )


def test_successful_generation_returns_ok(monkeypatch):
    models = fake_client(monkeypatch, FakeResponse("  Here is the answer.  "))

    assert gemini.generate_reply([("user", "hello")]) == LlmReply(
        "Here is the answer.", ok=True
    )
    assert models.calls[0]["model"] == gemini.settings.gemini_model
    config = models.calls[0]["config"]
    assert config.system_instruction == prompts.SYSTEM_INSTRUCTION
    # Pinned deliberately: Gemini 3.x rejects 2.5's thinking_budget, and a
    # silently drifting default moves latency and cost.
    assert (
        config.thinking_config.thinking_level.value
        == gemini.settings.gemini_thinking_level.upper()
    )


def test_empty_text_is_a_failure_not_an_empty_answer(monkeypatch):
    """A safety block or an exhausted output cap returns no text. Treating that
    as success would show the customer a blank bubble."""
    fake_client(monkeypatch, FakeResponse(""))

    assert gemini.generate_reply([("user", "hello")]) == LlmReply(
        FALLBACK_REPLY, ok=False
    )


def test_api_exception_falls_back(monkeypatch):
    fake_client(monkeypatch, RuntimeError("connection reset"))

    assert gemini.generate_reply([("user", "hello")]) == LlmReply(
        FALLBACK_REPLY, ok=False
    )


def test_no_usable_turns_is_not_sent(monkeypatch):
    models = fake_client(monkeypatch, FakeResponse("should not be reached"))

    assert gemini.generate_reply([("system", "only this")]) == LlmReply(
        FALLBACK_REPLY, ok=False
    )
    assert models.calls == []
