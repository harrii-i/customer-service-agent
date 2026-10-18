"""Phase 10 acceptance: a bounded number of ungrounded attempts, then a human.

The agent is allowed to answer without sources — `SYSTEM_INSTRUCTION` lets it
offer the general troubleshooting a technician would, and on the first try
that is genuinely useful. Doing it repeatedly, while the customer waits for
something only a human can tell them, is not. So the policy is a *run* of
ungrounded replies, not a single one, and most of these tests exist to pin
down what does and does not extend that run.

Four outcomes look identical in the response body — all of them have no
sources and no memories — and they must not be counted alike: an ordinary
ungrounded answer, a guardrail deflection, an escalation, and a failed
generation. That distinction is what `escalation.classify_outcome` records on
each message, and most of the risk in this feature lives there.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import escalation, guardrail, nodes
from app.agent.escalation import (
    BLOCKED,
    ESCALATED,
    FAILED,
    GROUNDED,
    HANDOFF_REPLY,
    UNGROUNDED,
    UNKNOWN,
)
from app.agent.graph import (
    ESCALATE,
    GENERATE,
    RETRIEVE_KNOWLEDGE,
    build_graph,
)
from app.agent.state import initial_state
from app.config import get_settings
from app.llm.gemini import FALLBACK_REPLY, LlmReply
from app.memory import store
from app.rag.ingestion import ingest
from tests.conftest import STUB_REPLY
from tests.test_phase3_graph import make_context, make_state, runtime

LIMIT = get_settings().escalation_ungrounded_limit


@pytest.fixture
def ingested():
    """Function-scoped: `_clean_vectors` empties the collections between
    tests, so anything wider would silently serve a later test nothing."""
    return ingest()


def start_conversation(client: TestClient) -> str:
    return client.post("/conversations", json={}).json()["conversation_id"]


def ask(client: TestClient, conversation_id: str, message: str) -> dict:
    response = client.post(
        "/chat", json={"conversation_id": conversation_id, "message": message}
    )
    assert response.status_code == 200, response.text
    return response.json()


UNANSWERABLE = "What is the maximum spin speed of the WM-900?"
# A question the documentation genuinely answers, phrased so it still
# retrieves after the unanswerable turns above it. `retrieval_query` prepends
# the last two customer turns, so a topic switch is searched together with
# what came before it — a pre-existing trade-off that follow-up questions
# depend on. Every test using this asserts the retrieval actually happened,
# so it fails loudly rather than quietly proving nothing.
GROUNDED_QUESTION = "My washing machine bangs loudly on the spin cycle."
GROUNDED_SOURCE = "Washing Machine Troubleshooting"


def message(role: str, outcome: str | None):
    """A stored message row, as `outcomes_from` sees it."""
    context = None if outcome is None else {"outcome": outcome}
    return SimpleNamespace(role=role, context=context)


# --- the counting rule, in isolation ---------------------------------------


def test_the_default_limit_leaves_room_for_real_attempts():
    """A limit of 1 would escalate the first unanswerable question, which the
    system prompt is explicitly written to answer with general advice."""
    assert LIMIT >= 2


@pytest.mark.parametrize(
    "outcomes, expected",
    [
        ([], 0),
        ([GROUNDED], 0),
        ([UNGROUNDED], 1),
        ([UNGROUNDED, UNGROUNDED], 2),
        ([UNGROUNDED, GROUNDED], 0),
        ([UNGROUNDED, UNGROUNDED, GROUNDED, UNGROUNDED], 1),
        ([UNGROUNDED, BLOCKED], 0),
        ([UNGROUNDED, ESCALATED], 0),
        ([UNGROUNDED, FAILED], 0),
        ([UNGROUNDED, UNKNOWN], 0),
    ],
)
def test_consecutive_ungrounded_counts_backwards_from_the_end(outcomes, expected):
    """Only a run that reaches the *present* counts. An old run that was
    already interrupted is spent."""
    assert escalation.consecutive_ungrounded(outcomes) == expected


@pytest.mark.parametrize(
    "state, outcome",
    [
        (make_state(blocked=True, llm_success=True), BLOCKED),
        (make_state(escalated=True, llm_success=True), ESCALATED),
        (make_state(llm_success=False, response=FALLBACK_REPLY), FAILED),
        (make_state(llm_success=True, documents=[{"title": "W"}]), GROUNDED),
        (make_state(llm_success=True, memories=[{"content": "m"}]), GROUNDED),
        (make_state(llm_success=True), UNGROUNDED),
    ],
)
def test_classify_outcome_separates_the_four_empty_handed_cases(state, outcome):
    """All of these produce a reply with no sources and no memories. Only one
    of them is an ungrounded troubleshooting attempt."""
    assert escalation.classify_outcome(state) == outcome


def test_outcomes_from_ignores_user_messages():
    """A customer repeating themselves is not the agent failing. Only the
    agent's own turns can count toward a run of failed attempts."""
    rows = [
        message("user", None),
        message("assistant", UNGROUNDED),
        message("user", None),
        message("assistant", UNGROUNDED),
    ]
    assert escalation.outcomes_from(rows) == [UNGROUNDED, UNGROUNDED]


def test_an_assistant_message_without_an_outcome_is_not_assumed_ungrounded():
    """Rows written before outcomes were recorded say nothing about what
    happened. Treating them as ungrounded would escalate a customer on the
    strength of history that predates the feature."""
    rows = [message("assistant", None), message("assistant", UNGROUNDED)]
    assert escalation.outcomes_from(rows) == [UNKNOWN, UNGROUNDED]
    assert escalation.consecutive_ungrounded([UNKNOWN, UNKNOWN]) == 0


@pytest.mark.parametrize(
    "prior, expected",
    [([], False), ([UNGROUNDED], False), ([UNGROUNDED, UNGROUNDED], True)],
)
def test_should_escalate_counts_the_current_turn_toward_the_limit(prior, expected):
    """The limit is reached *by* this turn, not after it: with a limit of 3,
    two prior ungrounded replies plus this one is the trigger."""
    state = make_state()
    context = make_context(recent_outcomes=prior)
    assert escalation.should_escalate(state, context, limit=3) is expected


def test_grounding_prevents_escalation_however_long_the_run():
    """Finding something is the whole point. A run of failures does not make
    a successful retrieval a failure too."""
    context = make_context(recent_outcomes=[UNGROUNDED] * 10)
    assert escalation.should_escalate(
        make_state(documents=[{"title": "Warranty Policy"}]), context, limit=3
    ) is False
    assert escalation.should_escalate(
        make_state(memories=[{"content": "Owns a WM-200"}]), context, limit=3
    ) is False


def test_a_limit_of_zero_disables_escalation():
    context = make_context(recent_outcomes=[UNGROUNDED] * 10)
    assert escalation.should_escalate(make_state(), context, limit=0) is False


# --- topology --------------------------------------------------------------


def test_knowledge_retrieval_branches_to_generation_or_escalation():
    edges = {
        (edge.source, edge.target)
        for edge in build_graph().compile().get_graph().edges
    }
    assert (RETRIEVE_KNOWLEDGE, GENERATE) in edges
    assert (RETRIEVE_KNOWLEDGE, ESCALATE) in edges
    # A handover is terminal: no generation, no extraction.
    assert (ESCALATE, "__end__") in edges
    assert (ESCALATE, GENERATE) not in edges


# --- the node --------------------------------------------------------------


def test_escalate_node_returns_a_deterministic_handoff():
    result = nodes.escalate(make_state(), runtime())
    assert result["escalated"] is True
    assert result["response"] == HANDOFF_REPLY
    # Reported as a successful turn, so it is persisted and rendered like any
    # other reply rather than treated as an outage.
    assert result["llm_success"] is True


# --- end to end ------------------------------------------------------------


def test_consecutive_ungrounded_replies_accumulate_across_requests(
    client: TestClient, account, ingested, prompt_contexts
):
    """The counter has to survive separate HTTP requests. It cannot live in
    the checkpoint — every run is handed a fresh `initial_state()` — so this
    is the test that would fail if the outcome were not persisted."""
    conversation_id = start_conversation(client)

    for turn in range(1, LIMIT):
        body = ask(client, conversation_id, UNANSWERABLE)
        assert body["message"] == STUB_REPLY, f"turn {turn} should be answered"

    # The model was asked on every one of those turns, and never with context.
    assert prompt_contexts == [None] * (LIMIT - 1)

    body = ask(client, conversation_id, UNANSWERABLE)
    assert body["message"] == HANDOFF_REPLY


def test_the_escalation_path_calls_neither_gemini_nor_the_extractor(
    client: TestClient, account, ingested, prompt_contexts, monkeypatch
):
    """The handover is deterministic: no generation, and nothing learned from
    a turn in which the agent failed to answer."""
    extracted = []
    monkeypatch.setattr(
        nodes.extractor,
        "extract",
        lambda *a, **k: extracted.append(a) or [],
    )

    conversation_id = start_conversation(client)
    for _ in range(LIMIT):
        body = ask(client, conversation_id, UNANSWERABLE)

    assert body["message"] == HANDOFF_REPLY
    # One generation per answered turn, and none for the escalated one.
    assert len(prompt_contexts) == LIMIT - 1
    assert len(extracted) == LIMIT - 1
    assert store.all_for_user(account.id) == []


def test_a_grounded_reply_resets_the_run(
    client: TestClient, account, ingested, prompt_contexts
):
    """A question we can answer is not a failed attempt, and it clears what
    came before it."""
    conversation_id = start_conversation(client)

    for _ in range(LIMIT - 1):
        ask(client, conversation_id, UNANSWERABLE)

    grounded = ask(client, conversation_id, GROUNDED_QUESTION)
    assert grounded["sources"], "this question must retrieve, or the test proves nothing"
    assert grounded["message"] == STUB_REPLY

    # The run is spent: the next ungrounded turn starts from one again.
    body = ask(client, conversation_id, UNANSWERABLE)
    assert body["message"] == STUB_REPLY


def test_a_grounded_question_never_escalates(
    client: TestClient, account, ingested, prompt_contexts
):
    """Even at the limit, retrieval finding something routes to generation."""
    conversation_id = start_conversation(client)
    for _ in range(LIMIT - 1):
        ask(client, conversation_id, UNANSWERABLE)

    body = ask(client, conversation_id, GROUNDED_QUESTION)
    assert body["message"] == STUB_REPLY
    assert body["message"] != HANDOFF_REPLY
    assert GROUNDED_SOURCE in [source["title"] for source in body["sources"]]


def test_guardrail_deflections_do_not_count_as_attempts(
    client: TestClient, account, ingested, monkeypatch
):
    """An off-topic message is not a troubleshooting attempt that failed. A
    customer who asks several out-of-scope things must not be handed to a
    human because of it."""
    conversation_id = start_conversation(client)

    monkeypatch.setattr(
        guardrail,
        "generate_json",
        lambda *a, **k: guardrail.GuardrailVerdict(category="off_topic"),
    )
    for _ in range(LIMIT + 2):
        body = ask(client, conversation_id, "Write me some Python.")
        assert body["message"] == guardrail.deflection("off_topic")

    # Back in scope, the customer still gets real attempts rather than an
    # immediate handover.
    monkeypatch.setattr(guardrail, "generate_json", lambda *a, **k: None)
    body = ask(client, conversation_id, UNANSWERABLE)
    assert body["message"] == STUB_REPLY


def test_a_failed_generation_does_not_count_as_an_attempt(
    client: TestClient, account, ingested, stub_llm
):
    """The fallback is an outage, not an answer. Counting it would escalate
    customers because Gemini had a bad minute."""
    conversation_id = start_conversation(client)
    stub_llm.setattr(
        nodes,
        "generate_reply",
        lambda turns, context=None: LlmReply(FALLBACK_REPLY, ok=False),
    )
    for _ in range(LIMIT + 2):
        body = ask(client, conversation_id, UNANSWERABLE)
        assert body["message"] == FALLBACK_REPLY

    stub_llm.setattr(
        nodes, "generate_reply", lambda turns, context=None: LlmReply(STUB_REPLY, ok=True)
    )
    body = ask(client, conversation_id, UNANSWERABLE)
    assert body["message"] == STUB_REPLY


def test_an_escalation_is_not_itself_another_failed_attempt(
    client: TestClient, account, ingested
):
    """Having handed over once, the agent starts again rather than escalating
    on every subsequent message."""
    conversation_id = start_conversation(client)
    for _ in range(LIMIT):
        body = ask(client, conversation_id, UNANSWERABLE)
    assert body["message"] == HANDOFF_REPLY

    body = ask(client, conversation_id, UNANSWERABLE)
    assert body["message"] == STUB_REPLY, "escalation restarted the run"


def test_the_escalation_reply_survives_a_reload(
    client: TestClient, account, ingested
):
    """It is persisted exactly like any other assistant turn, so the
    transcript matches what the customer saw."""
    conversation_id = start_conversation(client)
    for _ in range(LIMIT):
        ask(client, conversation_id, UNANSWERABLE)

    detail = client.get(f"/conversations/{conversation_id}").json()
    assistant = [m for m in detail["messages"] if m["role"] == "assistant"]
    assert assistant[-1]["content"] == HANDOFF_REPLY
    assert assistant[-1]["sources"] == []
    assert assistant[-1]["memories"] == []
    # Both halves of the conversation are intact, in order.
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"] * LIMIT


def test_escalation_is_scoped_to_one_conversation(
    client: TestClient, account, ingested
):
    """The run is a property of a conversation, not of a customer. Starting a
    new conversation starts the attempts over."""
    first = start_conversation(client)
    for _ in range(LIMIT):
        body = ask(client, first, UNANSWERABLE)
    assert body["message"] == HANDOFF_REPLY

    second = start_conversation(client)
    assert ask(client, second, UNANSWERABLE)["message"] == STUB_REPLY


def test_the_api_contract_is_unchanged(client: TestClient, account, ingested):
    """Escalation is internal. The response keeps exactly the fields the
    frontend already reads."""
    conversation_id = start_conversation(client)
    for _ in range(LIMIT):
        body = ask(client, conversation_id, UNANSWERABLE)

    assert body["message"] == HANDOFF_REPLY
    assert set(body) == {
        "conversation_id",
        "message",
        "memory_used",
        "memories",
        "sources",
    }
