"""Phase 10b: the two ways the escalation counter misread a conversation.

Both bugs had one cause. `classify_outcome` used "did retrieval return
anything" as a proxy for "did the agent answer the question", and that proxy
is wrong in both directions:

- A pleasantry retrieves nothing, so "thanks" looked like a failed support
  attempt. Three of them handed the customer to a human.
- `memory_always_types` surfaces the customer's name and appliances on every
  turn regardless of relevance, so any remembered customer looked answered on
  every turn and could never be escalated at all.

The first is a false positive, the second a false negative, and the fixes are
tested together here because they share `has_grounding`.

Nothing in these tests reaches Gemini: the guardrail classifier is stubbed by
`conftest` and re-stubbed per test, exactly as Phase 9 does it.
"""

import pytest
from fastapi.testclient import TestClient

from app.agent import escalation, guardrail, nodes
from app.agent.escalation import (
    BLOCKED,
    COURTESY,
    ESCALATED,
    FAILED,
    GROUNDED,
    HANDOFF_REPLY,
    UNGROUNDED,
)
from app.agent.guardrail import GuardrailVerdict
from app.config import get_settings
from app.memory import store
from app.rag.ingestion import ingest
from tests.conftest import STUB_REPLY
from tests.test_phase3_graph import make_context, make_state, runtime

SETTINGS = get_settings()
LIMIT = SETTINGS.escalation_ungrounded_limit
ALWAYS = tuple(SETTINGS.memory_always_types)

UNANSWERABLE = "What is the maximum spin speed of the WM-900?"
THANKS = "Thanks."


@pytest.fixture
def ingested():
    return ingest()


def stub_verdict(monkeypatch, category="in_scope", is_support_request=True):
    """Pin the classifier's answer. The real one is a Gemini call."""
    monkeypatch.setattr(
        guardrail,
        "generate_json",
        lambda *a, **k: GuardrailVerdict(
            category=category, is_support_request=is_support_request
        ),
    )


def start_conversation(client: TestClient) -> str:
    return client.post("/conversations", json={}).json()["conversation_id"]


def ask(client: TestClient, conversation_id: str, message: str) -> dict:
    response = client.post(
        "/chat", json={"conversation_id": conversation_id, "message": message}
    )
    assert response.status_code == 200, response.text
    return response.json()


# --- Fix A: the verdict ----------------------------------------------------


def test_the_new_field_defaults_to_a_support_request():
    """Additive and defaulted, so a verdict built without it — an old stub, a
    model response from before the field existed — behaves as it always did."""
    assert GuardrailVerdict(category="in_scope").is_support_request is True


def test_the_four_categories_are_unchanged():
    """Courtesy is deliberately not a fifth category: the category drives the
    blocked/allowed decision, and that decision must not move."""
    for category in ("injection", "unsafe", "off_topic", "in_scope"):
        assert GuardrailVerdict(category=category).category == category
    with pytest.raises(ValueError):
        GuardrailVerdict(category="courtesy")


def test_classifier_failure_still_fails_open_as_a_support_request(monkeypatch):
    """Both fields take the conservative value: the message is answered, and
    it still counts as a real attempt rather than being quietly discounted."""
    monkeypatch.setattr(guardrail, "generate_json", lambda *a, **k: None)
    verdict = guardrail.classify_verdict("my washer leaks")
    assert verdict.category == "in_scope"
    assert verdict.is_support_request is True
    assert guardrail.classify("my washer leaks") == "in_scope"


@pytest.mark.parametrize("category", ["off_topic", "injection", "unsafe"])
def test_blocked_categories_keep_their_deflections(monkeypatch, category):
    """The safety routing is untouched by this change."""
    stub_verdict(monkeypatch, category=category, is_support_request=False)
    result = nodes.check_guardrail(make_state(), runtime())
    assert result["blocked"] is True
    assert result["response"] == guardrail.deflection(category)
    assert result["llm_success"] is True
    # A blocked message is not "courtesy" however the flag came back.
    assert result.get("courtesy", False) is False


def test_the_node_records_courtesy_without_blocking_it(monkeypatch):
    """A greeting is answered like anything else. The flag only stops it being
    counted as a failed support attempt."""
    stub_verdict(monkeypatch, is_support_request=False)
    assert nodes.check_guardrail(make_state(), runtime()) == {
        "blocked": False,
        "courtesy": True,
    }

    stub_verdict(monkeypatch, is_support_request=True)
    assert nodes.check_guardrail(make_state(), runtime()) == {
        "blocked": False,
        "courtesy": False,
    }


# --- Fix A: counting -------------------------------------------------------


def test_courtesy_is_its_own_outcome():
    assert escalation.classify_outcome(
        make_state(llm_success=True, courtesy=True)
    ) == COURTESY


@pytest.mark.parametrize(
    "state, outcome",
    [
        (make_state(blocked=True, llm_success=True, courtesy=True), BLOCKED),
        (make_state(escalated=True, llm_success=True, courtesy=True), ESCALATED),
        (make_state(llm_success=False, courtesy=True), FAILED),
    ],
)
def test_courtesy_does_not_override_the_outcomes_above_it(state, outcome):
    """Order is the policy, and courtesy sits below the three that describe
    what happened to the run rather than what the customer said."""
    assert escalation.classify_outcome(state) == outcome


@pytest.mark.parametrize(
    "outcomes, expected",
    [
        ([COURTESY], 0),
        ([UNGROUNDED, COURTESY], 1),
        # The sequence from the brief: leak, thanks, still leaking.
        ([UNGROUNDED, COURTESY, UNGROUNDED], 2),
        ([UNGROUNDED, COURTESY, COURTESY, UNGROUNDED], 2),
        ([COURTESY, COURTESY, COURTESY], 0),
        # Courtesy is transparent, not an eraser: what it sits between still
        # counts, but what was already reset stays reset.
        ([UNGROUNDED, GROUNDED, COURTESY, UNGROUNDED], 1),
        ([UNGROUNDED, COURTESY, GROUNDED], 0),
    ],
)
def test_courtesy_is_skipped_rather_than_counted_or_reset(outcomes, expected):
    assert escalation.consecutive_ungrounded(outcomes) == expected


def test_a_courtesy_turn_never_escalates_however_long_the_run():
    """Answering "thanks" with "let me pass you to an agent" would be absurd."""
    context = make_context(recent_outcomes=[UNGROUNDED] * 10)
    assert escalation.should_escalate(
        make_state(courtesy=True), context, limit=LIMIT
    ) is False
    # And the run is not spent — the next real question still sees it.
    assert escalation.should_escalate(
        make_state(courtesy=False), context, limit=LIMIT
    ) is True


# --- Fix B: grounding ------------------------------------------------------


def test_a_profile_fact_alone_does_not_ground_a_question():
    """The false negative. These are surfaced on every turn by design, so
    counting them as grounding made every remembered customer permanently
    "answered"."""
    for memory_type in ALWAYS:
        state = make_state(memories=[{"id": "1", "content": "x", "type": memory_type}])
        assert escalation.has_grounding(state) is False
        assert escalation.classify_outcome(
            make_state(
                llm_success=True,
                memories=[{"id": "1", "content": "x", "type": memory_type}],
            )
        ) == UNGROUNDED


def test_a_similarity_gated_memory_does_ground_a_question():
    """An `issue` had to clear the relevance floor to be here, so it was
    retrieved *for this message*."""
    state = make_state(
        memories=[{"id": "1", "content": "Washer leaks.", "type": "issue"}]
    )
    assert escalation.has_grounding(state) is True


def test_a_document_always_grounds():
    assert escalation.has_grounding(
        make_state(documents=[{"title": "Warranty Policy"}])
    ) is True
    # Even alongside nothing but profile facts.
    assert escalation.has_grounding(
        make_state(
            documents=[{"title": "Warranty Policy"}],
            memories=[{"id": "1", "content": "x", "type": ALWAYS[0]}],
        )
    ) is True


def test_a_memory_of_unknown_type_is_treated_as_grounding():
    """We cannot tell whether it was surfaced unconditionally, and the
    conservative direction is to under-escalate rather than hand someone over
    on the strength of a fact we cannot interpret."""
    assert escalation.has_grounding(make_state(memories=[{"content": "x"}])) is True
    assert escalation.has_grounding(
        make_state(memories=[{"content": "x", "type": "fact"}])
    ) is True


# --- end to end ------------------------------------------------------------


def test_courtesy_between_two_failures_preserves_the_count(
    client: TestClient, account, ingested, monkeypatch, prompt_contexts
):
    """The headline sequence. A polite customer must reach a human in the same
    number of real attempts as a terse one."""
    conversation_id = start_conversation(client)

    stub_verdict(monkeypatch, is_support_request=True)
    assert ask(client, conversation_id, UNANSWERABLE)["message"] == STUB_REPLY

    stub_verdict(monkeypatch, is_support_request=False)
    assert ask(client, conversation_id, THANKS)["message"] == STUB_REPLY

    # Attempt two of three, not one: the pleasantry did not reset anything.
    stub_verdict(monkeypatch, is_support_request=True)
    assert ask(client, conversation_id, UNANSWERABLE)["message"] == STUB_REPLY

    assert ask(client, conversation_id, UNANSWERABLE)["message"] == HANDOFF_REPLY


def test_courtesy_alone_never_escalates(
    client: TestClient, account, ingested, monkeypatch
):
    """The false positive this milestone exists to remove."""
    conversation_id = start_conversation(client)
    stub_verdict(monkeypatch, is_support_request=False)

    for message in ["Hello", "Thanks", "Okay", "That worked", "Got it", "Thanks!"]:
        body = ask(client, conversation_id, message)
        assert body["message"] == STUB_REPLY, f"{message!r} escalated"


def test_a_customer_with_a_remembered_name_can_still_escalate(
    client: TestClient, account, ingested, monkeypatch
):
    """The false negative, end to end. Before this fix the stored identity
    made every turn look grounded and this conversation never escalated."""
    store.save(account.id, "Customer's name is Rahul.", "identity")
    store.save(account.id, "Customer owns a WM-200 washing machine.", "product")
    stub_verdict(monkeypatch, is_support_request=True)

    conversation_id = start_conversation(client)
    for _ in range(LIMIT - 1):
        assert ask(client, conversation_id, UNANSWERABLE)["message"] == STUB_REPLY

    assert ask(client, conversation_id, UNANSWERABLE)["message"] == HANDOFF_REPLY


def test_the_profile_facts_still_reach_the_prompt(
    client: TestClient, account, ingested, monkeypatch, prompt_contexts
):
    """Fix B changes how grounding is *judged*, not what is retrieved. The
    customer must still be recognised — that is the whole project."""
    store.save(account.id, "Customer's name is Rahul.", "identity")
    stub_verdict(monkeypatch, is_support_request=True)

    conversation_id = start_conversation(client)
    body = ask(client, conversation_id, "What do you know about me?")

    assert any("Rahul" in memory["content"] for memory in body["memories"])
    assert body["memory_used"] is True
    assert "Rahul" in (prompt_contexts[-1] or "")


def test_a_gated_issue_memory_still_resets_the_run(
    client: TestClient, account, ingested, monkeypatch
):
    """A memory retrieved *for this message* is real grounding, and clears
    what came before it."""
    store.save(
        account.id, "Customer reported the washing machine bangs on spin.", "issue"
    )
    stub_verdict(monkeypatch, is_support_request=True)
    conversation_id = start_conversation(client)

    for _ in range(LIMIT - 1):
        ask(client, conversation_id, UNANSWERABLE)

    grounded = ask(client, conversation_id, "My washing machine bangs on spin.")
    assert any(
        memory["type"] == "issue" for memory in grounded["memories"]
    ), "this message must retrieve the issue, or the test proves nothing"
    assert grounded["message"] == STUB_REPLY

    # The run is spent.
    assert ask(client, conversation_id, UNANSWERABLE)["message"] == STUB_REPLY
