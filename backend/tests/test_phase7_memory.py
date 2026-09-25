"""Phase 7 acceptance: the agent remembers a customer across conversations.

This is the claim the project exists to make, so the headline test is the one
that spans two *separate* conversations — and, just as importantly, the one
that proves a second customer sees none of it.
"""

from datetime import date

from fastapi.testclient import TestClient

from app.agent import nodes
from app.agent.prompts import SYSTEM_INSTRUCTION, build_context_block
from app.memory import extractor, retriever, store
from app.memory.extractor import ExtractedMemory, Extraction
from tests.test_phase3_graph import make_context, make_state, runtime


# --- the store -------------------------------------------------------------


def test_saved_memory_is_retrievable_by_its_owner():
    store.save("u1", "Customer owns a WM-200 washing machine.", "product")
    hits = store.search("u1", "what appliance do they own", top_k=5)
    assert hits and "WM-200" in hits[0]["content"]
    assert hits[0]["type"] == "product"


def test_one_customer_cannot_retrieve_anothers_memory():
    """The isolation guarantee, at the memory level — the counterpart to the
    conversation-level test in Phase 1."""
    store.save("owner", "Customer owns a WM-200 washing machine.", "product")
    assert store.search("intruder", "what appliance do they own", top_k=5) == []
    assert store.all_for_user("intruder") == []


def test_search_on_an_empty_store_is_empty():
    assert store.search("u1", "anything", top_k=5) == []


# --- extraction ------------------------------------------------------------


def stub_extraction(monkeypatch, *memories):
    monkeypatch.setattr(
        extractor,
        "generate_json",
        lambda *a, **k: Extraction(
            memories=[ExtractedMemory(content=c, type=t) for c, t in memories]
        ),
    )


def test_extraction_returns_candidates(monkeypatch):
    stub_extraction(monkeypatch, ("Customer's name is Rahul.", "identity"))
    assert extractor.extract("I'm Rahul", "Hello Rahul") == [
        {"content": "Customer's name is Rahul.", "type": "identity"}
    ]


def test_extraction_failure_is_not_an_error(monkeypatch):
    """Failing to remember must never fail the customer's reply."""
    monkeypatch.setattr(extractor, "generate_json", lambda *a, **k: None)
    assert extractor.extract("hi", "hello") == []


def test_blank_candidates_are_discarded(monkeypatch):
    stub_extraction(monkeypatch, ("   ", "identity"))
    assert extractor.extract("hi", "hello") == []


# --- retrieval threshold ---------------------------------------------------


def test_irrelevant_memories_are_not_retrieved():
    """Without a floor, every message recalls the nearest memory whether or
    not it relates — the agent opens a delivery question by mentioning a
    washing-machine noise."""
    store.save("u1", "Customer's washing machine bangs loudly on spin.", "issue")
    assert retriever.retrieve("u1", "When do you deliver?") == []
    assert retriever.retrieve("u1", "My washing machine is noisy again.")


def test_retrieved_shape_matches_the_api_contract():
    store.save("u1", "Customer owns a WM-200.", "product")
    hit = retriever.retrieve("u1", "what do they own")[0]
    assert set(hit) == {"id", "content", "type"}


def test_identity_is_recalled_for_a_meta_question():
    """Regression. "who am I?" / "what do you know about me?" share no words
    with "Customer's name is …" and score far below the relevance floor, so a
    similarity-only retriever handed the agent nothing and it truthfully denied
    knowing the customer. Identity facts must surface regardless of phrasing."""
    store.save("u1", "Customer's name is Harishankar.", "identity")
    store.save("u1", "Customer lives in Trivandrum, Kerala.", "identity")
    for question in ("who am i?", "what is your info about me?", "what is my name?"):
        contents = [m["content"] for m in retriever.retrieve("u1", question)]
        assert any("Harishankar" in c for c in contents), question


def test_products_owned_surface_regardless_of_phrasing():
    """The other stable profile fact: what the customer owns, recalled even by
    a bare "what do you know about me?" that matches nothing lexically."""
    store.save("u1", "Customer owns a WM-200 washing machine.", "product")
    contents = [
        m["content"] for m in retriever.retrieve("u1", "what do you know about me?")
    ]
    assert any("WM-200" in c for c in contents)


def test_issues_stay_similarity_gated_off_topic():
    """Issues are not profile facts: an old complaint must not leak into an
    unrelated message just because it is the customer's."""
    store.save("u1", "Customer reported a water leak from the washing machine.", "issue")
    assert retriever.retrieve("u1", "when do you deliver?") == []


def test_no_duplicate_when_a_profile_fact_also_matches():
    """A product asked about directly is both a profile fact and a strong
    similarity hit; it must appear once, not twice."""
    store.save("u1", "Customer owns a WM-200 washing machine.", "product")
    results = retriever.retrieve("u1", "what appliance do I own?")
    assert sum("WM-200" in m["content"] for m in results) == 1


# --- nodes -----------------------------------------------------------------


def test_retrieve_memories_node_filters_by_the_runtime_user():
    store.save("owner", "Customer owns a WM-200 washing machine.", "product")
    result = nodes.retrieve_memories(
        make_state(),
        runtime(make_context(user_id="intruder", user_message="what do I own")),
    )
    assert result == {"memories": []}


def test_retrieval_failure_degrades_instead_of_breaking(monkeypatch):
    """ChromaDB being down costs context, not the reply."""
    monkeypatch.setattr(
        nodes.memory_retriever,
        "retrieve",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("chroma down")),
    )
    assert nodes.retrieve_memories(make_state(), runtime()) == {"memories": []}


def test_extract_node_stores_what_it_finds(monkeypatch):
    stub_extraction(monkeypatch, ("Customer owns a WM-200.", "product"))
    nodes.extract_memories(
        make_state(response="Sure.", llm_success=True),
        runtime(make_context(user_id="u1", user_message="I own a WM-200")),
    )
    assert store.all_for_user("u1")


def test_nothing_is_learned_from_a_failed_generation(monkeypatch):
    """The fallback text is not something the customer said, and there is no
    real exchange to learn from."""
    stub_extraction(monkeypatch, ("Customer owns a WM-200.", "product"))
    nodes.extract_memories(
        make_state(response="Sorry, I can't reach…", llm_success=False),
        runtime(make_context(user_id="u1")),
    )
    assert store.all_for_user("u1") == []


# --- the headline claim ----------------------------------------------------


def test_memory_persists_across_separate_conversations(
    client: TestClient, account, monkeypatch
):
    """Conversation 1 teaches the agent something; conversation 2 is a brand
    new conversation that recalls it. This is the whole point of the project."""
    stub_extraction(
        monkeypatch, ("Customer owns a WM-200 washing machine.", "product")
    )

    first = client.post("/conversations", json={}).json()["conversation_id"]
    client.post("/chat", json={
        "conversation_id": first,
        "message": "My name is Rahul and I own a WM-200 washing machine.",
    })

    # A different conversation, sharing nothing but the customer.
    second = client.post("/conversations", json={}).json()["conversation_id"]
    assert second != first
    body = client.post("/chat", json={
        "conversation_id": second,
        "message": "My washing machine is making a loud noise again.",
    }).json()

    assert body["memory_used"] is True
    assert any("WM-200" in m["content"] for m in body["memories"])


def test_a_different_customer_recalls_nothing(
    client: TestClient, account, other_account, monkeypatch
):
    stub_extraction(
        monkeypatch, ("Customer owns a WM-200 washing machine.", "product")
    )
    conversation = client.post("/conversations", json={}).json()["conversation_id"]
    client.post("/chat", json={
        "conversation_id": conversation,
        "message": "I own a WM-200 washing machine.",
    })

    other_conversation = other_account.client.post(
        "/conversations", json={}
    ).json()["conversation_id"]
    body = other_account.client.post("/chat", json={
        "conversation_id": other_conversation,
        "message": "My washing machine is making a loud noise.",
    }).json()

    assert body["memory_used"] is False
    assert body["memories"] == []


# --- regressions -----------------------------------------------------------


def test_context_grants_permission_to_use_what_was_retrieved():
    """Regression. The system prompt forbids stating customer specifics, so a
    context block that merely *lists* memories loses to that prohibition and
    the agent answers "I can't look that up" about a fact it was just handed.
    The block must be labelled as context the agent actually has."""
    block = build_context_block(
        [{"content": "Customer bought an air conditioner on 2026-09-18.",
          "type": "product"}],
        [],
    )
    assert block.startswith("Context")
    assert "answer from them directly" in block


def test_the_system_prompt_defers_to_context():
    """The other half of the same regression: the prohibition must be scoped
    to what the context does *not* cover, not absolute."""
    assert "Never tell a customer you cannot look something up" in SYSTEM_INSTRUCTION
    assert "Where the context does not cover it" in SYSTEM_INSTRUCTION


def test_extraction_is_told_todays_date():
    """A memory saying "yesterday" is wrong the next time it is read. The
    model cannot resolve that without knowing when today is."""
    instruction = extractor.extraction_instruction(date(2026, 9, 19))
    assert "2026-09-19" in instruction
    assert "yesterday" in instruction  # named as the thing to avoid


def test_follow_up_questions_retrieve_using_conversation_context():
    """Regression. "What was the issue reported?" has no subject of its own —
    the subject is in the previous turn. Searched alone it scores 0.17 against
    the right memory and retrieves nothing, so the agent truthfully reports it
    cannot find what it was told a minute earlier."""
    store.save("u1", "Customer reported that their air conditioner is not cooling.", "issue")

    bare = make_context(
        user_id="u1",
        user_message="What was the issue reported?",
        history=[("user", "What was the issue reported?")],
    )
    assert nodes.retrieve_memories(make_state(), runtime(bare)) == {"memories": []}

    with_context = make_context(
        user_id="u1",
        user_message="What was the issue reported?",
        history=[
            ("user", "I had bought an air conditioner, can you tell me the purchase date?"),
            ("assistant", "You bought it on 2026-09-18."),
            ("user", "What was the issue reported?"),
        ],
    )
    found = nodes.retrieve_memories(make_state(), runtime(with_context))["memories"]
    assert any("not cooling" in m["content"] for m in found)


def test_retrieval_query_uses_only_customer_turns():
    """The assistant's replies are long; letting them in would let the agent's
    own wording decide what gets retrieved."""
    query = nodes.retrieval_query(
        make_context(
            user_message="and the issue?",
            history=[
                ("user", "my air conditioner"),
                ("assistant", "A LONG ASSISTANT REPLY ABOUT LEVELLING FEET"),
                ("user", "and the issue?"),
            ],
        )
    )
    assert "my air conditioner" in query
    assert "and the issue?" in query
    assert "LEVELLING FEET" not in query


def test_retrieval_query_is_bounded():
    """An unbounded query would eventually be the whole conversation, and the
    current question would stop deciding what is retrieved."""
    history = [("user", f"turn {i}") for i in range(10)]
    query = nodes.retrieval_query(
        make_context(user_message="turn 9", history=history), turns=2
    )
    assert query.count("\n") == 2
    assert query.endswith("turn 9")


def test_retrieval_query_falls_back_to_the_bare_message():
    assert nodes.retrieval_query(
        make_context(user_message="hello", history=[])
    ) == "hello"
