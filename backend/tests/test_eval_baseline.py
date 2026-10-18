"""Evaluation baseline: a whole `/chat` turn, judged on what grounded it.

The phase suites test the retrievers directly. These are the first tests that
take an authenticated turn end to end and check *both* halves of the claim a
response makes:

- what the API reports it used — `sources` and `memories`, the same objects
  the UI shows the customer;
- what the model was actually handed — the context block built from them.

Both matter, and neither is sufficient. A response can report the right
citation while the retrieved text never reaches the prompt, and a prompt can
carry a memory the response never admits to. An evaluation that reads only one
side would pass in either case, so every test here asserts on both.

Nothing new is stubbed: `conftest` already makes the suite offline and free,
and `prompt_contexts` records generation through the same seam.
"""

import re

import pytest
from fastapi.testclient import TestClient

from app.memory import store
from app.rag.chunker import document_title
from app.rag.ingestion import KNOWLEDGE_DIR, ingest


@pytest.fixture
def ingested():
    """The shipped documentation, in the vector store for one test.

    Function-scoped on purpose: `_clean_vectors` empties both collections
    between tests, so a wider scope would ingest once and then silently serve
    the rest of the module an empty collection, making every assertion below
    depend on execution order.
    """
    return ingest()


def start_conversation(client: TestClient) -> str:
    return client.post("/conversations", json={}).json()["conversation_id"]


def ask(client: TestClient, conversation_id: str, message: str) -> dict:
    response = client.post(
        "/chat", json={"conversation_id": conversation_id, "message": message}
    )
    assert response.status_code == 200, response.text
    return response.json()


def assert_sources_reached_the_prompt(sources: list[dict], context: str | None):
    """Every citation the customer is shown was in front of the model.

    `build_context_block` renders a document as `- [Title] snippet`, so the
    exact rendered pair is what gets looked for — not merely the title, which
    could match a passing mention.
    """
    assert context is not None, "documents were retrieved but no context was sent"
    for source in sources:
        assert f"[{source['title']}] {source['snippet']}" in context


def normalise(text: str) -> str:
    """Whitespace-insensitive comparison.

    Chunking strips paragraphs and re-joins them with a blank line, so a chunk
    is not a byte-for-byte slice of the file it came from. The words and their
    order are, and that is what "this quote is really in that document" means.
    """
    return re.sub(r"\s+", " ", text).strip()


# --- 1. a question the documentation answers -------------------------------


def test_a_documented_question_cites_the_right_document(
    client: TestClient, account, ingested, prompt_contexts
):
    """The baseline positive case. Asking something the knowledge base covers
    must produce the matching citation *and* ground the model in it.

    Note what this is not: a test of the answer's wording. The model is
    stubbed, so what is evaluated is the evidence the answer was built on —
    the part that is deterministic and therefore repeatable.
    """
    conversation_id = start_conversation(client)
    body = ask(client, conversation_id, "How long is the warranty on my washing machine?")

    titles = [source["title"] for source in body["sources"]]
    assert titles, "a documented question returned no sources"
    assert titles[0] == "Warranty Policy", f"best source was {titles[0]!r}"

    assert_sources_reached_the_prompt(body["sources"], prompt_contexts[-1])

    # Grounding came from documentation, not from a memory this fresh account
    # does not have.
    assert body["memory_used"] is False
    assert body["memories"] == []


# --- 2. a question the documentation does not answer -----------------------


def test_an_undocumented_question_cites_nothing_and_sends_no_context(
    client: TestClient, account, ingested, prompt_contexts
):
    """In scope, but unanswerable from what we hold.

    The nearest chunks always exist, so the failure this guards against is the
    agent citing a plausible-looking policy at a question it cannot answer. No
    sources is the correct result — and the context block must be absent
    entirely, because an empty "Context" heading is an invitation to invent
    one.
    """
    conversation_id = start_conversation(client)
    body = ask(client, conversation_id, "What is the maximum spin speed of the WM-900?")

    assert body["sources"] == []
    assert body["memories"] == []
    assert body["memory_used"] is False
    assert prompt_contexts == [None], "the model was given context it should not have"


# --- 4. a memory that has nothing to do with the question ------------------


def test_an_irrelevant_memory_never_reaches_the_prompt(
    client: TestClient, account, ingested, prompt_contexts
):
    """Holding a fact about a customer is not a reason to raise it.

    `issue` is the type under test on purpose: `identity` and `product` are
    profile facts and bypass the relevance floor by design, so they would
    prove nothing here. An old complaint must not surface in a delivery
    question — and the stronger claim is the prompt one, since retrieval
    returning nothing is not by itself proof that the prompt stayed clean.
    """
    store.save(
        account.id, "Customer's washing machine bangs loudly on spin.", "issue"
    )

    conversation_id = start_conversation(client)
    body = ask(client, conversation_id, "When do you deliver?")

    assert body["memories"] == []
    assert body["memory_used"] is False
    context = prompt_contexts[-1] or ""
    assert "bangs loudly on spin" not in context
    # Not merely absent from the text: the whole "what you know about this
    # customer" section must not have been built.
    assert "remembered from earlier conversations" not in context

    # The unrelated question is still answered from documentation: this test
    # must fail because the memory leaked, never because retrieval went quiet.
    assert "Delivery and Installation" in [s["title"] for s in body["sources"]]


# --- 5. citation integrity -------------------------------------------------


def test_cited_snippets_come_from_the_document_they_name(
    client: TestClient, account, ingested, prompt_contexts
):
    """A citation points at text that genuinely exists in the named file.

    Shape is already tested elsewhere; this is about truthfulness. For every
    source returned: the file exists, its own heading is the title the
    customer is shown, and every paragraph of the quoted snippet really
    appears in it.
    """
    conversation_id = start_conversation(client)
    body = ask(client, conversation_id, "My washing machine bangs loudly on the spin cycle.")

    sources = body["sources"]
    assert sources
    assert sources[0]["title"] == "Washing Machine Troubleshooting"

    titles = [source["title"] for source in sources]
    assert len(titles) == len(set(titles)), "the same document was cited twice"

    for source in sources:
        path = KNOWLEDGE_DIR / source["source"]
        assert path.is_file(), f"cited a file that does not exist: {source['source']}"

        markdown = path.read_text(encoding="utf-8")
        assert document_title(markdown, source["source"]) == source["title"]

        # A snippet may open with an overlap tail carried from the previous
        # chunk and close mid-sentence at the length cap, so it is checked
        # paragraph by paragraph: each one is contiguous text from this file.
        document = normalise(markdown)
        for paragraph in source["snippet"].split("\n\n"):
            if paragraph.strip():
                assert normalise(paragraph) in document, (
                    f"{source['title']} does not contain {normalise(paragraph)[:60]!r}"
                )

    assert_sources_reached_the_prompt(sources, prompt_contexts[-1])


def test_citations_survive_a_reload_unchanged(
    client: TestClient, account, ingested, prompt_contexts
):
    """What the customer sees after a refresh is what the agent actually used.

    The citations are persisted on the assistant message rather than
    recomputed, so a reload must not be able to show a different, newer or
    emptier set than the live response did.
    """
    conversation_id = start_conversation(client)
    body = ask(client, conversation_id, "My washing machine bangs loudly on the spin cycle.")
    assert body["sources"]

    detail = client.get(f"/conversations/{conversation_id}").json()
    assistant = [m for m in detail["messages"] if m["role"] == "assistant"][-1]

    assert assistant["sources"] == body["sources"]
    assert assistant["memories"] == body["memories"]
    # The customer's own turn carries no provenance of its own.
    user_message = [m for m in detail["messages"] if m["role"] == "user"][-1]
    assert user_message["sources"] == []
    assert user_message["memories"] == []
