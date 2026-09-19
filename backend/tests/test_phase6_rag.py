"""Phase 6 acceptance: answers grounded in the company's documentation."""

from pathlib import Path

import pytest

from app.rag import retriever
from app.rag.chunker import chunk_document, document_title, split_text
from app.rag.ingestion import KNOWLEDGE_DIR, ingest, load_chunks


# --- chunking --------------------------------------------------------------


def test_title_comes_from_the_first_heading():
    assert document_title("# Warranty Policy\n\nText.", "f.md") == "Warranty Policy"


def test_title_falls_back_to_the_filename():
    """A chunk is cited to the customer, so it always needs a name."""
    assert document_title("No heading here.", "warranty.md") == "warranty.md"


def test_split_never_cuts_mid_sentence():
    text = "\n\n".join(f"Paragraph number {i} with some words in it." for i in range(20))
    chunks = split_text(text, size=120, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip().endswith(".")


def test_consecutive_chunks_overlap():
    """A fact straddling a boundary must be whole in at least one chunk."""
    text = "\n\n".join(f"Paragraph {i} carries distinct content." for i in range(12))
    chunks = split_text(text, size=100, overlap=30)
    tail = chunks[0][-30:]
    assert tail in chunks[1]


def test_a_paragraph_longer_than_the_chunk_size_is_still_emitted():
    """Greedy packing must not silently drop an oversized paragraph."""
    long_paragraph = "word " * 400
    chunks = split_text(long_paragraph, size=100, overlap=10)
    assert chunks and "word" in chunks[0]


def test_chunk_carries_title_and_source():
    chunks = chunk_document("# Returns\n\nSome text.", "returns.md")
    assert chunks[0].title == "Returns"
    assert chunks[0].source == "returns.md"


# --- ingestion -------------------------------------------------------------


def test_shipped_knowledge_base_is_present_and_ingestible():
    assert KNOWLEDGE_DIR.is_dir()
    assert list(KNOWLEDGE_DIR.glob("*.md")), "no documentation shipped"
    assert load_chunks()


def test_ingest_is_a_rebuild_not_an_append():
    """Re-running after an edit must not leave stale chunks that still answer
    questions."""
    first = ingest()
    second = ingest()
    assert first == second

    from app.chroma import knowledge_collection

    assert knowledge_collection().count() == second


def test_ingest_of_an_empty_directory_stores_nothing(tmp_path: Path):
    assert ingest(tmp_path) == 0


# --- retrieval -------------------------------------------------------------


@pytest.fixture
def ingested():
    ingest()


def test_retrieval_finds_the_right_document(ingested):
    hits = retriever.retrieve("How long is the warranty on my appliance?")
    assert hits
    assert any(h["title"] == "Warranty Policy" for h in hits)
    assert all({"title", "source", "snippet"} <= set(h) for h in hits)


def test_retrieval_returns_nothing_for_an_unrelated_question(ingested):
    """Without a similarity floor the nearest chunks are returned regardless,
    and the agent cites a delivery policy when asked about geography."""
    assert retriever.retrieve("What is the capital of France?") == []


def test_retrieval_without_an_ingested_collection_is_empty():
    assert retriever.retrieve("How long is the warranty?") == []


def test_blank_question_retrieves_nothing(ingested):
    assert retriever.retrieve("   ") == []


def test_sources_are_documents_not_chunks(ingested):
    """One document usually matches several chunks. The customer is shown
    "Source: Warranty Policy", so the same document must not appear twice."""
    hits = retriever.retrieve("How long is the warranty and can I return it?")
    titles = [h["title"] for h in hits]
    assert len(titles) == len(set(titles))


def test_results_are_ordered_best_first(ingested):
    hits = retriever.retrieve("My washing machine bangs loudly on the spin cycle.")
    assert hits[0]["title"] == "Washing Machine Troubleshooting"


def test_a_passing_remark_cites_nothing(ingested):
    """"Thanks, that fixed it" is not a question, and an answer that appears
    sourced when it is not is worse than an unsourced one."""
    assert retriever.retrieve("Thanks, that fixed it!") == []


def test_internal_score_is_not_leaked_to_the_api(ingested):
    """`_score` is a ranking detail; SourceOut has no such field."""
    for hit in retriever.retrieve("How long is the warranty?"):
        assert set(hit) == {"title", "source", "snippet"}
