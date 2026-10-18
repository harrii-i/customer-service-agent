"""Retrieval evaluation against `backend/evals/dataset.json`.

Two layers, measuring different things:

  Layer A, chunk level. The raw ranking out of ChromaDB, before the
  similarity floor. Answers "is the right chunk near the top?" — Recall@k and
  MRR. Diagnostic: it explains *why* layer B did what it did.

  Layer B, document level. What `rag.retriever.retrieve` actually returns, so
  after the floor and after deduplication to one entry per document. This is
  what the agent is given and what the customer is shown, so it is the layer
  that describes the product.

Questions are asked bare, exactly as the dataset words them, rather than
through `nodes.retrieval_query` — that prepends the previous two customer
turns, which is right in a conversation and would make these results depend
on the order they were run in.

No numeric quality floors are asserted. The scores are reported, and the only
regressions pinned are the unambiguous ones: that the ground truth still maps
onto the documents, and that questions nothing in the corpus answers retrieve
nothing. Asserting invented thresholds before measuring is how a suite ends
up defending its own bugs.

Nothing here reaches Gemini, and nothing touches the developer's ChromaDB:
`conftest` points CHROMA_PATH at a throwaway directory for the whole run and
empties both collections between tests.
"""

import pytest

from app.chroma import knowledge_collection, similarity
from app.config import get_settings
from app.rag import retriever
from app.rag.ingestion import ingest
from evals import gold as gold_module
from evals import metrics

pytestmark = pytest.mark.evals

SETTINGS = get_settings()
TOP_K = SETTINGS.knowledge_top_k
K_VALUES = (1, 3, 5)


@pytest.fixture
def ingested():
    """The shipped documentation in the throwaway collection, for one test.

    Function-scoped because `_clean_vectors` empties the collections between
    tests; anything wider would quietly serve a later test an empty corpus.
    """
    return ingest()


@pytest.fixture(scope="module")
def gold():
    """Ground truth. No ChromaDB involved — pure dataset plus chunker."""
    return gold_module.build_gold()


def ranked_chunk_ids(question: str, k: int = TOP_K) -> list[str]:
    """Layer A: the raw ranking, unfiltered.

    No similarity floor on purpose. Layer B measures what survives the floor;
    this measures the ranking the floor is applied to, so that a bad score
    can be read as either "ranked badly" or "ranked fine, filtered out".
    """
    collection = knowledge_collection()
    count = collection.count()
    if count == 0:
        return []
    result = collection.query(query_texts=[question], n_results=min(k, count))
    return list(result["ids"][0])


def retrieved_documents(question: str) -> list[str]:
    """Layer B: the documents the agent is actually handed, best first."""
    return [hit["source"] for hit in retriever.retrieve(question)]


# --- the ground truth itself ----------------------------------------------


def test_every_quote_still_maps_onto_a_chunk(gold):
    """A regression on the dataset, not the retriever. An unmappable quote
    means a document was edited and the ground truth now describes text that
    no longer exists — which would silently corrupt every score below."""
    broken = {
        question.id: question.unmapped_quotes
        for question in gold
        if question.unmapped_quotes
    }
    assert not broken, f"dataset quotes no longer found in knowledge/: {broken}"


def test_questions_with_sources_have_gold_chunks(gold):
    for question in gold:
        if question.documents:
            assert question.facts, f"{question.id} names sources but derived no facts"
            assert not question.expects_abstention


def test_abstention_cases_are_exactly_the_sourceless_questions(gold):
    abstaining = {question.id for question in gold if question.expects_abstention}
    assert abstaining == {"UNA-03", "OOS-01", "OOS-02", "OOS-03"}


# --- the unambiguous regression -------------------------------------------


def test_questions_with_no_supporting_document_retrieve_nothing(gold, ingested):
    """The one retrieval behaviour worth asserting outright.

    For the out-of-scope three this is defence in depth: the guardrail
    normally deflects them before retrieval runs, but it fails open, so on a
    bad Gemini minute the similarity floor is what stops a fabricated
    citation. UNA-03 is the real retrieval case — nothing in the corpus
    describes a spin speed, so nothing should come back.
    """
    for question in gold:
        if not question.expects_abstention:
            continue
        documents = retrieved_documents(question.question)
        assert metrics.abstention_is_correct(documents, True), (
            f"{question.id} cited {documents} for a question with no "
            f"supporting document: {question.question!r}"
        )


def test_retrieved_documents_are_always_real_files(gold, ingested):
    """Whatever the scores, a citation must name a document that exists."""
    known = set(gold_module.load_chunks())
    for question in gold:
        for document in retrieved_documents(question.question):
            assert document in known, f"{question.id} cited unknown {document!r}"


# --- the measurement run ---------------------------------------------------


def test_report_retrieval_metrics(gold, ingested, capsys):
    """Measure and print. Asserts nothing numeric, by design.

    Run with `-s` to see the table:
        pytest tests/test_eval_retrieval.py -m evals -s
    """
    scored = [question for question in gold if question.counts_toward_retrieval_quality]
    rows = []

    for question in gold:
        ranked = ranked_chunk_ids(question.question)
        documents = retrieved_documents(question.question)
        rows.append(
            {
                "id": question.id,
                "category": question.category,
                "recall": {
                    k: metrics.fact_recall_at_k(ranked, question.facts, k)
                    for k in K_VALUES
                },
                "mrr": metrics.reciprocal_rank(ranked, question.facts),
                "precision": metrics.document_precision(documents, question.documents),
                "doc_recall": metrics.document_recall(documents, question.documents),
                "abstained": metrics.abstention_is_correct(
                    documents, question.expects_abstention
                ),
                "documents": documents,
                "facts": len(question.facts),
            }
        )

    by_id = {row["id"]: row for row in rows}
    scored_rows = [by_id[question.id] for question in scored]

    def show(value):
        return " —  " if value is None else f"{value:.2f}"

    lines = [
        "",
        "Retrieval evaluation — backend/evals/dataset.json",
        f"corpus: {ingested} chunks   top_k: {TOP_K}   "
        f"similarity floor: {SETTINGS.knowledge_similarity_threshold}",
        "",
        f"{'id':9} {'category':13} {'facts':>5} {'R@1':>5} {'R@3':>5} {'R@5':>5} "
        f"{'MRR':>5} {'P':>5} {'docR':>5}  documents cited",
    ]
    for row in rows:
        lines.append(
            f"{row['id']:9} {row['category']:13} {row['facts']:>5} "
            f"{show(row['recall'][1]):>5} {show(row['recall'][3]):>5} "
            f"{show(row['recall'][5]):>5} {show(row['mrr']):>5} "
            f"{show(row['precision']):>5} {show(row['doc_recall']):>5}  "
            + (", ".join(row["documents"]) or "(none)")
        )

    lines += [
        "",
        f"averages over {len(scored_rows)} scored questions "
        "(out-of-scope and abstention cases excluded):",
    ]
    for k in K_VALUES:
        lines.append(
            f"  fact Recall@{k}        {show(mean_of(scored_rows, lambda r, k=k: r['recall'][k]))}"
        )
    lines += [
        f"  MRR                  {show(mean_of(scored_rows, lambda r: r['mrr']))}",
        f"  document precision   {show(mean_of(scored_rows, lambda r: r['precision']))}",
        f"  document recall      {show(mean_of(scored_rows, lambda r: r['doc_recall']))}",
        "",
        "abstention cases (asserted separately, not averaged):",
    ]
    for row in rows:
        if row["abstained"] is not None:
            verdict = "ok" if row["abstained"] else "FAILED"
            lines.append(
                f"  {row['id']:9} {verdict:7} cited: "
                + (", ".join(row["documents"]) or "(none)")
            )

    with capsys.disabled():
        print("\n".join(lines))

    # Structural only: every question was evaluated, and the scored set is
    # the one the audit defined. No numeric floor is asserted here.
    assert len(rows) == len(gold)
    assert {row["id"] for row in scored_rows}.isdisjoint(
        {question.id for question in gold if question.is_out_of_scope}
    )


def mean_of(rows, pick):
    return metrics.mean(pick(row) for row in rows)


# --- metric unit tests (no ChromaDB, no dataset) ---------------------------

A, B, C = "doc.md::0", "doc.md::1", "other.md::0"


def test_recall_counts_each_fact_once_when_covered():
    facts = [frozenset({A}), frozenset({C})]
    assert metrics.fact_recall_at_k([A, C], facts, 5) == 1.0
    assert metrics.fact_recall_at_k([A], facts, 5) == 0.5
    assert metrics.fact_recall_at_k([], facts, 5) == 0.0


def test_recall_respects_the_cut_off():
    facts = [frozenset({C})]
    ranked = [A, B, C]
    assert metrics.fact_recall_at_k(ranked, facts, 1) == 0.0
    assert metrics.fact_recall_at_k(ranked, facts, 3) == 1.0


def test_a_fact_in_either_of_two_overlapping_chunks_counts_once():
    """Consecutive chunks overlap, so one fact can sit in both. Retrieving
    either is success, and retrieving both is not double credit."""
    facts = [frozenset({A, B})]
    assert metrics.fact_recall_at_k([A], facts, 5) == 1.0
    assert metrics.fact_recall_at_k([B], facts, 5) == 1.0
    assert metrics.fact_recall_at_k([A, B], facts, 5) == 1.0


def test_multi_fact_questions_score_partially():
    """The reason recall is fact-level: two of three found is not the same as
    none found, and an all-or-nothing score would hide the difference."""
    facts = [frozenset({A}), frozenset({B}), frozenset({C})]
    assert metrics.fact_recall_at_k([A, B], facts, 5) == pytest.approx(2 / 3)


def test_recall_is_undefined_with_no_expected_facts():
    """Not 0.0 — there was nothing to recall. Scoring it would punish a
    question that was never supposed to retrieve anything."""
    assert metrics.fact_recall_at_k([A], [], 5) is None
    assert metrics.fact_recall_at_k([], [], 5) is None


def test_recall_at_zero_is_zero_not_undefined():
    """k=0 is a real cut-off that retrieves nothing, unlike an empty gold."""
    assert metrics.fact_recall_at_k([A], [frozenset({A})], 0) == 0.0


def test_document_precision_and_recall():
    assert metrics.document_precision(["a.md", "b.md"], {"a.md"}) == 0.5
    assert metrics.document_recall(["a.md"], {"a.md", "b.md"}) == 0.5
    assert metrics.document_recall(["a.md", "b.md"], {"a.md"}) == 1.0


def test_precision_is_undefined_when_nothing_was_retrieved():
    """No claims were made, so none can be wrong. Whether retrieving nothing
    was *correct* is a different question — `abstention_is_correct`."""
    assert metrics.document_precision([], {"a.md"}) is None


def test_document_recall_is_undefined_with_no_expected_documents():
    assert metrics.document_recall(["a.md"], set()) is None
    assert metrics.document_recall([], set()) is None


def test_reciprocal_rank_rewards_position():
    facts = [frozenset({C})]
    assert metrics.reciprocal_rank([C, A, B], facts) == 1.0
    assert metrics.reciprocal_rank([A, C, B], facts) == 0.5
    assert metrics.reciprocal_rank([A, B, C], facts) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_gold_exists_but_is_missed():
    """Defined and bad, which is not the same as undefined."""
    assert metrics.reciprocal_rank([A, B], [frozenset({C})]) == 0.0
    assert metrics.reciprocal_rank([], [frozenset({C})]) == 0.0


def test_reciprocal_rank_is_undefined_without_gold():
    assert metrics.reciprocal_rank([A], []) is None


def test_abstention_only_applies_where_it_is_expected():
    assert metrics.abstention_is_correct([], True) is True
    assert metrics.abstention_is_correct(["a.md"], True) is False
    assert metrics.abstention_is_correct(["a.md"], False) is None
    assert metrics.abstention_is_correct([], False) is None


def test_mean_skips_undefined_values():
    assert metrics.mean([1.0, None, 0.0]) == 0.5
    assert metrics.mean([None, None]) is None
    assert metrics.mean([]) is None


# --- gold-derivation unit tests (no ChromaDB) ------------------------------


def test_quote_maps_to_the_chunk_containing_it():
    chunks = {"a.md": ["first chunk text", "second  chunk\ntext"]}
    assert gold_module.chunks_containing("second chunk text", "a.md", chunks) == {
        "a.md::1"
    }


def test_quote_mapping_is_whitespace_insensitive():
    """Markdown is hard-wrapped and chunking re-joins paragraphs, so raw
    substring matching would miss quotes that span a line break."""
    chunks = {"a.md": ["a two-year\nmanufacturer's warranty applies"]}
    assert gold_module.chunks_containing(
        "two-year manufacturer's warranty", "a.md", chunks
    ) == {"a.md::0"}


def test_an_unlocatable_quote_maps_to_nothing_rather_than_guessing():
    chunks = {"a.md": ["nothing relevant here"]}
    assert gold_module.chunks_containing("absent text", "a.md", chunks) == frozenset()
    assert gold_module.chunks_containing("x", "missing.md", chunks) == frozenset()


def test_gold_records_unmapped_quotes_instead_of_dropping_them(tmp_path):
    """An unmappable quote is a fault in the ground truth. Silently skipping
    it would leave the question scored against fewer facts than it needs."""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "a.md").write_text("# A\n\nreal sentence here.\n", encoding="utf-8")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        '{"questions": [{"id": "X-1", "category": "answerable",'
        ' "question": "q?", "sources": ["a.md"], "source_quotes": ['
        '{"file": "a.md", "quote": "real sentence"},'
        '{"file": "a.md", "quote": "invented sentence"}]}]}',
        encoding="utf-8",
    )

    question = gold_module.build_gold(dataset, knowledge)[0]
    assert question.facts == (frozenset({"a.md::0"}),)
    assert question.unmapped_quotes == (("a.md", "invented sentence"),)


def test_a_question_without_sources_expects_abstention(tmp_path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "a.md").write_text("# A\n\ntext.\n", encoding="utf-8")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        '{"questions": [{"id": "X-2", "category": "unanswerable",'
        ' "question": "q?", "sources": [], "source_quotes": []}]}',
        encoding="utf-8",
    )

    question = gold_module.build_gold(dataset, knowledge)[0]
    assert question.expects_abstention is True
    assert question.facts == ()
    assert question.counts_toward_retrieval_quality is False


def test_out_of_scope_questions_are_excluded_from_quality_scoring(gold):
    for question in gold:
        if question.is_out_of_scope:
            assert question.counts_toward_retrieval_quality is False
