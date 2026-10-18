"""Retrieval metrics. Pure functions over ids — no ChromaDB, no dataset, no I/O.

Undefined is not zero, and the distinction matters here more than usual. A
question with no expected facts has no recall: scoring it 0.0 would drag the
average down for a question that was never supposed to retrieve anything, and
scoring it 1.0 would reward a retriever that returned the whole corpus. So
every metric returns `None` when it is undefined for the case in hand, and
`mean` skips those rather than counting them.

The three division-by-zero cases, decided explicitly:

  recall, no expected facts      -> None  (nothing to recall)
  precision, nothing retrieved   -> None  (no claims made, so none can be
                                    wrong; whether retrieving nothing was
                                    *right* is `abstention_is_correct`)
  reciprocal rank, gold exists
  but none retrieved             -> 0.0   (defined and genuinely bad, which
                                    is not the same as undefined)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def fact_recall_at_k(
    ranked_chunk_ids: Sequence[str], facts: Sequence[frozenset[str]], k: int
) -> float | None:
    """Fraction of required facts covered by the top `k` chunks.

    Fact-level rather than set-level on purpose. A multi-part question needs
    three facts from three documents; finding two of them is materially
    better than finding none, and an all-or-nothing score would hide that.

    A fact counts as covered when *any* chunk containing it is retrieved —
    consecutive chunks overlap, so one fact can sit in two of them and
    requiring both would penalise a correct result.
    """
    if not facts:
        return None
    if k <= 0:
        return 0.0
    top = set(ranked_chunk_ids[:k])
    covered = sum(1 for fact in facts if fact & top)
    return covered / len(facts)


def document_precision(
    retrieved_documents: Sequence[str], expected_documents: Iterable[str]
) -> float | None:
    """Of the documents cited to the customer, how many were expected.

    Document-level, not chunk-level, and that is a deliberate limit. The
    ground truth marks chunks that carry a required fact; a chunk can be
    relevant without carrying one, so chunk precision would count correct
    retrievals as errors. At document level the expectation is complete.
    """
    if not retrieved_documents:
        return None
    expected = set(expected_documents)
    hits = sum(1 for document in retrieved_documents if document in expected)
    return hits / len(retrieved_documents)


def document_recall(
    retrieved_documents: Iterable[str], expected_documents: Sequence[str]
) -> float | None:
    """How much of what should have been cited was."""
    expected = set(expected_documents)
    if not expected:
        return None
    retrieved = set(retrieved_documents)
    return len(expected & retrieved) / len(expected)


def reciprocal_rank(
    ranked_chunk_ids: Sequence[str], facts: Sequence[frozenset[str]]
) -> float | None:
    """1 / the rank of the first chunk that covers any required fact.

    Measures whether the right chunk is at the top rather than merely
    present, which is what matters when only the strongest chunk per document
    survives deduplication into the prompt.
    """
    if not facts:
        return None
    wanted = set().union(*facts)
    for position, chunk in enumerate(ranked_chunk_ids, start=1):
        if chunk in wanted:
            return 1.0 / position
    # Defined, and bad: the gold exists and was not found.
    return 0.0


def abstention_is_correct(
    retrieved_documents: Sequence[str], expects_abstention: bool
) -> bool | None:
    """Whether a question that should retrieve nothing did.

    `None` for questions that are supposed to retrieve something — this is a
    pass/fail for the abstention cases only, not a score everything gets.
    """
    if not expects_abstention:
        return None
    return not retrieved_documents


def mean(values: Iterable[float | None]) -> float | None:
    """Average of the defined values. `None` when none are defined.

    Skipping rather than zero-filling is the whole point of returning `None`
    above; averaging it back in as 0.0 here would undo it.
    """
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)
