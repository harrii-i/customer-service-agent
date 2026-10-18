"""Turning the dataset's quotes into expected chunk ids.

The dataset records, for each question, verbatim extracts from the knowledge
documents that a correct answer has to rest on. Chunking is deterministic, so
those extracts can be located in the chunks they ended up in, and the dataset
yields chunk-level ground truth without anyone hand-labelling 234
question-chunk pairs — and without that labelling going stale the next time a
document is edited.

Nothing here touches ChromaDB or Gemini. It reads two things off disk, the
dataset and the Markdown, and runs the same `chunk_document` the ingestion
script runs. That keeps the ground truth honest: if the chunker changes, the
expected ids change with it, rather than silently describing a corpus that no
longer exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.chunker import chunk_document
from app.rag.ingestion import KNOWLEDGE_DIR

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"

# Categories whose questions never reach the retriever in production: the
# guardrail deflects them at the first node. They are still checked for
# abstention — the guardrail fails open, so the similarity floor is the second
# line of defence — but they are not retrieval-quality cases and are kept out
# of the averages.
OUT_OF_SCOPE = "out_of_scope"


def normalise(text: str) -> str:
    """Whitespace-insensitive comparison.

    Chunking strips paragraphs and re-joins them with a blank line, so a chunk
    is not a byte-for-byte slice of its file. The words and their order are,
    and that is what "this quote is in that chunk" has to mean.
    """
    return re.sub(r"\s+", " ", text).strip()


def chunk_id(source: str, index: int) -> str:
    """The id ingestion gives a chunk. Defined here rather than spelled out,
    so the evaluator and `app.rag.ingestion` cannot drift apart."""
    return f"{source}::{index}"


@dataclass(frozen=True)
class GoldQuestion:
    """One question, with what retrieval is expected to find for it."""

    id: str
    category: str
    question: str
    # One entry per required fact: the set of chunk ids that contain that
    # fact. A fact is covered when *any* of its chunks is retrieved, because
    # the overlap between consecutive chunks means one fact can legitimately
    # live in two of them.
    facts: tuple[frozenset[str], ...]
    documents: frozenset[str]
    expects_abstention: bool
    # Quotes that could not be located in any chunk. Never silently dropped:
    # an unmappable quote means the dataset and the documents have diverged,
    # which is a fault in the ground truth and has to be visible.
    unmapped_quotes: tuple[tuple[str, str], ...] = ()

    @property
    def is_out_of_scope(self) -> bool:
        return self.category == OUT_OF_SCOPE

    @property
    def counts_toward_retrieval_quality(self) -> bool:
        """Whether this question's scores belong in the averages.

        A question with expected sources does, even when the *answer* is
        unanswerable: UNA-01 asks for a figure the documents omit, but the
        surrounding policy is documented and retrieval is supposed to find it.
        Unanswerability is a property of the generated answer, not of
        retrieval, and scoring it here would punish correct behaviour.
        """
        return bool(self.facts) and not self.is_out_of_scope


def load_chunks(knowledge_dir: Path | None = None) -> dict[str, list[str]]:
    """Every document's chunks, keyed by filename, in ingestion order.

    Deliberately re-derived rather than read back from ChromaDB: the point is
    to know what *should* be in the collection, independently of what is.
    """
    directory = knowledge_dir or KNOWLEDGE_DIR
    chunks: dict[str, list[str]] = {}
    for path in sorted(directory.glob("*.md")):
        document = chunk_document(path.read_text(encoding="utf-8"), path.name)
        chunks[path.name] = [chunk.text for chunk in document]
    return chunks


def chunks_containing(quote: str, source: str, chunks: dict[str, list[str]]) -> frozenset[str]:
    """Which chunks of `source` contain `quote`. Empty when none do."""
    needle = normalise(quote)
    texts = chunks.get(source, [])
    return frozenset(
        chunk_id(source, index)
        for index, text in enumerate(texts)
        if needle in normalise(text)
    )


def build_gold(
    dataset_path: Path | None = None, knowledge_dir: Path | None = None
) -> list[GoldQuestion]:
    """The dataset, resolved against the documents as they are on disk now."""
    data = json.loads((dataset_path or DATASET_PATH).read_text(encoding="utf-8"))
    chunks = load_chunks(knowledge_dir)

    gold: list[GoldQuestion] = []
    for item in data["questions"]:
        facts: list[frozenset[str]] = []
        unmapped: list[tuple[str, str]] = []
        for quote in item.get("source_quotes", []):
            found = chunks_containing(quote["quote"], quote["file"], chunks)
            if found:
                facts.append(found)
            else:
                unmapped.append((quote["file"], quote["quote"]))

        documents = frozenset(item.get("sources", []))
        gold.append(
            GoldQuestion(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                facts=tuple(facts),
                documents=documents,
                # No expected source document means nothing in the corpus
                # answers this, so the correct retrieval is none at all.
                expects_abstention=not documents,
                unmapped_quotes=tuple(unmapped),
            )
        )
    return gold
