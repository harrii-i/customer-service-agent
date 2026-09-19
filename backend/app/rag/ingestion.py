"""Loading the support documentation into ChromaDB.

Ingestion is a full rebuild, not an append: chunk ids are derived from the
source filename and chunk index, so re-running after editing a document
replaces its chunks rather than leaving stale copies alongside the new ones.
Stale documentation that still answers questions is worse than none.
"""

import logging
from pathlib import Path

from app.chroma import knowledge_collection
from app.rag.chunker import Chunk, chunk_document

logger = logging.getLogger("csam.rag.ingestion")

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"


def load_chunks(directory: Path | None = None) -> list[Chunk]:
    directory = directory or KNOWLEDGE_DIR
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_document(path.read_text(encoding="utf-8"), path.name))
    return chunks


def ingest(directory: Path | None = None) -> int:
    """Rebuild the knowledge collection. Returns the number of chunks stored."""
    chunks = load_chunks(directory)
    if not chunks:
        logger.warning("no markdown found to ingest")
        return 0

    collection = knowledge_collection()
    # Drop whatever was there before: a document renamed or a section deleted
    # must not survive as an orphaned chunk.
    existing = collection.get(include=[])["ids"]
    if existing:
        collection.delete(ids=existing)

    collection.add(
        ids=[f"{c.source}::{c.index}" for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {"title": c.title, "source": c.source, "index": c.index} for c in chunks
        ],
    )
    logger.info("ingested %d chunks from %d documents", len(chunks),
                len({c.source for c in chunks}))
    return len(chunks)
