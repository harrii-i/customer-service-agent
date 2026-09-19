"""Splitting a support document into retrievable chunks.

Retrieval returns chunks, not documents, so the chunk is the unit the customer
effectively sees quoted. Two rules follow from that:

- Split on blank lines, never mid-sentence. A chunk cut through the middle of
  a sentence reads as nonsense when it is quoted back.
- Overlap consecutive chunks, so a fact that straddles a boundary is complete
  in at least one of them.
"""

import re
from dataclasses import dataclass

from app.config import get_settings

settings = get_settings()


@dataclass(frozen=True)
class Chunk:
    title: str
    source: str
    text: str
    index: int


def document_title(markdown: str, fallback: str) -> str:
    """The first `# ` heading. It is what the UI cites as the source, so a
    document without one falls back to its filename rather than to nothing."""
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def split_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Greedily pack paragraphs up to `size`, then start a new chunk carrying
    `overlap` characters of tail context."""
    size = size or settings.knowledge_chunk_size
    overlap = overlap if overlap is not None else settings.knowledge_chunk_overlap

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= size or not current:
            current = candidate
            continue
        chunks.append(current)
        # Carry the tail forward so a fact spanning the boundary survives whole.
        tail = current[-overlap:] if overlap else ""
        current = f"{tail}\n\n{paragraph}" if tail else paragraph

    if current:
        chunks.append(current)
    return chunks


def chunk_document(markdown: str, source: str) -> list[Chunk]:
    title = document_title(markdown, fallback=source)
    return [
        Chunk(title=title, source=source, text=text, index=i)
        for i, text in enumerate(split_text(markdown))
    ]
