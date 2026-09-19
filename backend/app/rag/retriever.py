"""Retrieving support documentation for a question.

Returns plain dicts shaped like the API's `SourceOut`, so what the agent used
and what the customer is shown are the same objects — the UI can never display
a citation the agent did not actually receive.
"""

import logging

from app.chroma import knowledge_collection, similarity
from app.config import get_settings

logger = logging.getLogger("csam.rag")

settings = get_settings()

# A chunk long enough to be useful in the prompt, short enough that the UI can
# show it without becoming a wall of text.
SNIPPET_CHARS = 400


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """Top matching documentation chunks, weakest matches discarded.

    Without a floor, every question retrieves *something* — the nearest five
    chunks exist whether or not they are relevant — and the agent would then
    cite a delivery policy when asked about a noise. An empty list is the
    correct answer to a question the documentation does not cover.
    """
    question = question.strip()
    if not question:
        return []

    collection = knowledge_collection()
    if collection.count() == 0:
        logger.warning("knowledge collection is empty; run scripts/ingest_knowledge.py")
        return []

    top_k = top_k or settings.knowledge_top_k
    result = collection.query(
        query_texts=[question], n_results=min(top_k, collection.count())
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    # One document usually matches on several chunks. The customer is shown
    # "Source: Warranty Policy", not "Source: Warranty Policy (chunk 3 of 4)",
    # so collapse to one entry per document, keeping its strongest chunk as the
    # snippet. Results stay ordered best-first.
    best: dict[str, dict] = {}
    for text, metadata, distance in zip(documents, metadatas, distances):
        score = similarity(distance)
        if score < settings.knowledge_similarity_threshold:
            continue
        title = metadata["title"]
        if title in best and best[title]["_score"] >= score:
            continue
        best[title] = {
            "title": title,
            "source": metadata["source"],
            "snippet": text[:SNIPPET_CHARS].strip(),
            "_score": score,
        }

    ranked = sorted(best.values(), key=lambda h: h["_score"], reverse=True)
    return [{k: v for k, v in hit.items() if k != "_score"} for hit in ranked]
