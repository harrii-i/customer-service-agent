"""Storage for long-term customer memories in ChromaDB.

Every function here takes a `user_id` and every query filters on it. That is
the isolation guarantee, and it is enforced in the query rather than by a
caller remembering to check afterwards — the same reasoning as
`get_conversation_for_user` on the SQL side.
"""

import logging
import uuid
from collections.abc import Sequence

from app.chroma import memory_collection, similarity
from app.config import get_settings

logger = logging.getLogger("csam.memory.store")

settings = get_settings()


def save(user_id: str, content: str, memory_type: str) -> str:
    """Store one durable fact. Returns its id."""
    memory_id = str(uuid.uuid4())
    memory_collection().add(
        ids=[memory_id],
        documents=[content],
        metadatas=[{"user_id": user_id, "type": memory_type}],
    )
    # The fact itself is not logged: it is customer data.
    logger.info("memory saved user_id=%s type=%s", user_id, memory_type)
    return memory_id


def save_many(user_id: str, memories: Sequence[dict]) -> list[str]:
    if not memories:
        return []
    ids = [str(uuid.uuid4()) for _ in memories]
    memory_collection().add(
        ids=ids,
        documents=[m["content"] for m in memories],
        metadatas=[{"user_id": user_id, "type": m["type"]} for m in memories],
    )
    logger.info("memories saved user_id=%s count=%d", user_id, len(ids))
    return ids


def search(user_id: str, query: str, top_k: int) -> list[dict]:
    """Nearest memories *for this user*, with their similarity attached.

    Filtering happens inside the Chroma query via `where`, so another
    customer's memory is never a candidate in the first place.
    """
    collection = memory_collection()
    if collection.count() == 0 or not query.strip():
        return []

    result = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        where={"user_id": user_id},
    )
    if not result["ids"] or not result["ids"][0]:
        return []

    return [
        {
            "id": memory_id,
            "content": document,
            "type": metadata.get("type", "fact"),
            "similarity": similarity(distance),
        }
        for memory_id, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def all_for_user(user_id: str) -> list[dict]:
    """Every memory held for one customer. Used by tests and by the
    deduplicator's cold path, not on the request path."""
    result = memory_collection().get(where={"user_id": user_id})
    return [
        {"id": i, "content": d, "type": m.get("type", "fact")}
        for i, d, m in zip(result["ids"], result["documents"], result["metadatas"])
    ]


def delete_for_user(user_id: str) -> int:
    existing = memory_collection().get(where={"user_id": user_id}, include=[])["ids"]
    if existing:
        memory_collection().delete(ids=existing)
    return len(existing)
