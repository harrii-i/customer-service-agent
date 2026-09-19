"""Retrieving a customer's long-term memories for the current message.

Returns plain dicts shaped like the API's `MemoryOut`, so what the agent was
given and what the customer is shown are the same objects. The UI can never
display a memory the agent did not actually receive.
"""

import logging

from app.config import get_settings
from app.memory import store

logger = logging.getLogger("csam.memory.retriever")

settings = get_settings()


def retrieve(user_id: str, query: str, top_k: int | None = None) -> list[dict]:
    """This user's most relevant memories, weak matches discarded.

    Without a floor every message retrieves the nearest memories whether or not
    they relate to it, and the agent would open a delivery question by
    recalling a washing-machine noise.
    """
    hits = store.search(user_id, query, top_k=top_k or settings.memory_top_k)
    return [
        {"id": h["id"], "content": h["content"], "type": h["type"]}
        for h in hits
        if h["similarity"] >= settings.memory_similarity_threshold
    ]
