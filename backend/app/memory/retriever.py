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
    """This user's relevant memories for the current message.

    Two sources, merged:

    - The stable profile facts (identity, products owned) are surfaced
      unconditionally. A customer asking "who am I?" or "what do you know about
      me?" shares no words with "Customer's name is …", so those facts score
      far below any usable relevance floor; gating them on similarity made the
      agent deny knowing what it held. See `memory_always_types`.
    - Everything else (issues, preferences) stays similarity-gated. Without a
      floor every message recalls the nearest memory whether or not it relates,
      and the agent would open a delivery question by mentioning an old noise.
    """
    always = tuple(settings.memory_always_types)

    profile = store.by_types(user_id, always)[: settings.memory_profile_limit]

    hits = store.search(user_id, query, top_k=top_k or settings.memory_top_k)
    gated = [
        {"id": h["id"], "content": h["content"], "type": h["type"]}
        for h in hits
        if h["similarity"] >= settings.memory_similarity_threshold
        and h["type"] not in always
    ]

    # Profile first; a gated hit that is also a profile fact is already present.
    seen = {m["id"] for m in profile}
    return profile + [m for m in gated if m["id"] not in seen]
