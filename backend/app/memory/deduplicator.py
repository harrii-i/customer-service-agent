"""Keeping the memory store from filling up with the same fact.

A customer mentions their WM-200 in three conversations. Without this, that is
three memories, and retrieval then spends its whole budget re-telling the agent
one thing it already knew.

The test is semantic, not textual: "Customer owns a WM-200 washing machine" and
"Customer has a WM-200 washer" are the same fact in different words, and only a
similarity comparison catches that. The threshold is deliberately much higher
than the retrieval threshold — *relevant* and *the same fact* are different
questions, and sharing one number for both would either merge distinct facts or
store every rephrasing.
"""

import logging
from collections.abc import Sequence

from app.config import get_settings
from app.memory import store

logger = logging.getLogger("csam.memory.dedup")

settings = get_settings()


def is_duplicate(user_id: str, content: str) -> bool:
    """True when this user already holds a memory that says the same thing."""
    neighbours = store.search(user_id, content, top_k=1)
    if not neighbours:
        return False
    return neighbours[0]["similarity"] >= settings.memory_dedup_threshold


def filter_new(user_id: str, candidates: Sequence[dict]) -> list[dict]:
    """Drop candidates already known, and drop duplicates *within* one batch.

    The within-batch check matters: a single exchange can easily yield "Customer
    owns a WM-200" twice in slightly different words, and neither would be in
    the store yet.
    """
    kept: list[dict] = []
    for candidate in candidates:
        content = candidate["content"]
        if is_duplicate(user_id, content):
            logger.info("skipping duplicate memory user_id=%s", user_id)
            continue
        if any(_same(content, k["content"]) for k in kept):
            logger.info("skipping in-batch duplicate memory user_id=%s", user_id)
            continue
        kept.append(candidate)
    return kept


def _same(a: str, b: str) -> bool:
    """Cheap within-batch comparison. The batch is a handful of items and none
    are stored yet, so this avoids embedding round-trips for an exact or
    near-exact repeat."""
    return a.strip().casefold() == b.strip().casefold()
