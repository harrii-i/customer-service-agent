"""Phase 8 acceptance: the same fact is stored once."""

from app.config import get_settings
from app.memory import deduplicator, store

settings = get_settings()


def test_the_same_fact_reworded_is_a_duplicate():
    """Textual comparison would miss this; that is why the test is semantic."""
    store.save("u1", "Customer owns a WM-200 washing machine.", "product")
    assert deduplicator.is_duplicate("u1", "Customer has a WM-200 washer.")


def test_a_different_fact_is_not_a_duplicate():
    store.save("u1", "Customer owns a WM-200 washing machine.", "product")
    assert not deduplicator.is_duplicate("u1", "Customer owns a dishwasher.")


def test_nothing_is_a_duplicate_of_an_empty_store():
    assert not deduplicator.is_duplicate("u1", "Customer owns a WM-200.")


def test_another_users_identical_memory_is_not_a_duplicate():
    """Dedup must not leak across customers any more than retrieval does."""
    store.save("someone_else", "Customer owns a WM-200 washing machine.", "product")
    assert not deduplicator.is_duplicate("u1", "Customer owns a WM-200 washing machine.")


def test_filter_new_drops_what_is_already_known():
    store.save("u1", "Customer owns a WM-200 washing machine.", "product")
    kept = deduplicator.filter_new("u1", [
        {"content": "Customer has a WM-200 washer.", "type": "product"},
        {"content": "Customer prefers email contact.", "type": "preference"},
    ])
    assert [k["content"] for k in kept] == ["Customer prefers email contact."]


def test_filter_new_drops_duplicates_within_one_batch():
    """Neither is in the store yet, so the store check alone cannot catch it."""
    kept = deduplicator.filter_new("u1", [
        {"content": "Customer owns a WM-200.", "type": "product"},
        {"content": "  customer owns a wm-200.  ", "type": "product"},
    ])
    assert len(kept) == 1


def test_repeating_a_fact_across_conversations_stores_it_once():
    """The behaviour that matters: mentioning the WM-200 three times must not
    spend the whole retrieval budget re-telling the agent one thing."""
    candidate = [{"content": "Customer owns a WM-200 washing machine.", "type": "product"}]
    for _ in range(3):
        fresh = deduplicator.filter_new("u1", candidate)
        store.save_many("u1", fresh)
    assert len(store.all_for_user("u1")) == 1


def test_dedup_bar_is_higher_than_the_relevance_bar():
    """Sharing one threshold would either merge distinct facts or store every
    rephrasing of the same one."""
    assert settings.memory_dedup_threshold > settings.memory_similarity_threshold
