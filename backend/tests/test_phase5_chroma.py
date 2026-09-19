"""Phase 5 acceptance: two ChromaDB collections, cosine, locally embedded."""

from app.chroma import (
    KNOWLEDGE_COLLECTION,
    MEMORY_COLLECTION,
    get_client,
    knowledge_collection,
    memory_collection,
    similarity,
)


def test_both_collections_exist_and_are_separate():
    assert knowledge_collection().name == KNOWLEDGE_COLLECTION
    assert memory_collection().name == MEMORY_COLLECTION
    names = {c.name for c in get_client().list_collections()}
    assert {KNOWLEDGE_COLLECTION, MEMORY_COLLECTION} <= names


def test_collections_use_cosine_distance():
    """The configured thresholds are similarities. Under Chroma's L2 default
    they would be meaningless numbers that still compared successfully."""
    for collection in (knowledge_collection(), memory_collection()):
        assert collection.metadata["hnsw:space"] == "cosine"


def test_similarity_inverts_distance():
    assert similarity(0.0) == 1.0
    assert similarity(1.0) == 0.0


def test_semantic_search_beats_keyword_overlap():
    """The reason for a vector store at all: a query that shares no words with
    the right answer must still find it."""
    collection = memory_collection()
    collection.add(
        ids=["a", "b"],
        documents=[
            "Customer owns a WM-200 washing machine.",
            "Customer prefers to be contacted by email.",
        ],
        metadatas=[{"user_id": "u1"}, {"user_id": "u1"}],
    )
    result = collection.query(query_texts=["what laundry appliance do they have"],
                              n_results=1)
    assert result["ids"][0] == ["a"]


def test_metadata_filter_partitions_by_user():
    """The mechanism the whole memory-isolation guarantee rests on."""
    collection = memory_collection()
    collection.add(
        ids=["mine", "theirs"],
        documents=["Customer owns a WM-200.", "Customer owns a WM-200."],
        metadatas=[{"user_id": "u1"}, {"user_id": "u2"}],
    )
    result = collection.query(
        query_texts=["what do they own"], n_results=5, where={"user_id": "u1"}
    )
    assert result["ids"][0] == ["mine"]
