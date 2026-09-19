"""Look inside the vector store.

    python scripts/inspect_vectors.py                 # summary + everything
    python scripts/inspect_vectors.py --user <id>     # one customer's memories
    python scripts/inspect_vectors.py --query "warranty"   # what a search returns

Chroma has no UI, and reading chroma.sqlite3 by hand tells you little — the
documents are there but the embeddings are opaque blobs. This shows the two
things actually worth seeing: what is stored, and what a given query retrieves
with its similarity score.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chroma import (  # noqa: E402
    knowledge_collection,
    memory_collection,
    similarity,
)
from app.config import get_settings  # noqa: E402

settings = get_settings()


def summary() -> None:
    print(f"store: {settings.chroma_path}\n")
    for collection in (memory_collection(), knowledge_collection()):
        print(f"  {collection.name}: {collection.count()} items")
    print()


def dump_memories(user_id: str | None) -> None:
    where = {"user_id": user_id} if user_id else None
    result = memory_collection().get(where=where)
    print(f"--- customer_memories ({len(result['ids'])}) ---")
    for doc, meta in zip(result["documents"], result["metadatas"]):
        print(f"  [{meta.get('type','?'):<10}] user={meta.get('user_id','?')[:8]}  {doc}")
    print()


def dump_knowledge() -> None:
    result = knowledge_collection().get()
    by_document: dict[str, int] = {}
    for meta in result["metadatas"]:
        by_document[meta["title"]] = by_document.get(meta["title"], 0) + 1
    print(f"--- support_knowledge ({len(result['ids'])} chunks) ---")
    for title, count in sorted(by_document.items()):
        print(f"  {count} chunk(s)  {title}")
    print()


def run_query(text: str, user_id: str | None) -> None:
    """Shows the scores the thresholds are compared against — the fastest way
    to tell a retrieval miss from a prompt problem."""
    print(f'--- query: "{text}" ---\n')

    print(f"  memories (floor {settings.memory_similarity_threshold}):")
    where = {"user_id": user_id} if user_id else None
    memories = memory_collection()
    if memories.count():
        result = memories.query(
            query_texts=[text], n_results=min(5, memories.count()), where=where
        )
        for doc, distance in zip(result["documents"][0], result["distances"][0]):
            score = similarity(distance)
            mark = "keep" if score >= settings.memory_similarity_threshold else "drop"
            print(f"    {score:.3f} {mark}  {doc[:70]}")
    else:
        print("    (empty)")

    print(f"\n  documents (floor {settings.knowledge_similarity_threshold}):")
    knowledge = knowledge_collection()
    if knowledge.count():
        result = knowledge.query(
            query_texts=[text], n_results=min(5, knowledge.count())
        )
        for meta, distance in zip(result["metadatas"][0], result["distances"][0]):
            score = similarity(distance)
            mark = "keep" if score >= settings.knowledge_similarity_threshold else "drop"
            print(f"    {score:.3f} {mark}  {meta['title']}")
    else:
        print("    (empty — run scripts/ingest_knowledge.py)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", help="only this customer's memories")
    parser.add_argument("--query", help="show what this search retrieves, with scores")
    args = parser.parse_args()

    summary()
    if args.query:
        run_query(args.query, args.user)
    else:
        dump_memories(args.user)
        dump_knowledge()
