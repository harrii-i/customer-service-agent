"""Rebuild the support-knowledge collection from backend/knowledge/*.md.

    python scripts/ingest_knowledge.py

Run it after editing any document. Embeddings are computed on this machine.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.ingestion import KNOWLEDGE_DIR, ingest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

if __name__ == "__main__":
    count = ingest()
    if count == 0:
        print(f"No markdown documents found in {KNOWLEDGE_DIR}")
        raise SystemExit(1)
    print(f"Ingested {count} chunks from {KNOWLEDGE_DIR}")
