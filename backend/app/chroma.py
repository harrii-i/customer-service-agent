"""The ChromaDB client and the two collections.

Both the RAG layer and the memory layer need vector storage, so the client
lives here rather than in either of them — one persistent client per process,
because opening two against the same path is asking for trouble.

Two collections, deliberately separate:

  `support_knowledge`  chunked company documentation. Global, read-only at
                       query time, written only by the ingestion script.
  `customer_memories`  distilled durable facts. Partitioned by `user_id` in
                       metadata, and every query filters on it.

Embeddings are computed locally by Chroma's default model (ONNX
all-MiniLM-L6-v2). No text — least of all a customer memory — is sent to a
third-party embedding API.
"""

import logging
from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from app.config import get_settings

logger = logging.getLogger("csam.chroma")

settings = get_settings()

KNOWLEDGE_COLLECTION = "support_knowledge"
MEMORY_COLLECTION = "customer_memories"

# Cosine, not Chroma's L2 default. MiniLM embeddings are compared by angle, and
# a cosine distance maps to a similarity of `1 - distance`, which is what the
# configured thresholds are expressed in. With L2 those thresholds would be
# meaningless numbers.
_COSINE = {"hnsw:space": "cosine"}


@lru_cache
def get_client() -> chromadb.ClientAPI:
    """One persistent client per process, created on first use."""
    logger.info("opening chroma at %s", settings.chroma_path)
    return chromadb.PersistentClient(path=settings.chroma_path)


@lru_cache
def _embedding_function():
    """Loaded once — instantiating it loads the ONNX model."""
    return embedding_functions.DefaultEmbeddingFunction()


def _collection(name: str) -> Collection:
    return get_client().get_or_create_collection(
        name=name,
        metadata=_COSINE,
        embedding_function=_embedding_function(),
    )


def knowledge_collection() -> Collection:
    return _collection(KNOWLEDGE_COLLECTION)


def memory_collection() -> Collection:
    return _collection(MEMORY_COLLECTION)


def similarity(distance: float) -> float:
    """Chroma returns cosine *distance*; thresholds are expressed as
    similarity. Kept in one place so the conversion is never done by hand at a
    call site, where an inverted comparison is easy to miss and hard to see."""
    return 1.0 - distance
