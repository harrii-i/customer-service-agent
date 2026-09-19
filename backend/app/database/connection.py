"""SQLAlchemy engine / session wiring.

Sync engine on purpose: FastAPI runs `def` path operations in a threadpool, and
the LangGraph PostgreSQL checkpointer and sentence-transformers embedding calls
added in later phases are blocking anyway. One concurrency model, no async/sync
bridging.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # recover transparently from dropped connections
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
