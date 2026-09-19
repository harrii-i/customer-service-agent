"""The LangGraph PostgreSQL checkpointer.

Workflow state and application state share one durable store, so there is one
thing to back up and one thing to lose. The checkpointer keeps its own tables
(`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`); it manages their
schema itself via `setup()`, which is why they are absent from `models.py`.

The thread id is the conversation id: one conversation is one resumable run.
"""

import logging
from functools import lru_cache

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

logger = logging.getLogger("csam.checkpointer")

settings = get_settings()


def psycopg_conninfo() -> str:
    """`DATABASE_URL` is a SQLAlchemy URL; psycopg wants a plain one."""
    return settings.database_url.replace("postgresql+psycopg://", "postgresql://")


@lru_cache
def get_checkpointer() -> PostgresSaver:
    """Built lazily and once.

    Lazily, so importing the agent package never opens a database connection —
    tests and tooling import it freely. Once, because the pool is the
    expensive part.

    `autocommit` and `dict_row` are required by PostgresSaver, not
    preferences.

    This deliberately does *not* call `setup()`. See `setup_checkpointer`.
    """
    pool = ConnectionPool(
        conninfo=psycopg_conninfo(),
        min_size=1,
        max_size=settings.checkpointer_pool_size,
        open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    logger.info("checkpointer pool opened")
    return PostgresSaver(pool)


def setup_checkpointer() -> None:
    """Create/migrate the checkpointer's tables. Idempotent.

    This runs at application startup and *only* there. LangGraph's migration
    uses `CREATE INDEX CONCURRENTLY`, which waits for every other open
    transaction in the database to finish. Called lazily from inside a request,
    it waits on that request's own still-open session and the process hangs
    forever. At startup there is no request in flight and nothing to wait on.
    """
    get_checkpointer().setup()
    logger.info("checkpointer schema ensured")
