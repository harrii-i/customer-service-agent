"""Test fixtures.

Tests run against a real PostgreSQL database (never SQLite) so that the
UUID columns, server-side defaults and cascade behaviour exercised here are
the same ones production uses. Point TEST_DATABASE_URL at a throwaway DB.
"""

import os
import tempfile

import pytest

# Must be set before app modules are imported: the engine is built at import
# time from the settings.
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/csam_test",
)
# A throwaway vector store per run. Tests must never read, write or wipe the
# developer's real ./data/chroma.
os.environ["CHROMA_PATH"] = tempfile.mkdtemp(prefix="csam-chroma-test-")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.agent import nodes as agent_nodes  # noqa: E402
from app.chroma import knowledge_collection, memory_collection  # noqa: E402
from app.memory import extractor as memory_extractor  # noqa: E402
from app.agent.checkpointer import setup_checkpointer  # noqa: E402
from app.database.connection import engine  # noqa: E402
from app.database.models import Base  # noqa: E402
from app.llm.gemini import LlmReply  # noqa: E402
from app.main import app  # noqa: E402

STUB_REPLY = "Stubbed assistant reply."

# LangGraph owns these and creates them via its own setup(); they are not in
# models.py, so Base.metadata neither creates nor cleans them.
CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


@pytest.fixture(scope="session", autouse=True)
def _schema():
    Base.metadata.create_all(bind=engine)
    # Once per session, before any test opens a transaction — LangGraph's
    # migration uses CREATE INDEX CONCURRENTLY and would block on one.
    setup_checkpointer()
    yield
    Base.metadata.drop_all(bind=engine)
    with engine.begin() as connection:
        for table in CHECKPOINT_TABLES + ("checkpoint_migrations",):
            connection.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))


@pytest.fixture(autouse=True)
def _clean_tables():
    with engine.begin() as connection:
        connection.execute(
            text("TRUNCATE messages, conversations, users RESTART IDENTITY CASCADE")
        )
        # Checkpoints must not leak between tests either: a stale thread would
        # let a test pass on a previous test's workflow state.
        for table in CHECKPOINT_TABLES:
            exists = connection.execute(
                text("SELECT to_regclass(:name)"), {"name": f"public.{table}"}
            ).scalar()
            if exists:
                connection.execute(text(f"TRUNCATE {table} CASCADE"))
    yield


@pytest.fixture(autouse=True)
def _clean_vectors():
    """Chroma is process-global; without this, memories and documents leak
    between tests and a test can pass on another test's data."""
    for collection in (memory_collection(), knowledge_collection()):
        ids = collection.get(include=[])["ids"]
        if ids:
            collection.delete(ids=ids)
    yield


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    """No test ever calls Gemini. The suite must be deterministic, offline and
    free, and it must stay that way even on a machine whose .env holds a real
    API key. Tests that care about the LLM re-patch this in their own body."""
    monkeypatch.setattr(
        agent_nodes,
        "generate_reply",
        lambda turns, context=None: LlmReply(STUB_REPLY, ok=True),
    )
    # Extraction is a second Gemini call. Off by default — a test that wants
    # memories written says so explicitly.
    monkeypatch.setattr(
        memory_extractor, "generate_json", lambda *a, **k: None
    )
    return monkeypatch


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def user_id(client: TestClient) -> str:
    return client.post("/users").json()["user_id"]
