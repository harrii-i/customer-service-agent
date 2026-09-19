"""Phase 4 acceptance: workflow state is checkpointed to PostgreSQL.

The point of the checkpointer is that a run is durable and resumable, keyed by
conversation. The point of *how* it is wired is that it does not become a
second copy of the conversation — that is what the context/state split exists
for, and it is asserted here rather than left as an intention.
"""

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.agent.checkpointer import get_checkpointer, psycopg_conninfo
from app.agent.graph import get_graph
from app.agent.state import AgentContext, initial_state
from app.config import get_settings
from app.database.connection import engine


def run_graph(conversation_id: str, history, user_message="hello"):
    return get_graph().invoke(
        initial_state(),
        context=AgentContext(
            user_id="u-1",
            conversation_id=conversation_id,
            user_message=user_message,
            history=history,
        ),
        config={"configurable": {"thread_id": conversation_id}},
    )


def test_conninfo_strips_the_sqlalchemy_driver():
    """psycopg cannot parse SQLAlchemy's `+psycopg` dialect URL."""
    assert psycopg_conninfo().startswith("postgresql://")
    assert "+psycopg" not in psycopg_conninfo()


def test_checkpointer_creates_its_own_tables():
    """They are absent from models.py by design — LangGraph migrates them."""
    get_checkpointer()
    with engine.connect() as connection:
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            assert connection.execute(
                text("SELECT to_regclass(:n)"), {"n": f"public.{table}"}
            ).scalar() is not None, f"{table} missing"


def test_a_run_is_checkpointed_under_its_conversation():
    thread = "11111111-1111-1111-1111-111111111111"
    run_graph(thread, [("user", "hello")])

    snapshot = get_graph().get_state({"configurable": {"thread_id": thread}})
    assert snapshot is not None
    # The workflow's own output survived the run.
    assert snapshot.values["response"]
    assert snapshot.values["llm_success"] is True


def test_separate_conversations_have_separate_checkpoints():
    """Thread id is the conversation id, so two conversations must not share
    workflow state."""
    a = "22222222-2222-2222-2222-222222222222"
    b = "33333333-3333-3333-3333-333333333333"
    run_graph(a, [("user", "first")])
    run_graph(b, [("user", "second")])

    graph = get_graph()
    state_a = graph.get_state({"configurable": {"thread_id": a}})
    state_b = graph.get_state({"configurable": {"thread_id": b}})
    assert state_a.config["configurable"]["thread_id"] == a
    assert state_b.config["configurable"]["thread_id"] == b


def test_the_checkpoint_does_not_become_a_second_transcript():
    """The design guarantee. Conversation history is runtime context, not
    checkpointed state, so a checkpoint can never drift from `messages` or be
    mistaken for it."""
    thread = "44444444-4444-4444-4444-444444444444"
    secret = "a-distinctive-phrase-only-in-the-transcript"
    run_graph(thread, [("user", secret)], user_message=secret)

    snapshot = get_graph().get_state({"configurable": {"thread_id": thread}})
    assert "history" not in snapshot.values
    assert "user_message" not in snapshot.values
    assert set(snapshot.values) == {
        "memories",
        "documents",
        "response",
        "llm_success",
    }

    # And the phrase is nowhere in the checkpoint tables at all.
    with engine.connect() as connection:
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            rows = connection.execute(
                text(f"SELECT count(*) FROM {table} WHERE CAST({table} AS text) LIKE :p"),
                {"p": f"%{secret}%"},
            ).scalar()
            assert rows == 0, f"transcript leaked into {table}"


def test_chat_endpoint_writes_a_checkpoint(client: TestClient, user_id: str):
    conversation_id = client.post(
        "/conversations", json={"user_id": user_id}
    ).json()["conversation_id"]
    client.post(
        "/chat",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "My washer is noisy.",
        },
    )

    snapshot = get_graph().get_state(
        {"configurable": {"thread_id": conversation_id}}
    )
    assert snapshot.values["response"]
    assert get_settings().checkpointer_pool_size >= 1
