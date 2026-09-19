"""Phase 3 acceptance: the workflow is a LangGraph, not a function.

Answering a message is a sequence of named steps. These tests assert the shape
of that sequence, that each node is callable on its own, and that the endpoint
runs the graph — plus the thing that must stay true throughout: inert nodes
return *empty* results, never invented ones.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.agent import nodes
from app.agent.graph import (
    EXTRACT_MEMORIES,
    GENERATE,
    RETRIEVE_KNOWLEDGE,
    RETRIEVE_MEMORIES,
    build_graph,
)
from app.agent.prompts import build_context_block
from app.agent.state import AgentContext, AgentState, initial_state
from app.llm.gemini import LlmReply
from tests.conftest import STUB_REPLY


def make_state(**overrides) -> AgentState:
    state = initial_state()
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def make_context(**overrides) -> AgentContext:
    defaults = dict(
        user_id="u-1",
        conversation_id="c-1",
        user_message="My washer is noisy.",
        history=[("user", "My washer is noisy.")],
    )
    defaults.update(overrides)
    return AgentContext(**defaults)  # type: ignore[arg-type]


def runtime(context: AgentContext | None = None):
    """Nodes only ever read `runtime.context`, so a test does not need a real
    LangGraph Runtime to call one directly."""
    return SimpleNamespace(context=context or make_context())


# --- topology --------------------------------------------------------------


def test_graph_has_the_four_named_steps():
    nodes_in_graph = set(build_graph().compile().get_graph().nodes)
    assert {
        RETRIEVE_MEMORIES,
        RETRIEVE_KNOWLEDGE,
        GENERATE,
        EXTRACT_MEMORIES,
    } <= nodes_in_graph


def test_retrieval_precedes_generation_and_extraction_follows_it():
    """Order is the design, not an accident: retrieved material has to exist
    before it can be prompt context, and extraction reads what generation
    produced."""
    edges = {
        (edge.source, edge.target)
        for edge in build_graph().compile().get_graph().edges
    }
    assert ("__start__", RETRIEVE_MEMORIES) in edges
    assert (RETRIEVE_MEMORIES, RETRIEVE_KNOWLEDGE) in edges
    assert (RETRIEVE_KNOWLEDGE, GENERATE) in edges
    assert (GENERATE, EXTRACT_MEMORIES) in edges
    assert (EXTRACT_MEMORIES, "__end__") in edges


# --- nodes, individually ---------------------------------------------------


def test_generate_node_returns_the_reply_and_its_success_flag(monkeypatch):
    monkeypatch.setattr(
        nodes, "generate_reply", lambda turns, context=None: LlmReply("hi", ok=True)
    )
    assert nodes.generate(make_state(), runtime()) == {
        "response": "hi",
        "llm_success": True,
    }


def test_generate_node_reads_history_from_runtime_context(monkeypatch):
    """History is runtime input, not checkpointed state — the node must take
    it from the context."""
    seen = {}

    def capture(turns, context=None):
        seen["turns"] = list(turns)
        return LlmReply("ok", ok=True)

    monkeypatch.setattr(nodes, "generate_reply", capture)
    nodes.generate(
        make_state(),
        runtime(make_context(history=[("user", "a"), ("assistant", "b")])),
    )
    assert seen["turns"] == [("user", "a"), ("assistant", "b")]


def test_generate_node_passes_retrieved_context_to_the_llm(monkeypatch):
    """Phases 6 and 7 fill the retrieval nodes; this asserts the wiring that
    lets them do so without touching this node."""
    seen = {}

    def capture(turns, context=None):
        seen["context"] = context
        return LlmReply("ok", ok=True)

    monkeypatch.setattr(nodes, "generate_reply", capture)

    nodes.generate(make_state(), runtime())
    assert seen["context"] is None, "nothing retrieved must mean no context"

    nodes.generate(
        make_state(
            memories=[{"content": "Owns a WM-200"}],
            documents=[{"title": "Warranty Policy", "snippet": "Two years."}],
        ),
        runtime(),
    )
    assert "Owns a WM-200" in seen["context"]
    assert "Warranty Policy" in seen["context"]


def test_empty_retrieval_produces_no_context_block():
    """An empty 'Context:' heading is an invitation to fill it in, which is
    the failure this project is built to avoid."""
    assert build_context_block([], []) is None


# --- the endpoint runs the graph -------------------------------------------


def test_chat_endpoint_runs_the_workflow(client: TestClient, user_id: str):
    conversation_id = client.post(
        "/conversations", json={"user_id": user_id}
    ).json()["conversation_id"]

    body = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "My washer is noisy.",
        },
    ).json()

    assert body["message"] == STUB_REPLY
    assert body["memory_used"] is False
    assert body["memories"] == []
    assert body["sources"] == []
