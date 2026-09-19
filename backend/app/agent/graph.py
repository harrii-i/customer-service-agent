"""The workflow, assembled.

    retrieve_memories → retrieve_knowledge → generate → extract_memories

Answering one support message is not one LLM call, and writing it as one
function hides that. As a graph, each step is a named node with typed state:
independently testable, individually observable, and durably checkpointed so a
run can resume rather than restart.

The two retrieval steps run before generation because their results become
prompt context. Extraction runs after, because it reads the exchange that
generation produced.
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agent.checkpointer import get_checkpointer
from app.agent.nodes import (
    extract_memories,
    generate,
    retrieve_knowledge,
    retrieve_memories,
)
from app.agent.state import AgentContext, AgentState

# Node names are referenced by tests and appear in logs and checkpoints, so
# they are defined once rather than spelled out at each call site.
RETRIEVE_MEMORIES = "retrieve_memories"
RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
GENERATE = "generate"
EXTRACT_MEMORIES = "extract_memories"


def build_graph() -> StateGraph:
    """The uncompiled graph. Separate from `compile()` so a test can inspect
    the topology, and so the same graph can be compiled with or without a
    checkpointer."""
    builder = StateGraph(AgentState, context_schema=AgentContext)

    builder.add_node(RETRIEVE_MEMORIES, retrieve_memories)
    builder.add_node(RETRIEVE_KNOWLEDGE, retrieve_knowledge)
    builder.add_node(GENERATE, generate)
    builder.add_node(EXTRACT_MEMORIES, extract_memories)

    builder.add_edge(START, RETRIEVE_MEMORIES)
    builder.add_edge(RETRIEVE_MEMORIES, RETRIEVE_KNOWLEDGE)
    builder.add_edge(RETRIEVE_KNOWLEDGE, GENERATE)
    builder.add_edge(GENERATE, EXTRACT_MEMORIES)
    builder.add_edge(EXTRACT_MEMORIES, END)

    return builder


@lru_cache
def get_graph():
    """Compiled once, lazily.

    Lazily because compiling attaches the checkpointer, which opens a
    connection pool — importing this module must not do that. Once because
    rebuilding the topology per message buys nothing.
    """
    return build_graph().compile(checkpointer=get_checkpointer())
