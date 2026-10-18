"""The workflow, assembled.

    guardrail ─(in scope)→ retrieve_memories → retrieve_knowledge ─(grounded)→ generate → extract_memories → END
            │                                                     └─(out of tries)→ escalate ──────────────→ END
            └─(blocked)────────────────────────────────────────────────────────────────────────────────────→ END

Answering one support message is not one LLM call, and writing it as one
function hides that. As a graph, each step is a named node with typed state:
independently testable, individually observable, and durably checkpointed so a
run can resume rather than restart.

The guardrail runs first: an out-of-scope, injection or unsafe message is
deflected before any retrieval or generation happens, and the run ends. An
in-scope message follows the full path — the two retrieval steps run before
generation because their results become prompt context, and extraction runs
after, because it reads the exchange that generation produced.

The second branch sits after retrieval for the same reason: it is the first
point at which it is known whether the agent has anything to answer *from*.
A turn that found nothing, after a run of turns that also found nothing, is
handed to a human rather than answered again. See `agent.escalation`.
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.agent import escalation
from app.agent.checkpointer import get_checkpointer
from app.agent.nodes import (
    check_guardrail,
    escalate,
    extract_memories,
    generate,
    retrieve_knowledge,
    retrieve_memories,
)
from app.agent.state import AgentContext, AgentState

# Node names are referenced by tests and appear in logs and checkpoints, so
# they are defined once rather than spelled out at each call site.
GUARDRAIL = "guardrail"
RETRIEVE_MEMORIES = "retrieve_memories"
RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
GENERATE = "generate"
EXTRACT_MEMORIES = "extract_memories"
ESCALATE = "escalate"


def _route_after_guardrail(state: AgentState) -> str:
    """A blocked message already has its reply; send it straight to the end.
    Anything else enters the normal retrieve-generate-extract path."""
    return END if state["blocked"] else RETRIEVE_MEMORIES


def _route_after_knowledge(state: AgentState, runtime: Runtime[AgentContext]) -> str:
    """Answer, or hand over.

    Both retrieval steps have run by now, so this is the first point at which
    it is known whether the agent has anything to answer *from*. The decision
    itself lives in `agent.escalation` as a pure function; this only maps it
    onto an edge.
    """
    if escalation.should_escalate(state, runtime.context):
        return ESCALATE
    return GENERATE


def build_graph() -> StateGraph:
    """The uncompiled graph. Separate from `compile()` so a test can inspect
    the topology, and so the same graph can be compiled with or without a
    checkpointer."""
    builder = StateGraph(AgentState, context_schema=AgentContext)

    builder.add_node(GUARDRAIL, check_guardrail)
    builder.add_node(RETRIEVE_MEMORIES, retrieve_memories)
    builder.add_node(RETRIEVE_KNOWLEDGE, retrieve_knowledge)
    builder.add_node(GENERATE, generate)
    builder.add_node(EXTRACT_MEMORIES, extract_memories)
    builder.add_node(ESCALATE, escalate)

    builder.add_edge(START, GUARDRAIL)
    builder.add_conditional_edges(
        GUARDRAIL,
        _route_after_guardrail,
        {RETRIEVE_MEMORIES: RETRIEVE_MEMORIES, END: END},
    )
    builder.add_edge(RETRIEVE_MEMORIES, RETRIEVE_KNOWLEDGE)
    builder.add_conditional_edges(
        RETRIEVE_KNOWLEDGE,
        _route_after_knowledge,
        {GENERATE: GENERATE, ESCALATE: ESCALATE},
    )
    builder.add_edge(GENERATE, EXTRACT_MEMORIES)
    builder.add_edge(EXTRACT_MEMORIES, END)
    # A handover is terminal: nothing is generated and nothing is learned.
    builder.add_edge(ESCALATE, END)

    return builder


@lru_cache
def get_graph():
    """Compiled once, lazily.

    Lazily because compiling attaches the checkpointer, which opens a
    connection pool — importing this module must not do that. Once because
    rebuilding the topology per message buys nothing.
    """
    return build_graph().compile(checkpointer=get_checkpointer())
