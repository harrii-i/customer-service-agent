"""What flows through the graph, split by lifetime.

Two shapes, and the split is the point:

`AgentContext` is *runtime* input — who is asking, what they said, and the
recent transcript. LangGraph does not checkpoint it. That matters, because the
transcript is already owned by the `messages` table, and a checkpoint holding
its own copy would be a second source of truth for the same conversation. The
README is explicit that the four kinds of state are never copies of each other;
this is where that is enforced rather than merely intended.

`AgentState` is what the *workflow itself* produces and what gets checkpointed:
what was retrieved, what was generated, whether generation succeeded. That is
"where the workflow got to", which is exactly what a checkpoint should hold.
"""

from dataclasses import dataclass, field
from typing import TypedDict


@dataclass
class AgentContext:
    """Per-run input. Never checkpointed, never a source of truth."""

    user_id: str
    conversation_id: str
    user_message: str
    # Recent conversation, oldest first, already truncated by the caller and
    # already ending with `user_message`.
    history: list[tuple[str, str]] = field(default_factory=list)
    # What each earlier *assistant* turn in this conversation amounted to,
    # oldest first — see `agent.escalation`. It is derived from the same rows
    # as `history` and lives here, not in `AgentState`, for the same reason
    # the transcript does: a checkpoint must not become a second copy of
    # something the `messages` table already owns. Each run is handed a fresh
    # `AgentState`, so a run of ungrounded replies could not be counted from
    # the checkpoint even if it were kept there.
    recent_outcomes: list[str] = field(default_factory=list)


class AgentState(TypedDict):
    """The workflow's own output. Checkpointed to PostgreSQL per thread."""

    memories: list[dict]   # long-term memory hits (Phase 7)
    documents: list[dict]  # support-documentation hits (Phase 6)
    response: str
    llm_success: bool
    # True when the input guardrail deflected the message: retrieval and
    # generation are skipped and `response` already holds the canned reply.
    blocked: bool
    # True when the run was handed to a human instead of answered: generation
    # and extraction are skipped and `response` holds the handoff reply.
    escalated: bool
    # True when the message was pure courtesy — a greeting, thanks, an
    # acknowledgement — rather than a request for support. It is answered
    # exactly like any other in-scope message; the flag only stops a
    # pleasantry from being counted as a failed support attempt.
    courtesy: bool


def initial_state() -> AgentState:
    """Every run starts from empty retrieval. A node that finds nothing must
    leave nothing behind — never last turn's results."""
    return {
        "memories": [],
        "documents": [],
        "response": "",
        "llm_success": False,
        "blocked": False,
        "escalated": False,
        "courtesy": False,
    }
