"""Deciding when to hand the customer to a human.

The agent is allowed to answer without sources: `SYSTEM_INSTRUCTION` lets it
offer the general troubleshooting a knowledgeable technician would, and that is
genuinely useful on the first try. What is not useful is doing it over and over
while the customer waits for an answer only a human can give. So escalation is
not triggered by *an* ungrounded reply, but by a run of them.

Pure functions, no I/O and no model call, so the policy can be read and tested
on its own:

  `has_grounding`      whether this turn was answered from anything
  `classify_outcome`   what a finished run was, as one word
  `consecutive_ungrounded`  how long the current run of them is
  `should_escalate`    whether this turn ends it

Two kinds of turn look like failures and are not, and keeping them out of the
count is most of what this module does:

- A pleasantry. "Thanks" retrieves nothing, but the customer asked for
  nothing either. The guardrail already reads every message and now says
  whether it was a support request at all, so this costs no extra model call.
- A profile fact. `memory_always_types` surfaces the customer's name and the
  appliances they own on every single turn, by design, so that "who am I?"
  can be answered. Treating that as grounding would mean any remembered
  customer is permanently "answered" and could never be escalated.

What does count as grounding is a support document, or a memory that had to
clear the relevance floor to be here — something retrieved *for this message*.
"""

import logging
from collections.abc import Sequence

from app.agent.state import AgentContext, AgentState
from app.config import get_settings

logger = logging.getLogger("csam.escalation")

settings = get_settings()

# The outcome of one assistant turn, persisted alongside the message so the
# next turn can count without re-deriving anything. Deliberately a closed set
# of strings rather than booleans: "why was there no context block" has four
# different answers and they must not collapse into one.
GROUNDED = "grounded"      # retrieval found something that answers the question
UNGROUNDED = "ungrounded"  # answered, but from general knowledge only
BLOCKED = "blocked"        # the guardrail deflected it
ESCALATED = "escalated"    # handed to a human
FAILED = "failed"          # generation failed; the customer saw the fallback
COURTESY = "courtesy"      # a pleasantry; the customer asked for nothing

# An assistant message written before outcomes were recorded. It is not
# evidence of anything, so it ends a run rather than extending one — the
# conservative direction, since the cost of under-counting is one extra
# ordinary reply and the cost of over-counting is escalating a customer who
# never asked for a human.
UNKNOWN = "unknown"

HANDOFF_REPLY = (
    "I'm sorry — I haven't been able to find what you need in our "
    "documentation or your account history, and I don't want to keep guessing "
    "at it. Let me pass you to a human agent who can look into this properly. "
    "They'll be able to see your account and pick up from here."
)


def has_grounding(state: AgentState) -> bool:
    """Whether retrieval found anything that actually answers this message.

    Not simply "the lists are non-empty". `memory_always_types` surfaces the
    stable profile facts — the customer's name, the appliances they own —
    unconditionally, bypassing the relevance floor, so that the agent can
    answer "who am I?". Those facts are present on *every* turn for any
    customer who has ever been remembered, and counting them as grounding
    made every such turn look answered: escalation could then only ever fire
    for a customer with an empty profile.

    So a profile fact does not ground a question. A document does, and so
    does a memory that had to clear the similarity floor to be here, because
    that one was retrieved *for this message*.

    This changes only how grounding is judged. The profile facts are still
    retrieved and still go into the prompt — personalisation is untouched.
    """
    if state["documents"]:
        return True
    always = tuple(settings.memory_always_types)
    # A memory with no recorded type is counted as grounding: we cannot tell
    # whether it was surfaced unconditionally, and the conservative direction
    # is to under-escalate rather than hand someone to a human on the
    # strength of a fact we cannot interpret. `store.by_types` already
    # defaults a missing type to "fact", which is not an always-type, so this
    # is the same choice made consistently.
    return any(memory.get("type") not in always for memory in state["memories"])


def classify_outcome(state: AgentState) -> str:
    """What this run was, from the state it finished in.

    Order is the policy. A blocked message retrieved nothing and generated
    nothing, and a failed generation produced no real answer — neither is a
    troubleshooting attempt, so neither may look like one to the next turn.
    Courtesy is checked before grounding for the same reason: "thanks" is not
    an attempt at anything, whatever retrieval happened to turn up alongside
    it.
    """
    if state["blocked"]:
        return BLOCKED
    if state["escalated"]:
        return ESCALATED
    if not state["llm_success"]:
        return FAILED
    if state["courtesy"]:
        return COURTESY
    if has_grounding(state):
        return GROUNDED
    return UNGROUNDED


def outcomes_from(messages: Sequence) -> list[str]:
    """Prior assistant outcomes, oldest first.

    Takes the message rows the endpoint already loaded for history, so this
    costs no extra query. User messages are skipped outright: only the agent
    makes attempts, so only the agent's turns can count toward a run of failed
    ones.
    """
    outcomes = []
    for message in messages:
        if message.role != "assistant":
            continue
        context = message.context or {}
        outcomes.append(context.get("outcome", UNKNOWN))
    return outcomes


def consecutive_ungrounded(outcomes: Sequence[str]) -> int:
    """How many ungrounded replies end this list, counting backwards.

    Courtesy is *skipped* rather than counted or treated as an interruption.
    This is the whole point of recording it separately: a customer who says
    "thanks" between two unanswered questions has not made a third failed
    attempt, but neither have they resolved anything, and a reply that reset
    the count would mean a polite customer could never be escalated at all.
    So the run reads straight through it.

    Anything else stops the count — a grounded answer because it means the
    agent did find what was asked for, and a block, an escalation or a failure
    because none of them was an attempt at troubleshooting.
    """
    run = 0
    for outcome in reversed(outcomes):
        if outcome == COURTESY:
            continue
        if outcome != UNGROUNDED:
            break
        run += 1
    return run


def should_escalate(
    state: AgentState, context: AgentContext, limit: int | None = None
) -> bool:
    """Whether this turn should be handed over instead of answered.

    The current turn counts toward the limit, so a limit of 3 means the third
    consecutive ungrounded reply is the one that escalates — the customer gets
    two real attempts first.
    """
    # A pleasantry is never the turn that hands someone over, however long the
    # run behind it. Answering "thanks" with "let me pass you to an agent"
    # would be absurd, and the run is not spent — the next real question still
    # sees it.
    if state["courtesy"]:
        return False
    if has_grounding(state):
        return False
    limit = settings.escalation_ungrounded_limit if limit is None else limit
    if limit <= 0:
        return False
    return consecutive_ungrounded(context.recent_outcomes) + 1 >= limit
