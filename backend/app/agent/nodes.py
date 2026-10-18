"""The graph's nodes.

Each is an ordinary function of `(state, runtime) -> partial state`. Nothing
here knows it is running inside LangGraph beyond the `Runtime` type, which is
what makes every node callable directly from a test.

`runtime.context` carries who is asking and what was said; the return value
carries what the workflow found or produced. Nodes never receive a database
session: graph state is checkpointed and must stay serialisable, so nodes that
need their own I/O (ChromaDB) open it themselves.

Retrieval and extraction are best-effort. If ChromaDB is unavailable the
customer still gets an answer, with less context, and the failure is logged —
never swallowed, and never allowed to take down the reply.
"""

import logging

from langgraph.runtime import Runtime

from app.agent import escalation, guardrail
from app.agent.prompts import build_context_block
from app.agent.state import AgentContext, AgentState
from app.config import get_settings
from app.llm.gemini import generate_reply
from app.memory import deduplicator, extractor
from app.memory import retriever as memory_retriever
from app.memory import store as memory_store
from app.rag import retriever as knowledge_retriever

logger = logging.getLogger("csam.agent")

settings = get_settings()


def check_guardrail(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """First node: decide whether the message is allowed to reach the agent.

    An in-scope message passes straight through. Anything else is deflected
    here with a deterministic canned reply, and the graph then skips retrieval,
    generation and extraction — there is nothing to retrieve for, and a blocked
    request is not an exchange worth remembering.

    Classification failing is not a block: `guardrail.classify_verdict` fails
    open, so the worst case is that the normal agent handles a message the
    filter could not judge, still bounded by the system prompt.

    The verdict also says whether the message was a support request at all.
    That changes nothing about how it is answered — a greeting gets a reply
    like anything else — and is used only so that being polite is not counted
    as the agent failing to help. See `agent.escalation`.
    """
    verdict = guardrail.classify_verdict(runtime.context.user_message)
    if verdict.category == "in_scope":
        # The same classification already being made, now also recording
        # whether the customer asked for anything. No extra model call: this
        # rides along on the verdict the guardrail always fetches.
        return {"blocked": False, "courtesy": not verdict.is_support_request}
    category = verdict.category
    logger.info("guardrail blocked message category=%s", category)
    return {
        "blocked": True,
        "response": guardrail.deflection(category),
        # A deliberate, correct reply — not the LLM fallback. Reported as a
        # successful turn so it is persisted and shown like any other.
        "llm_success": True,
    }


def retrieval_query(context: AgentContext, turns: int | None = None) -> str:
    """The text to search memories and documentation with.

    Not simply the current message. Customers ask follow-ups — "what was the
    issue reported?", "how long is it covered?" — whose subject sits in the
    previous turn rather than in the words being searched. Embedded alone,
    such a fragment matches nothing and the agent then truthfully reports that
    it cannot find what it was told minutes earlier.

    Only the customer's own turns are used. The assistant's replies are long,
    and letting them into the query lets the agent's own wording, rather than
    the customer's question, decide what gets retrieved.
    """
    count = settings.retrieval_context_turns if turns is None else turns
    customer_turns = [text for role, text in context.history if role == "user"]
    if not customer_turns:
        return context.user_message

    # `history` already ends with the current message.
    recent = customer_turns[-(count + 1):]
    if recent[-1] != context.user_message:
        recent.append(context.user_message)
    return "\n".join(recent)


def retrieve_memories(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """What we already know about this customer, from earlier conversations.

    Filtered by user_id inside the Chroma query, so one customer's memories can
    never surface in another's conversation.
    """
    context = runtime.context
    try:
        memories = memory_retriever.retrieve(
            context.user_id, retrieval_query(context)
        )
    except Exception:
        logger.exception("memory retrieval failed; continuing without memories")
        return {"memories": []}
    logger.info("retrieved %d memories", len(memories))
    return {"memories": memories}


def retrieve_knowledge(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """Company documentation relevant to the question."""
    try:
        documents = knowledge_retriever.retrieve(retrieval_query(runtime.context))
    except Exception:
        logger.exception("knowledge retrieval failed; continuing without sources")
        return {"documents": []}
    logger.info("retrieved %d documents", len(documents))
    return {"documents": documents}


def escalate(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """Hand the conversation to a human instead of answering it again.

    Reached only when this turn is ungrounded and completes a run of them, so
    by this point the agent has already tried general troubleshooting and the
    customer is still asking. Continuing would mean another answer built on
    nothing, which is how a support bot wastes someone's afternoon.

    Deterministic by design, like `guardrail.deflection`: the customer being
    handed over should not depend on what a model happens to say, and the
    graph skips generation and extraction from here. Nothing is learned from
    a handover — it is the agent failing to answer, not the customer telling
    it something.
    """
    logger.info(
        "escalating to a human after %d consecutive ungrounded replies",
        escalation.consecutive_ungrounded(runtime.context.recent_outcomes) + 1,
    )
    return {
        "escalated": True,
        "response": escalation.HANDOFF_REPLY,
        # A deliberate, correct reply — persisted and shown like any other
        # turn, exactly as a guardrail deflection is.
        "llm_success": True,
    }


def generate(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """Ask Gemini for the reply, with whatever was retrieved as context."""
    context = build_context_block(state["memories"], state["documents"])
    reply = generate_reply(runtime.context.history, context=context)
    return {"response": reply.text, "llm_success": reply.ok}


def extract_memories(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """Decide what in this exchange is worth remembering, and store it.

    Runs after generate on purpose: remembering is a side effect of the
    conversation, and the customer should never wait on it to see a reply.

    Nothing is extracted from a failed generation — there is no real exchange
    to learn from, and the fallback text is not something the customer said.
    """
    if not state["llm_success"]:
        return {}

    context = runtime.context
    try:
        candidates = extractor.extract(context.user_message, state["response"])
        fresh = deduplicator.filter_new(context.user_id, candidates)
        if fresh:
            memory_store.save_many(context.user_id, fresh)
        logger.info(
            "memory extraction: %d candidate(s), %d stored",
            len(candidates),
            len(fresh),
        )
    except Exception:
        logger.exception("memory extraction failed; reply is unaffected")
    return {}
