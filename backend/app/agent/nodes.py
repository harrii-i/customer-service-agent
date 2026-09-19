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
