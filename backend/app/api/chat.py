import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agent.graph import get_graph
from app.auth.dependencies import get_current_user
from app.agent.state import AgentContext, initial_state
from app.config import get_settings
from app.database.connection import get_db
from app.database.models import User
from app.database.repositories import conversations as conversation_repo
from app.database.repositories import messages as message_repo
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger("csam.chat")

settings = get_settings()

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    started = time.perf_counter()

    user_message = payload.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(user_message) > settings.max_message_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Message cannot exceed {settings.max_message_chars} characters",
        )

    conversation = conversation_repo.get_conversation_for_user(
        db, payload.conversation_id, current_user.id
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation_repo.set_title_if_empty(db, conversation, user_message)
    message_repo.add_message(db, conversation.id, "user", user_message)

    # --- Run the LangGraph workflow. ------------------------------------------
    # The user message is persisted *before* this, so a failed run never loses
    # what the customer typed. History therefore already ends with it.
    #
    # Database reads stay here rather than inside a node: a Session is not
    # serialisable and must never reach graph state. Nodes that need their own
    # I/O (ChromaDB) open it themselves.
    history = message_repo.recent_messages(
        db, conversation.id, settings.history_message_limit
    )
    result = get_graph().invoke(
        initial_state(),
        context=AgentContext(
            # The identity handed to the workflow — and therefore to
            # ChromaDB's per-user filter — is the authenticated one.
            user_id=str(current_user.id),
            conversation_id=str(conversation.id),
            user_message=user_message,
            history=[(m.role, m.content) for m in history],
        ),
        # One conversation is one resumable run. The checkpoint records where
        # the workflow got to; the transcript itself stays in `messages`.
        config={"configurable": {"thread_id": str(conversation.id)}},
    )
    response_text = result["response"]
    llm_success = result["llm_success"]
    # Empty until Phase 7 (memory) and Phase 6 (documents) fill the retrieval
    # nodes. The UI must never display memory or sources that do not exist.
    memories = result["memories"]
    sources = result["documents"]
    # --------------------------------------------------------------------------

    # The fallback is persisted too, so the stored transcript matches what the
    # customer actually saw on screen. What the agent used is persisted with
    # it, so the UI can show the same provenance after a reload.
    message_repo.add_message(
        db,
        conversation.id,
        "assistant",
        response_text,
        context={"memories": memories, "sources": sources} if (memories or sources) else None,
    )

    # Structured observability. Message contents are deliberately not logged.
    logger.info(
        "chat_completed conversation_id=%s user_id=%s memories=%d documents=%d "
        "llm_success=%s latency=%.2fs",
        conversation.id,
        current_user.id,
        len(memories),
        len(sources),
        llm_success,
        time.perf_counter() - started,
    )

    return ChatResponse(
        conversation_id=conversation.id,
        message=response_text,
        memory_used=bool(memories),
        memories=memories,
        sources=sources,
    )
