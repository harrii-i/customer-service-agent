import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import Conversation, Message


def add_message(
    db: Session,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
    context: dict | None = None,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        context=context,
    )
    db.add(message)
    # Touch the conversation so the sidebar can order by recency.
    conversation = db.get(Conversation, conversation_id)
    if conversation is not None:
        conversation.updated_at = func.now()
    db.commit()
    db.refresh(message)
    return message


def list_messages(db: Session, conversation_id: uuid.UUID) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.seq)
    )
    return list(db.scalars(stmt))


def recent_messages(
    db: Session, conversation_id: uuid.UUID, limit: int
) -> list[Message]:
    """Last `limit` messages in chronological order. The LLM never receives an
    unbounded history."""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.seq.desc())
        .limit(limit)
    )
    return list(reversed(list(db.scalars(stmt))))
