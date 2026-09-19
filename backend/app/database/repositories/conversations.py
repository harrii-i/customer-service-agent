import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Conversation


def create_conversation(
    db: Session, user_id: uuid.UUID, title: str | None = None
) -> Conversation:
    conversation = Conversation(user_id=user_id, title=title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def list_conversations(db: Session, user_id: uuid.UUID) -> list[Conversation]:
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(db.scalars(stmt))


def get_conversation_for_user(
    db: Session, conversation_id: uuid.UUID, user_id: uuid.UUID
) -> Conversation | None:
    """Ownership is enforced in the query itself, not by a later `if` that an
    edit could drop. A conversation is only ever reachable through its owner."""
    stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id,
    )
    return db.scalars(stmt).first()


def set_title_if_empty(db: Session, conversation: Conversation, text: str) -> None:
    """Title comes from the first user message, truncated. No LLM call."""
    if conversation.title:
        return
    title = " ".join(text.split())
    conversation.title = title[:37] + "..." if len(title) > 40 else title
    db.commit()
