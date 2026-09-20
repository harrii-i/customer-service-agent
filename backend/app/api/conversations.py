"""Conversations, owned by the authenticated account.

No endpoint here accepts a user id. Ownership comes from the token, so a
client cannot create a conversation for someone else, and cannot read one by
guessing its id.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.connection import get_db
from app.database.models import User
from app.database.repositories import conversations as conversation_repo
from app.database.repositories import messages as message_repo
from app.schemas import (
    ConversationCreate,
    ConversationCreated,
    ConversationDetail,
    ConversationSummary,
    MessageOut,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationCreated, status_code=201)
def create_conversation(
    payload: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationCreated:
    conversation = conversation_repo.create_conversation(
        db, current_user.id, payload.title
    )
    return ConversationCreated(conversation_id=conversation.id)


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ConversationSummary]:
    """Only this account's conversations. The caller cannot ask for anyone
    else's, because there is nowhere to say whose."""
    return [
        ConversationSummary.model_validate(c)
        for c in conversation_repo.list_conversations(db, current_user.id)
    ]


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationDetail:
    conversation = conversation_repo.get_conversation_for_user(
        db, conversation_id, current_user.id
    )
    if conversation is None:
        # Deliberately the same 404 whether the conversation does not exist or
        # belongs to someone else: a mismatched id must not confirm existence.
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = message_repo.list_messages(db, conversation_id)
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[MessageOut.from_model(m) for m in messages],
    )
