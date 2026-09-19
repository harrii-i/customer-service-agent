import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.repositories import conversations as conversation_repo
from app.database.repositories import messages as message_repo
from app.database.repositories import users as user_repo
from app.schemas import (
    ConversationCreate,
    ConversationCreated,
    ConversationDetail,
    MessageOut,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationCreated, status_code=201)
def create_conversation(
    payload: ConversationCreate, db: Session = Depends(get_db)
) -> ConversationCreated:
    user_repo.get_or_create_user(db, payload.user_id)
    conversation = conversation_repo.create_conversation(
        db, payload.user_id, payload.title
    )
    return ConversationCreated(conversation_id=conversation.id)


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="Owning user id"),
    db: Session = Depends(get_db),
) -> ConversationDetail:
    conversation = conversation_repo.get_conversation_for_user(
        db, conversation_id, user_id
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
