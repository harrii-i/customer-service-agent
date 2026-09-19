import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.repositories import conversations as conversation_repo
from app.database.repositories import users as user_repo
from app.schemas import ConversationSummary, UserCreated

router = APIRouter(tags=["users"])


@router.post("/users", response_model=UserCreated, status_code=201)
def create_user(db: Session = Depends(get_db)) -> UserCreated:
    """MVP identity: no auth. The frontend stores the returned id in
    localStorage under `csam_user_id` and sends it with every request. It marks
    *which* data belongs to whom; it is not an authorisation claim."""
    user = user_repo.create_user(db)
    return UserCreated(user_id=user.id)


@router.get(
    "/users/{user_id}/conversations", response_model=list[ConversationSummary]
)
def list_user_conversations(
    user_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[ConversationSummary]:
    user_repo.get_or_create_user(db, user_id)
    return [
        ConversationSummary.model_validate(c)
        for c in conversation_repo.list_conversations(db, user_id)
    ]
