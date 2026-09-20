"""Pydantic request/response models.

These are the *only* shapes the frontend ever sees. Internal LangGraph state is
never serialised out of the backend.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreate(BaseModel):
    # No user id: the owner is the authenticated account, never a body field.
    title: str | None = None


class ConversationCreated(BaseModel):
    conversation_id: uuid.UUID


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class MemoryOut(BaseModel):
    id: str | None = None
    content: str
    type: str


class SourceOut(BaseModel):
    title: str
    source: str
    snippet: str | None = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    # What the agent used for this message. Empty for user messages, and for
    # assistant messages that recalled and cited nothing.
    memories: list[MemoryOut] = Field(default_factory=list)
    sources: list[SourceOut] = Field(default_factory=list)

    @classmethod
    def from_model(cls, message) -> "MessageOut":
        context = message.context or {}
        return cls(
            id=message.id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            memories=context.get("memories", []),
            sources=context.get("sources", []),
        )


class ConversationDetail(ConversationSummary):
    messages: list[MessageOut]


class ChatRequest(BaseModel):
    # No user id: taken from the token, never from the client.
    conversation_id: uuid.UUID
    message: str


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    message: str
    memory_used: bool = False
    memories: list[MemoryOut] = Field(default_factory=list)
    sources: list[SourceOut] = Field(default_factory=list)
