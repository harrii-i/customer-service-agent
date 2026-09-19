import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.agent.checkpointer import setup_checkpointer
from app.api import chat, conversations, users
from app.config import get_settings
from app.database.connection import engine
from app.database.models import Base

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("csam")

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # MVP schema management. A migration tool is warranted once the schema is
    # shared with other people, not before.
    Base.metadata.create_all(bind=engine)
    # LangGraph manages its own checkpoint tables. This must happen at startup,
    # never inside a request: its migration blocks on open transactions.
    setup_checkpointer()
    logger.info("startup complete: schema ensured")
    yield


app = FastAPI(
    title="CSAM — Customer Support Agent with Memory",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(SQLAlchemyError)
def handle_database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """Surface DB outages as 503 rather than a bare 500 — and log them. Errors
    are never silently swallowed."""
    logger.exception("database error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ok"}


app.include_router(users.router)
app.include_router(conversations.router)
app.include_router(chat.router)
