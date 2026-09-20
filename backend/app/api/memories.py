"""The customer's own memories: see them, delete them.

Every route takes its user id from the token. There is no `{user_id}` in any
path, which is the strongest form of the rule in the spec — an endpoint that
cannot name another user cannot be tricked into serving one.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import get_current_user
from app.database.models import User
from app.memory import store
from app.schemas import MemoryOut

logger = logging.getLogger("csam.memories")

router = APIRouter(prefix="/memories", tags=["memories"])


@router.get("", response_model=list[MemoryOut])
def list_memories(current_user: User = Depends(get_current_user)) -> list[MemoryOut]:
    """Everything the agent remembers about the signed-in customer.

    People should be able to see what a system has stored about them, and —
    below — delete it.
    """
    return [
        MemoryOut(id=m["id"], content=m["content"], type=m["type"])
        for m in store.all_for_user(str(current_user.id))
    ]


@router.delete("/{memory_id}", status_code=204)
def delete_memory(
    memory_id: str, current_user: User = Depends(get_current_user)
) -> None:
    if not store.delete_one(str(current_user.id), memory_id):
        # Same 404 for "no such memory" and "not yours": a different response
        # would confirm that some other customer holds that id.
        raise HTTPException(status_code=404, detail="Memory not found")


@router.delete("", status_code=204)
def forget_everything(current_user: User = Depends(get_current_user)) -> None:
    deleted = store.delete_for_user(str(current_user.id))
    logger.info("forgot all memories user_id=%s count=%d", current_user.id, deleted)
