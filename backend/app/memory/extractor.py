"""Deciding what is worth remembering about a customer.

Conversation history is *what was said*; a memory is *what is worth
remembering*. This module draws that line, and it draws it narrowly: a memory
must still be true and still be useful in a different conversation weeks later.

Extraction runs on the exchange that just happened, not on the whole
transcript, because everything earlier has already been offered for extraction
once.
"""

import logging
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from app.llm.gemini import generate_json

logger = logging.getLogger("csam.memory.extractor")

MemoryType = Literal["identity", "product", "issue", "preference"]

EXTRACTION_TEMPLATE = """\
You extract durable facts about a customer from a support exchange, for recall \
in *future, separate* conversations.

Record a fact only if all of these hold:
- It is about the customer, their appliance, or their situation — not about \
what the assistant said.
- It will still be true and still be useful weeks from now.
- The customer stated it. Never infer, guess, or embellish.

Today's date is {today}. Never write a relative time reference — "yesterday", \
"last week", "this morning" — because a memory is read back weeks later, when \
it is no longer true. Convert it to an absolute date: a purchase made \
"yesterday" is recorded as the calendar date before today.

Use these types:
- identity: who they are (name, household, location).
- product: an appliance they own, including model where given.
- issue: a problem they reported, including when it happens.
- preference: how they want to be treated or contacted.

Do NOT record: greetings, thanks, one-off pleasantries, anything the assistant \
said, questions the customer asked, or transient state ("the machine is \
running right now").

Write each fact as one short, self-contained sentence that makes sense alone, \
starting with "Customer". If nothing qualifies, return an empty list. An empty \
list is a good and common answer.
"""


class ExtractedMemory(BaseModel):
    content: str = Field(description="One short self-contained fact.")
    type: MemoryType


class Extraction(BaseModel):
    memories: list[ExtractedMemory]


def extraction_instruction(today: date | None = None) -> str:
    """The instruction, dated. The model cannot resolve "yesterday" without
    knowing when today is, and a memory holding a relative date is simply
    wrong the next time it is read."""
    return EXTRACTION_TEMPLATE.format(today=(today or date.today()).isoformat())


def extract(user_message: str, assistant_message: str) -> list[dict]:
    """Candidate memories from one exchange. Never raises.

    Returns [] when nothing qualifies *and* when extraction fails. Both mean
    the same thing to the caller: there is nothing new to remember. Failing to
    remember must never fail the customer's reply.
    """
    exchange = f"Customer said:\n{user_message}\n\nAssistant replied:\n{assistant_message}"
    result = generate_json(extraction_instruction(), exchange, Extraction)

    if not isinstance(result, Extraction):
        logger.warning("memory extraction produced nothing usable")
        return []

    candidates = [
        {"content": m.content.strip(), "type": m.type}
        for m in result.memories
        if m.content.strip()
    ]
    logger.info("memory extraction produced %d candidate(s)", len(candidates))
    return candidates
