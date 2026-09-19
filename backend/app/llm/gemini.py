"""The only place Gemini is called.

Everything above this module deals in `(role, content)` turns and plain
strings; nothing else in the codebase imports `google.genai`. That keeps the
provider swappable and keeps the API key in exactly one place.

This layer never raises. A support chat that 500s because a third-party API
had a bad minute is worse than one that says so and keeps the conversation
alive, so every failure path returns `LlmReply(FALLBACK_REPLY, ok=False)` and
logs the cause. `ok` is what the caller reports as `llm_success`; it is never
inferred from the text.
"""

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import NamedTuple

from google import genai
from google.genai import types

from app.agent.prompts import SYSTEM_INSTRUCTION
from app.config import get_settings

logger = logging.getLogger("csam.llm")

settings = get_settings()

FALLBACK_REPLY = (
    "Sorry — I can't reach the support assistant right now. Please try "
    "again in a moment."
)

# Gemini names the assistant role "model"; our database calls it "assistant".
# Roles outside this map (a future "system" / "tool") are not conversation
# turns, so they are dropped rather than guessed at.
_ROLE_TO_GEMINI = {"user": "user", "assistant": "model"}


class LlmReply(NamedTuple):
    """`ok` is False whenever the text is the fallback rather than a model
    response — the caller logs it and must not present it as an AI answer."""

    text: str
    ok: bool


@lru_cache
def _client() -> genai.Client | None:
    """Built once, lazily. Lazily so that importing this module — which the
    test suite and any tooling does — never requires an API key, and once
    because the client holds a connection pool."""
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY is not set; /chat will use the fallback reply")
        return None
    return genai.Client(api_key=settings.gemini_api_key)


def build_contents(turns: Sequence[tuple[str, str]]) -> list[types.Content]:
    """Map stored `(role, content)` turns onto Gemini's content list.

    The caller passes history that already ends with the current user message,
    so no turn is special-cased here.
    """
    contents: list[types.Content] = []
    for role, text in turns:
        gemini_role = _ROLE_TO_GEMINI.get(role)
        if gemini_role is None or not text.strip():
            continue
        contents.append(
            types.Content(role=gemini_role, parts=[types.Part(text=text)])
        )
    return contents


def generate_json(
    system_instruction: str, user_text: str, response_schema: type
) -> object | None:
    """Ask Gemini for structured JSON matching `response_schema`.

    Used for extraction rather than conversation, so it takes a single block of
    text instead of a turn list. Returns None on any failure — a caller that
    cannot parse a result must not guess one, and memory extraction failing is
    never a reason to fail the customer's reply.
    """
    client = _client()
    if client is None:
        return None

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,  # extraction is not a creative task
                response_mime_type="application/json",
                response_schema=response_schema,
                thinking_config=types.ThinkingConfig(
                    thinking_level=settings.gemini_thinking_level
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
    except Exception:
        logger.exception("Gemini structured call failed")
        return None

    parsed = getattr(response, "parsed", None)
    if parsed is None:
        logger.error("Gemini returned no parsable JSON")
    return parsed


def generate_reply(
    turns: Sequence[tuple[str, str]], context: str | None = None
) -> LlmReply:
    """Generate the assistant's next turn from recent conversation history.

    `turns` is `(role, content)` oldest-first, already truncated to
    `history_message_limit` by the caller: this module never reads the
    database and never decides how much history is affordable.

    `context` is retrieved memories and documents, already worded by
    `agent.prompts`, appended to the system instruction. None until Phases 6
    and 7 have something to retrieve.
    """
    client = _client()
    if client is None:
        return LlmReply(FALLBACK_REPLY, ok=False)

    contents = build_contents(turns)
    if not contents:
        logger.error("refusing to call Gemini with no usable turns")
        return LlmReply(FALLBACK_REPLY, ok=False)

    system_instruction = SYSTEM_INSTRUCTION
    if context:
        system_instruction = f"{SYSTEM_INSTRUCTION}\n{context}\n"

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.gemini_max_output_tokens,
        # Pinned so latency and cost stay predictable rather than tracking
        # whatever the model default happens to be. Gemini 3.x takes a level;
        # 2.5's thinking_budget is rejected by these models.
        thinking_config=types.ThinkingConfig(
            thinking_level=settings.gemini_thinking_level
        ),
        # We pass no tools, so automatic function calling has nothing to do.
        # Saying so explicitly stops the SDK warning it emits on every call.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    )

    try:
        response = client.models.generate_content(
            model=settings.gemini_model, contents=contents, config=config
        )
    except Exception:
        # Deliberately broad: a transport error, a 429 and a malformed request
        # all mean the same thing to the customer. Logged with a traceback, so
        # the cause is never lost.
        logger.exception("Gemini call failed")
        return LlmReply(FALLBACK_REPLY, ok=False)

    text = (response.text or "").strip()
    if not text:
        # Empty text is not success. It usually means a safety block or the
        # output token cap was hit, both of which are visible on the candidate.
        finish_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
        logger.error("Gemini returned no text (finish_reason=%s)", finish_reason)
        return LlmReply(FALLBACK_REPLY, ok=False)

    return LlmReply(text, ok=True)
