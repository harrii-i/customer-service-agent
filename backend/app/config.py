"""Application configuration.

Everything tunable lives here and is read from the environment, so no
thresholds / model names / paths are hard-coded across the codebase.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # PostgreSQL: application data + LangGraph checkpoints.
    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/csam"
    )
    # Separate from SQLAlchemy's pool: the checkpointer holds its own psycopg
    # connections and competes for the same server max_connections.
    checkpointer_pool_size: int = 5

    # Gemini. Backend-only: never exposed to the frontend.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    # Low but not zero: support answers should be consistent, not robotic.
    gemini_temperature: float = 0.3
    gemini_max_output_tokens: int = 1024
    # MINIMAL | LOW | MEDIUM | HIGH. Pinned so latency and cost do not move
    # when the model default does. Gemini 3.x replaced 2.5's thinking_budget
    # with this and rejects the old field outright.
    gemini_thinking_level: str = "minimal"

    # ChromaDB (Phase 5): local persistent client.
    chroma_path: str = "./data/chroma"

    # Long-term memory retrieval.
    memory_top_k: int = 5
    # Types that are always surfaced, bypassing the similarity floor. These are
    # the stable "profile" facts — who the customer is and what they own — that
    # they expect recalled however they ask ("who am I?", "what do you know
    # about me?"). Such meta-questions share no words with a declarative fact
    # ("Customer's name is …") and score far below any usable relevance floor,
    # so gating them on similarity made the agent deny knowing facts it held.
    # Issues and preferences stay similarity-gated: they are many and only
    # relevant in context, and always surfacing them would leak an unrelated
    # complaint into an off-topic message.
    memory_always_types: tuple[str, ...] = ("identity", "product")
    # Ceiling on the always-surfaced profile, so a customer with a long
    # appliance history does not swamp the prompt.
    memory_profile_limit: int = 6
    # Floor for a memory to count as relevant to the current message.
    # Calibrated against real extracted memories and the MiniLM embeddings
    # actually in use: a directly relevant memory scores 0.53-0.71, an
    # indirect one ("what is my name?") 0.36, and an unrelated question 0.09.
    # 0.30 clears the indirect case with headroom and still rejects noise.
    memory_similarity_threshold: float = 0.30
    # Floor for two memories to be *the same fact*. Much higher than the
    # relevance floor on purpose: relevant and identical are different
    # questions, and one number cannot answer both.
    memory_dedup_threshold: float = 0.85

    # RAG retrieval / ingestion.
    knowledge_top_k: int = 5
    # Floor on cosine similarity for a documentation chunk to be usable. The
    # nearest chunks always exist; without a floor the agent cites a delivery
    # policy when asked about a noise. Calibrated: a real question's best
    # chunk scores 0.55-0.65, a passing remark ("thanks, that fixed it")
    # peaks at 0.21, and an off-topic question below 0.01.
    knowledge_similarity_threshold: float = 0.35
    knowledge_chunk_size: int = 700
    knowledge_chunk_overlap: int = 80

    # How much conversation history is handed to the LLM.
    history_message_limit: int = 20

    # How many earlier customer turns join the current one to form a retrieval
    # query. A follow-up ("what was the issue reported?") carries no subject —
    # the subject is in the conversation — and on its own it retrieves nothing.
    # Measured: that question alone scores 0.17 against the right memory; with
    # the previous turn attached it scores 0.49.
    retrieval_context_turns: int = 2

    # Ceiling on a single inbound message. This is a cost and abuse control on
    # the Gemini call, not a body-size limit — the request is already parsed by
    # the time it is checked.
    max_message_chars: int = 4000

    frontend_origin: str = "http://localhost:3000"

    # --- Authentication -----------------------------------------------------
    # Signing key for JWTs. There is no usable default on purpose: a shared
    # fallback secret is a forged-token vulnerability, so the app refuses to
    # start without one rather than quietly accepting anybody's tokens.
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # a week; a support chat is not a bank
    password_min_length: int = 8

    # The token is delivered as an httpOnly cookie, so JavaScript — and any
    # script injected into the page — cannot read it.
    auth_cookie_name: str = "csam_token"
    # Must be True wherever the site is served over HTTPS. Left False so the
    # http://localhost dev setup works; the README says to turn it on.
    auth_cookie_secure: bool = False
    auth_cookie_samesite: str = "lax"


@lru_cache
def get_settings() -> Settings:
    return Settings()
