# CSAM — Customer Support Agent with Memory

A full-stack customer-support assistant that remembers its customers.

> **Status: all 9 phases complete.** The agent answers with Gemini 3.5
> Flash-Lite through a LangGraph workflow, checkpointed to PostgreSQL. It
> recalls durable facts about a customer across *separate* conversations,
> answers policy questions from the company's documentation, and shows which
> memories and which documents each answer drew on.
>
> Two things remain true by design. Memories and sources shown in the UI are
> exactly what the agent received — never invented, and never displayed when
> nothing was retrieved. And there is still **no authentication**: the user id
> identifies whose data this is, it does not authorise access.

## What it does

A customer chats with a support agent. Across *separate* conversations the
agent recalls durable facts about that customer ("you own a WM-200, you
reported a noise issue"), answers policy and troubleshooting questions from the
company's support documentation, and shows the customer exactly which memories
and which documents it used.

## Architecture

```text
                    USER
                     │
                     ▼
                  Next.js          browser; holds the user id, never a secret
                     │
                     ▼
                  FastAPI          the only process that talks to Gemini
                     │
                     ▼
                LangGraph          orchestrates the multi-step workflow
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   PostgreSQL     ChromaDB      Gemini
        │            │            │
        │       ┌────┴─────┐      └── response generation
        │       │          │           memory extraction
        │  memories   knowledge
        │
   users, conversations, messages, LangGraph checkpoints
```

**PostgreSQL** stores structured application and conversation data.
**ChromaDB** provides semantic retrieval for long-term memories and support
documentation. **LangGraph** orchestrates the multi-step AI workflow.
**Gemini** generates responses and extracts candidate memories.

### Why each piece

- **LangGraph** — answering one message is not one LLM call. It is retrieve
  memories → retrieve documents → generate → extract memories → save. LangGraph
  makes those steps explicit named nodes with typed state, so each is testable
  on its own and the whole run is checkpointed and resumable.
- **PostgreSQL** — conversations and messages are relational, queried by owner,
  and must survive a restart. It also backs LangGraph's checkpointer, so
  workflow state and application state share one durable store.
- **ChromaDB** — memories and documentation are retrieved by *meaning*, not by
  key. Chroma stores embeddings with metadata and filters on that metadata at
  query time, which is what makes per-user memory isolation enforceable.

### Four kinds of state — do not conflate them

| | Lives in | Holds | Scope |
|---|---|---|---|
| Conversation history | PostgreSQL `messages` | verbatim turns | one conversation |
| LangGraph checkpoint | PostgreSQL (checkpointer tables) | workflow state per `thread_id` | one conversation |
| Long-term memory | ChromaDB `customer_memories` | distilled durable facts | one **user**, across conversations |
| RAG knowledge | ChromaDB `support_knowledge` | chunked company docs | global, read-only |

Conversation history is *what was said*. Long-term memory is *what is worth
remembering*. The checkpoint is *where the workflow got to*. They are stored
separately and are never copies of each other.

### User identity

MVP only, no authentication. The browser calls `POST /users`, stores the
returned UUID in `localStorage` under `csam_user_id`, and sends it with each
request. It identifies *whose* data this is; it is not an authorisation claim,
and it must be replaced by real auth before any real deployment. Every
conversation query already filters by owner, and an id that does not own a
conversation gets a 404 rather than a hint that it exists.

## Install

Requirements: Python 3.11+, Node 20+, PostgreSQL 15+.

```bash
git clone <this repo> && cd csam
```

### 1. PostgreSQL

With Docker:

```bash
docker compose up -d postgres     # postgres:16 on :5432, db "csam"
```

Or with a PostgreSQL you already run locally:

```bash
createdb -h localhost -U postgres csam
createdb -h localhost -U postgres csam_test   # for the test suite
```

Docker is used for PostgreSQL only. FastAPI, Next.js and ChromaDB run locally —
containerising them adds operational complexity without teaching anything about
the architecture.

### 2. Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set DATABASE_URL and GEMINI_API_KEY
uvicorn app.main:app --reload --port 8000
```

Tables are created on startup. Open http://localhost:8000/docs.

Schema management is `create_all`, not a migration tool — deliberate while the
schema is not yet shared (see [Layout](#layout)). The practical consequence: a
*new* database gets the current schema automatically, but an existing one does
not pick up a column added later. Until a migration tool is warranted, recreate
the database or apply the change by hand.

### 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env.local    # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open http://localhost:3000.

### 4. Configure Gemini

Put `GEMINI_API_KEY` in `backend/.env` (get one from
[Google AI Studio](https://aistudio.google.com/apikey)). It is read only by
FastAPI. It is never sent to the browser and must never appear in any
`NEXT_PUBLIC_*` variable.

Without a key the app still runs: `/chat` persists the conversation and returns
a fallback message, logging `llm_success=false`. It never 500s because a
third-party API had a bad minute.

### 5. Ingest knowledge documents

```bash
python scripts/ingest_knowledge.py
```

Markdown under `backend/knowledge/` → chunked → embedded locally with
`sentence-transformers/all-MiniLM-L6-v2` → stored in ChromaDB. Embeddings are
generated on this machine; customer memories are never sent to a third-party
embedding API.

## Run the tests

```bash
cd backend
TEST_DATABASE_URL=postgresql+psycopg://postgres:<password>@localhost:5432/csam_test \
  .venv/bin/pytest -q
```

Tests run against a real PostgreSQL database, never SQLite, so they exercise the
same UUID columns, server defaults and cascades as production.

They never call Gemini: `conftest.py` stubs the LLM for every test, so the
suite is deterministic, offline and free even on a machine whose `.env` holds a
real API key. `app/llm/gemini.py`'s own failure paths are covered against a
fake client.

## Configuration

Nothing tunable is hard-coded. See `backend/.env.example`:
`DATABASE_URL`, `CHECKPOINTER_POOL_SIZE`, `GEMINI_MODEL`,
`GEMINI_TEMPERATURE`, `GEMINI_MAX_OUTPUT_TOKENS`, `GEMINI_THINKING_LEVEL`,
`CHROMA_PATH`, `MEMORY_TOP_K`, `MEMORY_SIMILARITY_THRESHOLD`,
`MEMORY_DEDUP_THRESHOLD`, `KNOWLEDGE_TOP_K`, `KNOWLEDGE_SIMILARITY_THRESHOLD`,
`KNOWLEDGE_CHUNK_SIZE`, `KNOWLEDGE_CHUNK_OVERLAP`, `HISTORY_MESSAGE_LIMIT`,
`MAX_MESSAGE_CHARS`.

### The similarity thresholds are measured, not guessed

Three of these decide what the agent sees, and they are not interchangeable.
Against the MiniLM embeddings actually in use:

| | Setting | Observed |
|---|---|---|
| Is this memory relevant? | `MEMORY_SIMILARITY_THRESHOLD` 0.30 | direct hit 0.53-0.71, indirect 0.36, unrelated 0.09 |
| Is this the *same* memory? | `MEMORY_DEDUP_THRESHOLD` 0.85 | reworded duplicate 0.88, different fact 0.58 |
| Is this document relevant? | `KNOWLEDGE_SIMILARITY_THRESHOLD` 0.35 | real question 0.55-0.65, passing remark 0.21, off-topic <0.01 |

*Relevant* and *identical* are different questions, which is why memory has two
thresholds rather than one. Re-measure after changing the embedding model:
these numbers belong to MiniLM, not to the code.

## Layout

```text
backend/
  app/
    main.py            FastAPI app, CORS, error handlers, /health
    config.py          all tunables, from the environment
    api/               chat.py conversations.py users.py
    agent/             state.py nodes.py graph.py prompts.py
    database/          connection.py models.py repositories/
    memory/            store.py retriever.py extractor.py deduplicator.py
    rag/               chunker.py ingestion.py retriever.py
    chroma.py          the ChromaDB client and its two collections
    llm/gemini.py      the only place Gemini is called
  knowledge/           support documentation (markdown)
  scripts/             ingest_knowledge.py
  tests/
frontend/
  app/                 layout.tsx page.tsx globals.css
  components/          chat/ conversations/ memory/ sources/
  lib/                 api.ts user.ts
  types/api.ts
```

## Development phases

| # | Phase | Status |
|---|---|---|
| 1 | Next.js + FastAPI + PostgreSQL | ✅ done |
| 2 | Gemini 3.5 Flash-Lite | ✅ done |
| 3 | LangGraph | ✅ done |
| 4 | LangGraph PostgreSQL checkpointing | ✅ done |
| 5 | ChromaDB collections | ✅ done |
| 6 | RAG over support docs | ✅ done |
| 7 | Long-term memory (extract + save + retrieve) | ✅ done |
| 8 | Memory deduplication | ✅ done |
| 9 | Memory / source UI, tests, polish | ✅ done |

Built in this order deliberately: each technology is added and verified on its
own, so a failure has one plausible cause.

## Demo

**Conversation 1** — "My name is Rahul. I own a WM-200 washing machine."
→ memory saved: user name, product.

**Conversation 2** (a fresh conversation) — "My washing machine is making a
loud noise again." → the agent retrieves the WM-200 and the earlier noise
issue, and the UI shows `🧠 2 memories used`.

**RAG** — "How long is the warranty?" → answered from the warranty document,
with `📚 Source: Warranty Policy` shown.

### What the badges mean

`🧠` lists the memories retrieved for that message; `📚` lists the documents
retrieved for it. They are what the agent *received*, which is the honest claim
the system can make — not a promise that the reply quotes every one. Both
render nothing when nothing was retrieved, and both are stored with the
message, so they survive a reload rather than vanishing.
