# CSAM — Architecture

How this system is put together, for someone who has to change it.

The [README](README.md) covers what CSAM does, why each technology was chosen,
and how to install and run it. This document is the level below that: the
request lifecycle, the agent graph, the retrieval and memory pipelines, the
invariants that hold them together, and where the known weak points are.

Read the README first. Nothing here repeats it except where a detail is needed
to make sense of the mechanism.

---

## Contents

1. [Runtime shape](#1-runtime-shape)
2. [The lifecycle of one message](#2-the-lifecycle-of-one-message)
3. [The agent graph](#3-the-agent-graph)
4. [State: what lives where](#4-state-what-lives-where)
5. [Retrieval (RAG)](#5-retrieval-rag)
6. [Long-term memory](#6-long-term-memory)
7. [Policy layers: guardrail and escalation](#7-policy-layers-guardrail-and-escalation)
8. [Identity and isolation](#8-identity-and-isolation)
9. [Configuration](#9-configuration)
10. [Testing architecture](#10-testing-architecture)
11. [Evaluation harness](#11-evaluation-harness)
12. [Invariants](#12-invariants)
13. [Known limitations](#13-known-limitations)
14. [Extension points](#14-extension-points)

---

## 1. Runtime shape

Four processes, three of them local in development:

| Process | Role |
|---|---|
| Next.js (`:3000`) | Browser UI. Holds no secret — the session token is an httpOnly cookie it cannot read. |
| FastAPI (`:8000`) | The only process that talks to Gemini, PostgreSQL or ChromaDB. |
| PostgreSQL (`:5432`) | Application data **and** LangGraph checkpoints. One durable store. |
| ChromaDB | Embedded, not a server. A persistent client inside the FastAPI process. |

Gemini is the one external dependency. Embeddings are **not** sent to it:
ChromaDB computes them locally with ONNX all-MiniLM-L6-v2
([`app/chroma.py`](backend/app/chroma.py)), so no customer memory ever leaves
the machine to be embedded.

### Module map

```text
backend/app/
  main.py              FastAPI app, CORS, SQLAlchemy error handler, /health,
                       lifespan (runs the checkpointer migration once)
  config.py            every tunable, read from the environment
  schemas.py           the only shapes the frontend ever sees

  api/                 auth.py  chat.py  conversations.py  memories.py
  auth/                security.py (Argon2id + JWT)  dependencies.py  schemas.py

  agent/
    state.py           AgentContext (per-run input) | AgentState (checkpointed)
    graph.py           topology: nodes, two conditional edges
    nodes.py           the node functions
    prompts.py         SYSTEM_INSTRUCTION, build_context_block
    guardrail.py       input classification + canned deflections
    escalation.py      ungrounded-run counting + handoff policy
    checkpointer.py    LangGraph PostgresSaver and its connection pool

  memory/              store.py  retriever.py  extractor.py  deduplicator.py
  rag/                 chunker.py  ingestion.py  retriever.py
  chroma.py            the ChromaDB client and its two collections
  llm/gemini.py        the only place google.genai is imported
  database/            connection.py  models.py  repositories/

backend/knowledge/     support documentation (markdown) — the RAG corpus
backend/evals/         dataset.json  gold.py  metrics.py
backend/migrations/    alembic
backend/tests/
frontend/              app/  components/  lib/api.ts  types/api.ts
```

Two boundaries are enforced by convention and worth preserving:

- **`llm/gemini.py` is the only module that imports `google.genai`.** Everything
  above it deals in `(role, content)` turns and plain strings. Swapping
  provider means rewriting one file.
- **`chroma.py` is the only module that constructs a Chroma client.** One
  persistent client per process; opening two against the same path is trouble.

---

## 2. The lifecycle of one message

`POST /chat` — [`app/api/chat.py`](backend/app/api/chat.py).

```text
 1. get_current_user          JWT from httpOnly cookie, else Bearer header.
                              The user id is NEVER read from the request body.
 2. validate                  non-empty, <= max_message_chars (4000) -> 400
 3. ownership check           get_conversation_for_user -> 404 if not yours
 4. persist the user message  BEFORE the graph runs, so a failed run never
                              loses what the customer typed
 5. load history              recent_messages(limit=20), oldest-first
 6. derive prior outcomes     escalation.outcomes_from(history) — no extra query
 7. graph.invoke(...)         context=AgentContext(...), thread_id=conversation.id
 8. classify the outcome      escalation.classify_outcome(result)
 9. persist the reply         with context={"outcome", "memories", "sources"}
10. structured log            message contents deliberately not logged
11. ChatResponse              message, memory_used, memories[], sources[]
```

Two decisions in that sequence carry weight:

**The user message is persisted before the graph runs.** So `history` already
ends with the current message, and no node has to special-case "the thing being
answered". `generate` passes the turn list to Gemini unmodified.

**No database session enters graph state.** A `Session` is not serialisable and
graph state is checkpointed, so all SQL happens in the endpoint. Nodes that
need I/O (ChromaDB) open it themselves.

---

## 3. The agent graph

[`app/agent/graph.py`](backend/app/agent/graph.py). Compiled once, lazily
(`@lru_cache`), because compiling attaches the checkpointer and opens a
connection pool — importing the module must not do that.

```text
START
  │
  ▼
guardrail ──(blocked)─────────────────────────────────────────────────► END
  │
  │ (in scope)
  ▼
retrieve_memories ──► retrieve_knowledge ──(grounded, or still trying)──► generate
                                        │                                   │
                                        └──(out of tries)──► escalate ──► END
                                                                            ▼
                                                                   extract_memories
                                                                            │
                                                                            ▼
                                                                           END
```

**Two conditional edges, and both decide on the same principle:** the decision
itself is a pure function elsewhere; the router only maps it onto an edge.

| Edge | Router | Decision lives in |
|---|---|---|
| after `guardrail` | `_route_after_guardrail(state)` | `state["blocked"]`, set by the node |
| after `retrieve_knowledge` | `_route_after_knowledge(state, runtime)` | `escalation.should_escalate` |

Everything else is a fixed line. Order is the design, not an accident:
retrieval runs before generation because its results *become* the prompt
context, and extraction runs after because it reads what generation produced.

### The nodes

[`app/agent/nodes.py`](backend/app/agent/nodes.py). Each is an ordinary
`(state, runtime) -> partial state` function, directly callable from a test
without a real LangGraph `Runtime`.

- **`check_guardrail`** — classifies the message. In-scope passes through,
  recording whether it was a support request at all. Anything else is deflected
  with a canned reply and the run ends.
- **`retrieve_memories`** — this customer's relevant memories. Best-effort: a
  ChromaDB outage costs context, not the reply.
- **`retrieve_knowledge`** — matching documentation chunks. Also best-effort.
- **`generate`** — builds the context block and calls Gemini. Never raises.
- **`escalate`** — deterministic handoff reply. No model call, no retrieval,
  nothing learned.
- **`extract_memories`** — distils the exchange into durable facts and stores
  them. Runs *after* generate so the customer never waits on it, and is skipped
  entirely when generation failed.

### Cost per turn

Up to **three Gemini calls**: guardrail classification, generation, memory
extraction. A blocked turn makes one. An escalated turn makes one. All three
are stubbed in tests.

---

## 4. State: what lives where

The README's "four kinds of state" table is the conceptual version. This is how
the split is *enforced*.

[`app/agent/state.py`](backend/app/agent/state.py) defines two shapes, and the
difference between them is the whole point:

**`AgentContext`** — per-run input, **never checkpointed**.

```python
user_id, conversation_id, user_message
history: list[tuple[str, str]]      # already truncated, ends with user_message
recent_outcomes: list[str]          # prior assistant outcomes, oldest first
```

The transcript is already owned by the `messages` table. A checkpoint holding
its own copy would be a second source of truth for the same conversation, so it
does not get one. This is asserted, not merely intended, by
`test_the_checkpoint_does_not_become_a_second_transcript`, which greps the
checkpoint tables for a phrase that appeared only in the transcript.

**`AgentState`** — what the workflow produces, checkpointed per `thread_id`.

```python
memories: list[dict]     documents: list[dict]
response: str            llm_success: bool
blocked: bool            escalated: bool        courtesy: bool
```

### The reset that surprises people

`chat.py` invokes the graph with a fresh `initial_state()` **every turn**.
`AgentState` is a plain `TypedDict` with no reducers, so that input overwrites
the checkpointed values before the first node runs.

**Consequence: the checkpoint is durable within a run, but reset at the start
of each request.** Nothing in `AgentState` survives into the next turn. Any
cross-turn counter must travel through `AgentContext` (as `recent_outcomes`
does) or be re-derived from the database. This is consistent with
`initial_state()`'s stated intent — "every run starts from empty retrieval" —
but it will silently defeat any design that assumes otherwise.

### Per-message provenance

`messages.context` is a nullable `JSONB` column holding what the agent used:

```json
{"outcome": "grounded", "memories": [...], "sources": [...]}
```

`MessageOut.from_model` reads only `memories` and `sources`, so additional keys
are invisible to the API. That is what made adding `outcome` a zero-migration
change. User messages keep `context = NULL`.

`outcome` is a closed set — `grounded`, `ungrounded`, `blocked`, `escalated`,
`failed`, `courtesy`, plus `unknown` for rows written before the field existed.
Four of those produce a reply with no sources and no memories, and conflating
them is exactly the bug the field exists to prevent.

### Transcript ordering

`messages.seq` is a database `Identity` column, and it is the ordering key —
not `created_at`. PostgreSQL's `now()` is the *transaction* timestamp, so
messages written in one transaction share it exactly, and the primary key is a
random UUID, so tie-breaking on it shuffles the conversation. This bit the
project once; `test_transcript_order_survives_identical_timestamps` keeps it
fixed.

---

## 5. Retrieval (RAG)

### Ingestion

`backend/knowledge/*.md` → [`rag/chunker.py`](backend/app/rag/chunker.py) →
[`rag/ingestion.py`](backend/app/rag/ingestion.py) → the `support_knowledge`
collection. Run with `scripts/ingest_knowledge.py`.

Chunking splits on blank lines, greedily packs paragraphs up to
`knowledge_chunk_size` (700 chars), and carries `knowledge_chunk_overlap` (80
chars) of tail into the next chunk so a fact straddling a boundary survives
whole in at least one of them. Titles come from the first `# ` heading, falling
back to the filename — a chunk is cited to a customer, so it always needs a
name.

Ingestion is a **full rebuild**, not an append: chunk ids are
`f"{source}::{index}"`, and existing ids are deleted first. Stale documentation
that still answers questions is worse than none.

Each chunk carries `{"title", "source", "index"}`. Source document and chunk
ordinal are both recoverable from the id alone.

> A chunk is **not** a byte-for-byte slice of its file — paragraphs are
> stripped and re-joined, and the overlap tail creates a seam. Any code
> comparing a chunk to its source must normalise whitespace and compare
> paragraph by paragraph. `test_cited_snippets_come_from_the_document_they_name`
> does exactly this.

### Query time

[`rag/retriever.py`](backend/app/rag/retriever.py):

1. Query the collection for `knowledge_top_k` (5) nearest chunks.
2. **Filter** — discard anything below `knowledge_similarity_threshold` (0.35).
   This is the abstention mechanism. Without it the nearest five chunks come
   back regardless of relevance, and the agent cites a delivery policy when
   asked about geography.
3. **Deduplicate by title** — keep each document's single strongest chunk. The
   customer is shown "Source: Warranty Policy", not "chunk 3 of 4".
4. **Rank** best-first and strip the internal `_score`.

Collections use **cosine** distance, not Chroma's L2 default
([`chroma.py`](backend/app/chroma.py)), and `similarity(d) = 1 - d` converts it.
Every threshold in `config.py` is expressed as a similarity. With L2 those
numbers would be meaningless.

The returned dicts are shaped exactly like the API's `SourceOut`, so what the
agent received and what the customer is shown are the same objects. The UI
cannot display a citation the agent did not actually get.

### Query construction

`nodes.retrieval_query` does **not** search with the bare message. It joins the
last `retrieval_context_turns` (2) customer turns to the current one, because
follow-ups carry no subject of their own — "what was the issue reported?"
scores 0.17 against the right memory alone, and 0.49 with the previous turn
attached. Assistant replies are deliberately excluded: they are long, and
letting them in lets the agent's own wording decide what gets retrieved.

This has a cost worth knowing about: a topic switch after two unrelated turns
can suppress retrieval that succeeds standalone. See
[Known limitations](#13-known-limitations).

---

## 6. Long-term memory

The `customer_memories` collection, partitioned by `user_id` in metadata. Every
query filters on it **inside the Chroma query**, not afterwards in Python —
another customer's memory is never a candidate in the first place.

```text
exchange ──► extractor ──► deduplicator ──► store ──► (later) retriever ──► prompt
             (Gemini)      (similarity)     (Chroma)
```

**Extract** ([`memory/extractor.py`](backend/app/memory/extractor.py)) — a
structured Gemini call over the exchange that just happened. A fact is recorded
only if it is about the customer, still true weeks later, and actually stated
rather than inferred. Types: `identity`, `product`, `issue`, `preference`. The
prompt is told today's date and forbidden from writing relative times — a
memory saying "yesterday" is wrong the next time it is read.

**Deduplicate** ([`memory/deduplicator.py`](backend/app/memory/deduplicator.py))
— semantic, not textual. "Owns a WM-200 washing machine" and "has a WM-200
washer" are one fact. `memory_dedup_threshold` (0.85) is deliberately far above
the retrieval threshold: *relevant* and *the same fact* are different questions
and one number cannot answer both. Within-batch duplicates are caught too,
since neither copy is in the store yet.

**Retrieve** ([`memory/retriever.py`](backend/app/memory/retriever.py)) — two
sources, merged:

- **Profile facts** (`memory_always_types` = `identity`, `product`) are surfaced
  **unconditionally**, capped at `memory_profile_limit` (6). "Who am I?" shares
  no words with "Customer's name is …" and scores far below any usable floor;
  gating these on similarity made the agent deny knowing what it held.
- **Everything else** stays similarity-gated at `memory_similarity_threshold`
  (0.30). Without a floor every message recalls the nearest memory whether or
  not it relates, and the agent opens a delivery question by mentioning an old
  noise.

> The unconditional half has a consequence that is easy to miss: for any
> customer who has ever been remembered, `memories` is **non-empty on every
> turn**. Any logic treating "memories exist" as "the question was answered"
> will be wrong for every returning customer. `escalation.has_grounding`
> exists precisely to not make that mistake.

---

## 7. Policy layers: guardrail and escalation

Two independent policies, both deliberately deterministic at the point where
the customer sees something.

### Guardrail — should this message reach the agent?

[`app/agent/guardrail.py`](backend/app/agent/guardrail.py). A structured Gemini
call classifies the message as `injection`, `unsafe`, `off_topic` or
`in_scope`, in that priority order, and reads it as *data to categorise, never
instructions to follow*. Blocked categories get a canned deflection from
`_DEFLECTIONS` — the reply a customer sees for an out-of-scope request is
decided in code, not by the model that is under attack.

It **fails open**. If classification cannot run, the message is treated as
in-scope and the system prompt remains the backstop. A guardrail that failed
closed would turn a bad Gemini minute into a bot that refuses everyone — worse
than the risk it guards against.

`GuardrailVerdict` carries a second, additive field, `is_support_request`,
defaulted to `True`. It marks pure courtesy ("thanks", "got it") without
touching the category, so the blocked/allowed decision is untouched. This rides
on the call the guardrail already makes — it is not a fourth Gemini call.

### Escalation — has the agent stopped being useful?

[`app/agent/escalation.py`](backend/app/agent/escalation.py). Four pure
functions, no I/O, no model call:

```python
has_grounding(state)              # documents, or a memory that cleared the floor
classify_outcome(state)           # one of the seven outcome words
consecutive_ungrounded(outcomes)  # how long the current run is
should_escalate(state, context)   # does this turn end it
```

The rule: after `escalation_ungrounded_limit` (3) **consecutive ungrounded**
assistant turns, hand over to a human instead of answering again. The current
turn counts toward the limit, so the customer gets two real attempts first.

What does *not* count is most of the design:

| Outcome | Effect on the run |
|---|---|
| `grounded` | resets — the agent found what was asked for |
| `ungrounded` | **extends** |
| `courtesy` | **skipped** — transparent, neither counted nor reset |
| `blocked` / `failed` / `escalated` / `unknown` | resets |

Courtesy being *transparent* rather than resetting is the subtle part. "My
machine leaks" → "Thanks." → "It still leaks" must count as **two** attempts,
not one. If a pleasantry reset the counter, a polite customer could never reach
a human.

`has_grounding` deliberately excludes `memory_always_types`, for the reason
given in §6. A memory with an unrecognised type counts *as* grounding — we
cannot tell whether it was surfaced unconditionally, and under-escalating is
the safer error.

---

## 8. Identity and isolation

The README covers the model. Two implementation notes:

`extract_token` ([`auth/dependencies.py`](backend/app/auth/dependencies.py))
checks the httpOnly cookie first, then `Authorization: Bearer`. The cookie is
how browsers authenticate; the header exists for API clients and tests, which
have no cookie jar and no XSS surface to protect.

The authenticated id reaches ChromaDB through `AgentContext.user_id` and is
used as the `where` filter. `test_the_graph_receives_the_authenticated_user_id`
posts a *different* user id in the request body and asserts the token's id is
what the retriever sees — if a body field could set it, every isolation
guarantee in the system would be bypassable in one request.

---

## 9. Configuration

Everything tunable is in [`app/config.py`](backend/app/config.py), read from
the environment via pydantic-settings. Nothing is hard-coded at a call site.

| Setting | Default | What it governs |
|---|---|---|
| `knowledge_top_k` | 5 | documentation candidates per query |
| `knowledge_similarity_threshold` | 0.35 | floor for a chunk to be usable |
| `knowledge_chunk_size` / `_overlap` | 700 / 80 | chunking |
| `memory_top_k` | 5 | memory candidates |
| `memory_similarity_threshold` | 0.30 | floor for a memory to be relevant |
| `memory_dedup_threshold` | 0.85 | floor for two memories to be one fact |
| `memory_always_types` | identity, product | surfaced regardless of similarity |
| `memory_profile_limit` | 6 | cap on the always-surfaced profile |
| `history_message_limit` | 20 | turns handed to the model |
| `retrieval_context_turns` | 2 | prior customer turns joined to the query |
| `escalation_ungrounded_limit` | 3 | consecutive ungrounded turns before handoff |
| `max_message_chars` | 4000 | inbound cost/abuse control |
| `gemini_thinking_level` | minimal | pinned so latency and cost do not drift |

**The similarity thresholds are measured, not guessed.** The reasoning behind
each number is in the comments next to it, with the observed score ranges that
justify it. Change one and the comment is the first thing to update.

`jwt_secret` has no usable default on purpose — a shared fallback secret is a
forged-token vulnerability, so the app refuses to start without one.

---

## 10. Testing architecture

```bash
cd backend
TEST_DATABASE_URL=postgresql+psycopg://postgres:<password>@localhost:5432/csam_test \
  .venv/bin/pytest -q
```

**241 tests** — 215 ordinary, 26 evaluation.

```bash
pytest -m "not evals"   # ordinary suite only
pytest -m evals         # evaluation only
pytest -m evals -s      # evaluation with the metrics table printed
```

The whole design is in [`tests/conftest.py`](backend/tests/conftest.py):

- **Real PostgreSQL, never SQLite**, so UUID columns, server defaults and
  cascades are the ones production uses. `DATABASE_URL` is set *before* app
  import, because the engine is built at import time.
- **A throwaway ChromaDB per run.** `CHROMA_PATH` is a fresh `mkdtemp`, so tests
  never read, write or wipe the developer's `./data/chroma`.
- **Gemini is stubbed for every test**, autouse: generation, extraction and the
  guardrail classifier. The suite is deterministic, offline and free even on a
  machine whose `.env` holds a real API key. A test that wants a real verdict
  re-patches in its own body.
- **Truncation between every test** — application tables *and* checkpoint
  tables, plus both Chroma collections. A stale checkpoint would let a test
  pass on a previous test's workflow state.
- **`setup_checkpointer()` runs once per session**, before any test opens a
  transaction. LangGraph's migration uses `CREATE INDEX CONCURRENTLY`, which
  waits on open transactions; called mid-request it waits on that request's own
  session and hangs forever.

Two fixtures are worth knowing before writing a test:

**`prompt_contexts`** records the context block handed to the model on each
generation, in order. `sources` and `memories` say what retrieval *found*; this
says what the model was actually *given*. A test asserting only the former
would pass even if the retrieved material never reached the prompt.

**`make_client`** gives each account its own cookie jar. Two accounts on one
client would share one, and the second signup would silently replace the
first — exactly the confusion an isolation test must not have.

Test files are named by the phase that introduced them
(`test_phase6_rag.py` … `test_phase10b_courtesy.py`), with cross-cutting ones
named by concern (`test_auth.py`, `test_transcript_integrity.py`,
`test_eval_baseline.py`).

---

## 11. Evaluation harness

Separate from `tests/` because the dataset and metrics are artefacts in their
own right — read by people, not only by pytest.

**[`evals/dataset.json`](backend/evals/dataset.json)** — 18 hand-written cases:
7 answerable, 4 unanswerable, 4 tricky/multi-part, 3 out-of-scope. Every
expected answer is traceable to a **verbatim quote** from a shipped knowledge
document. Nothing asserts a policy the documents do not contain.

**[`evals/gold.py`](backend/evals/gold.py)** — converts those quotes into
expected chunk ids by re-running `chunk_document`. Chunking is deterministic,
so chunk-level ground truth comes for free: no one hand-labels 234
question-chunk pairs, and the labels cannot go stale, because an edited
document shows up as a quote that no longer maps.

**[`evals/metrics.py`](backend/evals/metrics.py)** — pure functions.
Undefined is not zero: a question with no expected facts has no recall, and
returning `0.0` would punish a question that was never supposed to retrieve
anything. Every metric returns `None` when undefined and `mean` skips those.

[`tests/test_eval_retrieval.py`](backend/tests/test_eval_retrieval.py) measures
two layers:

- **Chunk level** (diagnostic) — raw ChromaDB ranking, *before* the similarity
  floor. Fact Recall@1/3/5 and MRR. Explains *why* the next layer did what it
  did.
- **Document level** (product) — what `retriever.retrieve` actually returns.
  Precision, recall, and abstention.

It asserts **no numeric quality floors**. Only unambiguous behaviours are
pinned: that every dataset quote still maps onto a chunk, that questions with
no supporting document retrieve nothing, and that every citation names a real
file. Inventing thresholds before measuring is how a suite ends up defending
its own bugs.

Last measured, 13 chunks, `top_k=5`, floor 0.35:

```
fact Recall@1 0.66   @3 0.80   @5 0.85   MRR 0.88
document precision 0.82   document recall 0.90
abstention 4/4 correct
```

What this harness does **not** measure: answer text. Gemini is stubbed suite-
wide, so the dataset's `required_facts` are not checked against generated
prose. That needs a live run or an LLM judge — neither offline nor free.

---

## 12. Invariants

Rules that hold across the codebase. Breaking one is how this system
regresses.

1. **The user id comes from the verified token. Never from a path, query or
   body.** Every isolation guarantee rests on this one.
2. **Nothing retrieved is invented.** `memories` and `sources` in the response
   are the same objects the agent received. The UI cannot show a citation the
   agent did not get.
3. **Empty retrieval means no context block.** `build_context_block` returns
   `None` rather than an empty "Context:" heading — an empty heading is an
   invitation for the model to fill it in, which is the exact failure this
   project exists to avoid.
4. **Retrieval and extraction are best-effort; generation is not optional.**
   A ChromaDB outage costs context, not the reply. `llm/gemini.py` never
   raises; it returns `LlmReply(FALLBACK_REPLY, ok=False)`.
5. **`ok` is never inferred from the text.** A fallback is reported as
   `llm_success=False` and must not be presented as an AI answer, or learned
   from.
6. **Nothing is learned from a failed or deflected turn.** There is no real
   exchange to learn from, and fallback text is not something the customer said.
7. **The checkpoint is not a transcript.** `history` and `user_message` live in
   `AgentContext` and must stay there.
8. **No database session enters graph state.**
9. **Tests never call Gemini, and never touch `./data/chroma`.**
10. **Message contents are never logged.** Structured logs carry ids, counts
    and outcomes only.

---

## 13. Known limitations

**Chunk boundaries are the dominant retrieval weakness.** Greedy 700-char
packing splits topics mid-section, so a fact can end up as the last clause of a
chunk otherwise about something else, and its embedding is a blend. Two
measured cases:

- *"How much is the call-out charge…"* — the gold chunk
  (`contacting-support.md::1`, which mixes an overlap fragment, Escalation and
  Engineer visits) ranks **6th at similarity 0.307**: outside `top_k=5` and
  below the 0.35 floor. Rephrasing to "engineer visit" moves it to rank 1.
- *"What is included if I pay for installation?"* — the inclusions list sits at
  the end of a chunk dominated by Delivery content and ranks **6th at 0.144**.
  The retriever returns the chunk saying what is *not* included.

Neither is fixable by raising `top_k` alone. Splitting on `##` headings is the
experiment to run, and the evaluator already exists to measure it.

**Deduplication can discard the chunk that mattered.** `retrieve()` keeps one
chunk per document. For *"How long is the warranty?"*, `warranty-policy.md::0`
(the two-year term) and `::2` (extended cover) both clear the floor; the
stronger `::2` wins, and the snippet handed to the model does not contain the
answer. Returning best-*N*-per-document, or choosing the snippet by query
relevance, would address it.

**Retrieval query dilution.** Joining two prior customer turns makes follow-ups
work and makes topic switches worse. A grounded question asked after two
unrelated turns can retrieve nothing.

**JWT revocation.** Logout clears the cookie, but the token stays valid until
it expires. Stateless tokens cost this. A denylist, or short-lived tokens with
refresh, is the right next step if this ever holds anything more sensitive than
support chat.

**Courtesy detection depends on a model.** `is_support_request` comes from
Gemini. During an outage the guardrail fails open and every message counts as a
support attempt — today's behaviour, so no regression, but no improvement
either.

**Answer quality is unmeasured.** See §11.

**The README is drifting.** Its `Layout` section predates `guardrail.py`,
`escalation.py`, `checkpointer.py` and `evals/`, and its phase table stops at
phase 9. Worth a pass.

---

## 14. Extension points

**Another LLM provider** — rewrite [`llm/gemini.py`](backend/app/llm/gemini.py).
It is the only module importing `google.genai`, and its interface is
`generate_reply(turns, context)` and `generate_json(instruction, text, schema)`.

**Another knowledge source** — add markdown to `backend/knowledge/` and re-run
`scripts/ingest_knowledge.py`. Ingestion is a rebuild, so editing a document is
safe. Then add dataset cases with verbatim quotes and re-run the evaluator.

**A new workflow step** — add the node function to `nodes.py`, register it in
`build_graph`, and add a name constant. If it branches, put the decision in a
pure function and keep the router trivial, as both existing branches do.

**A new piece of per-turn provenance** — add a key to `messages.context`. It is
`JSONB` and `MessageOut` ignores unknown keys, so no migration and no API
change. This is how `outcome` was added.

**A new retrieval signal** — note that `retrieve()` deliberately strips
`_score` and the chunk index. Anything needing them must query
`knowledge_collection()` directly, as the evaluator does, rather than widening
the public contract.

**Changing a threshold** — update the comment next to it in `config.py` with
the measurement that justifies the new number, then run `pytest -m evals -s`
and record what moved.
