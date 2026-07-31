# Local AI Assistant

A ChatGPT-style assistant that runs entirely on your own machine using a small
language model served by [Ollama](https://ollama.com) — no API keys, no cloud
calls, no data leaving your computer.

Built incrementally as a learning project covering the full stack of a
production-style AI application: layered backend architecture, RAG, tool use,
agentic reasoning, voice, and vision.

## Status: Phase 5 — Agent ✅

Streamlit UI → FastAPI backend → Ollama → `qwen2.5:3b-instruct`, with
streaming responses, SQLite-backed conversation history, document upload
(ChromaDB + `nomic-embed-text`) for RAG, and an agent (`src/agents/AgentRunner`)
that can chain up to 8 tool calls per turn — calculator, current
date/time, and an explicit document search — showing its full
step-by-step reasoning trace rather than just a final answer.

## Architecture

```
┌─────────────────────┐       HTTP        ┌──────────────────────┐
│  apps/streamlit      │  ───────────────▶ │   src/api (FastAPI)   │
│  (chat UI)            │  ◀─────────────── │   routes/chat.py       │
└─────────────────────┘   streamed text    └──────────┬───────────┘
                                                        │ Depends()
                                                        ▼
                                            ┌──────────────────────┐
                                            │  src/services          │
                                            │  ChatService            │
                                            │  (injects system prompt)│
                                            └──────────┬───────────┘
                                                        │
                                                        ▼
                                            ┌──────────────────────┐
                                            │  src/ai                │
                                            │  OllamaClient           │
                                            │  (only module that      │
                                            │  knows Ollama's format) │
                                            └──────────┬───────────┘
                                                        │ HTTP (localhost:11434)
                                                        ▼
                                            ┌──────────────────────┐
                                            │  Ollama                 │
                                            │  qwen2.5:3b-instruct     │
                                            │  (llama.cpp, Q4_K_M)     │
                                            └──────────────────────┘
```

Domain objects (`src/models/`) flow between layers; each layer only imports
the one directly below it. `src/api/schemas/` is a separate, deliberately
thin layer that exists only to validate/serialize HTTP traffic.

## Folder structure

```
local-ai-assistant/
├── apps/streamlit/       # Streamlit frontend (talks to FastAPI only)
├── src/
│   ├── api/               # FastAPI app, routes, request/response schemas
│   ├── services/          # Business logic / orchestration (ChatService)
│   ├── ai/                # Model client(s) — currently OllamaClient
│   ├── models/             # Framework-agnostic domain models (Message, Conversation)
│   ├── prompts/            # Prompt templates, loaded via src/utils/prompt_loader.py
│   ├── core/                # Config (pydantic-settings) and logging setup
│   ├── utils/                # Shared helpers
│   ├── repositories/         # ConversationRepository — persistence, abstracts SQLite away from services
│   ├── db/                    # SQLAlchemy async engine/session + ORM models
│   ├── memory/                 # (Later) context-window/summarization strategies
│   ├── rag/                     # chunker, VectorStore (Chroma), RagRetriever
│   ├── tools/                     # Tool (calculator, get_current_datetime, search_documents) + ToolRegistry
│   └── agents/                   # AgentRunner — the bounded, multi-step tool-calling loop
├── data/                    # Gitignored runtime data: uploads, chroma/, sqlite/
├── tests/
├── .env.example
└── pyproject.toml
```

## Installation

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com) installed.

```bash
# 1. Install dependencies
uv sync

# 2. Pull the chat and embedding models
ollama pull qwen2.5:3b-instruct
ollama pull nomic-embed-text

# 3. Copy environment config
cp .env.example .env
```

SQLite tables and the Chroma collection are created automatically at
backend startup (`data/sqlite/app.db`, `data/chroma/`) — no separate
migration step needed.

## Running

Two processes, two terminals:

```bash
# Terminal 1 — backend
uv run uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — frontend
uv run streamlit run apps/streamlit/app.py
```

Open http://localhost:8501.

## API reference

| Method | Path                       | Description                              |
|--------|----------------------------|--------------------------------------------|
| GET    | `/health`                  | Liveness check                            |
| POST   | `/chat`                    | Full conversation → complete reply + step trace |
| POST   | `/chat/stream`             | Full conversation → streamed NDJSON events  |
| GET    | `/conversations`           | List past conversations (id/title/created_at) |
| GET    | `/conversations/{id}`      | Load one conversation with its full message history |
| DELETE | `/conversations/{id}`      | Delete a conversation                     |
| POST   | `/documents`               | Upload a `.txt`/`.md`/`.pdf`/`.docx` file to the RAG knowledge base |
| GET    | `/documents`                | List uploaded documents (id/filename/chunk_count) |
| DELETE | `/documents/{id}`           | Delete a document and its chunks          |

Request body for both `/chat` endpoints — the client still resends the full
message history each turn; `conversation_id` ties it to a persisted row
(auto-generated server-side if omitted):

```json
{
  "messages": [{"role": "user", "content": "Hello"}],
  "model": null,
  "conversation_id": null
}
```

`/chat` echoes `conversation_id` back in the JSON body, plus a `steps`
array (empty if the turn needed no tools) — one entry per resolved tool
call: `{"tool_name": "calculator", "arguments": {...}, "result": "..."}`.
`/chat/stream` returns `conversation_id` in an `X-Conversation-Id`
response header. Its body is NDJSON, one event per line — either
`{"type": "content", "text": "..."}` (a real streamed token chunk) or
`{"type": "step", "tool_name": ..., "arguments": ..., "result": ...}`
(one per resolved tool call, before the final answer streams in).

If any documents have been uploaded, every chat turn automatically embeds
the user's message, retrieves the most relevant chunks (cosine distance
below a threshold), and folds them into the system prompt — no separate
"RAG mode" to turn on. On top of that, `AgentRunner` lets the model
deliberately chain up to 8 tool calls in a single turn — e.g. searching
the documents, then computing something from what it found, then
checking the date — with each step exposed in the trace. True token
streaming is preserved for ordinary questions that need no tool at all.

## Roadmap

- [x] Phase 1 — Core chat loop (Streamlit + FastAPI + Ollama)
- [x] Phase 2 — Conversation memory (SQLite)
- [x] Phase 3 — RAG (ChromaDB, document upload)
- [x] Phase 4 — Tool calling
- [x] Phase 5 — Agent (planning, multi-step reasoning)
- [ ] Phase 6 — Voice (Faster-Whisper, Piper)
- [ ] Phase 7 — Vision
- [ ] Phase 8 — Deployment (Docker, logging, testing)
