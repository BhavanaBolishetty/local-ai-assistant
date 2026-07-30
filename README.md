# Local AI Assistant

A ChatGPT-style assistant that runs entirely on your own machine using a small
language model served by [Ollama](https://ollama.com) — no API keys, no cloud
calls, no data leaving your computer.

Built incrementally as a learning project covering the full stack of a
production-style AI application: layered backend architecture, RAG, tool use,
agentic reasoning, voice, and vision.

## Status: Phase 4 — Tool calling ✅

Streamlit UI → FastAPI backend → Ollama → `qwen2.5:3b-instruct`, with
streaming responses, SQLite-backed conversation history, document upload
(ChromaDB + `nomic-embed-text`) for RAG, and native tool calling (a
sandboxed calculator and a current-date/time lookup) so the model can
reach for a real computation instead of guessing.

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
│   ├── tools/                     # Tool (calculator, get_current_datetime) + ToolRegistry
│   └── agents/                   # (Phase 5) planning, multi-step reasoning
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
| POST   | `/chat`                    | Full conversation → complete reply (tool calls resolved server-side) |
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

`/chat` echoes `conversation_id` back in the JSON body; `/chat/stream`
returns it in an `X-Conversation-Id` response header. `/chat/stream`'s
body is NDJSON, one event per line — either
`{"type": "content", "text": "..."}` (a real streamed token chunk) or
`{"type": "tool_call", "tool": "calculator"}` (fired when the model
invokes a tool, before it streams the final answer).

If any documents have been uploaded, every chat turn automatically embeds
the user's message, retrieves the most relevant chunks (cosine distance
below a threshold), and folds them into the system prompt — no separate
"RAG mode" to turn on. Tool calling works the same way: the model decides
on its own whether a turn needs the calculator or current-date/time tool,
executes it, and continues to a normal streamed answer — true token
streaming is preserved for ordinary questions that don't need a tool.

## Roadmap

- [x] Phase 1 — Core chat loop (Streamlit + FastAPI + Ollama)
- [x] Phase 2 — Conversation memory (SQLite)
- [x] Phase 3 — RAG (ChromaDB, document upload)
- [x] Phase 4 — Tool calling
- [ ] Phase 5 — Agent (planning, multi-step reasoning)
- [ ] Phase 6 — Voice (Faster-Whisper, Piper)
- [ ] Phase 7 — Vision
- [ ] Phase 8 — Deployment (Docker, logging, testing)
