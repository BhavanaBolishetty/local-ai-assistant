"""Streamlit chat UI — talks to the FastAPI backend, never to Ollama directly."""

import sys
from pathlib import Path

# apps/streamlit/app.py is not inside the `src` package, and Streamlit sets
# sys.path[0] to this file's own directory (not the project root) when it
# runs the script. Add the project root so `import src...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections.abc import Iterator  # noqa: E402

import httpx  # noqa: E402
import streamlit as st  # noqa: E402

from src.core.config import get_settings  # noqa: E402
from src.models.conversation import Conversation  # noqa: E402
from src.models.message import Message, MessageRole  # noqa: E402

settings = get_settings()
API_BASE_URL = f"http://{settings.api_host}:{settings.api_port}"

st.set_page_config(page_title="Local AI Assistant", page_icon="🤖", layout="wide")


def _new_conversation() -> Conversation:
    return Conversation()


if "conversation" not in st.session_state:
    st.session_state.conversation = _new_conversation()


def _stream_reply(conversation: Conversation) -> Iterator[str]:
    payload = {
        "messages": [
            {"role": m.role.value, "content": m.content} for m in conversation.messages
        ]
    }
    with httpx.stream(
        "POST", f"{API_BASE_URL}/chat/stream", json=payload, timeout=120.0
    ) as response:
        response.raise_for_status()
        yield from response.iter_text()


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.title("🤖 Local AI Assistant")
    st.caption(f"Model: `{settings.ollama_model}` (via Ollama, fully local)")

    if st.button("+ New Chat", use_container_width=True):
        st.session_state.conversation = _new_conversation()
        st.rerun()

    st.divider()
    with st.expander("📜 Chat History", expanded=False):
        st.caption("Coming in Phase 2 (SQLite-backed persistence).")
    with st.expander("📄 Upload Documents", expanded=False):
        st.caption("Coming in Phase 3 (RAG with ChromaDB).")

# --- Main chat area ----------------------------------------------------------
st.header(st.session_state.conversation.title)

for message in st.session_state.conversation.messages:
    if message.role == MessageRole.SYSTEM:
        continue
    with st.chat_message(message.role.value):
        st.markdown(message.content)

prompt = st.chat_input("Message the assistant...")

if prompt:
    conversation: Conversation = st.session_state.conversation
    conversation.add_message(Message(role=MessageRole.USER, content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        full_reply = ""
        try:
            for chunk in _stream_reply(conversation):
                full_reply += chunk
                placeholder.markdown(full_reply + "▌")
            placeholder.markdown(full_reply)
        except httpx.ConnectError:
            full_reply = ""
            placeholder.error(
                "Can't reach the backend API. Start it with:\n\n"
                "`uv run uvicorn src.api.main:app --reload`"
            )

    if full_reply:
        conversation.add_message(Message(role=MessageRole.ASSISTANT, content=full_reply))
