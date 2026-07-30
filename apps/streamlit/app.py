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
        "conversation_id": conversation.id,
        "messages": [
            {"role": m.role.value, "content": m.content} for m in conversation.messages
        ],
    }
    with httpx.stream(
        "POST", f"{API_BASE_URL}/chat/stream", json=payload, timeout=120.0
    ) as response:
        response.raise_for_status()
        yield from response.iter_text()


def _list_past_conversations() -> list[dict]:
    response = httpx.get(f"{API_BASE_URL}/conversations", timeout=10.0)
    response.raise_for_status()
    return response.json()


def _load_conversation(conversation_id: str) -> Conversation:
    response = httpx.get(f"{API_BASE_URL}/conversations/{conversation_id}", timeout=10.0)
    response.raise_for_status()
    data = response.json()
    return Conversation(
        id=data["id"],
        title=data["title"],
        messages=[
            Message(role=MessageRole(m["role"]), content=m["content"]) for m in data["messages"]
        ],
    )


def _delete_conversation(conversation_id: str) -> None:
    httpx.delete(f"{API_BASE_URL}/conversations/{conversation_id}", timeout=10.0)


def _upload_document(uploaded_file) -> None:
    content_type = uploaded_file.type or "application/octet-stream"
    response = httpx.post(
        f"{API_BASE_URL}/documents",
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), content_type)},
        timeout=120.0,
    )
    response.raise_for_status()


def _list_documents() -> list[dict]:
    response = httpx.get(f"{API_BASE_URL}/documents", timeout=10.0)
    response.raise_for_status()
    return response.json()


def _delete_document(document_id: str) -> None:
    httpx.delete(f"{API_BASE_URL}/documents/{document_id}", timeout=10.0)


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.title("🤖 Local AI Assistant")
    st.caption(f"Model: `{settings.ollama_model}` (via Ollama, fully local)")

    if st.button("+ New Chat", use_container_width=True):
        st.session_state.conversation = _new_conversation()
        st.rerun()

    st.divider()
    with st.expander("📜 Chat History", expanded=False):
        try:
            past_conversations = _list_past_conversations()
        except httpx.ConnectError:
            past_conversations = []
            st.caption("Can't reach the backend API.")

        if not past_conversations:
            st.caption("No past conversations yet.")
        for conv in past_conversations:
            title_col, delete_col = st.columns([5, 1])
            if title_col.button(conv["title"], key=f"load-{conv['id']}", use_container_width=True):
                st.session_state.conversation = _load_conversation(conv["id"])
                st.rerun()
            if delete_col.button("🗑", key=f"delete-{conv['id']}"):
                _delete_conversation(conv["id"])
                if st.session_state.conversation.id == conv["id"]:
                    st.session_state.conversation = _new_conversation()
                st.rerun()
    with st.expander("📄 Upload Documents", expanded=False):
        if "processed_uploads" not in st.session_state:
            st.session_state.processed_uploads = set()

        uploaded_files = st.file_uploader(
            "Add .txt, .md, .pdf, or .docx files to the assistant's knowledge base",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True,
            key="doc_uploader",
        )
        for uploaded in uploaded_files or []:
            if uploaded.file_id not in st.session_state.processed_uploads:
                with st.spinner(f"Embedding {uploaded.name}..."):
                    _upload_document(uploaded)
                st.session_state.processed_uploads.add(uploaded.file_id)
                st.rerun()

        try:
            documents = _list_documents()
        except httpx.ConnectError:
            documents = []
            st.caption("Can't reach the backend API.")

        if not documents:
            st.caption("No documents uploaded yet.")
        for doc in documents:
            name_col, delete_col = st.columns([5, 1])
            name_col.markdown(f"**{doc['filename']}** · {doc['chunk_count']} chunks")
            if delete_col.button("🗑", key=f"delete-doc-{doc['id']}"):
                _delete_document(doc["id"])
                st.rerun()

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
