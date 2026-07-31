"""Streamlit chat UI — talks to the FastAPI backend, never to Ollama directly."""

import json
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


def _stream_reply(conversation: Conversation) -> Iterator[tuple]:
    """Yields `("content", text)` or `("step", tool_name, arguments, result)`
    tuples, parsed from the backend's NDJSON stream (see `_to_ndjson` in
    `src/api/routes/chat.py`)."""
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
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "step":
                yield ("step", event["tool_name"], event["arguments"], event["result"])
            else:
                yield ("content", event["text"])


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


def _error_detail(exc: httpx.HTTPStatusError) -> str:
    """FastAPI error responses are `{"detail": "..."}`; fall back to raw
    text if the backend returned something else (e.g. a proxy's HTML page)."""
    try:
        return exc.response.json().get("detail", exc.response.text)
    except ValueError:
        return exc.response.text


def _transcribe_audio(audio_value) -> str:
    response = httpx.post(
        f"{API_BASE_URL}/voice/transcribe",
        files={"file": ("recording.wav", audio_value.getvalue(), "audio/wav")},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["text"]


def _render_play_button(text: str, key: str) -> None:
    if st.button("🔊 Play", key=key):
        try:
            response = httpx.post(
                f"{API_BASE_URL}/voice/speak", json={"text": text}, timeout=60.0
            )
            response.raise_for_status()
            st.audio(response.content, format="audio/wav", autoplay=True)
        except httpx.HTTPStatusError as exc:
            st.error(f"Couldn't generate speech: {_error_detail(exc)}")
        except httpx.ConnectError:
            st.error("Can't reach the backend API.")


def _ask_vision(conversation: Conversation, image_file, text: str) -> Iterator[str]:
    files = {"image": (image_file.name, image_file.getvalue(), image_file.type or "image/png")}
    data = {"text": text, "conversation_id": conversation.id}
    with httpx.stream(
        "POST", f"{API_BASE_URL}/vision/ask", files=files, data=data, timeout=120.0
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
                # Mark as seen before attempting the upload, win or lose —
                # otherwise a failed file would just retry (and re-fail)
                # forever on every rerun.
                st.session_state.processed_uploads.add(uploaded.file_id)
                try:
                    with st.spinner(f"Embedding {uploaded.name}..."):
                        _upload_document(uploaded)
                except httpx.HTTPStatusError as exc:
                    st.error(f"Couldn't add {uploaded.name}: {_error_detail(exc)}")
                except httpx.ConnectError:
                    st.error("Can't reach the backend API.")
                else:
                    st.rerun()

        try:
            documents = _list_documents()
        except httpx.ConnectError:
            documents = []
            st.caption("Can't reach the backend API.")

        st.caption(
            "The file picker above just remembers what you've selected in "
            "this browser tab — it doesn't reflect what's actually in the "
            "knowledge base. The list below is the real thing; use 🗑 there "
            "to remove something for good."
        )
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

for idx, message in enumerate(st.session_state.conversation.messages):
    if message.role == MessageRole.SYSTEM:
        continue
    with st.chat_message(message.role.value):
        st.markdown(message.content)
        if message.role == MessageRole.ASSISTANT:
            _render_play_button(message.content, key=f"play-history-{idx}")

chat_value = st.chat_input(
    "Message the assistant, attach an image, or record a voice message...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp"],
    accept_audio=True,
)

prompt = None
uploaded_image = None

if chat_value:
    if chat_value.audio is not None:
        try:
            with st.spinner("Transcribing..."):
                prompt = _transcribe_audio(chat_value.audio)
        except httpx.HTTPStatusError as exc:
            st.error(f"Couldn't transcribe that recording: {_error_detail(exc)}")
        except httpx.ConnectError:
            st.error("Can't reach the backend API.")
    else:
        prompt = chat_value.text
    if chat_value.files:
        uploaded_image = chat_value.files[0]
        if not prompt:
            prompt = "What do you see in this image?"

if prompt:
    conversation: Conversation = st.session_state.conversation

    # If the previous turn never got a reply (stopped mid-stream, or the
    # backend call failed), its user message is still the last entry with
    # nothing answering it. Sending two user turns in a row confuses the
    # model into answering both at once (and answering neither well) — but
    # the earlier question shouldn't just vanish either, so record an
    # honest placeholder reply rather than deleting it.
    if conversation.messages and conversation.messages[-1].role == MessageRole.USER:
        conversation.add_message(
            Message(role=MessageRole.ASSISTANT, content="*(interrupted — no response)*")
        )

    conversation.add_message(Message(role=MessageRole.USER, content=prompt))

    use_vision = uploaded_image is not None

    with st.chat_message("user"):
        if uploaded_image is not None:
            st.image(uploaded_image, width=200)
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        full_reply = ""
        status_box = None
        completed = False
        try:
            if use_vision:
                for text in _ask_vision(conversation, uploaded_image, prompt):
                    full_reply += text
                    placeholder.markdown(full_reply + "▌")
            else:
                for item in _stream_reply(conversation):
                    if item[0] == "step":
                        _, tool_name, arguments, result = item
                        if status_box is None:
                            status_box = st.status("🤖 Working...", expanded=True)
                        args_str = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
                        status_box.write(f"🔧 **{tool_name}**({args_str}) → {result}")
                    else:
                        _, text = item
                        full_reply += text
                        placeholder.markdown(full_reply + "▌")
            if status_box is not None:
                status_box.update(label="Done", state="complete", expanded=False)
            placeholder.markdown(full_reply)
            completed = True
        except httpx.ConnectError:
            placeholder.error(
                "Can't reach the backend API. Start it with:\n\n"
                "`uv run uvicorn src.api.main:app --reload`"
            )
        except httpx.HTTPStatusError as exc:
            placeholder.error(f"Request failed: {_error_detail(exc)}")
        finally:
            # Runs even when Streamlit's native Stop button interrupts the
            # loop above (it works by raising an exception into this
            # script), so whatever text had already streamed in is kept
            # instead of vanishing — matching ChatGPT's "keep what it had"
            # behavior on stop, instead of discarding it.
            if full_reply:
                stored_reply = full_reply if completed else f"{full_reply}\n\n*(interrupted)*"
                conversation.add_message(Message(role=MessageRole.ASSISTANT, content=stored_reply))

    if completed and full_reply:
        # Rerun so the history loop above renders this message (and its Play
        # button) as a normal, stable widget — a button created in this
        # branch would vanish next run (prompt/audio_value are now empty),
        # so its own click could never be detected. Skipped when interrupted
        # (completed=False) — the script is already being torn down by the
        # Stop signal at that point, so forcing a rerun here would fight it;
        # the next natural interaction will show the saved partial reply.
        st.rerun()
