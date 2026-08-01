"""Application configuration, loaded once from environment variables / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings.

    Values are read from environment variables first, falling back to a
    `.env` file in the project root. See `.env.example` for the full list
    of supported variables.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b-instruct"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_vision_model: str = "moondream"
    ollama_request_timeout_seconds: float = 120.0

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/sqlite/app.db"

    # RAG
    chroma_path: str = "./data/chroma"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_top_k: int = 6
    # nomic-embed-text cosine distances for genuinely relevant chunks in a
    # real resume measured 0.42-0.53 against conversational questions
    # (e.g. "how many years did she work") — a 0.5 cutoff was silently
    # dropping real matches on most turns. 0.75 leaves headroom.
    rag_max_distance: float = 0.75

    # Voice
    whisper_model_size: str = "base"
    whisper_download_root: str = "./data/whisper"
    piper_voice: str = "en_US-lessac-medium"
    piper_voices_dir: str = "./data/voices"

    # FastAPI backend
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"
    # Console logging is always on; set this to also write a rotating log
    # file (e.g. "./data/logs/app.log") — off by default, matching the
    # "logs go to stdout" convention Docker/most process managers expect.
    log_file: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings instance.

    `lru_cache` ensures the .env file is parsed once per process instead of
    on every call, and lets FastAPI routes depend on this function via
    `Depends(get_settings)` for testability (override the dependency in tests).
    """
    return Settings()
