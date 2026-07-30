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
    ollama_request_timeout_seconds: float = 120.0

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/sqlite/app.db"

    # FastAPI backend
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings instance.

    `lru_cache` ensures the .env file is parsed once per process instead of
    on every call, and lets FastAPI routes depend on this function via
    `Depends(get_settings)` for testability (override the dependency in tests).
    """
    return Settings()
