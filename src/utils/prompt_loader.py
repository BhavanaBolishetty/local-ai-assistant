"""Loads prompt templates from src/prompts/ by name."""

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache
def load_prompt(name: str) -> str:
    """Load a prompt template (without extension) from src/prompts/<name>.txt.

    Cached because prompt files don't change at runtime; restart the process
    to pick up edits made during development.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()
