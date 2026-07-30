"""Shared shape for a callable tool the model can invoke."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the function's arguments (Ollama's `tools` format)
    execute: Callable[[dict], str]
