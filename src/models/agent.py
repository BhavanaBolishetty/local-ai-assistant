"""Domain model for one resolved step of an agent's tool-calling loop."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentStep:
    tool_name: str
    arguments: dict
    result: str
