"""Small text-formatting helpers shared across services."""

_TITLE_MAX_LENGTH = 50


def derive_title(content: str, max_length: int = _TITLE_MAX_LENGTH) -> str:
    stripped = content.strip()
    if len(stripped) <= max_length:
        return stripped
    return stripped[: max_length - 3] + "..."
