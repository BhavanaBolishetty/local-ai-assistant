"""A tool returning the current local date/time.

Without this, the model has to guess "today" from its training data cutoff.
"""

from datetime import datetime

from src.tools.base import Tool


def _current_datetime(_arguments: dict) -> str:
    return datetime.now().strftime("%A, %B %d, %Y, %H:%M:%S")


current_datetime_tool = Tool(
    name="get_current_datetime",
    description="Get the current local date and time.",
    parameters={"type": "object", "properties": {}},
    execute=_current_datetime,
)
