"""A tool for correctly computing the duration between two dates.

Small local models reliably *read* dates out of a document correctly but
unreliably *subtract* them — verified repeatedly against real questions
about resume employment dates, where the model listed the right dates
but computed wildly inconsistent totals (0.67, 1, 2.67, 5 years across
retries on the same facts). Same fix as `calculator`: take the arithmetic
away from the model entirely and let deterministic code do it.
"""

from datetime import datetime

from src.tools.base import Tool

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m",
    "%B %Y",
    "%b %Y",
    "%m/%Y",
    "%Y",
)


def _parse_date(value: str) -> datetime:
    value = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"could not parse date {value!r} (try e.g. 'May 2021' or 'YYYY-MM-DD')")


def _format_duration(total_months: int) -> str:
    years, months = divmod(total_months, 12)
    parts = []
    if years:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if months:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    if not parts:
        parts.append("0 months")
    return " and ".join(parts)


async def _calculate_date_duration(arguments: dict) -> str:
    try:
        start = _parse_date(arguments.get("start_date", ""))
        end = _parse_date(arguments.get("end_date", ""))
    except ValueError as exc:
        return f"Error: {exc}"

    if end < start:
        return "Error: end_date is before start_date"

    total_months = (end.year - start.year) * 12 + (end.month - start.month)
    return f"{_format_duration(total_months)} ({total_months} months total)"


date_duration_tool = Tool(
    name="calculate_date_duration",
    description=(
        "Calculate the exact duration between two dates (e.g. employment start/end "
        "dates, or any 'how long between X and Y' question). Always use this instead "
        "of computing a date difference yourself — that kind of arithmetic is easy to "
        "get subtly wrong. Accepts dates like 'May 2021', 'Jul 2024', or 'YYYY-MM-DD'. "
        "If you need to add up several date ranges, call this once per range and sum "
        "the results."
    ),
    parameters={
        "type": "object",
        "properties": {
            "start_date": {"type": "string", "description": "Start date, e.g. 'May 2021'"},
            "end_date": {"type": "string", "description": "End date, e.g. 'Jul 2024'"},
        },
        "required": ["start_date", "end_date"],
    },
    execute=_calculate_date_duration,
)
