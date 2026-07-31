"""Tests for the date-duration tool — the fix for the model reliably
misjudging month/year arithmetic even when it has the correct dates."""

from src.tools.date_duration import date_duration_tool


async def _duration(start: str, end: str) -> str:
    return await date_duration_tool.execute({"start_date": start, "end_date": end})


async def test_exact_year_span() -> None:
    assert await _duration("Jan 2020", "Jan 2023") == "3 years (36 months total)"


async def test_years_and_months() -> None:
    # The real resume case that repeatedly tripped up the model's own
    # mental arithmetic (and, initially, mine too when drafting this test
    # by hand — exactly the point of not trusting mental date math).
    assert await _duration("May 2021", "Dec 2022") == "1 year and 7 months (19 months total)"
    assert await _duration("Jan 2023", "Jul 2024") == "1 year and 6 months (18 months total)"


async def test_months_only() -> None:
    assert await _duration("March 2024", "July 2024") == "4 months (4 months total)"


async def test_same_date_is_zero_months() -> None:
    assert await _duration("2021-05-01", "2021-05-15") == "0 months (0 months total)"


async def test_accepts_multiple_date_formats() -> None:
    assert await _duration("2021-05", "2022-12") == "1 year and 7 months (19 months total)"
    assert await _duration("05/2021", "12/2022") == "1 year and 7 months (19 months total)"


async def test_end_before_start_is_an_error() -> None:
    result = await _duration("Jan 2024", "Jan 2020")
    assert result.startswith("Error:")


async def test_unparseable_date_is_an_error() -> None:
    result = await _duration("not a date", "Jan 2024")
    assert result.startswith("Error:")
