"""Tests for the sandboxed calculator tool — especially that it can't
be used to execute arbitrary code."""

from src.tools.calculator import calculator_tool


async def _calc(expression: str) -> str:
    return await calculator_tool.execute({"expression": expression})


async def test_basic_arithmetic() -> None:
    assert await _calc("47 * 89") == "4183"
    assert await _calc("2 + 3 * 4") == "14"
    assert await _calc("10 / 4") == "2.5"
    assert await _calc("2 ** 10") == "1024"
    assert await _calc("-5 + 2") == "-3"


async def test_invalid_syntax_returns_error_string() -> None:
    result = await _calc("2 +")
    assert result.startswith("Error:")


async def test_name_lookup_is_rejected() -> None:
    result = await _calc("some_variable + 1")
    assert result.startswith("Error:")


async def test_function_call_is_rejected() -> None:
    result = await _calc("__import__('os').system('echo pwned')")
    assert result.startswith("Error:")


async def test_attribute_access_is_rejected() -> None:
    result = await _calc("(1).__class__")
    assert result.startswith("Error:")
