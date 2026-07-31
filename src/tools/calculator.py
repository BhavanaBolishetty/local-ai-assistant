"""A sandboxed arithmetic evaluator tool.

Never uses `eval()` — walks the parsed AST and only permits numeric
constants and `+ - * / % **` (binary and unary). Anything else (a name,
a call, attribute access, ...) is rejected before it can execute.
"""

import ast
import operator

from src.tools.base import Tool

_BIN_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPERATORS:
        return _BIN_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


async def _calculate(arguments: dict) -> str:
    expression = arguments.get("expression", "")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except Exception as exc:
        return f"Error: could not evaluate '{expression}' ({exc})"
    return str(result)


calculator_tool = Tool(
    name="calculator",
    description=(
        "Evaluate a basic arithmetic expression (+, -, *, /, %, **) and return the numeric result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A math expression, e.g. '47 * 89'",
            },
        },
        "required": ["expression"],
    },
    execute=_calculate,
)
