"""Calculator Tool - thực hiện phép tính toán."""
from __future__ import annotations

import ast
import operator
from typing import Dict, Any

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Safe operators
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "int": int, "float": float,
}


def _safe_eval(node):
    """Đánh giá AST node an toàn."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported op: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        fname = node.func.id
        if fname not in _FUNCS:
            raise ValueError(f"Function not allowed: {fname}")
        args = [_safe_eval(a) for a in node.args]
        return _FUNCS[fname](*args)
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(e) for e in node.elts)
    if isinstance(node, ast.List):
        return [_safe_eval(e) for e in node.elts]
    raise ValueError(f"Unsupported node: {type(node).__name__}")


class CalculatorTool(Tool):
    """Calculator an toàn - eval biểu thức toán học."""
    category = ToolCategory.MATH
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return (
            "Tính biểu thức toán học an toàn (+, -, *, /, **, %, //, abs, round, min, max, ...). "
            "Không thực thi code arbitrary."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Biểu thức toán học"},
                "precision": {"type": "integer", "default": 10},
            },
            "required": ["expression"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        expr = args["expression"]
        precision = args.get("precision", 10)
        
        try:
            tree = ast.parse(expr, mode="eval")
            result = _safe_eval(tree)
            
            if isinstance(result, float):
                result_str = f"{result:.{precision}g}"
            else:
                result_str = str(result)
            
            return ToolResult(
                success=True,
                output=f"{expr} = {result_str}",
                metadata={
                    "expression": expr,
                    "result": result,
                    "result_type": type(result).__name__,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Cannot evaluate '{expr}': {e}",
                return_code=1,
            )
