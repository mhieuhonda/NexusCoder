"""
Code Complexity Tool - Tính cyclomatic complexity cho Python functions.
Author: Hieu Louis (2026)

Cyclomatic complexity = 1 + số decision points:
  if/elif (each elif), for, while, except, with, assert,
  boolean and/or (n values → n-1), ternary if-exp, comprehension clauses.

Reference: McCabe (1976) — complexity ≥ 10 cần refactor.
"""
from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class _ComplexityVisitor(ast.NodeVisitor):
    """Đếm decision points trong một function body để tính cyclomatic complexity."""

    def __init__(self) -> None:
        self.complexity: int = 1  # baseline path

    def visit_If(self, n: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(n)

    def visit_For(self, n: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(n)

    def visit_AsyncFor(self, n: ast.AsyncFor) -> None:
        self.complexity += 1
        self.generic_visit(n)

    def visit_While(self, n: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(n)

    def visit_ExceptHandler(self, n: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(n)

    def visit_With(self, n: ast.With) -> None:
        self.complexity += 1
        self.generic_visit(n)

    def visit_AsyncWith(self, n: ast.AsyncWith) -> None:
        self.complexity += 1
        self.generic_visit(n)

    def visit_Assert(self, n: ast.Assert) -> None:
        self.complexity += 1
        self.generic_visit(n)

    def visit_BoolOp(self, n: ast.BoolOp) -> None:
        # `a and b and c` = 2 decision points
        self.complexity += max(0, len(n.values) - 1)
        self.generic_visit(n)

    def visit_IfExp(self, n: ast.IfExp) -> None:
        self.complexity += 1
        self.generic_visit(n)

    def visit_comprehension(self, n: ast.comprehension) -> None:
        # mỗi clause +1, mỗi if-condition +1
        self.complexity += 1 + len(n.ifs)
        self.generic_visit(n)


def _risk(c: int) -> str:
    """Phân loại risk theo complexity (McCabe thresholds)."""
    if c <= 5:
        return "low"
    if c <= 10:
        return "moderate"
    if c <= 20:
        return "high"
    return "very_high"


class CodeComplexityTool(Tool):
    """Tính cyclomatic complexity của các function trong Python file."""

    category = ToolCategory.CODE
    safety = ToolSafety.SAFE  # read-only analysis

    @property
    def name(self) -> str:
        return "code_complexity"

    @property
    def description(self) -> str:
        return (
            "Tính cyclomatic complexity của Python functions trong file. "
            "Complexity = 1 + số decision points (if/for/while/except/and/or...)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File Python (.py) để phân tích"},
                "function": {
                    "type": "string",
                    "description": "Tên function cụ thể (optional). Mặc định tính tất cả.",
                },
            },
            "required": ["path"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("path"):
            return "Missing required arg: path"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path: str = args["path"]
        target_fn: Optional[str] = args.get("function")

        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            return ToolResult(success=False, error=f"Không đọc được file: {e}", return_code=1)

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"SyntaxError line {e.lineno}: {e.msg}",
                return_code=1,
            )

        results: List[Dict[str, Any]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if target_fn and node.name != target_fn:
                continue
            visitor = _ComplexityVisitor()
            visitor.visit(node)
            results.append({
                "function": node.name,
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "complexity": visitor.complexity,
                "risk": _risk(visitor.complexity),
            })

        if target_fn and not results:
            return ToolResult(
                success=False,
                error=f"Không tìm thấy function '{target_fn}' trong {path}",
                return_code=1,
            )

        return ToolResult(
            success=True,
            output=json.dumps(results, indent=2, ensure_ascii=False),
            metadata={
                "path": path,
                "function": target_fn,
                "count": len(results),
                "max_complexity": max((r["complexity"] for r in results), default=0),
            },
        )
