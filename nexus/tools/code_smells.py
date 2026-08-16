"""
Code Smells Detector - Phát hiện code smells trong Python file.
Author: Hieu Louis (2026)

Phát hiện các smells:
- long_function       : quá nhiều statements/lines
- too_many_params     : > 5 parameters
- deep_nesting        : nesting > 4 levels
- long_class          : class có quá nhiều methods
- duplicate_string_literal : string literal xuất hiện ≥ 3 lần (len ≥ 5)

Dùng `ast` để walk tree. Read-only (SAFE).
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Ngưỡng smell // smell thresholds (tunable)
LONG_FUNCTION_STMTS = 50
LONG_FUNCTION_LINES = 50
TOO_MANY_PARAMS = 5
DEEP_NESTING = 4
LONG_CLASS_METHODS = 20
DUP_LIT_MIN_COUNT = 3
DUP_LIT_MIN_LEN = 5
DUP_LIT_MAX_REPORT = 50


class _SmellVisitor(ast.NodeVisitor):
    """Visitor quét AST để phát hiện code smells."""

    def __init__(self) -> None:
        self.smells: List[Dict[str, Any]] = []
        self._str_literals: List[str] = []

    def _record(self, kind: str, name: str, line: int, detail: Dict[str, Any]) -> None:
        self.smells.append({"kind": kind, "name": name, "line": line, **detail})

    def _max_nesting(self, node: ast.AST, depth: int = 0) -> int:
        """Tính độ sâu nesting tối đa trong block."""
        max_d = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)):
                d = self._max_nesting(child, depth + 1)
            else:
                d = self._max_nesting(child, depth)
            if d > max_d:
                max_d = d
        return max_d

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)

    def _check_function(self, node: ast.FunctionDef) -> None:
        name = node.name
        line = node.lineno
        end_line = getattr(node, "end_lineno", line)
        # Statement count (approximate)
        n_stmts = sum(1 for _ in ast.walk(node) if isinstance(_, ast.stmt))
        n_lines = max(1, end_line - line + 1)
        if n_stmts > LONG_FUNCTION_STMTS or n_lines > LONG_FUNCTION_LINES:
            self._record("long_function", name, line, {"statements": n_stmts, "lines": n_lines})
        # Parameters count
        n_args = (
            len(node.args.args)
            + len(node.args.kwonlyargs)
            + len(node.args.posonlyargs)
        )
        if node.args.vararg:
            n_args += 1
        if node.args.kwarg:
            n_args += 1
        if n_args > TOO_MANY_PARAMS:
            self._record("too_many_params", name, line, {"params": n_args})
        # Nesting depth
        nesting = self._max_nesting(node)
        if nesting > DEEP_NESTING:
            self._record("deep_nesting", name, line, {"depth": nesting})

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        n_methods = sum(
            1 for m in node.body
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        if n_methods > LONG_CLASS_METHODS:
            self._record("long_class", node.name, node.lineno, {"methods": n_methods})
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # ast.Str deprecated in 3.8+; use ast.Constant
        if isinstance(node.value, str) and len(node.value) >= DUP_LIT_MIN_LEN:
            self._str_literals.append(node.value)
        self.generic_visit(node)


class CodeSmellsTool(Tool):
    """Phát hiện code smells (long function, too many params, deep nesting, ...)."""

    category = ToolCategory.CODE
    safety = ToolSafety.SAFE  # read-only analysis

    @property
    def name(self) -> str:
        return "code_smells"

    @property
    def description(self) -> str:
        return (
            "Phát hiện code smells trong Python file: long function, too many params, "
            "deep nesting, long class, duplicate string literals."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File Python (.py) để phân tích"},
            },
            "required": ["path"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("path"):
            return "Missing required arg: path"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path: str = args["path"]
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

        visitor = _SmellVisitor()
        visitor.visit(tree)

        # Duplicate string literals analysis
        dup_lits = [
            {"literal": lit, "count": cnt}
            for lit, cnt in Counter(visitor._str_literals).most_common()
            if cnt >= DUP_LIT_MIN_COUNT
        ][:DUP_LIT_MAX_REPORT]
        for d in dup_lits:
            visitor.smells.append({
                "kind": "duplicate_string_literal",
                "name": "<literal>",
                "line": 0,
                **d,
            })

        by_kind: Dict[str, int] = {}
        for s in visitor.smells:
            by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1

        return ToolResult(
            success=True,
            output=json.dumps(
                {"smells": visitor.smells, "summary": by_kind},
                indent=2, ensure_ascii=False,
            ),
            metadata={
                "path": path,
                "smell_count": len(visitor.smells),
                "by_kind": by_kind,
            },
        )
