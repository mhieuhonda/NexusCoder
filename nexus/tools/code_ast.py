"""
Code AST Tool - Parse Python source thành AST bằng stdlib `ast`.
Author: Hieu Louis (2026)

Operations:
- dump_tree      : Dump toàn bộ AST tree (ast.dump indent=2)
- list_functions : Liệt kê FunctionDef / AsyncFunctionDef
- list_classes   : Liệt kê ClassDef + methods
- list_imports   : Liệt kê Import / ImportFrom
- list_calls     : Liệt kê Call sites

Tool read-only (SAFE). Trả về JSON summary trong `output`.
"""
from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional, Tuple

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Các operation được hỗ trợ // supported operations
OPERATIONS = {
    "dump_tree",
    "list_functions",
    "list_classes",
    "list_imports",
    "list_calls",
}


def _load_code(args: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Load source từ `path` hoặc `code`. Trả về (code, error)."""
    path = args.get("path")
    code = args.get("code")
    if not path and not code:
        return None, "Missing required arg: path hoặc code"
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read(), None
        except Exception as e:
            return None, f"Không đọc được file {path}: {e}"
    return code, None


class CodeASTTool(Tool):
    """Parse Python source thành AST và trả về summary theo operation."""

    category = ToolCategory.CODE
    safety = ToolSafety.SAFE  # read-only analysis

    @property
    def name(self) -> str:
        return "code_ast"

    @property
    def description(self) -> str:
        return (
            "Parse Python source thành AST. Hỗ trợ dump_tree, list_functions, "
            "list_classes, list_imports, list_calls. Trả về JSON summary."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Đường dẫn file Python (.py)"},
                "code": {"type": "string", "description": "Mã nguồn Python (nếu không dùng path)"},
                "operation": {
                    "type": "string",
                    "enum": sorted(OPERATIONS),
                    "description": "Operation (default list_functions)",
                },
            },
            "anyOf": [{"required": ["path"]}, {"required": ["code"]}],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        op = args.get("operation", "list_functions")
        if op not in OPERATIONS:
            return f"Unsupported operation: {op}. Chọn một trong: {sorted(OPERATIONS)}"
        if not args.get("path") and not args.get("code"):
            return "Missing required arg: path hoặc code"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        code, err = _load_code(args)
        if err:
            return ToolResult(success=False, error=err, return_code=1)

        op = args.get("operation", "list_functions")
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"SyntaxError (line {e.lineno}): {e.msg}",
                return_code=1,
                metadata={"path": args.get("path")},
            )

        try:
            if op == "dump_tree":
                dump = ast.dump(tree, indent=2)
                node_count = sum(1 for _ in ast.walk(tree))
                return ToolResult(
                    success=True,
                    output=dump,
                    metadata={
                        "operation": op,
                        "path": args.get("path"),
                        "nodes": node_count,
                    },
                )

            if op == "list_functions":
                items: List[Dict[str, Any]] = []
                for n in ast.walk(tree):
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        items.append({
                            "name": n.name,
                            "line": n.lineno,
                            "end_line": getattr(n, "end_lineno", n.lineno),
                            "args": [a.arg for a in n.args.args],
                            "decorators": [ast.unparse(d) for d in n.decorator_list],
                            "is_async": isinstance(n, ast.AsyncFunctionDef),
                        })

            elif op == "list_classes":
                items = []
                for n in ast.walk(tree):
                    if isinstance(n, ast.ClassDef):
                        methods = [
                            m.name for m in n.body
                            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                        ]
                        items.append({
                            "name": n.name,
                            "line": n.lineno,
                            "end_line": getattr(n, "end_lineno", n.lineno),
                            "bases": [ast.unparse(b) for b in n.bases],
                            "methods": methods,
                            "decorators": [ast.unparse(d) for d in n.decorator_list],
                        })

            elif op == "list_imports":
                items = []
                for n in ast.walk(tree):
                    if isinstance(n, ast.Import):
                        for alias in n.names:
                            items.append({
                                "line": n.lineno,
                                "module": alias.name,
                                "alias": alias.asname,
                                "type": "import",
                            })
                    elif isinstance(n, ast.ImportFrom):
                        mod = "." * (n.level or 0) + (n.module or "")
                        for alias in n.names:
                            items.append({
                                "line": n.lineno,
                                "module": mod,
                                "name": alias.name,
                                "alias": alias.asname,
                                "type": "from",
                            })

            elif op == "list_calls":
                items = []
                for n in ast.walk(tree):
                    if isinstance(n, ast.Call):
                        try:
                            func_repr = ast.unparse(n.func)
                        except Exception:
                            func_repr = "<unknown>"
                        items.append({"line": n.lineno, "func": func_repr})
            else:
                return ToolResult(success=False, error=f"Unknown operation: {op}", return_code=1)

            return ToolResult(
                success=True,
                output=json.dumps(items, indent=2, ensure_ascii=False),
                metadata={
                    "operation": op,
                    "path": args.get("path"),
                    "count": len(items),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
                return_code=1,
            )
