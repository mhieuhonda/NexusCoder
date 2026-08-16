"""
Code Dependency Tool - Trích xuất import graph từ Python file/thư mục.
Author: Hieu Louis (2026)

Trả về danh sách (file, imported_module) pairs. Hỗ trợ recursive scan
thư mục. Module string chứa dotted path đầy đủ (vd `os.path`, `.foo.bar`
cho relative imports).
"""
from __future__ import annotations

import ast
import json
import os
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


def _extract_imports(source: str) -> List[str]:
    """Trích danh sách module imported từ source code (dùng ast)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    modules: List[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for alias in n.names:
                modules.append(alias.name)
        elif isinstance(n, ast.ImportFrom):
            mod = "." * (n.level or 0) + (n.module or "")
            if mod:
                modules.append(mod)
    return modules


def _walk_python_files(path: str, recursive: bool) -> List[str]:
    """Tìm tất cả file .py trong path (file hoặc dir)."""
    if os.path.isfile(path):
        return [path]
    files: List[str] = []
    if not os.path.isdir(path):
        return files
    if recursive:
        for root, _dirs, names in os.walk(path):
            for name in sorted(names):
                if name.endswith(".py"):
                    files.append(os.path.join(root, name))
    else:
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full) and name.endswith(".py"):
                files.append(full)
    return files


class CodeDependencyTool(Tool):
    """Extract import graph từ Python file/thư mục."""

    category = ToolCategory.CODE
    safety = ToolSafety.SAFE  # read-only analysis

    @property
    def name(self) -> str:
        return "code_dependency"

    @property
    def description(self) -> str:
        return (
            "Extract import graph từ Python file/dir. Trả về list (file, imported_module). "
            "Hỗ trợ recursive scan và top-level package ranking."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File hoặc thư mục Python"},
                "recursive": {
                    "type": "boolean",
                    "description": "Scan đệ quy nếu là thư mục (default true)",
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
        recursive: bool = bool(args.get("recursive", True))

        if not os.path.exists(path):
            return ToolResult(
                success=False,
                error=f"Path không tồn tại: {path}",
                return_code=1,
            )

        files = _walk_python_files(path, recursive)
        if not files:
            return ToolResult(
                success=True,
                output="[]",
                metadata={"path": path, "file_count": 0, "edge_count": 0},
            )

        edges: List[Dict[str, str]] = []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    src = f.read()
            except Exception:
                continue
            for mod in _extract_imports(src):
                edges.append({"file": fp, "module": mod})

        # Top-level package ranking
        top_packages: Dict[str, int] = {}
        for e in edges:
            top = e["module"].lstrip(".").split(".")[0]
            if top:
                top_packages[top] = top_packages.get(top, 0) + 1

        ranked = dict(sorted(top_packages.items(), key=lambda kv: -kv[1])[:20])

        return ToolResult(
            success=True,
            output=json.dumps(edges, indent=2, ensure_ascii=False),
            metadata={
                "path": path,
                "recursive": recursive,
                "file_count": len(files),
                "edge_count": len(edges),
                "top_packages": ranked,
            },
        )
