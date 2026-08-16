"""
Code Metrics Tool - Tính metrics: LOC, SLOC, comments, blank lines, file count.
Author: Hieu Louis (2026)

Sử dụng stdlib `tokenize` cho Python files để đếm comment chính xác,
fallback heuristic (comment prefix) cho các ngôn ngữ khác.
"""
from __future__ import annotations

import io
import json
import os
import tokenize
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


DEFAULT_EXTENSIONS = [
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".sql", ".sh", ".yml", ".yaml",
]

# Map extension → comment prefix (cho non-Python files)
COMMENT_PREFIX = {
    ".js": "//", ".ts": "//", ".java": "//", ".c": "//", ".cpp": "//",
    ".h": "//", ".hpp": "//", ".go": "//", ".rs": "//", ".php": "//",
    ".sh": "#", ".yml": "#", ".yaml": "#",
    ".rb": "#", ".sql": "--",
}


def _metrics_python(source: str) -> Dict[str, int]:
    """Tính metrics cho file Python dùng tokenize (chính xác)."""
    lines = source.splitlines()
    loc = len(lines)
    blank = sum(1 for ln in lines if not ln.strip())
    comments = 0
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                comments += 1
    except tokenize.TokenError:
        # Fallback heuristic
        comments = sum(1 for ln in lines if ln.strip().startswith("#"))
    # SLOC = non-blank, non-pure-comment lines
    sloc = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        sloc += 1
    return {"loc": loc, "sloc": sloc, "comments": comments, "blank": blank}


def _metrics_text(source: str, comment_prefix: str = "#") -> Dict[str, int]:
    """Tính metrics cho file text generic (non-Python) theo prefix."""
    lines = source.splitlines()
    loc = len(lines)
    blank = sum(1 for ln in lines if not ln.strip())
    comments = sum(1 for ln in lines if ln.strip().startswith(comment_prefix))
    sloc = loc - blank - comments
    if sloc < 0:
        sloc = 0
    return {"loc": loc, "sloc": sloc, "comments": comments, "blank": blank}


def _walk_files(path: str, extensions: Optional[List[str]]) -> List[str]:
    """Walk tất cả files trong path (file hoặc dir), lọc theo extension."""
    if os.path.isfile(path):
        return [path]
    out: List[str] = []
    for root, _dirs, names in os.walk(path):
        for name in sorted(names):
            if extensions:
                if not any(name.endswith(ext) for ext in extensions):
                    continue
            out.append(os.path.join(root, name))
    return out


class CodeMetricsTool(Tool):
    """Tính LOC/SLOC/comments/blank lines/file count cho file hoặc thư mục."""

    category = ToolCategory.CODE
    safety = ToolSafety.SAFE  # read-only

    @property
    def name(self) -> str:
        return "code_metrics"

    @property
    def description(self) -> str:
        return (
            "Tính metrics: LOC, SLOC, comments, blank lines, file count. "
            "Hỗ trợ thư mục hoặc single file. Python dùng tokenize, "
            "ngôn ngữ khác dùng heuristic theo comment prefix."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File hoặc thư mục"},
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Lọc theo extension (vd ['.py', '.js']). "
                        "Mặc định auto-detect nhiều loại code files."
                    ),
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
        extensions: List[str] = args.get("extensions") or DEFAULT_EXTENSIONS

        if not os.path.exists(path):
            return ToolResult(
                success=False,
                error=f"Path không tồn tại: {path}",
                return_code=1,
            )

        files = _walk_files(path, extensions)
        if not files:
            return ToolResult(
                success=True,
                output="[]",
                metadata={"path": path, "file_count": 0, "totals": {}},
            )

        per_file: List[Dict[str, Any]] = []
        totals: Dict[str, int] = {"loc": 0, "sloc": 0, "comments": 0, "blank": 0}
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    src = f.read()
            except Exception:
                continue
            if fp.endswith(".py"):
                m = _metrics_python(src)
            else:
                # Tìm comment prefix phù hợp
                prefix = "#"
                for ext, p in COMMENT_PREFIX.items():
                    if fp.endswith(ext):
                        prefix = p
                        break
                m = _metrics_text(src, prefix)
            m["file"] = fp
            per_file.append(m)
            for k in totals:
                totals[k] += m.get(k, 0)

        return ToolResult(
            success=True,
            output=json.dumps(
                {"files": per_file, "totals": totals},
                indent=2, ensure_ascii=False,
            ),
            metadata={
                "path": path,
                "file_count": len(per_file),
                "totals": totals,
            },
        )
