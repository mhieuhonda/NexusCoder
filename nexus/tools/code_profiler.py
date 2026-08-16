"""
Code Profiler Tool - Profile Python code với cProfile + pstats.
Author: Hieu Louis (2026)

Chạy code trong cProfile context, xuất top-N functions theo cumulative time.
DANGEROUS (executes Python code), requires_confirmation.

Note: Sử dụng stdlib `cProfile` và `pstats` (không cần lazy import).
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class CodeProfilerTool(Tool):
    """Profile Python code với cProfile + pstats (stdlib)."""

    category = ToolCategory.EXEC
    safety = ToolSafety.DANGEROUS  # executes Python code
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "code_profiler"

    @property
    def description(self) -> str:
        return (
            "Profile Python code với cProfile + pstats. Trả về bảng top functions "
            "theo cumulative time. Hỗ trợ top_n (default 20)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File Python để profile"},
                "code": {"type": "string", "description": "Python code (nếu không dùng path)"},
                "top_n": {
                    "type": "integer",
                    "default": 20,
                    "description": "Số dòng top functions trong output",
                },
            },
            "anyOf": [{"required": ["path"]}, {"required": ["code"]}],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("path") and not args.get("code"):
            return "Missing required arg: path hoặc code"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path: Optional[str] = args.get("path")
        code: Optional[str] = args.get("code")
        top_n: int = int(args.get("top_n", 20))

        if path:
            if not os.path.isfile(path):
                return ToolResult(
                    success=False,
                    error=f"File không tồn tại: {path}",
                    return_code=1,
                )
            try:
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                return ToolResult(success=False, error=f"Đọc file lỗi: {e}", return_code=1)

        assert code is not None

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Sẽ profile {len(code)} bytes Python code (top_n={top_n})",
                metadata={"top_n": top_n, "dry_run": True, "code_length": len(code)},
            )

        # Compile code (catch SyntaxError before profiling)
        try:
            compiled = compile(code, "<profile>", "exec")
        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"SyntaxError line {e.lineno}: {e.msg}",
                return_code=1,
            )

        # Run with cProfile
        profiler = cProfile.Profile()
        globals_dict: Dict[str, Any] = {
            "__name__": "__nexus_profile__",
            "__builtins__": __builtins__,
        }
        try:
            profiler.enable()
            exec(compiled, globals_dict)
            profiler.disable()
        except Exception as e:
            profiler.disable()
            # Vẫn xuất partial stats
            stats_buf = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_buf)
            stats.sort_stats("cumulative").print_stats(top_n)
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
                return_code=1,
                output=stats_buf.getvalue(),
                metadata={
                    "top_n": top_n,
                    "path": path,
                    "partial": True,
                    "total_calls": getattr(stats, "total_calls", 0),
                    "total_time": round(getattr(stats, "total_tt", 0.0), 6),
                },
            )

        # Build stats output
        stats_buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buf)
        stats.sort_stats("cumulative").print_stats(top_n)
        output = stats_buf.getvalue()

        total_calls = getattr(stats, "total_calls", 0)
        total_time = getattr(stats, "total_tt", 0.0)

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "top_n": top_n,
                "total_calls": total_calls,
                "total_time": round(total_time, 6),
                "path": path,
            },
        )
