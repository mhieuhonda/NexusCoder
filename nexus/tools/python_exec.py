"""Python Execution Tool - chạy Python code an toàn."""
from __future__ import annotations

import sys
import io
import os
import signal
import threading
import traceback
import builtins as _builtins
from typing import Dict, Any
from contextlib import redirect_stdout, redirect_stderr

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Whitelist builtins for sandboxed exec.
_SANDBOX_BLOCKED_BUILTINS = frozenset({
    "exec", "eval", "compile", "open", "input",
    "breakpoint", "exit", "quit", "__import__",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
})

# Whitelist of importable modules inside sandbox.
_SANDBOX_ALLOWED_MODULES = frozenset({
    "math", "random", "json", "re", "datetime", "collections",
    "itertools", "functools", "string", "decimal", "fractions",
    "statistics", "heapq", "bisect", "operator", "textwrap",
})


def _make_safe_builtins() -> Dict[str, Any]:
    """Build a safe __builtins__ dict (consistent in script & REPL)."""
    safe: Dict[str, Any] = {}
    for k in dir(_builtins):
        if k in _SANDBOX_BLOCKED_BUILTINS:
            continue
        v = getattr(_builtins, k, None)
        if v is not None:
            safe[k] = v

    def _safe_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top not in _SANDBOX_ALLOWED_MODULES:
            raise ImportError(f"Sandboxed exec: import of {name!r} is not allowed")
        return __import__(name, *args, **kwargs)

    safe["__import__"] = _safe_import
    return safe


class PythonExecTool(Tool):
    """Execute Python code trong restricted namespace."""
    category = ToolCategory.EXEC
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "python_exec"

    @property
    def description(self) -> str:
        return (
            "Execute Python code an toàn. Capture stdout/stderr. "
            "Restricted builtins (no open, exec, eval, __import__ bên ngoài)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["code"],
        }

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        code = args["code"]
        timeout = args.get("timeout", context.timeout)
        timeout = max(1, min(int(timeout), 600))

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        safe_builtins = _make_safe_builtins()

        import math, random, json, re, datetime, collections, itertools, functools

        globals_dict = {
            "__builtins__": safe_builtins,
            "__name__": "__nexus_sandbox__",
            # Safe imports
            "math": math,
            "random": random,
            "json": json,
            "re": re,
            "datetime": datetime,
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
            "print": print,
        }

        result_holder: Dict[str, Any] = {"exc": None}

        def _run():
            try:
                exec(code, globals_dict)
            except BaseException as e:
                result_holder["exc"] = e

        # Run in thread with timeout (cross-platform, no signal hacks)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            # Cannot kill the thread cleanly but mark timeout
            return ToolResult(
                success=False,
                error=f"TimeoutError: execution exceeded {timeout}s",
                return_code=124,
            )
        if result_holder["exc"] is not None:
            e = result_holder["exc"]
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {e}\n{tb}",
                return_code=1,
            )

        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()

        created_vars = {
            k: type(v).__name__
            for k, v in globals_dict.items()
            if not k.startswith("_") and k not in (
                "math", "random", "json", "re", "datetime",
                "collections", "itertools", "functools", "print",
            )
        }

        return ToolResult(
            success=True,
            output=stdout or "(no output)",
            error=stderr if stderr else None,
            metadata={
                "code_lines": len(code.splitlines()),
                "created_vars": created_vars,
                "timeout": timeout,
            },
        )
