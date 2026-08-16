"""Python Execution Tool - chạy Python code an toàn."""
from __future__ import annotations

import sys
import io
import traceback
from typing import Dict, Any
from contextlib import redirect_stdout, redirect_stderr

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


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
            "Restricted builtins (no open, exec, eval bên ngoài)."
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
        
        # Capture stdout/stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        # Restricted globals
        safe_builtins = {
            k: v for k, v in __builtins__.items() if k not in (
                "exec", "eval", "compile", "open", "input",
                "breakpoint", "exit", "quit",
            )
        } if isinstance(__builtins__, dict) else __builtins__
        
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
        
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, globals_dict)
            
            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue()
            
            # Capture variables created
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
                },
            )
        except Exception as e:
            tb = traceback.format_exc()
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {e}\n{tb}",
                return_code=1,
            )
