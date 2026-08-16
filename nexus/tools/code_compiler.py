"""
Code Compiler Tool - Compile C/C++/Rust/Go source code.
Author: Hieu Louis (2026)

Wraps:
- C   : gcc      (gcc -o <out> <src> [-O2])
- C++ : g++      (g++ -o <out> <src> [-O2])
- Rust: rustc    (rustc -O -o <out> <src>)
- Go  : go build (go build -o <out> <src>)

Subprocess có timeout, capture stdout+stderr. Trả về binary path trong artifacts.
DANGEROUS (executes compiler), requires_confirmation.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Map language → (cmd, ext, optional subcommand)
COMPILERS: Dict[str, Dict[str, Any]] = {
    "c":    {"cmd": "gcc",    "ext": ".c",   "subcmd": None},
    "cpp":  {"cmd": "g++",    "ext": ".cpp", "subcmd": None},
    "rust": {"cmd": "rustc",  "ext": ".rs",  "subcmd": None},
    "go":   {"cmd": "go",     "ext": ".go",  "subcmd": "build"},
}


class CodeCompilerTool(Tool):
    """Compile C/C++/Rust/Go source code."""

    category = ToolCategory.EXEC
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "code_compiler"

    @property
    def description(self) -> str:
        return (
            "Compile C/C++/Rust/Go source code. Wraps gcc, g++, rustc, go build. "
            "Hỗ trợ optimize flag (-O2 cho C/C++, -O cho Rust). "
            "Trả về binary path trong artifacts."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Source file để compile"},
                "language": {
                    "type": "string",
                    "enum": sorted(COMPILERS.keys()),
                    "description": "Ngôn ngữ",
                },
                "output": {
                    "type": "string",
                    "description": "Binary output path (default: <source>.out)",
                },
                "optimize": {"type": "boolean", "description": "Enable -O2/-O (default false)"},
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Compiler flags bổ sung",
                },
            },
            "required": ["path", "language"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("path"):
            return "Missing required arg: path"
        lang = args.get("language")
        if not lang:
            return "Missing required arg: language"
        if lang not in COMPILERS:
            return f"Unsupported language: {lang}. Chọn: {sorted(COMPILERS.keys())}"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path: str = args["path"]
        lang: str = args["language"]
        optimize: bool = bool(args.get("optimize", False))
        extra_args: List[str] = list(args.get("extra_args", []))

        if not os.path.isfile(path):
            return ToolResult(success=False, error=f"Source file không tồn tại: {path}", return_code=1)

        spec = COMPILERS[lang]
        cmd: str = spec["cmd"]
        if not shutil.which(cmd):
            return ToolResult(
                success=False,
                error=f"Compiler '{cmd}' không tìm thấy. Cài đặt để compile {lang}.",
                return_code=127,
                metadata={"language": lang, "missing": cmd},
            )

        # Output binary path (default next to source)
        out_path: str = args.get("output") or os.path.splitext(path)[0] + ".out"

        # Build command
        full_cmd: List[str] = [cmd]
        if spec.get("subcmd"):
            full_cmd.append(spec["subcmd"])
        if optimize and lang in ("c", "cpp"):
            full_cmd.append("-O2")
        if optimize and lang == "rust":
            full_cmd.append("-O")
        full_cmd += ["-o", out_path, path]
        full_cmd += extra_args

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Sẽ compile: {' '.join(shlex.quote(c) for c in full_cmd)}",
                metadata={
                    "language": lang,
                    "command": full_cmd,
                    "output": out_path,
                    "optimize": optimize,
                    "dry_run": True,
                },
            )

        try:
            cp = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=context.timeout,
                cwd=context.working_dir,
                env={**os.environ, **context.env},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Compile timeout ({context.timeout}s)",
                return_code=124,
                metadata={"language": lang, "command": " ".join(full_cmd)},
            )
        except FileNotFoundError:
            return ToolResult(success=False, error=f"{cmd} not found", return_code=127)

        artifacts: List[str] = []
        if cp.returncode == 0 and os.path.isfile(out_path):
            artifacts.append(out_path)

        return ToolResult(
            success=(cp.returncode == 0),
            output=cp.stdout,
            error=cp.stderr if cp.stderr else None,
            return_code=cp.returncode,
            artifacts=artifacts,
            metadata={
                "language": lang,
                "command": " ".join(full_cmd),
                "output": out_path,
                "optimize": optimize,
            },
        )
