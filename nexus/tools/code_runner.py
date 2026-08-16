"""
Code Runner Tool - Run code trong nhiều ngôn ngữ.
Author: Hieu Louis (2026)

Hỗ trợ:
- python      : python3 <file>
- javascript  : node <file>
- go          : go run <file>
- rust        : rustc -o <bin> <file> && <bin>
- c           : gcc -o <bin> <file> && <bin>
- cpp         : g++ -o <bin> <file> && <bin>

Subprocess có timeout=context.timeout, capture stdout+stderr.
DANGEROUS (executes arbitrary code), requires_confirmation.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Map language → interpreter/compiler command, file extension, mode
LANGUAGE_RUNNERS: Dict[str, Dict[str, str]] = {
    "python":     {"cmd": "python3", "ext": ".py",  "mode": "interpret"},
    "javascript": {"cmd": "node",    "ext": ".js",  "mode": "interpret"},
    "go":         {"cmd": "go",      "ext": ".go",  "mode": "interpret"},   # go run
    "rust":       {"cmd": "rustc",   "ext": ".rs",  "mode": "compile_run"},
    "c":          {"cmd": "gcc",     "ext": ".c",   "mode": "compile_run"},
    "cpp":        {"cmd": "g++",     "ext": ".cpp", "mode": "compile_run"},
}


class CodeRunnerTool(Tool):
    """Run code Python/JS/Go/Rust/C/C++ trong subprocess sandbox."""

    category = ToolCategory.EXEC
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "code_runner"

    @property
    def description(self) -> str:
        return (
            "Run code trong nhiều ngôn ngữ (Python, JavaScript/node, Go, Rust, C, C++). "
            "Capture stdout+stderr. Có timeout. Subprocess sandbox."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Source code để chạy"},
                "language": {
                    "type": "string",
                    "enum": sorted(LANGUAGE_RUNNERS.keys()),
                    "description": "Ngôn ngữ của code",
                },
                "stdin": {"type": "string", "description": "Stdin input (optional)"},
            },
            "required": ["code", "language"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("code"):
            return "Missing required arg: code"
        lang = args.get("language")
        if not lang:
            return "Missing required arg: language"
        if lang not in LANGUAGE_RUNNERS:
            return f"Unsupported language: {lang}. Chọn: {sorted(LANGUAGE_RUNNERS.keys())}"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        code: str = args["code"]
        lang: str = args["language"]
        stdin_data: Optional[str] = args.get("stdin")

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Sẽ chạy {lang} code ({len(code)} bytes)",
                metadata={"language": lang, "dry_run": True, "code_length": len(code)},
            )

        spec = LANGUAGE_RUNNERS[lang]
        cmd = spec["cmd"]
        if not shutil.which(cmd):
            return ToolResult(
                success=False,
                error=(
                    f"Interpreter/Compiler '{cmd}' không tìm thấy trong PATH. "
                    f"Cài đặt để chạy {lang}."
                ),
                return_code=127,
                metadata={"language": lang, "missing": cmd},
            )

        # Tạo temp dir + temp source file
        tmpdir = tempfile.mkdtemp(prefix="nexus_runner_")
        src_path = os.path.join(tmpdir, f"main{spec['ext']}")
        try:
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            return ToolResult(success=False, error=f"Write temp file lỗi: {e}", return_code=1)

        # Build command
        if spec["mode"] == "interpret":
            if lang == "go":
                full_cmd: List[str] = [cmd, "run", src_path]
            else:
                full_cmd = [cmd, src_path]
            cwd = tmpdir
        else:  # compile_run
            binary = os.path.join(tmpdir, "main_bin")
            compile_cmd = [cmd, "-o", binary, src_path]
            # Step 1: compile
            try:
                cp_compile = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=context.timeout,
                    cwd=tmpdir,
                    env={**os.environ, **context.env},
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    success=False,
                    error=f"Compile timeout ({context.timeout}s)",
                    return_code=124,
                    metadata={"language": lang, "phase": "compile"},
                )
            except FileNotFoundError:
                return ToolResult(success=False, error=f"{cmd} not found", return_code=127)
            if cp_compile.returncode != 0:
                return ToolResult(
                    success=False,
                    output=cp_compile.stdout,
                    error=cp_compile.stderr,
                    return_code=cp_compile.returncode,
                    metadata={"phase": "compile", "language": lang, "command": " ".join(compile_cmd)},
                )
            full_cmd = [binary]
            cwd = tmpdir

        # Step 2: run
        try:
            cp = subprocess.run(
                full_cmd,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=context.timeout,
                cwd=cwd,
                env={**os.environ, **context.env},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Runtime timeout ({context.timeout}s)",
                return_code=124,
                metadata={"language": lang, "phase": "run", "command": " ".join(full_cmd)},
            )
        except FileNotFoundError:
            return ToolResult(success=False, error=f"{full_cmd[0]} not found", return_code=127)

        return ToolResult(
            success=(cp.returncode == 0),
            output=cp.stdout,
            error=cp.stderr if cp.stderr else None,
            return_code=cp.returncode,
            metadata={
                "language": lang,
                "command": " ".join(full_cmd),
                "phase": "run",
                "timeout": context.timeout,
            },
        )
