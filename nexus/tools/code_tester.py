"""
Code Tester Tool - Run tests với nhiều frameworks.
Author: Hieu Louis (2026)

Frameworks:
- pytest   : pytest -v <path>
- unittest : python3 -m unittest -v <path>
- jest     : npx jest <path>
- go       : go test -v <path>
- cargo    : cargo test <path>
- rspec    : rspec <path>

Auto-detect framework dựa trên config files (pyproject.toml, go.mod,
Cargo.toml, package.json, Gemfile) hoặc file extensions.
DANGEROUS (executes test commands), requires_confirmation.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Map framework → (cmd, default args)
FRAMEWORKS: Dict[str, Dict[str, Any]] = {
    "pytest":   {"cmd": "pytest",    "args": ["-v"]},
    "unittest": {"cmd": "python3",   "args": ["-m", "unittest", "-v"]},
    "jest":     {"cmd": "npx",       "args": ["jest"]},
    "go":       {"cmd": "go",        "args": ["test", "-v"]},
    "cargo":    {"cmd": "cargo",     "args": ["test"]},
    "rspec":    {"cmd": "rspec",     "args": []},
}


def _detect_framework(path: str) -> str:
    """Auto-detect framework dựa trên file structure."""
    # Check config files trong dir hoặc parent dir
    check_dir = path if os.path.isdir(path) else os.path.dirname(path) or "."
    if os.path.isfile(os.path.join(check_dir, "pytest.ini")):
        return "pytest"
    if os.path.isfile(os.path.join(check_dir, "pyproject.toml")):
        return "pytest"
    if os.path.isfile(os.path.join(check_dir, "go.mod")):
        return "go"
    if os.path.isfile(os.path.join(check_dir, "Cargo.toml")):
        return "cargo"
    if os.path.isfile(os.path.join(check_dir, "package.json")):
        return "jest"
    if os.path.isfile(os.path.join(check_dir, "Gemfile")):
        return "rspec"
    if os.path.isfile(os.path.join(check_dir, ".rspec")):
        return "rspec"
    # Fallback: check file extensions
    files: List[str] = []
    if os.path.isdir(path):
        try:
            files = os.listdir(path)
        except OSError:
            files = []
    elif os.path.isfile(path):
        files = [path]
    if any(f.endswith(".py") for f in files):
        return "pytest"
    if any(f.endswith(".go") for f in files):
        return "go"
    if any(f.endswith(".rs") for f in files):
        return "cargo"
    if any(f.endswith((".js", ".ts", ".jsx", ".tsx")) for f in files):
        return "jest"
    if any(f.endswith(".rb") for f in files):
        return "rspec"
    return "pytest"  # default


class CodeTesterTool(Tool):
    """Run tests với pytest/unittest/jest/go test/cargo test/rspec."""

    category = ToolCategory.EXEC
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "code_tester"

    @property
    def description(self) -> str:
        return (
            "Run tests với pytest, unittest, jest, go test, cargo test, rspec. "
            "Auto-detect framework nếu không chỉ định. Subprocess có timeout."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File hoặc thư mục chứa tests"},
                "framework": {
                    "type": "string",
                    "enum": sorted(FRAMEWORKS.keys()),
                    "description": "Framework (auto-detect nếu không chỉ định)",
                },
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Args bổ sung cho test runner",
                },
            },
            "required": ["path"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("path"):
            return "Missing required arg: path"
        fw = args.get("framework")
        if fw and fw not in FRAMEWORKS:
            return f"Unsupported framework: {fw}. Chọn: {sorted(FRAMEWORKS.keys())}"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path: str = args["path"]
        fw: Optional[str] = args.get("framework")
        extra_args: List[str] = list(args.get("extra_args", []))

        if not os.path.exists(path):
            return ToolResult(success=False, error=f"Path không tồn tại: {path}", return_code=1)

        if not fw:
            fw = _detect_framework(path)

        spec = FRAMEWORKS[fw]
        cmd: str = spec["cmd"]

        if context.dry_run:
            full = [cmd, *spec["args"], path, *extra_args]
            return ToolResult(
                success=True,
                output=f"[dry-run] Sẽ chạy: {' '.join(shlex.quote(c) for c in full)}",
                metadata={"framework": fw, "command": full, "dry_run": True},
            )

        if not shutil.which(cmd):
            return ToolResult(
                success=False,
                error=f"Framework CLI '{cmd}' không tìm thấy. Cài đặt để chạy {fw}.",
                return_code=127,
                metadata={"framework": fw, "missing": cmd},
            )

        full_cmd: List[str] = [cmd, *spec["args"], path, *extra_args]
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
                error=f"Test timeout ({context.timeout}s)",
                return_code=124,
                metadata={"framework": fw, "command": " ".join(full_cmd)},
            )
        except FileNotFoundError:
            return ToolResult(success=False, error=f"{cmd} not found", return_code=127)

        return ToolResult(
            success=(cp.returncode == 0),
            output=cp.stdout,
            error=cp.stderr if cp.stderr else None,
            return_code=cp.returncode,
            metadata={
                "framework": fw,
                "command": " ".join(full_cmd),
                "path": path,
                "timeout": context.timeout,
            },
        )
