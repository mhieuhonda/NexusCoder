"""Code Tools - code search, lint, format."""
from __future__ import annotations

import os
import re
import subprocess
from typing import Dict, Any, List

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class CodeSearchTool(Tool):
    """Search code trong files với regex."""
    category = ToolCategory.CODE
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "code_search"
    
    @property
    def description(self) -> str:
        return "Search trong code files bằng regex. Hỗ trợ file pattern, context lines."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern"},
                "path": {"type": "string", "default": "."},
                "file_pattern": {"type": "string", "default": "*.py"},
                "case_insensitive": {"type": "boolean", "default": False},
                "context": {"type": "integer", "default": 0, "description": "Lines of context"},
                "max_results": {"type": "integer", "default": 50},
            },
            "required": ["pattern"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = args["pattern"]
        path = args.get("path", ".")
        file_pattern = args.get("file_pattern", "*.py")
        case_insensitive = args.get("case_insensitive", False)
        context_lines = args.get("context", 0)
        max_results = args.get("max_results", 50)
        
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(success=False, error=f"Invalid regex: {e}", return_code=2)
        
        full_path = path if os.path.isabs(path) else os.path.join(context.working_dir, path)
        
        matches = []
        files_scanned = 0
        
        for root, dirs, files in os.walk(full_path):
            # Skip hidden dirs, venv, __pycache__, .git
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in (
                "venv", "__pycache__", "node_modules", ".git", "dist", "build",
            )]
            for fname in files:
                if not _matches_pattern(fname, file_pattern):
                    continue
                fpath = os.path.join(root, fname)
                files_scanned += 1
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    for i, line in enumerate(lines):
                        if regex.search(line):
                            start = max(0, i - context_lines)
                            end = min(len(lines), i + context_lines + 1)
                            context_text = "".join(
                                f"  {j+1}: {lines[j]}" for j in range(start, end)
                            )
                            matches.append({
                                "file": fpath,
                                "line": i + 1,
                                "match": line.rstrip(),
                                "context": context_text,
                            })
                            if len(matches) >= max_results:
                                return ToolResult(
                                    success=True,
                                    output=_format_matches(matches),
                                    metadata={
                                        "total_matches": len(matches),
                                        "files_scanned": files_scanned,
                                        "truncated": True,
                                    },
                                )
                except Exception:
                    continue
        
        return ToolResult(
            success=True,
            output=_format_matches(matches) if matches else "No matches found.",
            metadata={"total_matches": len(matches), "files_scanned": files_scanned},
        )


def _matches_pattern(fname: str, pattern: str) -> bool:
    """Simple glob matching."""
    import fnmatch
    return fnmatch.fnmatch(fname, pattern)


def _format_matches(matches: List[Dict]) -> str:
    lines = []
    for m in matches:
        lines.append(f"📄 {m['file']}:{m['line']}")
        lines.append(f"  → {m['match']}")
        if m.get("context"):
            lines.append(m["context"])
        lines.append("")
    return "\n".join(lines)


class CodeLintTool(Tool):
    """Lint code với nhiều linters."""
    category = ToolCategory.CODE
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "code_lint"
    
    @property
    def description(self) -> str:
        return "Lint Python code với pyflakes, pycodestyle, hoặc pylint."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "linter": {"type": "string", "default": "auto", "enum": ["auto", "pyflakes", "pycodestyle", "pylint", "flake8", "ruff"]},
            },
            "required": ["path"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path = args["path"]
        linter = args.get("linter", "auto")
        
        linters_to_try = ["ruff", "flake8", "pyflakes", "pycodestyle"] if linter == "auto" else [linter]
        
        for l in linters_to_try:
            try:
                result = subprocess.run(
                    [l, path],
                    capture_output=True,
                    text=True,
                    timeout=context.timeout,
                    check=False,
                )
                if result.returncode == 0 or result.stdout or result.stderr:
                    return ToolResult(
                        success=(result.returncode == 0),
                        output=result.stdout or "(no issues)",
                        error=result.stderr if result.stderr else None,
                        return_code=result.returncode,
                        metadata={"linter": l, "path": path},
                    )
            except FileNotFoundError:
                continue
            except Exception:
                continue
        
        return ToolResult(
            success=False,
            error="No linter available. Install: pip install ruff flake8",
            return_code=1,
        )


class CodeFormatTool(Tool):
    """Format code với black, autopep8, hoặc isort."""
    category = ToolCategory.CODE
    safety = ToolSafety.MODERATE
    
    @property
    def name(self) -> str:
        return "code_format"
    
    @property
    def description(self) -> str:
        return "Format Python code với black / autopep8 / isort."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "formatter": {"type": "string", "default": "auto", "enum": ["auto", "black", "autopep8", "isort"]},
                "check_only": {"type": "boolean", "default": False},
            },
            "required": ["path"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path = args["path"]
        formatter = args.get("formatter", "auto")
        check_only = args.get("check_only", False)
        
        formatters = ["black", "autopep8", "isort"] if formatter == "auto" else [formatter]
        
        for fmt in formatters:
            cmd = [fmt]
            if fmt == "black":
                cmd.append("--check" if check_only else "--write")
            elif fmt == "autopep8":
                cmd.append("--in-place" if not check_only else "--diff")
            elif fmt == "isort":
                cmd.append("--check-only" if check_only else "--write")
            cmd.append(path)
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=context.timeout,
                    check=False,
                )
                return ToolResult(
                    success=(result.returncode == 0),
                    output=result.stdout or f"Formatted with {fmt}",
                    error=result.stderr if result.stderr else None,
                    return_code=result.returncode,
                    metadata={"formatter": fmt, "path": path, "check_only": check_only},
                )
            except FileNotFoundError:
                continue
            except Exception:
                continue
        
        return ToolResult(
            success=False,
            error="No formatter available. Install: pip install black",
            return_code=1,
        )
