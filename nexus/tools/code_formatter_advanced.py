"""
Code Formatter Advanced Tool - Format code trong nhiều ngôn ngữ.
Author: Hieu Louis (2026)

- Python : stdlib ast + simple normalizer (rstrip, blank-line cleanup, trailing newline)
- JSON   : stdlib json.dumps(indent=2)
- JS     : lazy import `jsbeautifier`
- SQL    : lazy import `sqlparse`
- HTML   : lazy import `beautifulsoup4` (bs4)
- CSS    : lazy import `cssbeautifier`

In-place edit nếu cung cấp `path`. MODERATE safety, requires_confirmation.
"""
from __future__ import annotations

import ast
import json
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


SUPPORTED_LANGS = {"python", "javascript", "sql", "html", "css", "json"}


def _format_python(source: str) -> str:
    """Format Python cơ bản: validate syntax, rstrip lines, cleanup blank runs."""
    ast.parse(source)  # raise SyntaxError if invalid
    lines = source.splitlines()
    out: list[str] = []
    blank_run = 0
    for ln in lines:
        stripped = ln.rstrip()
        if stripped == "":
            blank_run += 1
            # Tối đa 2 blank lines liên tiếp
            if blank_run <= 2:
                out.append("")
            continue
        blank_run = 0
        out.append(stripped)
    formatted = "\n".join(out).rstrip() + "\n"
    return formatted


def _format_json(source: str) -> str:
    data = json.loads(source)
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


class CodeFormatterAdvancedTool(Tool):
    """Format code multi-language: Python/JSON (stdlib) + JS/SQL/HTML/CSS (lazy deps)."""

    category = ToolCategory.CODE
    safety = ToolSafety.MODERATE  # có thể sửa file in-place
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "code_formatter_advanced"

    @property
    def description(self) -> str:
        return (
            "Format code multi-language: Python (ast), JSON (stdlib), "
            "JavaScript (jsbeautifier), SQL (sqlparse), HTML (bs4), CSS (cssbeautifier). "
            "In-place edit nếu cung cấp path."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File để format (in-place)"},
                "code": {"type": "string", "description": "Code để format (nếu không dùng path)"},
                "language": {
                    "type": "string",
                    "enum": sorted(SUPPORTED_LANGS),
                    "description": "Ngôn ngữ (mặc định python)",
                },
            },
            "anyOf": [{"required": ["path"]}, {"required": ["code"]}],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("path") and not args.get("code"):
            return "Missing required arg: path hoặc code"
        lang = args.get("language", "python")
        if lang not in SUPPORTED_LANGS:
            return f"Unsupported language: {lang}. Chọn: {sorted(SUPPORTED_LANGS)}"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        lang = args.get("language", "python")
        path = args.get("path")
        code: Optional[str] = args.get("code")

        if path:
            if context.dry_run:
                return ToolResult(
                    success=True,
                    output=f"[dry-run] Sẽ format {path} ({lang})",
                    metadata={"path": path, "language": lang, "dry_run": True},
                )
            try:
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                return ToolResult(success=False, error=f"Đọc file lỗi: {e}", return_code=1)

        assert code is not None  # validated above

        try:
            if lang == "python":
                formatted = _format_python(code)
            elif lang == "json":
                formatted = _format_json(code)
            elif lang == "javascript":
                try:
                    import jsbeautifier  # type: ignore
                except ImportError as e:
                    return ToolResult(
                        success=False,
                        error=f"jsbeautifier not installed: {e}. Cài: pip install jsbeautifier",
                        return_code=127,
                    )
                formatted = jsbeautifier.beautify(code)
            elif lang == "sql":
                try:
                    import sqlparse  # type: ignore
                except ImportError as e:
                    return ToolResult(
                        success=False,
                        error=f"sqlparse not installed: {e}. Cài: pip install sqlparse",
                        return_code=127,
                    )
                formatted = sqlparse.format(
                    code, reindent=True, keyword_case="upper", indent_width=2,
                )
            elif lang == "html":
                try:
                    from bs4 import BeautifulSoup  # type: ignore
                except ImportError as e:
                    return ToolResult(
                        success=False,
                        error=f"beautifulsoup4 not installed: {e}. Cài: pip install beautifulsoup4",
                        return_code=127,
                    )
                soup = BeautifulSoup(code, "html.parser")
                formatted = soup.prettify()
            elif lang == "css":
                try:
                    import cssbeautifier  # type: ignore
                except ImportError as e:
                    return ToolResult(
                        success=False,
                        error=f"cssbeautifier not installed: {e}. Cài: pip install cssbeautifier",
                        return_code=127,
                    )
                formatted = cssbeautifier.beautify(code)
            else:
                return ToolResult(success=False, error=f"Unsupported language: {lang}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}", return_code=1)

        # Write back if path provided
        artifacts = []
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(formatted)
                artifacts.append(path)
            except Exception as e:
                return ToolResult(success=False, error=f"Write file lỗi: {e}", return_code=1)

        return ToolResult(
            success=True,
            output=formatted,
            artifacts=artifacts,
            metadata={
                "language": lang,
                "path": path,
                "input_length": len(code),
                "output_length": len(formatted),
                "in_place": bool(path),
            },
        )
