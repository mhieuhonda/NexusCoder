"""
Code Minifier Tool - Minify JS/CSS/HTML code.
Author: Hieu Louis (2026)

Lazy import:
- JavaScript : `jsmin`
- CSS        : `cssmin`
- HTML       : `htmlmin`

MODERATE safety (in-place edit possible), requires_confirmation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


SUPPORTED_LANGS = {"javascript", "css", "html"}


class CodeMinifierTool(Tool):
    """Minify JavaScript/CSS/HTML code (lazy deps)."""

    category = ToolCategory.CODE
    safety = ToolSafety.MODERATE  # in-place edit
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "code_minifier"

    @property
    def description(self) -> str:
        return (
            "Minify code JS/CSS/HTML. Sử dụng jsmin (JS), cssmin (CSS), htmlmin (HTML). "
            "In-place edit nếu cung cấp path. Trả về reduction %."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File để minify (in-place)"},
                "code": {"type": "string", "description": "Code để minify (nếu không dùng path)"},
                "language": {
                    "type": "string",
                    "enum": sorted(SUPPORTED_LANGS),
                    "description": "Ngôn ngữ: javascript/css/html",
                },
            },
            "anyOf": [{"required": ["path"]}, {"required": ["code"]}],
            "required": ["language"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        lang = args.get("language")
        if not lang:
            return "Missing required arg: language"
        if lang not in SUPPORTED_LANGS:
            return f"Unsupported language: {lang}. Chọn: {sorted(SUPPORTED_LANGS)}"
        if not args.get("path") and not args.get("code"):
            return "Missing required arg: path hoặc code"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        lang: str = args["language"]
        path = args.get("path")
        code: Optional[str] = args.get("code")

        if path:
            if context.dry_run:
                return ToolResult(
                    success=True,
                    output=f"[dry-run] Sẽ minify {path} ({lang})",
                    metadata={"path": path, "language": lang, "dry_run": True},
                )
            try:
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                return ToolResult(success=False, error=f"Đọc file lỗi: {e}", return_code=1)

        assert code is not None

        try:
            if lang == "javascript":
                try:
                    import jsmin  # type: ignore
                except ImportError as e:
                    return ToolResult(
                        success=False,
                        error=f"jsmin not installed: {e}. Cài: pip install jsmin",
                        return_code=127,
                    )
                minified = jsmin.jsmin(code)
            elif lang == "css":
                try:
                    import cssmin  # type: ignore
                except ImportError as e:
                    return ToolResult(
                        success=False,
                        error=f"cssmin not installed: {e}. Cài: pip install cssmin",
                        return_code=127,
                    )
                minified = cssmin.cssmin(code)
            elif lang == "html":
                try:
                    import htmlmin  # type: ignore
                except ImportError as e:
                    return ToolResult(
                        success=False,
                        error=f"htmlmin not installed: {e}. Cài: pip install htmlmin",
                        return_code=127,
                    )
                minifier = htmlmin.Minifier(
                    remove_comments=True,
                    remove_empty_space=True,
                    remove_optional_attribute_quotes=False,
                )
                minified = minifier.minify(code)
            else:
                return ToolResult(success=False, error=f"Unsupported language: {lang}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}", return_code=1)

        # Write back if path provided
        artifacts = []
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(minified)
                artifacts.append(path)
            except Exception as e:
                return ToolResult(success=False, error=f"Write file lỗi: {e}", return_code=1)

        ratio = (len(minified) / max(1, len(code))) * 100.0
        return ToolResult(
            success=True,
            output=minified,
            artifacts=artifacts,
            metadata={
                "language": lang,
                "path": path,
                "input_length": len(code),
                "output_length": len(minified),
                "reduction_pct": round(100.0 - ratio, 2),
                "in_place": bool(path),
            },
        )
