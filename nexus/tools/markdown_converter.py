"""
Markdown Converter Tool - Convert Markdown sang HTML/PDF/LaTeX/DOCX.
===========================================
Dùng `markdown` lib cho HTML, `pandoc` subprocess cho PDF/LaTeX/DOCX.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


SUPPORTED_FORMATS = {"html", "pdf", "latex", "docx"}


class MarkdownConverterTool(Tool):
    """Convert Markdown → HTML / PDF / LaTeX / DOCX."""

    category = ToolCategory.CONVERT
    safety = ToolSafety.SAFE

    @property
    def name(self) -> str:
        return "markdown_converter"

    @property
    def description(self) -> str:
        return "Convert Markdown sang HTML (markdown lib), PDF/LaTeX/DOCX (pandoc subprocess)."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "Đường dẫn file .md (hoặc dùng input_text)"},
                "input_text": {"type": "string", "description": "Markdown content trực tiếp"},
                "output_format": {
                    "type": "string",
                    "enum": sorted(SUPPORTED_FORMATS),
                    "default": "html",
                },
                "output_path": {"type": "string", "description": "File output (bỏ qua → trả về content)"},
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Markdown extensions (tables, fenced_code, codehilite, ...)",
                },
            },
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("input_path") and not args.get("input_text"):
            return "Missing arg: cần 'input_path' hoặc 'input_text'"
        fmt = args.get("output_format", "html")
        if fmt not in SUPPORTED_FORMATS:
            return f"Invalid output_format='{fmt}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        return None

    def _read_input(self, args: Dict[str, Any]) -> str:
        if args.get("input_text"):
            return args["input_text"]
        with open(args["input_path"], "r", encoding="utf-8") as f:
            return f.read()

    def _to_html(self, md_text: str, extensions: List[str]) -> str:
        """Convert Markdown → HTML. Ưu tiên `markdown` lib, fallback regex cơ bản."""
        try:
            import markdown as md  # type: ignore
            return md.markdown(md_text, extensions=extensions or ["tables", "fenced_code"])
        except ImportError:
            pass
        # Fallback HTML escape + conversion rất cơ bản / minimal regex fallback
        import html as _html
        out = _html.escape(md_text)
        out = __import__("re").sub(r"^### (.+)$", r"<h3>\1</h3>", out, flags=__import__("re").MULTILINE)
        out = __import__("re").sub(r"^## (.+)$", r"<h2>\1</h2>", out, flags=__import__("re").MULTILINE)
        out = __import__("re").sub(r"^# (.+)$", r"<h1>\1</h1>", out, flags=__import__("re").MULTILINE)
        out = __import__("re").sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
        out = __import__("re").sub(r"\*(.+?)\*", r"<em>\1</em>", out)
        out = __import__("re").sub(r"`(.+?)`", r"<code>\1</code>", out)
        # Wrap paragraphs
        lines = out.split("\n\n")
        wrapped = "\n".join(
            ln if ln.startswith("<") else f"<p>{ln}</p>"
            for ln in lines if ln.strip()
        )
        return f"<!-- fallback converter (markdown lib missing) -->\n{wrapped}"

    def _pandoc_convert(
        self,
        md_text: str,
        fmt: str,
        output_path: str,
        timeout: int,
        cwd: str,
    ) -> ToolResult:
        """Dùng pandoc subprocess cho PDF/LaTeX/DOCX."""
        if not shutil.which("pandoc"):
            return ToolResult(
                success=False,
                error="pandoc không có trong PATH. Cài đặt: https://pandoc.org/installing.html",
                return_code=127,
            )
        tmp_md = os.path.join(cwd, f".nexus_md_{os.getpid()}.md")
        try:
            with open(tmp_md, "w", encoding="utf-8") as f:
                f.write(md_text)
            cmd = ["pandoc", tmp_md, "-f", "markdown", "-t", fmt, "-o", output_path]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                check=False,
            )
            if proc.returncode != 0:
                return ToolResult(
                    success=False,
                    error=proc.stderr.strip() or f"pandoc exit code {proc.returncode}",
                    return_code=proc.returncode,
                )
            return ToolResult(
                success=True,
                output=f"Converted → {output_path}",
                artifacts=[output_path],
                metadata={"format": fmt, "backend": "pandoc", "output_path": output_path},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="pandoc timeout", return_code=124)
        finally:
            if os.path.exists(tmp_md):
                os.remove(tmp_md)

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        fmt = args.get("output_format", "html")
        extensions = args.get("extensions", []) or []
        try:
            md_text = self._read_input(args)
        except Exception as e:
            return ToolResult(success=False, error=f"Read input failed: {e}", return_code=1)

        cwd = context.working_dir or os.getcwd()
        output_path = args.get("output_path")

        if context.dry_run and output_path:
            return ToolResult(
                success=True,
                output=f"[dry-run] Sẽ convert {len(md_text)} chars Markdown → {fmt} tại {output_path}",
                metadata={"format": fmt, "output_path": output_path, "dry_run": True},
            )

        # HTML: dùng lib markdown
        if fmt == "html":
            html = self._to_html(md_text, extensions)
            if output_path:
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    return ToolResult(
                        success=True,
                        output=f"Converted → {output_path}",
                        artifacts=[output_path],
                        metadata={"format": "html", "output_path": output_path},
                    )
                except Exception as e:
                    return ToolResult(success=False, error=f"Write failed: {e}", return_code=1)
            return ToolResult(success=True, output=html, metadata={"format": "html", "length": len(html)})

        # PDF / LaTeX / DOCX: cần pandoc
        if not output_path:
            # Sinh output_path mặc định / generate default output path
            base = args.get("input_path") or "markdown_output"
            stem = os.path.splitext(os.path.basename(base))[0]
            output_path = os.path.join(cwd, f"{stem}.{fmt}")
        return self._pandoc_convert(md_text, fmt, output_path, context.timeout, cwd)
