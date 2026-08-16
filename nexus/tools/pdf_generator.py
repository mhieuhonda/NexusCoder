"""
PDF Generator Tool - Sinh PDF từ text/HTML/markdown.
===========================================
Tool sinh PDF từ nội dung text thuần, HTML hoặc Markdown.
Backend: `reportlab` (text/markdown trực tiếp) hoặc `weasyprint` (HTML→PDF).

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


PAGE_SIZES = {"A4", "A3", "LETTER", "LEGAL"}


class PDFGeneratorTool(Tool):
    """Sinh PDF từ text/HTML/markdown content."""

    category = ToolCategory.CONVERT
    safety = ToolSafety.MODERATE
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "pdf_generator"

    @property
    def description(self) -> str:
        return "Sinh PDF từ text/HTML/markdown. Backend: reportlab (text/md) hoặc weasyprint (HTML)."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Nội dung cần convert"},
                "output_path": {"type": "string", "description": "Đường dẫn file PDF output"},
                "format": {
                    "type": "string",
                    "enum": ["text", "html", "markdown"],
                    "default": "text",
                },
                "page_size": {
                    "type": "string",
                    "enum": sorted(PAGE_SIZES),
                    "default": "A4",
                },
                "title": {"type": "string", "description": "Metadata title của PDF"},
                "author": {"type": "string", "description": "Metadata author của PDF"},
            },
            "required": ["content", "output_path"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("content"):
            return "Missing required arg: content"
        if not args.get("output_path"):
            return "Missing required arg: output_path"
        page = args.get("page_size", "A4")
        if page not in PAGE_SIZES:
            return f"Invalid page_size='{page}'. Supported: {sorted(PAGE_SIZES)}"
        return None

    # ---- Backends -------------------------------------------------------

    def _gen_reportlab(
        self,
        content: str,
        output_path: str,
        fmt: str,
        page_size: str,
        title: Optional[str],
        author: Optional[str],
    ) -> ToolResult:
        """Sinh PDF bằng reportlab (plaintext hoặc markdown đơn giản)."""
        try:
            from reportlab.lib.pagesizes import A4, A3, letter, legal  # type: ignore
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer  # type: ignore
            from reportlab.lib.styles import getSampleStyleSheet  # type: ignore
            from reportlab.lib.units import inch  # type: ignore  # noqa: F401
            from reportlab.lib import colors  # type: ignore  # noqa: F401
        except ImportError:
            return ToolResult(
                success=False,
                error="reportlab chưa cài. Cài đặt: pip install reportlab",
                return_code=127,
            )

        size_map = {"A4": A4, "A3": A3, "LETTER": letter, "LEGAL": legal}
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=size_map[page_size],
                title=title or "Nexus PDF",
                author=author or "Nexus Coder",
            )
            styles = getSampleStyleSheet()
            story = []
            if fmt == "markdown":
                # Parse markdown rất cơ bản: ## → Heading2, # → Heading1, else Paragraph
                for line in content.split("\n"):
                    stripped = line.strip()
                    if not stripped:
                        story.append(Spacer(1, 6))
                    elif stripped.startswith("# "):
                        story.append(Paragraph(stripped[2:], styles["Heading1"]))
                    elif stripped.startswith("## "):
                        story.append(Paragraph(stripped[3:], styles["Heading2"]))
                    elif stripped.startswith("### "):
                        story.append(Paragraph(stripped[4:], styles["Heading3"]))
                    else:
                        # Escape XML chars / escape XML special chars
                        safe = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        story.append(Paragraph(safe, styles["Normal"]))
            else:
                for line in content.split("\n"):
                    safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") or "&nbsp;"
                    story.append(Paragraph(safe, styles["Normal"]))
            doc.build(story)
            return ToolResult(
                success=True,
                output=f"PDF generated → {output_path}",
                artifacts=[output_path],
                metadata={"backend": "reportlab", "page_size": page_size, "format": fmt},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"reportlab build failed: {e}", return_code=1)

    def _gen_weasyprint(self, html_content: str, output_path: str, page_size: str, title: Optional[str], author: Optional[str]) -> ToolResult:
        """Sinh PDF từ HTML bằng weasyprint."""
        try:
            from weasyprint import HTML  # type: ignore
        except ImportError:
            return ToolResult(
                success=False,
                error="weasyprint chưa cài. Cài đặt: pip install weasyprint",
                return_code=127,
            )
        try:
            html_obj = HTML(string=html_content)
            html_obj.write_pdf(output_path)
            return ToolResult(
                success=True,
                output=f"PDF generated → {output_path}",
                artifacts=[output_path],
                metadata={"backend": "weasyprint", "page_size": page_size, "format": "html"},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"weasyprint build failed: {e}", return_code=1)

    # ---- Execute --------------------------------------------------------

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        content = args["content"]
        output_path = args["output_path"]
        fmt = args.get("format", "text")
        page_size = args.get("page_size", "A4")
        title = args.get("title")
        author = args.get("author")

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Sẽ sinh PDF ({fmt}, {page_size}) → {output_path} ({len(content)} chars)",
                metadata={"output_path": output_path, "format": fmt, "page_size": page_size, "dry_run": True},
            )

        # Đảm bảo thư mục cha tồn tại / ensure parent dir exists
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)

        if fmt == "html":
            return self._gen_weasyprint(content, output_path, page_size, title, author)

        if fmt == "markdown":
            # Thử pandoc trước (chất lượng cao) / try pandoc first
            import shutil
            if shutil.which("pandoc"):
                tmp_md = os.path.join(parent, f".nexus_pdf_{os.getpid()}.md")
                try:
                    with open(tmp_md, "w", encoding="utf-8") as f:
                        f.write(content)
                    cmd = ["pandoc", tmp_md, "-f", "markdown", "-t", "pdf", "-o", output_path, "-V", f"geometry:{page_size}paper"]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=context.timeout, check=False)
                    if proc.returncode == 0:
                        return ToolResult(
                            success=True,
                            output=f"PDF generated (pandoc) → {output_path}",
                            artifacts=[output_path],
                            metadata={"backend": "pandoc", "format": "markdown"},
                        )
                    # pandoc fail → fallback reportlab
                except subprocess.TimeoutExpired:
                    return ToolResult(success=False, error="pandoc timeout", return_code=124)
                finally:
                    if os.path.exists(tmp_md):
                        os.remove(tmp_md)
            return self._gen_reportlab(content, output_path, fmt, page_size, title, author)

        # format == "text"
        return self._gen_reportlab(content, output_path, fmt, page_size, title, author)
