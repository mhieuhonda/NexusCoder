"""
Web Scraper Tool - Scrape HTML bằng CSS/XPath selector, có JS rendering.
Author: Hieu Louis (2026)

Backend:
  - Mặc định: `requests` + `BeautifulSoup` (lxml parser).
  - render_js=True: `playwright` (lazy import, headless Chromium).
Extract modes: text, html, attr.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class WebScraperTool(Tool):
    """Scrape HTML page, extract data theo CSS/XPath selector."""

    category = ToolCategory.WEB
    safety = ToolSafety.MODERATE

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 NexusCoder/0.3"
    )

    @property
    def name(self) -> str:
        return "web_scraper"

    @property
    def description(self) -> str:
        return (
            "Scrape nội dung HTML. Hỗ trợ CSS selector (BeautifulSoup) hoặc "
            "XPath (lxml.etree). Extract mode: text/html/attr. Optional "
            "JS rendering qua Playwright (headless Chromium)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL cần scrape"},
                "selector": {
                    "type": "string",
                    "description": "CSS (vd 'div.article h2') hoặc XPath (vd '//h2[@class=\"title\"]')",
                },
                "extract": {
                    "type": "string",
                    "enum": ["text", "html", "attr"],
                    "default": "text",
                    "description": "Extract mode",
                },
                "attr_name": {
                    "type": "string",
                    "description": "Tên attr nếu extract=attr (vd 'href')",
                },
                "render_js": {
                    "type": "boolean",
                    "default": False,
                    "description": "Dùng Playwright render JS (chậm hơn, cần deps)",
                },
                "headers": {
                    "type": "object",
                    "description": "Custom HTTP headers (User-Agent, Authorization, ...)",
                },
                "timeout": {"type": "integer", "default": 30, "description": "HTTP timeout (giây)"},
                "max_results": {"type": "integer", "default": 100, "description": "Số kết quả tối đa trả về"},
            },
            "required": ["url", "selector"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("url"):
            return "Missing required arg: url"
        if not str(args["url"]).startswith(("http://", "https://")):
            return "url phải là http(s)://"
        if not args.get("selector"):
            return "Missing required arg: selector"
        if args.get("extract") == "attr" and not args.get("attr_name"):
            return "extract=attr yêu cầu arg 'attr_name'"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        url: str = args["url"]
        selector: str = args["selector"]
        extract: str = str(args.get("extract", "text"))
        attr_name: Optional[str] = args.get("attr_name")
        render_js: bool = bool(args.get("render_js", False))
        headers: Dict[str, str] = args.get("headers") or {}
        timeout = int(args.get("timeout") or context.timeout or 30)
        max_results = int(args.get("max_results", 100))

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would scrape {url} (selector={selector!r}, render_js={render_js})",
                metadata={
                    "dry_run": True,
                    "url": url,
                    "selector": selector,
                    "render_js": render_js,
                },
            )

        # Lấy HTML // fetch HTML
        try:
            if render_js:
                html = self._fetch_with_playwright(url, headers, timeout)
            else:
                html = self._fetch_with_requests(url, headers, timeout)
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                success=False,
                error=f"Fetch failed: {e}",
                return_code=1,
                metadata={"url": url},
            )

        # Extract data // extract data
        try:
            results = self._extract(html, selector, extract, attr_name, max_results)
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                success=False,
                error=f"Extract failed: {e}",
                return_code=1,
                metadata={"url": url, "html_size": len(html)},
            )

        # Output format
        if extract == "text":
            output = "\n---\n".join(results) if results else "(no matches)"
        elif extract == "html":
            output = "\n---\n".join(results) if results else "(no matches)"
        else:  # attr
            output = "\n".join(results) if results else "(no matches)"

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "url": url,
                "selector": selector,
                "extract": extract,
                "attr_name": attr_name,
                "render_js": render_js,
                "html_size": len(html),
                "match_count": len(results),
                "results": results[:max_results],
            },
        )

    # ---------- HTML fetch ----------
    def _fetch_with_requests(self, url: str, headers: Dict[str, str], timeout: int) -> str:
        try:
            import requests  # type: ignore
        except ImportError:
            # Fallback stdlib urllib // urllib fallback
            import urllib.request
            req_headers = {"User-Agent": self.DEFAULT_USER_AGENT, **headers}
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.read().decode("utf-8", errors="replace")

        final_headers = {"User-Agent": self.DEFAULT_USER_AGENT, **headers}
        resp = requests.get(url, headers=final_headers, timeout=timeout)
        resp.raise_for_status()
        return resp.text

    def _fetch_with_playwright(self, url: str, headers: Dict[str, str], timeout: int) -> str:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError:
            raise RuntimeError(
                "render_js=True cần playwright. Cài: pip install playwright && playwright install chromium"
            )

        extra_headers = {"User-Agent": self.DEFAULT_USER_AGENT, **headers}
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                ctx = browser.new_context(extra_http_headers=extra_headers)
                page = ctx.new_page()
                page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
                # Trả HTML sau khi JS render // return rendered HTML
                return page.content()
            finally:
                browser.close()

    # ---------- Extract ----------
    def _extract(
        self, html: str, selector: str, extract: str,
        attr_name: Optional[str], max_results: int,
    ) -> List[str]:
        # XPath selector → lxml.etree
        if selector.startswith("/"):
            return self._extract_xpath(html, selector, extract, attr_name, max_results)
        # CSS selector → BeautifulSoup
        return self._extract_css(html, selector, extract, attr_name, max_results)

    def _extract_css(
        self, html: str, selector: str, extract: str,
        attr_name: Optional[str], max_results: int,
    ) -> List[str]:
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError:
            raise RuntimeError("Cần beautifulsoup4. Cài: pip install beautifulsoup4 lxml")

        soup = BeautifulSoup(html, "lxml")
        nodes = soup.select(selector)
        out: List[str] = []
        for n in nodes[:max_results]:
            if extract == "text":
                out.append(n.get_text(separator=" ", strip=True))
            elif extract == "html":
                out.append(str(n))
            else:  # attr
                val = n.get(attr_name or "") if hasattr(n, "get") else None
                if val is not None:
                    out.append(str(val))
        return out

    def _extract_xpath(
        self, html: str, xpath: str, extract: str,
        attr_name: Optional[str], max_results: int,
    ) -> List[str]:
        try:
            from lxml import etree  # type: ignore
        except ImportError:
            raise RuntimeError("XPath cần lxml. Cài: pip install lxml")

        tree = etree.HTML(html)
        if tree is None:
            return []
        nodes = tree.xpath(xpath)
        out: List[str] = []
        for n in nodes[:max_results]:
            if extract == "text":
                txt = n.text if hasattr(n, "text") else str(n)
                out.append((txt or "").strip())
            elif extract == "html":
                if hasattr(n, "tag"):
                    out.append(etree.tostring(n, encoding="unicode"))
                else:
                    out.append(str(n))
            else:  # attr
                if isinstance(n, str):
                    out.append(n)
                elif hasattr(n, "get"):
                    v = n.get(attr_name or "")
                    if v is not None:
                        out.append(str(v))
        return out
