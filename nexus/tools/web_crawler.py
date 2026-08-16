"""
Web Crawler Tool - Crawl website theo BFS lên tới N depth.
Author: Hieu Louis (2026)

`requests` + `BeautifulSoup`. Tôn trọng same_domain_only, max_pages.
Ghi JSON crawl index vào output_dir.
"""
from __future__ import annotations

import json
import os
from collections import deque
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class WebCrawlerTool(Tool):
    """Crawl website BFS: thu thập URLs + page titles + link graph."""

    category = ToolCategory.WEB
    safety = ToolSafety.MODERATE
    timeout = 120

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 NexusCrawlerBot/0.3"
    )

    @property
    def name(self) -> str:
        return "web_crawler"

    @property
    def description(self) -> str:
        return (
            "Crawl website theo BFS với giới hạn depth + max_pages. "
            "Thu thập URL, title, status, content_type, link graph. "
            "Optionally filter same-domain only. Lưu JSON index."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "start_url": {"type": "string", "description": "URL khởi đầu"},
                "max_depth": {"type": "integer", "default": 2, "description": "Độ sâu tối đa (BFS)"},
                "max_pages": {"type": "integer", "default": 50, "description": "Số page tối đa crawl"},
                "same_domain_only": {"type": "boolean", "default": True, "description": "Chỉ crawl cùng domain"},
                "output_dir": {
                    "type": "string",
                    "description": "Thư mục lưu crawl_index.json (mặc định = cwd)",
                },
                "headers": {"type": "object", "description": "Custom HTTP headers"},
                "timeout_per_page": {"type": "integer", "default": 15, "description": "HTTP timeout (giây)"},
                "respect_robots": {
                    "type": "boolean",
                    "default": True,
                    "description": "Skip paths disallowed trong robots.txt",
                },
            },
            "required": ["start_url"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        url = str(args.get("start_url", "")).strip()
        if not url:
            return "Missing required arg: start_url"
        if not url.startswith(("http://", "https://")):
            return "start_url phải là http(s)://"
        if int(args.get("max_depth", 2)) < 0:
            return "max_depth phải >= 0"
        if int(args.get("max_pages", 50)) < 1:
            return "max_pages phải >= 1"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        start_url: str = str(args["start_url"]).strip()
        max_depth: int = int(args.get("max_depth", 2))
        max_pages: int = int(args.get("max_pages", 50))
        same_domain_only: bool = bool(args.get("same_domain_only", True))
        output_dir: str = str(args.get("output_dir") or context.working_dir or ".")
        headers: Dict[str, str] = args.get("headers") or {}
        timeout_per_page = int(args.get("timeout_per_page", 15))
        respect_robots: bool = bool(args.get("respect_robots", True))

        start_domain = urlparse(start_url).netloc
        if context.dry_run:
            return ToolResult(
                success=True,
                output=(
                    f"[dry-run] Would crawl from {start_url} "
                    f"(max_depth={max_depth}, max_pages={max_pages}, "
                    f"same_domain={same_domain_only})"
                ),
                metadata={
                    "dry_run": True,
                    "start_url": start_url,
                    "max_depth": max_depth,
                    "max_pages": max_pages,
                },
            )

        try:
            import requests  # type: ignore
        except ImportError:
            return ToolResult(
                success=False,
                error="Cần requests. Cài: pip install requests beautifulsoup4",
                return_code=1,
            )
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError:
            return ToolResult(
                success=False,
                error="Cần beautifulsoup4. Cài: pip install beautifulsoup4 lxml",
                return_code=1,
            )

        session = requests.Session()
        final_headers = {"User-Agent": self.DEFAULT_USER_AGENT, **headers}
        session.headers.update(final_headers)

        # robots.txt (best-effort) // best-effort robots.txt
        robots_disallowed: List[str] = []
        if respect_robots:
            robots_disallowed = self._fetch_robots(session, start_url, timeout_per_page)

        visited: Set[str] = set()
        pages: List[Dict[str, Any]] = []
        queue: deque[tuple[str, int]] = deque([(start_url, 0)])

        while queue and len(pages) < max_pages:
            url, depth = queue.popleft()
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            if self._is_robots_disallowed(url, robots_disallowed):
                continue

            page_info: Dict[str, Any] = {
                "url": url,
                "depth": depth,
                "status": None,
                "title": None,
                "content_type": None,
                "out_links": [],
                "error": None,
            }
            try:
                resp = session.get(url, timeout=timeout_per_page, allow_redirects=True)
                page_info["status"] = resp.status_code
                page_info["content_type"] = resp.headers.get("Content-Type", "").split(";")[0]
                final_url = resp.url
                if resp.status_code == 200 and "html" in (page_info["content_type"] or ""):
                    soup = BeautifulSoup(resp.text, "lxml")
                    title_tag = soup.find("title")
                    page_info["title"] = (title_tag.get_text(strip=True) if title_tag else "")[:200]
                    # Extract links for next depth
                    if depth < max_depth:
                        new_links = self._extract_links(soup, final_url, start_domain, same_domain_only)
                        page_info["out_links"] = new_links
                        for link in new_links:
                            if link not in visited:
                                queue.append((link, depth + 1))
            except Exception as e:  # noqa: BLE001
                page_info["error"] = str(e)[:200]

            pages.append(page_info)

        # Save crawl index // save JSON
        try:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "crawl_index.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "start_url": start_url,
                        "max_depth": max_depth,
                        "max_pages": max_pages,
                        "pages_crawled": len(pages),
                        "pages": pages,
                    },
                    f, ensure_ascii=False, indent=2,
                )
        except OSError as e:
            return ToolResult(
                success=False,
                error=f"Cannot save crawl index: {e}",
                return_code=1,
                metadata={"pages_crawled": len(pages)},
            )

        # Build summary
        success_count = sum(1 for p in pages if p["status"] == 200)
        domains_seen = sorted({urlparse(p["url"]).netloc for p in pages})

        lines = [
            f"Crawl summary for {start_url}",
            f"  Pages crawled: {len(pages)}",
            f"  Successful (200): {success_count}",
            f"  Max depth reached: {max(p['depth'] for p in pages) if pages else 0}",
            f"  Domains seen: {domains_seen}",
        ]
        return ToolResult(
            success=True,
            output="\n".join(lines),
            artifacts=[out_path],
            metadata={
                "start_url": start_url,
                "pages_crawled": len(pages),
                "success_count": success_count,
                "domains_seen": domains_seen,
                "output_file": out_path,
            },
        )

    @staticmethod
    def _fetch_robots(session: Any, base_url: str, timeout: int) -> List[str]:
        """Best-effort robots.txt fetch. Returns list of disallowed path prefixes."""
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            resp = session.get(robots_url, timeout=timeout)
            if resp.status_code != 200:
                return []
            disallowed: List[str] = []
            for line in resp.text.splitlines():
                line = line.strip()
                if line.lower().startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    if path:
                        disallowed.append(path)
            return disallowed
        except Exception:
            return []

    @staticmethod
    def _is_robots_disallowed(url: str, disallowed: List[str]) -> bool:
        if not disallowed:
            return False
        path = urlparse(url).path or "/"
        for d in disallowed:
            if d == "/" or path.startswith(d):
                return True
        return False

    @staticmethod
    def _extract_links(
        soup: Any, base_url: str, start_domain: str, same_domain_only: bool,
    ) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
                continue
            full = urljoin(base_url, href)
            # Strip fragment
            full = full.split("#", 1)[0]
            if not full.startswith(("http://", "https://")):
                continue
            if same_domain_only and urlparse(full).netloc != start_domain:
                continue
            if full not in seen:
                seen.add(full)
                out.append(full)
        return out
