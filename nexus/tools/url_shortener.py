"""
URL Shortener Tool - Rút gọn URL qua is.gd hoặc tinyurl.
Author: Hieu Louis (2026)

Dùng stdlib urllib.request (không cần deps ngoài).
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class URLShortenerTool(Tool):
    """Rút gọn URL bằng dịch vụ is.gd hoặc tinyurl."""

    category = ToolCategory.WEB
    safety = ToolSafety.SAFE

    SUPPORTED_SERVICES = ("is.gd", "tinyurl")
    USER_AGENT = "NexusCoder/0.3 (URL Shortener)"

    @property
    def name(self) -> str:
        return "url_shortener"

    @property
    def description(self) -> str:
        return (
            "Rút gọn URL dài thành URL ngắn qua is.gd hoặc tinyurl API. "
            "Không cần API key, dùng stdlib urllib."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL cần rút gọn"},
                "service": {
                    "type": "string",
                    "enum": list(self.SUPPORTED_SERVICES),
                    "default": "is.gd",
                    "description": "Dịch vụ rút gọn",
                },
            },
            "required": ["url"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        url = str(args.get("url", ""))
        if not url:
            return "Missing required arg: url"
        if not url.startswith(("http://", "https://")):
            return "url phải bắt đầu bằng http:// hoặc https://"
        service = args.get("service", "is.gd")
        if service not in self.SUPPORTED_SERVICES:
            return f"service phải là {self.SUPPORTED_SERVICES}, nhận được '{service}'"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        url: str = args["url"]
        service: str = args.get("service", "is.gd")
        timeout = int(context.timeout or 30)

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would shorten {url} via {service}",
                metadata={"dry_run": True, "url": url, "service": service},
            )

        try:
            short_url = self._shorten(url, service, timeout)
            return ToolResult(
                success=True,
                output=short_url,
                metadata={
                    "original_url": url,
                    "short_url": short_url,
                    "service": service,
                },
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                success=False,
                error=f"Shorten failed ({service}): {e}",
                return_code=1,
            )

    def _shorten(self, url: str, service: str, timeout: int) -> str:
        """Gọi API tương ứng // call the proper service."""
        headers = {"User-Agent": self.USER_AGENT}

        if service == "is.gd":
            # is.gd: GET https://is.gd/create.php?format=simple&url=<URL>
            api_url = (
                "https://is.gd/create.php?format=simple&"
                + urllib.parse.urlencode({"url": url})
            )
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                body = resp.read().decode("utf-8", errors="replace").strip()
            if not body or body.lower().startswith("error"):
                raise RuntimeError(f"is.gd trả về: {body!r}")
            return body

        # tinyurl: GET https://tinyurl.com/api-create.php?url=<URL>
        api_url = (
            "https://tinyurl.com/api-create.php?"
            + urllib.parse.urlencode({"url": url})
        )
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace").strip()
        if not body or "error" in body.lower()[:32]:
            raise RuntimeError(f"tinyurl trả về: {body!r}")
        return body
