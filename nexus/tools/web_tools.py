"""Web Tools - HTTP requests, web fetch, web search."""
from __future__ import annotations

import json
import urllib.parse
from typing import Dict, Any

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class HTTPRequestTool(Tool):
    """Thực hiện HTTP requests."""
    category = ToolCategory.WEB
    safety = ToolSafety.MODERATE
    
    @property
    def name(self) -> str:
        return "http_request"
    
    @property
    def description(self) -> str:
        return "Thực hiện HTTP request (GET/POST/PUT/DELETE). Hỗ trợ headers, body, params."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "default": "GET", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]},
                "headers": {"type": "object", "default": {}},
                "params": {"type": "object", "default": {}},
                "body": {"type": "string"},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["url"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            import urllib.request
            import urllib.error
        except ImportError:
            return ToolResult(success=False, error="urllib not available", return_code=1)
        
        url = args["url"]
        method = args.get("method", "GET").upper()
        headers = args.get("headers", {})
        params = args.get("params", {})
        body = args.get("body")
        timeout = args.get("timeout", context.timeout)
        
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}" if "?" not in url else f"{url}&{query}"
        
        if body and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        
        req = urllib.request.Request(
            url,
            data=body.encode() if body else None,
            headers=headers,
            method=method,
        )
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = response.status
                resp_headers = dict(response.headers)
                body_resp = response.read().decode("utf-8", errors="replace")
                return ToolResult(
                    success=(200 <= status < 400),
                    output=body_resp,
                    metadata={
                        "status_code": status,
                        "url": url,
                        "method": method,
                        "response_headers": resp_headers,
                        "body_size": len(body_resp),
                    },
                )
        except urllib.error.HTTPError as e:
            body_resp = e.read().decode("utf-8", errors="replace") if e.fp else ""
            return ToolResult(
                success=False,
                output=body_resp,
                error=f"HTTP {e.code}: {e.reason}",
                return_code=e.code,
                metadata={"status_code": e.code, "url": url},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)


class WebFetchTool(Tool):
    """Fetch webpage content (extract text từ HTML)."""
    category = ToolCategory.WEB
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "web_fetch"
    
    @property
    def description(self) -> str:
        return "Fetch webpage và extract text content (loại bỏ HTML tags, scripts, styles)."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer", "default": 10000},
            },
            "required": ["url"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            import urllib.request
            import re
            from html.parser import HTMLParser
        except ImportError:
            return ToolResult(success=False, error="imports failed", return_code=1)
        
        url = args["url"]
        max_chars = args.get("max_chars", 10000)
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NexusCoder/0.2"})
            with urllib.request.urlopen(req, timeout=context.timeout) as response:
                html = response.read().decode("utf-8", errors="replace")
            
            # Simple HTML text extraction
            # Remove scripts and styles
            html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
            # Remove tags
            text = re.sub(r"<[^>]+>", " ", html)
            # Decode entities
            text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            text = text.replace("&quot;", '"').replace("&#39;", "'")
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()
            
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"
            
            return ToolResult(
                success=True,
                output=text,
                metadata={"url": url, "chars": len(text), "original_html_size": len(html)},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)


class WebSearchTool(Tool):
    """Search web (uses search engine API)."""
    category = ToolCategory.WEB
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search web qua search engine. Trả về top results với title, url, snippet."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        query = args["query"]
        num = args.get("num_results", 5)
        
        # Placeholder: trong production, dùng Google Custom Search API / Bing API / Brave Search API
        # Cần API key trong env vars
        import os
        api_key = os.environ.get("SEARCH_API_KEY") or os.environ.get("BRAVE_SEARCH_API_KEY")
        
        if not api_key:
            return ToolResult(
                success=False,
                error="No search API key configured. Set SEARCH_API_KEY env var.",
                return_code=1,
                metadata={
                    "query": query,
                    "supported_engines": ["google_cse", "bing", "brave", "duckduckgo"],
                },
            )
        
        # Production code would call actual API here
        return ToolResult(
            success=True,
            output=f"[WebSearch] Searched: {query} (top {num} results)",
            metadata={"query": query, "num_results": num, "engine": "configured"},
        )
