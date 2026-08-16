"""
GraphQL Client Tool - Gửi GraphQL query/mutation tới một endpoint.
Author: Hieu Louis (2026)
Dùng stdlib urllib (fallback) hoặc requests nếu có. Hỗ trợ variables + headers.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class GraphQLClientTool(Tool):
    """Gửi GraphQL query/mutation tới một endpoint HTTP."""

    category = ToolCategory.WEB  # theo spec: category=WEB
    safety = ToolSafety.MODERATE  # network call nhưng query GraphQL
    requires_confirmation = False

    @property
    def name(self) -> str:
        return "graphql_client"

    @property
    def description(self) -> str:
        return (
            "Gửi GraphQL query hoặc mutation tới một endpoint. Hỗ trợ variables, "
            "headers (auth), timeout. Trả về JSON response."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "GraphQL endpoint URL (https://...)"},
                "query": {"type": "string", "description": "GraphQL query/mutation string"},
                "variables": {"type": "object", "description": "Biến cho GraphQL operation"},
                "operation_name": {"type": "string", "description": "Tên operation (nếu nhiều op trong query)"},
                "headers": {"type": "object", "description": "HTTP headers (Authorization, Content-Type, ...)"},
                "method": {"type": "string", "enum": ["POST", "GET"], "description": "HTTP method (default POST)"},
                "timeout": {"type": "integer", "description": "Request timeout (s)"},
            },
            "required": ["endpoint", "query"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("endpoint"):
            return "Missing required arg: endpoint"
        if not args.get("query"):
            return "Missing required arg: query"
        if not str(args["endpoint"]).startswith(("http://", "https://")):
            return "endpoint phải là URL http(s)://"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        endpoint: str = args["endpoint"]
        query: str = args["query"]
        variables = args.get("variables") or {}
        operation_name = args.get("operation_name")
        headers: Dict[str, str] = args.get("headers") or {}
        method = str(args.get("method") or "POST").upper()
        timeout = int(args.get("timeout") or context.timeout or 30)

        # Dry-run // dry-run
        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would send {method} GraphQL to {endpoint}",
                metadata={"dry_run": True, "endpoint": endpoint, "method": method, "operation_name": operation_name},
            )

        payload = {
            "query": query,
            "variables": variables,
        }
        if operation_name:
            payload["operationName"] = operation_name

        # Ưu tiên requests (nếu có), fallback urllib // prefer requests, fallback urllib
        try:
            import requests  # type: ignore
            use_requests = True
        except ImportError:
            use_requests = False

        try:
            if use_requests:
                # POST application/json (chuẩn GraphQL) // standard JSON POST
                if method == "POST":
                    resp = requests.post(  # type: ignore[union-attr]
                        endpoint,
                        json=payload,
                        headers=headers,
                        timeout=timeout,
                    )
                else:
                    # GET với query string // GET with querystring
                    import urllib.parse as up
                    qs = up.urlencode({"query": query, "variables": json.dumps(variables)})
                    resp = requests.get(  # type: ignore[union-attr]
                        f"{endpoint}?{qs}",
                        headers=headers,
                        timeout=timeout,
                    )
                status = resp.status_code
                try:
                    body = resp.json()
                except Exception:
                    body = {"raw": resp.text}
            else:
                # Fallback urllib // urllib fallback
                import urllib.request as ur
                import urllib.parse as up

                if method == "POST":
                    data = json.dumps(payload).encode("utf-8")
                    req_headers = dict(headers)
                    req_headers.setdefault("Content-Type", "application/json")
                    req = ur.Request(endpoint, data=data, headers=req_headers, method="POST")
                else:
                    qs = up.urlencode({"query": query, "variables": json.dumps(variables)})
                    req = ur.Request(f"{endpoint}?{qs}", headers=headers, method="GET")

                with ur.urlopen(req, timeout=timeout) as r:  # noqa: S310
                    status = r.status
                    raw = r.read().decode("utf-8", errors="replace")
                try:
                    body = json.loads(raw)
                except Exception:
                    body = {"raw": raw}

            # GraphQL trả về 200 ngay cả khi có errors // GraphQL may have errors
            has_errors = isinstance(body, dict) and bool(body.get("errors"))
            return ToolResult(
                success=(200 <= status < 300) and not has_errors,
                output=json.dumps(body, ensure_ascii=False, indent=2),
                error=(json.dumps(body.get("errors"), ensure_ascii=False, indent=2) if has_errors else None),
                return_code=status,
                metadata={
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": status,
                    "has_errors": has_errors,
                    "operation_name": operation_name,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
