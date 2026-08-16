"""
Web Auth Tool - Hỗ trợ xác thực HTTP: Basic, Bearer, OAuth2 client_credentials.
Author: Hieu Louis (2026)

Lazy import `requests`. Trả về response status + JSON body + token metadata.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class WebAuthTool(Tool):
    """Web auth helpers: Basic / Bearer / OAuth2 client_credentials."""

    category = ToolCategory.WEB
    safety = ToolSafety.MODERATE
    timeout = 30

    SUPPORTED_AUTH_TYPES = ("basic", "bearer", "oauth2_client")

    @property
    def name(self) -> str:
        return "web_auth"

    @property
    def description(self) -> str:
        return (
            "Hỗ trợ HTTP authentication: Basic, Bearer token, hoặc "
            "OAuth2 client_credentials grant. Thực hiện request authenticated "
            "tới URL và trả về response."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL endpoint cần gọi"},
                "auth_type": {
                    "type": "string",
                    "enum": list(self.SUPPORTED_AUTH_TYPES),
                    "description": "Loại auth: basic / bearer / oauth2_client",
                },
                "method": {"type": "string", "default": "GET", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                "username": {"type": "string", "description": "Username (cho basic)"},
                "password": {"type": "string", "description": "Password (cho basic)"},
                "token": {"type": "string", "description": "Bearer token (cho bearer)"},
                "client_id": {"type": "string", "description": "OAuth2 client_id"},
                "client_secret": {"type": "string", "description": "OAuth2 client_secret"},
                "token_url": {"type": "string", "description": "OAuth2 token endpoint URL"},
                "scope": {"type": "string", "description": "OAuth2 scope (optional)"},
                "body": {"type": "object", "description": "JSON body cho request"},
                "headers": {"type": "object", "description": "Custom headers (sẽ merge)"},
                "timeout": {"type": "integer", "default": 30, "description": "Timeout (giây)"},
            },
            "required": ["url", "auth_type"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("url"):
            return "Missing required arg: url"
        if not str(args["url"]).startswith(("http://", "https://")):
            return "url phải là http(s)://"
        auth_type = str(args.get("auth_type", "")).lower()
        if auth_type not in self.SUPPORTED_AUTH_TYPES:
            return f"auth_type phải là {self.SUPPORTED_AUTH_TYPES}"
        if auth_type == "basic":
            if not args.get("username") or not args.get("password"):
                return "auth_type=basic cần username và password"
        elif auth_type == "bearer":
            if not args.get("token"):
                return "auth_type=bearer cần token"
        elif auth_type == "oauth2_client":
            if not args.get("client_id") or not args.get("client_secret") or not args.get("token_url"):
                return "auth_type=oauth2_client cần client_id, client_secret, token_url"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        url: str = args["url"]
        auth_type: str = str(args["auth_type"]).lower()
        method: str = str(args.get("method", "GET")).upper()
        body: Optional[Dict[str, Any]] = args.get("body")
        extra_headers: Dict[str, str] = args.get("headers") or {}
        timeout = int(args.get("timeout") or context.timeout or 30)

        # Dry-run
        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would {method} {url} (auth={auth_type})",
                metadata={
                    "dry_run": True,
                    "url": url,
                    "auth_type": auth_type,
                    "method": method,
                },
            )

        try:
            import requests  # type: ignore
        except ImportError:
            return ToolResult(
                success=False,
                error="Cần requests. Cài: pip install requests",
                return_code=1,
            )

        # Build auth headers / session
        auth_meta: Dict[str, Any] = {"auth_type": auth_type}
        try:
            if auth_type == "basic":
                username: str = args["username"]
                password: str = args["password"]
                token_b64 = base64.b64encode(
                    f"{username}:{password}".encode("utf-8")
                ).decode("ascii")
                headers = {"Authorization": f"Basic {token_b64}", **extra_headers}
                auth_meta["username"] = username

            elif auth_type == "bearer":
                token: str = args["token"]
                headers = {"Authorization": f"Bearer {token}", **extra_headers}
                auth_meta["token_redacted"] = token[:8] + "..." if len(token) > 12 else "***"

            else:  # oauth2_client
                token_data = self._fetch_oauth2_token(
                    requests,
                    str(args["token_url"]),
                    str(args["client_id"]),
                    str(args["client_secret"]),
                    args.get("scope"),
                    timeout,
                )
                access_token = token_data["access_token"]
                headers = {"Authorization": f"Bearer {access_token}", **extra_headers}
                auth_meta.update({
                    "token_url": args["token_url"],
                    "expires_in": token_data.get("expires_in"),
                    "token_type": token_data.get("token_type", "Bearer"),
                    "scope_granted": token_data.get("scope"),
                })

            # Make request
            req_kwargs: Dict[str, Any] = {"headers": headers, "timeout": timeout}
            if body is not None and method in ("POST", "PUT", "PATCH"):
                req_kwargs["json"] = body

            resp = requests.request(method, url, **req_kwargs)

            # Parse response body
            try:
                body_out = resp.json()
                body_text = json.dumps(body_out, ensure_ascii=False, indent=2)
                is_json = True
            except ValueError:
                body_out = None
                body_text = resp.text[:5000]
                is_json = False

            return ToolResult(
                success=(200 <= resp.status_code < 300),
                output=body_text,
                error=None if resp.status_code < 400 else f"HTTP {resp.status_code}",
                return_code=resp.status_code,
                metadata={
                    **auth_meta,
                    "url": url,
                    "method": method,
                    "status_code": resp.status_code,
                    "is_json": is_json,
                    "response_headers": dict(resp.headers),
                    "body_size": len(resp.content),
                },
            )
        except requests.exceptions.Timeout:
            return ToolResult(
                success=False,
                error=f"Request timeout sau {timeout}s",
                return_code=124,
                metadata={"url": url, "auth_type": auth_type},
            )
        except requests.exceptions.RequestException as e:
            return ToolResult(
                success=False,
                error=f"Request failed: {e}",
                return_code=1,
                metadata={"url": url, "auth_type": auth_type},
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                success=False,
                error=f"Auth/request error: {e}",
                return_code=1,
                metadata={"url": url, "auth_type": auth_type},
            )

    @staticmethod
    def _fetch_oauth2_token(
        requests_module: Any,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: Optional[str],
        timeout: int,
    ) -> Dict[str, Any]:
        """Lấy OAuth2 access token qua client_credentials grant."""
        data = {"grant_type": "client_credentials"}
        if scope:
            data["scope"] = scope

        # RFC 6749: client credentials có thể gửi qua Basic auth hoặc form body
        # Dùng form body cho compatibility rộng hơn
        data["client_id"] = client_id
        data["client_secret"] = client_secret

        resp = requests_module.post(
            token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Accept": "application/json"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"OAuth2 token endpoint returned HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            token_data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"OAuth2 token endpoint trả về non-JSON: {e}") from e
        if "access_token" not in token_data:
            raise RuntimeError(f"OAuth2 response không có access_token: {token_data}")
        # Add a fetched_at timestamp để caller biết khi nào token lấy
        token_data["fetched_at"] = int(time.time())
        return token_data
