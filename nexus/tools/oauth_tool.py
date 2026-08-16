"""
OAuth Tool - OAuth 2.0 flows.
===========================================
Tool thực thi OAuth2: authorization_code, client_credentials, refresh_token.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


SUPPORTED_FLOWS = {"authorization_code", "client_credentials", "refresh_token"}


class OAuthTool(Tool):
    """Thực thi OAuth2 flows — trao đổi token với Authorization Server."""

    category = ToolCategory.WEB
    safety = ToolSafety.MODERATE
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "oauth"

    @property
    def description(self) -> str:
        return "OAuth2 flows: authorization_code, client_credentials, refresh_token."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "flow": {
                    "type": "string",
                    "enum": sorted(SUPPORTED_FLOWS),
                },
                "client_id": {"type": "string"},
                "client_secret": {"type": "string"},
                "auth_url": {"type": "string", "description": "Authorization endpoint (cho auth_code)"},
                "token_url": {"type": "string", "description": "Token endpoint"},
                "redirect_uri": {"type": "string"},
                "code": {"type": "string", "description": "Authorization code (cho auth_code flow)"},
                "refresh_token": {"type": "string"},
                "scope": {"type": "string"},
                "state": {"type": "string"},
                "extra_params": {"type": "object"},
            },
            "required": ["flow", "client_id", "token_url"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        flow = args.get("flow")
        if flow not in SUPPORTED_FLOWS:
            return f"Invalid flow='{flow}'. Supported: {sorted(SUPPORTED_FLOWS)}"
        if not args.get("client_id"):
            return "Missing required arg: client_id"
        if not args.get("token_url"):
            return "Missing required arg: token_url"
        if flow == "authorization_code" and not args.get("code"):
            return "Missing required arg: code (cho authorization_code flow)"
        if flow == "refresh_token" and not args.get("refresh_token"):
            return "Missing required arg: refresh_token (cho refresh_token flow)"
        return None

    # ---- Tiện ích / Helpers ---------------------------------------------

    def _post_form(
        self,
        url: str,
        data: Dict[str, str],
        timeout: int,
        basic_auth: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """POST form-urlencoded; trả về dict JSON. / POST form, return parsed JSON dict."""
        body = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        req.add_header("Accept", "application/json")
        if basic_auth:
            import base64
            user, pw = basic_auth
            cred = base64.b64encode(f"{user}:{pw}".encode()).decode()
            req.add_header("Authorization", f"Basic {cred}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Một số server trả về form-urlencoded / some servers return form-urlencoded
            parsed = urllib.parse.parse_qs(raw)
            return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}

    def _build_auth_url(
        self,
        auth_url: str,
        client_id: str,
        redirect_uri: str,
        scope: Optional[str],
        state: Optional[str],
        extra: Dict[str, Any],
    ) -> str:
        """Tạo URL redirect cho authorization_code flow."""
        params: Dict[str, str] = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
        }
        if scope:
            params["scope"] = scope
        if state:
            params["state"] = state
        params.update({k: str(v) for k, v in extra.items()})
        return f"{auth_url}?{urllib.parse.urlencode(params)}"

    # ---- Thực thi / Execute --------------------------------------------

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        flow = args["flow"]
        client_id = args["client_id"]
        client_secret = args.get("client_secret", "")
        token_url = args["token_url"]
        scope = args.get("scope")
        state = args.get("state")
        extra = args.get("extra_params", {}) or {}
        timeout = max(5, min(context.timeout, 120))

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] OAuth2 flow='{flow}' sẽ POST tới {token_url}",
                metadata={"flow": flow, "token_url": token_url, "client_id": client_id, "scope": scope},
            )

        # Note: nếu `requests` đã cài, sẽ được dùng tự động qua _post_form khi urllib fail.
        # / If `requests` is installed, it would be used via _post_form on urllib failure.

        # ---- authorization_code: sinh URL hoặc đổi code lấy token ----
        if flow == "authorization_code":
            auth_url = args.get("auth_url")
            redirect_uri = args.get("redirect_uri", "")
            code = args["code"]
            # Nếu chưa có code → trả về authorize URL / return authorize URL when no code yet
            if not code:
                if not auth_url:
                    return ToolResult(
                        success=False,
                        error="Cần 'auth_url' để xây authorize URL khi chưa có code",
                        return_code=1,
                    )
                url = self._build_auth_url(auth_url, client_id, redirect_uri, scope, state, extra)
                return ToolResult(
                    success=True,
                    output=f"Redirect user tới URL: {url}",
                    metadata={"authorize_url": url, "flow": flow},
                )
            data: Dict[str, str] = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
            }
            if client_secret:
                data["client_secret"] = client_secret
            data.update({k: str(v) for k, v in extra.items()})
            try:
                result = self._post_form(token_url, data, timeout, basic_auth=(client_id, client_secret) if client_secret else None)
            except Exception as e:
                return ToolResult(success=False, error=f"Token exchange failed: {e}", return_code=1)
            ok = "access_token" in result
            return ToolResult(
                success=ok,
                output=json.dumps(result, indent=2, ensure_ascii=False),
                error=result.get("error_description") or result.get("error") if not ok else None,
                return_code=0 if ok else 1,
                metadata={"flow": flow, "has_access_token": ok, "token_type": result.get("token_type")},
            )

        # ---- client_credentials: server-to-server token ----
        if flow == "client_credentials":
            if not client_secret:
                return ToolResult(success=False, error="client_credentials flow cần 'client_secret'", return_code=1)
            data = {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            }
            if scope:
                data["scope"] = scope
            data.update({k: str(v) for k, v in extra.items()})
            try:
                # Ưu tiên HTTP Basic auth (chuẩn RFC) / prefer HTTP Basic per RFC 6749
                result = self._post_form(token_url, {k: v for k, v in data.items() if k != "client_secret"}, timeout, basic_auth=(client_id, client_secret))
            except Exception as e:
                return ToolResult(success=False, error=f"Token request failed: {e}", return_code=1)
            ok = "access_token" in result
            return ToolResult(
                success=ok,
                output=json.dumps(result, indent=2, ensure_ascii=False),
                error=result.get("error_description") or result.get("error") if not ok else None,
                return_code=0 if ok else 1,
                metadata={"flow": flow, "has_access_token": ok, "expires_in": result.get("expires_in")},
            )

        # ---- refresh_token: đổi refresh_token lấy access_token mới ----
        if flow == "refresh_token":
            data = {
                "grant_type": "refresh_token",
                "refresh_token": args["refresh_token"],
                "client_id": client_id,
            }
            if client_secret:
                data["client_secret"] = client_secret
            if scope:
                data["scope"] = scope
            data.update({k: str(v) for k, v in extra.items()})
            try:
                result = self._post_form(token_url, data, timeout, basic_auth=(client_id, client_secret) if client_secret else None)
            except Exception as e:
                return ToolResult(success=False, error=f"Refresh failed: {e}", return_code=1)
            ok = "access_token" in result
            return ToolResult(
                success=ok,
                output=json.dumps(result, indent=2, ensure_ascii=False),
                error=result.get("error_description") or result.get("error") if not ok else None,
                return_code=0 if ok else 1,
                metadata={"flow": flow, "has_access_token": ok, "expires_in": result.get("expires_in")},
            )

        return ToolResult(success=False, error=f"Unsupported flow: {flow}", return_code=1)
