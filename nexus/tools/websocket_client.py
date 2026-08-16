"""
WebSocket Client Tool - Kết nối & giao tiếp với WebSocket server.
Author: Hieu Louis (2026)

Hỗ trợ: send / receive / ping. Lazy import `websocket-client` (đồng bộ)
hoặc fallback sang stdlib `websockets` (async, chạy trong asyncio.run).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class WebSocketClientTool(Tool):
    """WebSocket client: connect, send, receive, ping.

    Ưu tiên `websocket-client` (sync). Nếu không có, fallback sang
    stdlib `websockets` chạy trong asyncio event loop.
    """

    category = ToolCategory.WEB
    safety = ToolSafety.MODERATE
    requires_confirmation = True
    timeout = 30

    @property
    def name(self) -> str:
        return "websocket_client"

    @property
    def description(self) -> str:
        return (
            "Kết nối tới WebSocket server (ws/wss) và thực hiện action: "
            "send (gửi message), receive (đợi 1 message), ping (health check). "
            "Hỗ trợ subprotocol và custom headers."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "WebSocket URL ws:// hoặc wss://",
                },
                "action": {
                    "type": "string",
                    "enum": ["send", "receive", "ping"],
                    "default": "send",
                    "description": "Hành động cần thực hiện",
                },
                "message": {
                    "type": "string",
                    "description": "Message cần gửi (cho action=send). Nếu là JSON sẽ tự serialize.",
                },
                "subprotocols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Danh sách subprotocol thương lượng",
                },
                "headers": {
                    "type": "object",
                    "description": "Custom HTTP headers cho handshake",
                },
                "timeout": {
                    "type": "integer",
                    "default": 30,
                    "description": "Timeout cho toàn bộ thao tác (giây)",
                },
            },
            "required": ["url", "action"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        url = str(args.get("url", ""))
        if not url:
            return "Missing required arg: url"
        if not url.startswith(("ws://", "wss://")):
            return "url phải bắt đầu bằng ws:// hoặc wss://"
        action = args.get("action")
        if action not in ("send", "receive", "ping"):
            return f"action phải là send/receive/ping, nhận được '{action}'"
        if action == "send" and not args.get("message"):
            return "action=send yêu cầu arg 'message'"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        url: str = args["url"]
        action: str = args["action"]
        message: Any = args.get("message")
        subprotocols: List[str] = args.get("subprotocols") or []
        headers: Dict[str, str] = args.get("headers") or {}
        timeout = int(args.get("timeout") or context.timeout or 30)

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would {action} on WebSocket {url}",
                metadata={
                    "dry_run": True,
                    "url": url,
                    "action": action,
                    "has_message": bool(message),
                },
            )

        # Serialize message // serialize payload
        payload: Any = None
        if action == "send":
            if isinstance(message, (dict, list)):
                payload = json.dumps(message, ensure_ascii=False)
            else:
                payload = str(message)

        # Ưu tiên websocket-client (sync) // prefer sync websocket-client
        try:
            import websocket  # type: ignore
            return self._run_sync(
                websocket, url, action, payload, subprotocols, headers, timeout
            )
        except ImportError:
            pass

        # Fallback stdlib websockets (async) // stdlib fallback
        try:
            import websockets  # type: ignore
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "Không có thư viện WebSocket. Cài: "
                    "pip install websocket-client (hoặc websockets)"
                ),
                return_code=1,
            )

        try:
            result = asyncio.run(
                self._run_async(
                    websockets, url, action, payload, subprotocols, headers, timeout
                )
            )
            return result
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=str(e), return_code=1)

    # ---- websocket-client (sync) ----
    def _run_sync(
        self,
        ws_module: Any,
        url: str,
        action: str,
        payload: Any,
        subprotocols: List[str],
        headers: Dict[str, str],
        timeout: int,
    ) -> ToolResult:
        try:
            ws = ws_module.create_connection(
                url,
                timeout=timeout,
                subprotocols=subprotocols or None,
                header=[f"{k}: {v}" for k, v in headers.items()] or None,
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                success=False,
                error=f"WebSocket connect failed: {e}",
                return_code=1,
            )
        try:
            if action == "ping":
                ws.ping()
                pong = ws.pong(ws_module.create_ping_payload() if hasattr(ws_module, "create_ping_payload") else b"")
                return ToolResult(
                    success=True,
                    output=f"Ping OK tới {url}",
                    metadata={"action": "ping", "url": url},
                )
            elif action == "send":
                ws.send(payload)
                return ToolResult(
                    success=True,
                    output=f"Sent: {payload}",
                    metadata={"action": "send", "url": url, "bytes_sent": len(str(payload))},
                )
            else:  # receive
                received = ws.recv()
                return ToolResult(
                    success=True,
                    output=str(received),
                    metadata={
                        "action": "receive",
                        "url": url,
                        "bytes_received": len(str(received)),
                    },
                )
        finally:
            try:
                ws.close()
            except Exception:
                pass

    # ---- websockets (async stdlib) ----
    async def _run_async(
        self,
        ws_module: Any,
        url: str,
        action: str,
        payload: Any,
        subprotocols: List[str],
        headers: Dict[str, str],
        timeout: int,
    ) -> ToolResult:
        extra_headers = ws_module.Headers(**headers) if headers and hasattr(ws_module, "Headers") else headers or None
        try:
            async with ws_module.connect(
                url,
                subprotocols=subprotocols or None,
                additional_headers=extra_headers,
                open_timeout=timeout,
            ) as ws:
                if action == "ping":
                    pong_waiter = await ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=timeout)
                    return ToolResult(
                        success=True,
                        output=f"Ping/pong OK tới {url}",
                        metadata={"action": "ping", "url": url},
                    )
                elif action == "send":
                    await ws.send(payload)
                    return ToolResult(
                        success=True,
                        output=f"Sent: {payload}",
                        metadata={"action": "send", "url": url},
                    )
                else:  # receive
                    received = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    return ToolResult(
                        success=True,
                        output=str(received),
                        metadata={"action": "receive", "url": url},
                    )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"WebSocket {action} timed out sau {timeout}s",
                return_code=124,
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=str(e), return_code=1)
