"""
gRPC Client Tool - Gửi gRPC unary request tới một gRPC server.
Author: Hieu Louis (2026)

Vì gRPC yêu cầu compiled protobuf stubs nên tool này dùng grpcio's
generic `grpc.channel.unary_unary` (channel+method path, không cần stub).
Lazy import `grpc`.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class GRPCClientTool(Tool):
    """Gửi gRPC unary-unary request qua reflection-free generic call.

    Cách gọi: truyền host, port, service (full method path vd
    `/helloworld.Greeter/SayHello`), request_json (JSON serialised → bytes).
    TLS optional.
    """

    category = ToolCategory.WEB
    safety = ToolSafety.MODERATE
    requires_confirmation = True
    timeout = 30

    @property
    def name(self) -> str:
        return "grpc_client"

    @property
    def description(self) -> str:
        return (
            "Gửi gRPC unary request. Sử dụng generic channel (không cần "
            "protobuf stubs compiled). Truyền method path dạng "
            "/package.Service/Method và request_json. Hỗ trợ TLS."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "gRPC server host"},
                "port": {"type": "integer", "default": 50051, "description": "gRPC server port"},
                "service": {
                    "type": "string",
                    "description": "Full method path vd /helloworld.Greeter/SayHello",
                },
                "method": {
                    "type": "string",
                    "description": "Method name (nếu service đã là full path, để trống)",
                },
                "request_json": {
                    "type": "string",
                    "description": "Request payload dạng JSON string",
                },
                "use_tls": {"type": "boolean", "default": False, "description": "Dùng TLS channel"},
                "metadata": {
                    "type": "object",
                    "description": "gRPC metadata (headers) key→value",
                },
                "timeout": {"type": "integer", "default": 30, "description": "Timeout (giây)"},
            },
            "required": ["host", "service"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("host"):
            return "Missing required arg: host"
        if not args.get("service"):
            return "Missing required arg: service"
        port = args.get("port", 50051)
        if not isinstance(port, int) or not (1 <= port <= 65535):
            return "port phải là integer trong [1, 65535]"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            import grpc  # type: ignore
        except ImportError:
            return ToolResult(
                success=False,
                error="grpc không khả dụng. Cài: pip install grpcio",
                return_code=1,
            )

        host: str = args["host"]
        port: int = int(args.get("port", 50051))
        service: str = args["service"]
        method: str = args.get("method", "")
        request_json: str = args.get("request_json", "")
        use_tls: bool = bool(args.get("use_tls", False))
        metadata: Dict[str, str] = args.get("metadata") or {}
        timeout = int(args.get("timeout") or context.timeout or 30)

        # Build full method path // build full method path
        full_method = service if service.startswith("/") else f"/{service}"
        if method and not full_method.rstrip("/").endswith(method):
            full_method = f"{full_method.rstrip('/')}/{method.lstrip('/')}"

        # Serialize payload // serialize JSON → bytes
        try:
            payload_bytes = (
                json.dumps(json.loads(request_json)).encode("utf-8")
                if request_json.strip()
                else b""
            )
        except json.JSONDecodeError as e:
            return ToolResult(
                success=False,
                error=f"request_json không hợp lệ: {e}",
                return_code=1,
            )

        if context.dry_run:
            return ToolResult(
                success=True,
                output=(
                    f"[dry-run] Would call gRPC {host}:{port}{full_method} "
                    f"(tls={use_tls}, payload={len(payload_bytes)} bytes)"
                ),
                metadata={
                    "dry_run": True,
                    "host": host,
                    "port": port,
                    "method": full_method,
                    "use_tls": use_tls,
                },
            )

        target = f"{host}:{port}"
        try:
            if use_tls:
                channel = grpc.secure_channel(target, grpc.ssl_channel_credentials())
            else:
                channel = grpc.insecure_channel(target)
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                success=False,
                error=f"Không tạo được channel: {e}",
                return_code=1,
            )

        try:
            grpc.channel_ready_future(channel).result(timeout=timeout)
        except grpc.FutureTimeoutError:
            return ToolResult(
                success=False,
                error=f"gRPC channel timeout (server không sẵn sàng sau {timeout}s)",
                return_code=124,
            )

        try:
            # Generic unary_unary call // generic unary-unary call
            response = channel.unary_unary(
                full_method,
                request_serializer=lambda payload: payload,
                response_deserializer=lambda data: data,
            )(
                payload_bytes,
                timeout=timeout,
                metadata=tuple(metadata.items()) if metadata else None,
            )

            # Best-effort decode response as JSON hoặc raw text
            try:
                decoded = response.decode("utf-8")
                try:
                    parsed = json.loads(decoded)
                    output = json.dumps(parsed, ensure_ascii=False, indent=2)
                    is_json = True
                except json.JSONDecodeError:
                    output = decoded
                    is_json = False
            except UnicodeDecodeError:
                output = f"<binary {len(response)} bytes>"
                is_json = False

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "host": host,
                    "port": port,
                    "method": full_method,
                    "use_tls": use_tls,
                    "is_json": is_json,
                    "response_bytes": len(response),
                },
            )
        except grpc.RpcError as e:
            code = e.code() if hasattr(e, "code") else None
            return ToolResult(
                success=False,
                error=f"gRPC error: {code} — {e.details() if hasattr(e, 'details') else str(e)}",
                return_code=int(code.value[0]) if code and hasattr(code.value, "__getitem__") else 2,
                metadata={
                    "host": host,
                    "port": port,
                    "method": full_method,
                    "grpc_code": str(code) if code else None,
                },
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=str(e), return_code=1)
        finally:
            try:
                channel.close()
            except Exception:
                pass
