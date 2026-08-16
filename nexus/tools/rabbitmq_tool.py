"""
RabbitMQ Tool - Publish/consume RabbitMQ messages qua pika.
Author: Hieu Louis (2026)
Operations: publish, consume, declare_queue, list_queues.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, unquote

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Write ops // write ops
WRITE_OPS = {"publish", "declare_queue"}


def _parse_amqp_url(url: str) -> Dict[str, Any]:
    """Parse amqp://user:pw@host:5672/vhost // parse AMQP URL."""
    p = urlparse(url)
    vhost = unquote(p.path[1:]) if p.path and len(p.path) > 1 else "/"
    return {
        "host": p.hostname or "localhost",
        "port": p.port or 5672,
        "username": unquote(p.username) if p.username else "guest",
        "password": unquote(p.password) if p.password else "guest",
        "virtual_host": vhost,
    }


class RabbitMQTool(Tool):
    """Publish/consume RabbitMQ: publish, consume, declare_queue, list_queues."""

    category = ToolCategory.DATABASE
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "rabbitmq"

    @property
    def description(self) -> str:
        return (
            "Publish/consume RabbitMQ messages qua pika: publish, consume (batch), "
            "declare_queue (durable), list_queues (via management hoặc AMQP)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "connection_string": {"type": "string", "description": "AMQP URL (amqp://user:pw@host:5672/vhost)"},
                "operation": {
                    "type": "string",
                    "enum": ["publish", "consume", "declare_queue", "list_queues"],
                    "description": "RabbitMQ operation",
                },
                "queue": {"type": "string", "description": "Queue name"},
                "exchange": {"type": "string", "description": "Exchange (default '')"},
                "routing_key": {"type": "string", "description": "Routing key (default = queue name)"},
                "message": {"type": "string", "description": "Message body (publish)"},
                "durable": {"type": "boolean", "description": "Queue durable (default true)"},
                "max_messages": {"type": "integer", "description": "Giới hạn số message consume (default 10)"},
                "timeout_ms": {"type": "integer", "description": "Consume timeout ms (default 5000)"},
            },
            "required": ["connection_string", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("connection_string"):
            return "Missing required arg: connection_string"
        op = args.get("operation")
        if not op:
            return "Missing required arg: operation"
        if op in {"publish", "consume", "declare_queue"} and not args.get("queue"):
            return f"Operation '{op}' requires 'queue' arg"
        if op == "publish" and args.get("message") is None:
            return "Operation 'publish' requires 'message' arg"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cs: str = args["connection_string"]
        op: str = args["operation"]

        # Lazy import pika // lazy import
        try:
            import pika  # type: ignore
        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"pika not installed: {e}. Cài: pip install pika",
                return_code=127,
            )

        # Dry-run cho write ops // dry-run
        if context.dry_run and op in WRITE_OPS:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would run {op} on queue={args.get('queue')!r}",
                metadata={"dry_run": True, "operation": op, "queue": args.get("queue")},
            )

        cfg = _parse_amqp_url(cs)
        try:
            creds = pika.PlainCredentials(cfg["username"], cfg["password"])
            params = pika.ConnectionParameters(
                host=cfg["host"], port=cfg["port"], virtual_host=cfg["virtual_host"],
                credentials=creds,
                heartbeat=30,
                blocked_connection_timeout=context.timeout or 30,
            )
            conn = pika.BlockingConnection(params)
            channel = conn.channel()

            if op == "declare_queue":
                durable = bool(args.get("durable", True))
                channel.queue_declare(queue=args["queue"], durable=durable)
                return ToolResult(
                    success=True,
                    output=f"Queue declared: {args['queue']} (durable={durable})",
                    metadata={"queue": args["queue"], "durable": durable},
                )

            if op == "publish":
                # Đảm bảo queue tồn tại // ensure queue exists
                channel.queue_declare(queue=args["queue"], durable=bool(args.get("durable", True)))
                routing_key = args.get("routing_key") or args["queue"]
                body = args["message"]
                if not isinstance(body, (bytes, bytearray)):
                    body = body if isinstance(body, str) else json.dumps(body)
                channel.basic_publish(
                    exchange=args.get("exchange") or "",
                    routing_key=routing_key,
                    body=body,
                    properties=pika.BasicProperties(delivery_mode=2),  # persistent
                )
                return ToolResult(
                    success=True,
                    output=f"Published to {args['queue']} (routing_key={routing_key})",
                    metadata={"queue": args["queue"], "routing_key": routing_key, "bytes": len(body if isinstance(body, (bytes, str)) else str(body))},
                )

            if op == "consume":
                channel.queue_declare(queue=args["queue"], durable=True)
                max_msgs = int(args.get("max_messages") or 10)
                timeout_ms = int(args.get("timeout_ms", 5000))
                messages: List[Dict[str, Any]] = []

                for method_frame, properties, body in channel.consume(
                    args["queue"], inactivity_timeout=timeout_ms / 1000.0, auto_ack=False
                ):
                    if method_frame is None:
                        break  # timeout, không có message
                    messages.append({
                        "delivery_tag": method_frame.delivery_tag,
                        "routing_key": method_frame.routing_key,
                        "headers": dict(properties.headers or {}),
                        "body": body.decode("utf-8", errors="replace") if isinstance(body, (bytes, bytearray)) else str(body),
                    })
                    channel.basic_ack(method_frame.delivery_tag)
                    if len(messages) >= max_msgs:
                        break
                # Huỷ consumer // cancel consumer
                try:
                    channel.cancel()
                except Exception:
                    pass
                payload = json.dumps(messages, ensure_ascii=False, indent=2)
                return ToolResult(
                    success=True,
                    output=payload,
                    metadata={"consumed": len(messages), "queue": args["queue"]},
                )

            if op == "list_queues":
                # AMQP không có list_queues native; dùng passive declare để probe
                # Sẽ list qua HTTP management API nếu có, không thì thông báo
                return ToolResult(
                    success=False,
                    error="list_queues yêu cầu RabbitMQ Management HTTP API. Dùng HTTP tool với /api/queues.",
                    return_code=2,
                )

            return ToolResult(success=False, error=f"Unknown operation: {op}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
        finally:
            try:
                if conn and conn.is_open:
                    conn.close()
            except Exception:
                pass
