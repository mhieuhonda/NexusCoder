"""
Kafka Tool - Produce / consume Kafka messages qua kafka-python.
Author: Hieu Louis (2026)
Operations: produce, consume, list_topics, describe_topic.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Produce = write (cần confirmation) // write ops
WRITE_OPS = {"produce"}


class KafkaTool(Tool):
    """Produce/consume Kafka messages: produce, consume, list_topics, describe_topic."""

    category = ToolCategory.DATABASE
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "kafka"

    @property
    def description(self) -> str:
        return (
            "Produce/consume Kafka messages qua kafka-python: produce (sync/async), "
            "consume (poll batch), list_topics, describe_topic."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bootstrap_servers": {
                    "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                    "description": "Kafka brokers, vd: localhost:9092",
                },
                "operation": {
                    "type": "string",
                    "enum": ["produce", "consume", "list_topics", "describe_topic"],
                    "description": "Kafka operation",
                },
                "topic": {"type": "string", "description": "Topic name"},
                "messages": {
                    "oneOf": [{"type": "string"}, {"type": "array", "items": {}}],
                    "description": "Message(s) cần produce (string hoặc list)",
                },
                "key": {"type": "string", "description": "Message key (produce)"},
                "group_id": {"type": "string", "description": "Consumer group (consume)"},
                "max_messages": {"type": "integer", "description": "Giới hạn số message consume (default 100)"},
                "timeout_ms": {"type": "integer", "description": "Poll timeout ms (default 5000)"},
                "sasl_username": {"type": "string", "description": "SASL username (optional)"},
                "sasl_password": {"type": "string", "description": "SASL password (optional)"},
            },
            "required": ["bootstrap_servers", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("bootstrap_servers"):
            return "Missing required arg: bootstrap_servers"
        op = args.get("operation")
        if not op:
            return "Missing required arg: operation"
        if op in {"produce", "consume", "describe_topic"} and not args.get("topic"):
            return f"Operation '{op}' requires 'topic' arg"
        if op == "produce" and args.get("messages") is None:
            return "Operation 'produce' requires 'messages' arg"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        bs = args["bootstrap_servers"]
        op: str = args["operation"]
        sasl_u = args.get("sasl_username")
        sasl_p = args.get("sasl_password")

        # Lazy import kafka-python // lazy import
        try:
            from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient  # type: ignore
            from kafka.admin import NewTopic  # type: ignore  # noqa: F401
        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"kafka-python not installed: {e}. Cài: pip install kafka-python",
                return_code=127,
            )

        # Dry-run cho produce // dry-run
        if context.dry_run and op in WRITE_OPS:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would produce to topic={args.get('topic')!r}",
                metadata={"dry_run": True, "operation": op, "topic": args.get("topic")},
            )

        common_kwargs: Dict[str, Any] = {
            "bootstrap_servers": bs if isinstance(bs, list) else [bs],
            "request_timeout_ms": (context.timeout * 1000) if context.timeout else 30000,
        }
        if sasl_u and sasl_p:
            common_kwargs.update({
                "security_protocol": "SASL_PLAINTEXT",
                "sasl_mechanism": "PLAIN",
                "sasl_plain_username": sasl_u,
                "sasl_plain_password": sasl_p,
            })

        try:
            if op == "list_topics":
                admin = KafkaAdminClient(**common_kwargs)
                topics = sorted(admin.list_topics())
                admin.close()
                payload = json.dumps(topics, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"count": len(topics)})

            if op == "describe_topic":
                admin = KafkaAdminClient(**common_kwargs)
                # Lấy partitions // get partitions via consumer
                consumer = KafkaConsumer(args["topic"], **{k: v for k, v in common_kwargs.items() if k != "request_timeout_ms"}, request_timeout_ms=common_kwargs["request_timeout_ms"])
                parts = sorted(consumer.partitions_for_topic(args["topic"]) or [])
                consumer.close()
                admin.close()
                info = {"topic": args["topic"], "partitions": list(parts), "partition_count": len(parts)}
                return ToolResult(success=True, output=json.dumps(info, ensure_ascii=False, indent=2), metadata=info)

            if op == "produce":
                producer = KafkaProducer(
                    bootstrap_servers=common_kwargs["bootstrap_servers"],
                    value_serializer=lambda v: v.encode("utf-8") if isinstance(v, str) else json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
                    request_timeout_ms=common_kwargs["request_timeout_ms"],
                )
                msgs = args["messages"]
                if not isinstance(msgs, list):
                    msgs = [msgs]
                futures = []
                for m in msgs:
                    futures.append(producer.send(args["topic"], key=args.get("key"), value=m))
                # flush chờ gửi xong // wait for all
                producer.flush()
                for f in futures:
                    f.get(timeout=context.timeout or 30)  # raises nếu lỗi
                producer.close()
                return ToolResult(
                    success=True,
                    output=f"Produced {len(msgs)} message(s) to {args['topic']}",
                    metadata={"topic": args["topic"], "count": len(msgs)},
                )

            if op == "consume":
                consumer = KafkaConsumer(
                    args["topic"],
                    group_id=args.get("group_id"),
                    bootstrap_servers=common_kwargs["bootstrap_servers"],
                    auto_offset_reset="earliest",
                    enable_auto_commit=False,
                    consumer_timeout_ms=int(args.get("timeout_ms", 5000)),
                    value_deserializer=lambda v: v.decode("utf-8", errors="replace"),
                )
                max_msgs = int(args.get("max_messages") or 100)
                records: List[Dict[str, Any]] = []
                for msg in consumer:
                    records.append({
                        "topic": msg.topic,
                        "partition": msg.partition,
                        "offset": msg.offset,
                        "key": msg.key.decode("utf-8", errors="replace") if msg.key else None,
                        "value": msg.value,
                    })
                    if len(records) >= max_msgs:
                        break
                consumer.close()
                payload = json.dumps(records, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"consumed": len(records), "topic": args["topic"]})

            return ToolResult(success=False, error=f"Unknown operation: {op}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
