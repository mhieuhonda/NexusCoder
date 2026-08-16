"""
Redis Tool - Quản lý Redis qua redis-py.
Author: Hieu Louis (2026)
Operations: get, set, hget, hset, lpush, rpop, publish, info, keys.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Operations ghi/xoá // write/delete ops (cần confirmation)
WRITE_OPS = {"set", "hset", "lpush", "rpush", "publish", "delete", "expire"}
# Operations đọc // read-only ops
READ_OPS = {"get", "hget", "rpop", "lpop", "info", "keys", "ttl", "type", "exists"}


class RedisTool(Tool):
    """Quản lý Redis: get, set, hget, hset, lpush, rpop, publish, info, keys."""

    category = ToolCategory.DATABASE
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "redis"

    @property
    def description(self) -> str:
        return (
            "Quản lý Redis qua redis-py: get/set, hget/hset, lpush/rpop, "
            "publish (pubsub), info, keys. Yêu cầu confirmation cho writes."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "connection_string": {
                    "type": "string",
                    "description": "Redis URL (redis://[:pw@]host:6379/0) hoặc redis://host:6379",
                },
                "operation": {
                    "type": "string",
                    "enum": sorted(WRITE_OPS | READ_OPS),
                    "description": "Redis operation",
                },
                "key": {"type": "string", "description": "Redis key"},
                "field": {"type": "string", "description": "Hash field (hget/hset)"},
                "value": {"type": "string", "description": "Giá trị cần set"},
                "ttl": {"type": "integer", "description": "TTL (giây) cho set/expire"},
                "pattern": {"type": "string", "description": "Pattern cho keys (default *)"},
                "channel": {"type": "string", "description": "Pubsub channel (publish)"},
                "message": {"type": "string", "description": "Pubsub message (publish)"},
            },
            "required": ["connection_string", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("connection_string"):
            return "Missing required arg: connection_string"
        op = args.get("operation")
        if not op:
            return "Missing required arg: operation"
        if op in {"get", "set", "hget", "hset", "lpush", "rpop", "delete", "expire", "ttl", "type", "exists"} and not args.get("key"):
            return f"Operation '{op}' requires 'key' arg"
        if op in {"hget", "hset"} and not args.get("field"):
            return f"Operation '{op}' requires 'field' arg"
        if op == "set" and args.get("value") is None:
            return "Operation 'set' requires 'value' arg"
        if op == "publish" and not args.get("channel"):
            return "Operation 'publish' requires 'channel' arg"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cs: str = args["connection_string"]
        op: str = args["operation"]

        # Lazy import redis // lazy import
        try:
            import redis  # type: ignore
        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"redis not installed: {e}. Cài: pip install redis",
                return_code=127,
            )

        # Dry-run cho write ops // dry-run
        if context.dry_run and op in WRITE_OPS:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would run {op} on key={args.get('key')!r}",
                metadata={"dry_run": True, "operation": op, "key": args.get("key")},
            )

        try:
            client = redis.Redis.from_url(cs, decode_responses=True)

            if op == "get":
                val = client.get(args["key"])
                return ToolResult(success=True, output=str(val) if val is not None else "", metadata={"key": args["key"], "found": val is not None})

            if op == "set":
                ttl = args.get("ttl")
                if ttl:
                    client.set(args["key"], args["value"], ex=int(ttl))
                else:
                    client.set(args["key"], args["value"])
                return ToolResult(success=True, output=f"OK: {args['key']}", metadata={"key": args["key"], "ttl": ttl})

            if op == "hget":
                val = client.hget(args["key"], args["field"])
                return ToolResult(success=True, output=str(val) if val is not None else "", metadata={"key": args["key"], "field": args["field"]})

            if op == "hset":
                client.hset(args["key"], args["field"], args["value"])
                return ToolResult(success=True, output=f"OK: {args['key']}.{args['field']}", metadata={"key": args["key"], "field": args["field"]})

            if op == "lpush":
                n = client.lpush(args["key"], args["value"])
                return ToolResult(success=True, output=f"Pushed, list len={n}", metadata={"key": args["key"], "length": n})

            if op in {"rpop", "lpop"}:
                val = client.rpop(args["key"]) if op == "rpop" else client.lpop(args["key"])
                return ToolResult(success=True, output=str(val) if val is not None else "", metadata={"key": args["key"]})

            if op == "publish":
                n = client.publish(args["channel"], args.get("message", ""))
                return ToolResult(success=True, output=f"Published to {args['channel']}, subscribers={n}", metadata={"channel": args["channel"], "subscribers": n})

            if op == "info":
                info = client.info()
                payload = json.dumps({k: (dict(v) if isinstance(v, dict) else v) for k, v in info.items()}, default=str, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"sections": len(info)})

            if op == "keys":
                pattern = args.get("pattern", "*")
                keys = client.keys(pattern)
                payload = json.dumps(keys, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"count": len(keys), "pattern": pattern})

            if op == "delete":
                n = client.delete(args["key"])
                return ToolResult(success=True, output=f"Deleted {n} keys", metadata={"deleted": n})

            if op == "expire":
                ok = client.expire(args["key"], int(args.get("ttl", 60)))
                return ToolResult(success=True, output=f"Expire set: {ok}", metadata={"key": args["key"], "ttl": args.get("ttl")})

            if op == "ttl":
                ttl = client.ttl(args["key"])
                return ToolResult(success=True, output=str(ttl), metadata={"key": args["key"], "ttl": ttl})

            if op == "type":
                t = client.type(args["key"])
                return ToolResult(success=True, output=str(t), metadata={"key": args["key"], "type": str(t)})

            if op == "exists":
                n = client.exists(args["key"])
                return ToolResult(success=True, output=str(bool(n)), metadata={"key": args["key"], "exists": bool(n)})

            return ToolResult(success=False, error=f"Unknown operation: {op}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
