"""
Elasticsearch Tool - Quản lý Elasticsearch qua elasticsearch-py.
Author: Hieu Louis (2026)
Operations: search, index, create, update, delete, bulk.
"""
from __future__ import annotations

import json
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Write/delete ops (cần confirmation) // write ops
WRITE_OPS = {"index", "create", "update", "delete", "bulk"}


def _json_default(o: Any) -> Any:
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, bytes):
        try:
            return o.decode("utf-8")
        except Exception:
            return o.hex()
    return str(o)


class ElasticsearchTool(Tool):
    """Quản lý Elasticsearch: search, index, create, update, delete, bulk."""

    category = ToolCategory.DATABASE
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "elasticsearch"

    @property
    def description(self) -> str:
        return (
            "Quản lý Elasticsearch qua elasticsearch-py: search, index, create, "
            "update, delete, bulk. Hỗ trợ auth header và cloud_id."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "hosts": {
                    "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                    "description": "ES host(s), vd: http://localhost:9200",
                },
                "api_key": {"type": "string", "description": "API key (optional)"},
                "operation": {
                    "type": "string",
                    "enum": ["search", "index", "create", "update", "delete", "bulk"],
                    "description": "ES operation",
                },
                "index": {"type": "string", "description": "Tên ES index"},
                "id": {"type": "string", "description": "Document ID (create/update/delete)"},
                "body": {"type": "object", "description": "Document body hoặc query DSL"},
                "query": {"type": "object", "description": "Query DSL (search)"},
                "doc": {"type": "object", "description": "Partial doc (update)"},
                "actions": {"type": "array", "items": {"type": "object"}, "description": "Bulk actions (vd: [{index:{_id:1}}, {doc:..}])"},
                "size": {"type": "integer", "description": "Số kết quả trả về (default 50)"},
            },
            "required": ["hosts", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("hosts"):
            return "Missing required arg: hosts"
        op = args.get("operation")
        if not op:
            return "Missing required arg: operation"
        if op in {"search", "index", "create", "update", "delete"} and not args.get("index"):
            return f"Operation '{op}' requires 'index' arg"
        if op in {"create", "update", "delete"} and not args.get("id"):
            return f"Operation '{op}' requires 'id' arg"
        if op == "search" and not (args.get("query") or args.get("body")):
            return "Operation 'search' requires 'query' or 'body' arg"
        if op == "bulk" and not args.get("actions"):
            return "Operation 'bulk' requires 'actions' arg"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        hosts = args["hosts"]
        api_key = args.get("api_key")
        op: str = args["operation"]

        # Lazy import elasticsearch // lazy import
        try:
            from elasticsearch import Elasticsearch  # type: ignore
        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"elasticsearch not installed: {e}. Cài: pip install elasticsearch",
                return_code=127,
            )

        # Dry-run cho write ops // dry-run
        if context.dry_run and op in WRITE_OPS:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would run {op} on index={args.get('index')!r}",
                metadata={"dry_run": True, "operation": op, "index": args.get("index")},
            )

        try:
            client = Elasticsearch(hosts, api_key=api_key, request_timeout=context.timeout)

            if op == "search":
                body = args.get("body") or {"query": args["query"]}
                body.setdefault("size", int(args.get("size") or 50))
                resp = client.search(index=args["index"], body=body)
                hits = resp.get("hits", {}).get("hits", [])
                payload = json.dumps(resp, default=_json_default, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"hits": len(hits), "took": resp.get("took")})

            if op == "index":
                resp = client.index(index=args["index"], id=args.get("id"), document=args["body"])
                return ToolResult(
                    success=resp.get("result") in {"created", "updated"},
                    output=json.dumps(resp, default=_json_default, ensure_ascii=False, indent=2),
                    metadata={"result": resp.get("result"), "id": resp.get("_id")},
                )

            if op == "create":
                resp = client.create(index=args["index"], id=args["id"], document=args["body"])
                return ToolResult(
                    success=resp.get("result") == "created",
                    output=json.dumps(resp, default=_json_default, ensure_ascii=False, indent=2),
                    metadata={"result": resp.get("result"), "id": resp.get("_id")},
                )

            if op == "update":
                resp = client.update(index=args["index"], id=args["id"], doc=args.get("doc") or args.get("body"))
                return ToolResult(
                    success=resp.get("result") == "updated",
                    output=json.dumps(resp, default=_json_default, ensure_ascii=False, indent=2),
                    metadata={"result": resp.get("result")},
                )

            if op == "delete":
                resp = client.delete(index=args["index"], id=args["id"])
                return ToolResult(
                    success=resp.get("result") == "deleted",
                    output=json.dumps(resp, default=_json_default, ensure_ascii=False, indent=2),
                    metadata={"result": resp.get("result")},
                )

            if op == "bulk":
                # actions là list các cặp action_header/doc // actions list
                body_lines: List[str] = []
                for item in args["actions"]:
                    body_lines.append(json.dumps(item, default=_json_default))
                body = "\n".join(body_lines) + "\n"
                resp = client.bulk(index=args.get("index"), body=body)
                errors = bool(resp.get("errors"))
                return ToolResult(
                    success=not errors,
                    output=json.dumps(resp, default=_json_default, ensure_ascii=False, indent=2),
                    metadata={"errors": errors, "took": resp.get("took")},
                )

            return ToolResult(success=False, error=f"Unknown operation: {op}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
