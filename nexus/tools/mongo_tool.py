"""
MongoDB Tool - Quản lý MongoDB qua pymongo.
Author: Hieu Louis (2026)
Operations: find, insert, update, delete, aggregate, create_index.
"""
from __future__ import annotations

import json
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Operations ghi/xoá // write ops
WRITE_OPS = {"insert", "update", "delete", "create_index"}


def _json_default(o: Any) -> Any:
    """Serializer cho BSON ObjectId / datetime // BSON fallback."""
    try:
        from bson import ObjectId  # type: ignore
        if isinstance(o, ObjectId):
            return str(o)
    except Exception:
        pass
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, bytes):
        try:
            return o.decode("utf-8")
        except Exception:
            return o.hex()
    return str(o)


class MongoTool(Tool):
    """Quản lý MongoDB: find, insert, update, delete, aggregate, create_index."""

    category = ToolCategory.DATABASE
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "mongo"

    @property
    def description(self) -> str:
        return (
            "Quản lý MongoDB qua pymongo: find, insert (one/many), update (one/many), "
            "delete (one/many), aggregate, create_index."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "connection_string": {"type": "string", "description": "MongoDB URL (mongodb://host:27017 hoặc mongodb+srv://...)"},
                "database": {"type": "string", "description": "Tên database"},
                "collection": {"type": "string", "description": "Tên collection"},
                "operation": {
                    "type": "string",
                    "enum": ["find", "insert", "update", "delete", "aggregate", "create_index"],
                    "description": "Mongo operation",
                },
                "filter": {"type": "object", "description": "Query filter (find/update/delete)"},
                "doc": {"type": "object", "description": "Document (insert one)"},
                "docs": {"type": "array", "items": {"type": "object"}, "description": "Documents (insert many)"},
                "update": {"type": "object", "description": "Update spec (vd: {$set: {..}})"},
                "pipeline": {"type": "array", "items": {}, "description": "Aggregation pipeline (aggregate)"},
                "index": {
                    "oneOf": [{"type": "string"}, {"type": "object"}],
                    "description": "Index spec (create_index, vd: {\"email\": 1})",
                },
                "many": {"type": "boolean", "description": "Update/Delete nhiều (default false)"},
                "limit": {"type": "integer", "description": "Giới hạn kết quả (default 100)"},
            },
            "required": ["connection_string", "database", "collection", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        for k in ("connection_string", "database", "collection", "operation"):
            if not args.get(k):
                return f"Missing required arg: {k}"
        op = args["operation"]
        if op == "insert" and not (args.get("doc") or args.get("docs")):
            return "Operation 'insert' requires 'doc' or 'docs' arg"
        if op == "update" and not args.get("update"):
            return "Operation 'update' requires 'update' arg"
        if op == "aggregate" and not args.get("pipeline"):
            return "Operation 'aggregate' requires 'pipeline' arg"
        if op == "create_index" and not args.get("index"):
            return "Operation 'create_index' requires 'index' arg"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cs: str = args["connection_string"]
        db_name: str = args["database"]
        coll_name: str = args["collection"]
        op: str = args["operation"]

        # Lazy import pymongo // lazy import
        try:
            from pymongo import MongoClient, ASCENDING  # type: ignore
        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"pymongo not installed: {e}. Cài: pip install pymongo",
                return_code=127,
            )

        # Dry-run cho write ops // dry-run
        if context.dry_run and op in WRITE_OPS:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would run {op} on {db_name}.{coll_name}",
                metadata={"dry_run": True, "operation": op, "database": db_name, "collection": coll_name},
            )

        try:
            client = MongoClient(cs, serverSelectionTimeoutMS=context.timeout * 1000)
            db = client[db_name]
            coll = db[coll_name]

            if op == "find":
                flt = args.get("filter") or {}
                limit = int(args.get("limit") or 100)
                docs = list(coll.find(flt).limit(limit))
                payload = json.dumps(docs, default=_json_default, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"count": len(docs)})

            if op == "insert":
                if args.get("docs"):
                    result = coll.insert_many(args["docs"])
                    return ToolResult(
                        success=True,
                        output=json.dumps({"inserted": len(result.inserted_ids)}, default=_json_default),
                        metadata={"inserted_ids": [str(i) for i in result.inserted_ids]},
                    )
                result = coll.insert_one(args["doc"])
                return ToolResult(
                    success=True,
                    output=json.dumps({"inserted_id": str(result.inserted_id)}, default=_json_default),
                    metadata={"inserted_id": str(result.inserted_id)},
                )

            if op == "update":
                flt = args.get("filter") or {}
                upd = args["update"]
                if args.get("many"):
                    result = coll.update_many(flt, upd)
                else:
                    result = coll.update_one(flt, upd)
                return ToolResult(
                    success=True,
                    output=json.dumps({"matched": result.matched_count, "modified": result.modified_count}),
                    metadata={"matched": result.matched_count, "modified": result.modified_count},
                )

            if op == "delete":
                flt = args.get("filter") or {}
                if args.get("many"):
                    result = coll.delete_many(flt)
                else:
                    result = coll.delete_one(flt)
                return ToolResult(
                    success=True,
                    output=json.dumps({"deleted": result.deleted_count}),
                    metadata={"deleted": result.deleted_count},
                )

            if op == "aggregate":
                pipeline = args["pipeline"]
                docs = list(coll.aggregate(pipeline))
                payload = json.dumps(docs, default=_json_default, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"count": len(docs)})

            if op == "create_index":
                spec = args["index"]
                if isinstance(spec, str):
                    spec = [(spec, ASCENDING)]
                name = coll.create_index(spec)
                return ToolResult(
                    success=True,
                    output=f"Index created: {name}",
                    metadata={"index_name": name},
                )

            return ToolResult(success=False, error=f"Unknown operation: {op}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
        finally:
            try:
                client.close()
            except Exception:
                pass
