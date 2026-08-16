"""
SQLite Tool - Quản lý SQLite database qua stdlib sqlite3.
Author: Hieu Louis (2026)
Operations: query, list_tables, schema, vacuum, backup.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime, decimal
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


def _json_default(o: Any) -> Any:
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, decimal.Decimal):
        return float(o)
    if isinstance(o, bytes):
        try:
            return o.decode("utf-8")
        except Exception:
            return o.hex()
    return str(o)


class SQLiteTool(Tool):
    """Quản lý SQLite DB (file): query, list_tables, schema, vacuum, backup."""

    category = ToolCategory.DATABASE
    safety = ToolSafety.MODERATE  # local writes
    requires_confirmation = False  # file local, an toàn tương đối

    @property
    def name(self) -> str:
        return "sqlite"

    @property
    def description(self) -> str:
        return (
            "Quản lý SQLite (file hoặc :memory:) qua stdlib sqlite3: "
            "query parameterized, list_tables, schema, vacuum, backup."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "database": {"type": "string", "description": "Đường dẫn file .sqlite/.db (hoặc :memory:)"},
                "operation": {
                    "type": "string",
                    "enum": ["query", "list_tables", "schema", "vacuum", "backup"],
                    "description": "SQLite operation",
                },
                "query": {"type": "string", "description": "SQL query (operation=query)"},
                "params": {
                    "oneOf": [{"type": "array", "items": {}}, {"type": "object"}],
                    "description": "Tham số bind (? hoặc :name)",
                },
                "table": {"type": "string", "description": "Table name (schema)"},
                "backup_path": {"type": "string", "description": "Đường dẫn file backup (operation=backup)"},
                "limit": {"type": "integer", "description": "Giới hạn rows (default 1000)"},
            },
            "required": ["database", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("database"):
            return "Missing required arg: database"
        op = args.get("operation")
        if not op:
            return "Missing required arg: operation"
        if op == "query" and not args.get("query"):
            return "Operation 'query' requires 'query' arg"
        if op == "backup" and not args.get("backup_path"):
            return "Operation 'backup' requires 'backup_path' arg"
        if op == "schema" and not args.get("table"):
            return "Operation 'schema' requires 'table' arg"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        db: str = args["database"]
        op: str = args["operation"]

        # Dry-run cho backup/vacuum // dry-run
        if context.dry_run and op in {"backup", "vacuum"}:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would run {op} on {db}",
                metadata={"dry_run": True, "operation": op},
            )

        try:
            # Row factory cho dict-like rows // dict rows
            conn = sqlite3.connect(db, timeout=context.timeout)
            conn.row_factory = sqlite3.Row

            if op == "list_tables":
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
                rows = [dict(r) for r in cur.fetchall()]
                payload = json.dumps(rows, default=_json_default, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"count": len(rows)})

            if op == "schema":
                cur = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type IN ('table','index','view','trigger') AND tbl_name=?",
                    (args["table"],),
                )
                rows = [dict(r) for r in cur.fetchall()]
                payload = json.dumps(rows, default=_json_default, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"table": args["table"], "count": len(rows)})

            if op == "query":
                params = args.get("params") or ()
                cur = conn.execute(args["query"], params)
                if cur.description:
                    rows = cur.fetchmany(int(args.get("limit") or 1000))
                    result = [dict(r) for r in rows]
                    payload = json.dumps(result, default=_json_default, ensure_ascii=False, indent=2)
                    conn.commit()
                    return ToolResult(success=True, output=payload, metadata={"rowcount": len(result)})
                conn.commit()
                return ToolResult(
                    success=True,
                    output=json.dumps({"rowcount": cur.rowcount}),
                    metadata={"rowcount": cur.rowcount},
                )

            if op == "vacuum":
                conn.isolation_level = None  # autocommit for VACUUM
                conn.execute("VACUUM")
                size = os.path.getsize(db) if os.path.exists(db) else 0
                return ToolResult(success=True, output="VACUUM executed", metadata={"db_size": size})

            if op == "backup":
                backup_path = args["backup_path"]
                os.makedirs(os.path.dirname(os.path.abspath(backup_path)) or ".", exist_ok=True)
                # Dùng SQLite backup API // use SQLite backup API
                src = conn
                dst = sqlite3.connect(backup_path)
                try:
                    src.backup(dst)
                finally:
                    dst.close()
                return ToolResult(
                    success=True,
                    output=f"Backup created: {backup_path}",
                    artifacts=[backup_path],
                    metadata={"source": db, "backup": backup_path},
                )

            return ToolResult(success=False, error=f"Unknown operation: {op}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
        finally:
            try:
                conn.close()
            except Exception:
                pass
