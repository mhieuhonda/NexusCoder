"""
PostgreSQL Tool - Quản lý PostgreSQL qua psycopg2.
Author: Hieu Louis (2026)
Các operation: query, list_tables, describe_table, create_index, vacuum, explain.
"""
from __future__ import annotations

import json
from datetime import date, datetime, decimal
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Operations yêu cầu confirmation // DDL ops
DDL_OPS = {"create_index"}
# Operations yêu cầu superuser / không transactional
MAINTENANCE_OPS = {"vacuum"}


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


class PostgresTool(Tool):
    """Quản lý PostgreSQL: query, list_tables, describe_table, create_index, vacuum, explain."""

    category = ToolCategory.DATABASE
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "postgres"

    @property
    def description(self) -> str:
        return (
            "Quản lý PostgreSQL qua psycopg2: query (parameterized), list_tables, "
            "describe_table, create_index, vacuum, explain."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "connection_string": {
                    "type": "string",
                    "description": "Postgres URL (postgresql://user:pw@host:5432/db) hoặc DSN",
                },
                "operation": {
                    "type": "string",
                    "enum": ["query", "list_tables", "describe_table", "create_index", "vacuum", "explain"],
                    "description": "Postgres operation",
                },
                "query": {"type": "string", "description": "SQL query (operation=query)"},
                "params": {
                    "oneOf": [{"type": "array", "items": {}}, {"type": "object"}],
                    "description": "Tham số bind cho parameterized query",
                },
                "table": {"type": "string", "description": "Tên table (describe_table/create_index)"},
                "index_name": {"type": "string", "description": "Tên index (create_index)"},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Cột cho index (create_index)",
                },
                "schema": {"type": "string", "description": "Schema (default public)"},
                "limit": {"type": "integer", "description": "Giới hạn rows (default 1000)"},
            },
            "required": ["connection_string", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("connection_string"):
            return "Missing required arg: connection_string"
        op = args.get("operation")
        if not op:
            return "Missing required arg: operation"
        if op == "query" and not args.get("query"):
            return "Operation 'query' requires 'query' arg"
        if op in {"describe_table", "create_index"} and not args.get("table"):
            return f"Operation '{op}' requires 'table' arg"
        if op == "create_index" and not args.get("columns"):
            return "Operation 'create_index' requires 'columns' arg"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cs: str = args["connection_string"]
        op: str = args["operation"]
        schema: str = args.get("schema", "public")

        # Lazy import psycopg2 // lazy import
        try:
            import psycopg2  # type: ignore
            import psycopg2.extras  # type: ignore
        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"psycopg2 not installed: {e}. Cài: pip install psycopg2-binary",
                return_code=127,
            )

        # Dry-run cho DDL/maintenance // dry-run for DDL/maintenance
        if context.dry_run and op in DDL_OPS:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would run {op} on {args.get('table')}",
                metadata={"dry_run": True, "operation": op},
            )

        try:
            # VACUUM yêu cầu autocommit // VACUUM needs autocommit
            conn = psycopg2.connect(cs)
            conn.autocommit = op in MAINTENANCE_OPS or op in DDL_OPS
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            if op == "list_tables":
                cur.execute(
                    "SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_schema NOT IN ('pg_catalog','information_schema') ORDER BY 1,2"
                )
                rows = cur.fetchall()
                payload = json.dumps(rows, default=_json_default, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"count": len(rows)})

            if op == "describe_table":
                cur.execute(
                    "SELECT column_name, data_type, is_nullable, column_default "
                    "FROM information_schema.columns WHERE table_schema=%s AND table_name=%s "
                    "ORDER BY ordinal_position",
                    (schema, args["table"]),
                )
                rows = cur.fetchall()
                payload = json.dumps(rows, default=_json_default, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"table": args["table"], "columns": len(rows)})

            if op == "query":
                params = args.get("params") or ()
                cur.execute(args["query"], params)
                if cur.description:
                    rows = cur.fetchmany(int(args.get("limit") or 1000))
                    payload = json.dumps(rows, default=_json_default, ensure_ascii=False, indent=2)
                    return ToolResult(success=True, output=payload, metadata={"rowcount": len(rows)})
                conn.commit()
                return ToolResult(success=True, output=json.dumps({"rowcount": cur.rowcount}), metadata={"rowcount": cur.rowcount})

            if op == "create_index":
                idx = args["index_name"] or f"idx_{args['table']}_" + "_".join(args["columns"])
                cols = ", ".join(args["columns"])
                cur.execute(f'CREATE INDEX IF NOT EXISTS "{idx}" ON "{schema}"."{args["table"]}" ({cols})')
                return ToolResult(success=True, output=f"Index created: {idx}", metadata={"index": idx})

            if op == "vacuum":
                cur.execute("VACUUM ANALYZE")
                return ToolResult(success=True, output="VACUUM ANALYZE executed", metadata={"operation": "vacuum"})

            if op == "explain":
                cur.execute("EXPLAIN ANALYZE " + args["query"])
                rows = cur.fetchall()
                payload = json.dumps(rows, default=_json_default, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"operation": "explain"})

            return ToolResult(success=False, error=f"Unknown operation: {op}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
        finally:
            try:
                cur.close(); conn.close()
            except Exception:
                pass
