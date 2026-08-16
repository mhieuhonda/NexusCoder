"""
MySQL Tool - Quản lý MySQL qua pymysql.
Author: Hieu Louis (2026)
Operations: query, list_tables, describe_table, create_index, optimize, explain.
"""
from __future__ import annotations

import json
from datetime import date, datetime, decimal
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


DDL_OPS = {"create_index"}
MAINTENANCE_OPS = {"optimize"}


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


def _parse_mysql_url(url: str) -> Dict[str, Any]:
    """Parse mysql://user:pw@host:3306/db // parse MySQL URL."""
    from urllib.parse import urlparse
    p = urlparse(url)
    return {
        "host": p.hostname or "localhost",
        "port": p.port or 3306,
        "user": p.username or "root",
        "password": p.password or "",
        "database": (p.path or "/").lstrip("/"),
    }


class MySQLTool(Tool):
    """Quản lý MySQL: query, list_tables, describe_table, create_index, optimize, explain."""

    category = ToolCategory.DATABASE
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "mysql"

    @property
    def description(self) -> str:
        return (
            "Quản lý MySQL qua pymysql: query (parameterized), list_tables, "
            "describe_table, create_index, optimize (ANALYZE TABLE), explain."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "connection_string": {"type": "string", "description": "MySQL URL (mysql://user:pw@host:3306/db)"},
                "operation": {
                    "type": "string",
                    "enum": ["query", "list_tables", "describe_table", "create_index", "optimize", "explain"],
                    "description": "MySQL operation",
                },
                "query": {"type": "string", "description": "SQL query (operation=query/explain)"},
                "params": {
                    "oneOf": [{"type": "array", "items": {}}, {"type": "object"}],
                    "description": "Tham số bind (%s hoặc %(name)s)",
                },
                "table": {"type": "string", "description": "Table name (describe_table/create_index)"},
                "index_name": {"type": "string", "description": "Tên index (create_index)"},
                "columns": {"type": "array", "items": {"type": "string"}, "description": "Cột cho index"},
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

        # Lazy import pymysql (fallback mysql.connector) // lazy import
        try:
            import pymysql  # type: ignore
            import pymysql.cursors  # type: ignore
            driver = "pymysql"
        except ImportError:
            try:
                import mysql.connector  # type: ignore
                import mysql.connector.cursor  # type: ignore
                driver = "mysql.connector"
            except ImportError as e:
                return ToolResult(
                    success=False,
                    error=f"Neither pymysql nor mysql-connector installed: {e}",
                    return_code=127,
                )

        # Dry-run cho DDL // dry-run
        if context.dry_run and op in DDL_OPS:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would run {op} on {args.get('table')}",
                metadata={"dry_run": True, "operation": op},
            )

        cfg = _parse_mysql_url(cs)
        try:
            if driver == "pymysql":
                conn = pymysql.connect(
                    host=cfg["host"], port=cfg["port"], user=cfg["user"],
                    password=cfg["password"], database=cfg["database"] or None,
                    cursorclass=pymysql.cursors.DictCursor, charset="utf8mb4",
                    autocommit=op in DDL_OPS,
                )
            else:
                conn = mysql.connector.connect(
                    host=cfg["host"], port=cfg["port"], user=cfg["user"],
                    password=cfg["password"], database=cfg["database"] or None,
                    charset="utf8mb4", autocommit=op in DDL_OPS,
                )
            cur = conn.cursor(dictionary=True) if driver == "mysql.connector" else conn.cursor()

            if op == "list_tables":
                cur.execute("SHOW TABLES")
                rows = cur.fetchall()
                payload = json.dumps(rows, default=_json_default, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"count": len(rows)})

            if op == "describe_table":
                cur.execute(f"DESCRIBE `{args['table']}`")
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
                cols = ", ".join(f"`{c}`" for c in args["columns"])
                cur.execute(f"CREATE INDEX `{idx}` ON `{args['table']}` ({cols})")
                return ToolResult(success=True, output=f"Index created: {idx}", metadata={"index": idx})

            if op == "optimize":
                cur.execute(f"ANALYZE TABLE `{args['table']}`" if args.get("table") else "ANALYZE TABLE")
                rows = cur.fetchall()
                payload = json.dumps(rows, default=_json_default, ensure_ascii=False, indent=2)
                return ToolResult(success=True, output=payload, metadata={"operation": "optimize"})

            if op == "explain":
                cur.execute("EXPLAIN " + args["query"])
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
