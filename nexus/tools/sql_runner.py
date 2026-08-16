"""
SQL Runner Tool - Execute SQL against any database via SQLAlchemy.
Author: Hieu Louis (2026)
Công cụ chạy SQL trên nhiều loại DB (postgresql, mysql, sqlite, mssql) qua SQLAlchemy.
"""
from __future__ import annotations

import json
from datetime import date, datetime, decimal
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Các operation viết dữ liệu // write operations (cần confirmation)
WRITE_OPS = {"insert", "update", "delete", "drop", "create", "alter", "truncate", "merge"}


def _classify(query: str) -> str:
    """Phân loại query (read vs write) // Classify query type."""
    stripped = (query or "").lstrip().lower()
    for kw in WRITE_OPS:
        if stripped.startswith(kw):
            return "write"
    return "read"


def _json_default(o: Any) -> Any:
    """Serializer fallback cho JSON // JSON serializer fallback."""
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


class SQLRunnerTool(Tool):
    """Chạy SQL parameterized trên bất kỳ DB nào thông qua SQLAlchemy."""

    category = ToolCategory.DATABASE
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "sql_runner"

    @property
    def description(self) -> str:
        return (
            "Thực thi SQL parameterized trên PostgreSQL/MySQL/SQLite/MSSQL "
            "qua SQLAlchemy. Trả về rows JSON. Chống SQL injection."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "connection_string": {
                    "type": "string",
                    "description": "SQLAlchemy URL (vd: postgresql://user:pw@host:5432/db)",
                },
                "query": {"type": "string", "description": "SQL query với placeholder (:name hoặc %s)"},
                "params": {
                    "oneOf": [
                        {"type": "array", "items": {}},
                        {"type": "object"},
                    ],
                    "description": "Tham số bind cho parameterized query (list hoặc dict)",
                },
                "limit": {"type": "integer", "description": "Giới hạn số rows trả về (default 1000)"},
            },
            "required": ["connection_string", "query"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("connection_string"):
            return "Missing required arg: connection_string"
        if not args.get("query"):
            return "Missing required arg: query"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cs: str = args["connection_string"]
        query: str = args["query"]
        params = args.get("params") or ()
        limit = int(args.get("limit") or 1000)
        op_kind = _classify(query)

        # Lazy import SQLAlchemy // lazy import
        try:
            from sqlalchemy import create_engine, text  # type: ignore
        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"SQLAlchemy not installed: {e}. Cài: pip install sqlalchemy",
                return_code=127,
            )

        # Dry-run // dry-run simulation
        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would execute ({op_kind}) query:\n{query}\nParams: {params!r}",
                metadata={"dry_run": True, "op_kind": op_kind, "limit": limit},
            )

        try:
            engine = create_engine(cs, future=True)
            with engine.connect() as conn:
                stmt = text(query)
                if op_kind == "write":
                    result = conn.execute(stmt, params)
                    conn.commit()
                    return ToolResult(
                        success=True,
                        output=json.dumps(
                            {"rowcount": result.rowcount},
                            default=_json_default,
                            ensure_ascii=False,
                        ),
                        metadata={"op_kind": "write", "rowcount": result.rowcount},
                    )
                # read
                result = conn.execute(stmt, params)
                rows: List[Dict[str, Any]] = []
                for i, row in enumerate(result):
                    if i >= limit:
                        break
                    rows.append({k: row[k] for k in row._mapping.keys()})  # type: ignore[attr-defined]
                payload = json.dumps(rows, default=_json_default, ensure_ascii=False, indent=2)
                return ToolResult(
                    success=True,
                    output=payload,
                    metadata={
                        "op_kind": "read",
                        "rowcount": len(rows),
                        "truncated": len(rows) >= limit,
                        "limit": limit,
                    },
                )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
