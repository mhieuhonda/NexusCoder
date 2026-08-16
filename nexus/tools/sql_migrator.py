"""
SQL Migrator Tool - Chạy DB migrations kiểu Alembic.
Author: Hieu Louis (2026)
Quản lý versioned SQL migrations trong thư mục migrations (up/down).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Pattern tên file migration // migration filename pattern
# Ví dụ: 0001_create_users.up.sql  /  0001_create_users.down.sql
MIGRATION_RE = re.compile(r"^(\d{4})_(.+?)\.(up|down)\.sql$", re.IGNORECASE)


def _discover(migrations_dir: str) -> List[Dict[str, str]]:
    """Quét thư mục migrations // scan migrations dir."""
    out: List[Dict[str, str]] = []
    if not os.path.isdir(migrations_dir):
        return out
    for fname in sorted(os.listdir(migrations_dir)):
        m = MIGRATION_RE.match(fname)
        if not m:
            continue
        version, name, direction = m.group(1), m.group(2), m.group(3).lower()
        out.append({
            "version": version,
            "name": name,
            "direction": direction,
            "path": os.path.join(migrations_dir, fname),
        })
    return out


class SQLMigratorTool(Tool):
    """Chạy DB migrations (up/down) từ thư mục migrations (Alembic-style)."""

    category = ToolCategory.DATABASE
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "sql_migrator"

    @property
    def description(self) -> str:
        return (
            "Chạy SQL migrations từ thư mục (file *_N_description.up.sql / .down.sql). "
            "Hỗ trợ up/down, steps, parameterized cho bất kỳ DB nào (qua SQLAlchemy)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "connection_string": {"type": "string", "description": "SQLAlchemy URL"},
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Hướng migrate (default up)",
                },
                "steps": {"type": "integer", "description": "Số bước áp dụng (default: all up = 0/all)"},
                "migrations_dir": {"type": "string", "description": "Thư mục chứa file migration"},
                "dry_run": {"type": "boolean", "description": "Simulate mà không ghi DB"},
            },
            "required": ["connection_string", "migrations_dir"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("connection_string"):
            return "Missing required arg: connection_string"
        if not args.get("migrations_dir"):
            return "Missing required arg: migrations_dir"
        direction = args.get("direction", "up")
        if direction not in {"up", "down"}:
            return f"Invalid direction: {direction}"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cs: str = args["connection_string"]
        direction: str = args.get("direction", "up")
        steps: Optional[int] = args.get("steps")
        migrations_dir: str = args["migrations_dir"]

        if not os.path.isdir(migrations_dir):
            return ToolResult(
                success=False,
                error=f"migrations_dir not found: {migrations_dir}",
                return_code=2,
            )

        all_migs = _discover(migrations_dir)
        # Lọc theo direction // filter by direction
        migs = [m for m in all_migs if m["direction"] == direction]
        if direction == "down":
            migs = list(reversed(migs))

        if steps is not None and steps >= 0:
            migs = migs[:steps]

        # Lazy import SQLAlchemy // lazy import
        try:
            from sqlalchemy import create_engine, text  # type: ignore
        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"SQLAlchemy not installed: {e}",
                return_code=127,
            )

        # Dry-run // dry-run
        if context.dry_run or args.get("dry_run"):
            return ToolResult(
                success=True,
                output=json.dumps(
                    {
                        "dry_run": True,
                        "direction": direction,
                        "migrations": [{"version": m["version"], "name": m["name"]} for m in migs],
                        "count": len(migs),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                metadata={
                    "dry_run": True,
                    "direction": direction,
                    "count": len(migs),
                    "migrations_dir": migrations_dir,
                },
            )

        applied: List[Dict[str, str]] = []
        try:
            engine = create_engine(cs, future=True)
            with engine.begin() as conn:
                for m in migs:
                    with open(m["path"], "r", encoding="utf-8") as f:
                        script = f.read()
                    # Tách theo ';' // split statements roughly
                    for stmt in [s.strip() for s in script.split(";") if s.strip()]:
                        conn.execute(text(stmt))
                    applied.append({"version": m["version"], "name": m["name"], "direction": direction})
            return ToolResult(
                success=True,
                output=json.dumps(
                    {"applied": applied, "count": len(applied), "direction": direction},
                    ensure_ascii=False,
                    indent=2,
                ),
                metadata={
                    "direction": direction,
                    "applied": len(applied),
                    "migrations_dir": migrations_dir,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Migration failed at step {len(applied)+1}: {e}",
                return_code=1,
                metadata={"applied_before_failure": applied},
            )
