"""
SQL Formatter Tool - Định dạng SQL cho đẹp và nhất quán.
Author: Hieu Louis (2026)
Dùng thư viện sqlparse để format SQL.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Các dialect được hỗ trợ // supported dialects
DIALECTS = {"ansi", "non-leaning", "postgres", "postgresql", "mysql", "mssql"}


class SQLFormatterTool(Tool):
    """Format SQL với sqlparse (keyword case, indentation, identifier quotes)."""

    category = ToolCategory.DATABASE
    safety = ToolSafety.SAFE  # read-only, không có side-effect

    @property
    def name(self) -> str:
        return "sql_formatter"

    @property
    def description(self) -> str:
        return "Định dạng (làm đẹp) SQL bằng sqlparse: in hoa keyword, indent, identifier_quotes."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "Chuỗi SQL cần format"},
                "dialect": {
                    "type": "string",
                    "enum": sorted(DIALECTS),
                    "description": "SQL dialect (default ansi)",
                },
                "identifier_case": {
                    "type": "string",
                    "enum": ["lower", "upper", "capitalize"],
                    "description": "Case cho identifiers (default lower)",
                },
                "keyword_case": {
                    "type": "string",
                    "enum": ["lower", "upper", "capitalize"],
                    "description": "Case cho keywords (default upper)",
                },
                "strip_comments": {"type": "boolean", "description": "Xoá comment (default false)"},
                "reindent": {"type": "boolean", "description": "Tự động indent (default true)"},
            },
            "required": ["sql"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("sql"):
            return "Missing required arg: sql"
        dialect = args.get("dialect")
        if dialect and dialect not in DIALECTS:
            return f"Unsupported dialect: {dialect}. Chọn một trong: {sorted(DIALECTS)}"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        sql: str = args["sql"]
        dialect = args.get("dialect", "ansi")
        keyword_case = args.get("keyword_case", "upper")
        identifier_case = args.get("identifier_case", "lower")
        strip_comments = bool(args.get("strip_comments", False))
        reindent = bool(args.get("reindent", True))

        # Lazy import sqlparse // lazy import
        try:
            import sqlparse  # type: ignore
        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"sqlparse not installed: {e}. Cài: pip install sqlparse",
                return_code=127,
            )

        try:
            formatted = sqlparse.format(
                sql,
                keyword_case=keyword_case,
                identifier_case=identifier_case,
                strip_comments=strip_comments,
                reindent=reindent,
                indent_width=2,
                comma_first=False,
                use_2c_for_indent=False,
            )
            # Không thay đổi identifier cho dialect non-ansi (chỉ thông báo metadata)
            return ToolResult(
                success=True,
                output=formatted,
                metadata={
                    "dialect": dialect,
                    "keyword_case": keyword_case,
                    "identifier_case": identifier_case,
                    "strip_comments": strip_comments,
                    "reindent": reindent,
                    "input_length": len(sql),
                    "output_length": len(formatted),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
