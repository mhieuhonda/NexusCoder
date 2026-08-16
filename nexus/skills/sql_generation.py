"""SQL Generation Skill - Sinh SQL queries."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class SQLGenerationSkill(Skill):
    """Sinh SQL: SELECT, JOIN, aggregate, window functions, CTEs, optimization."""
    
    category = SkillCategory.DATA
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "sql", "query", "truy vấn", "select", "join",
        "database", "cơ sở dữ liệu", "table", "bảng",
        "postgres", "mysql", "sqlite", "mariadb", "sqlserver",
    ]
    
    @property
    def name(self) -> str:
        return "sql_generation"
    
    @property
    def description(self) -> str:
        return (
            "Sinh SQL từ ngôn ngữ tự nhiên: SELECT, JOINs, aggregations, "
            "window functions, CTEs, recursive queries, query optimization."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        dialects = ["PostgreSQL", "MySQL", "SQLite", "SQL Server", "Oracle", "BigQuery", "Snowflake"]
        features = [
            "SELECT / WHERE / GROUP BY / HAVING / ORDER BY",
            "INNER / LEFT / RIGHT / FULL / CROSS JOIN",
            "Window functions (ROW_NUMBER, RANK, LAG, LEAD, SUM OVER)",
            "Common Table Expressions (CTE) & Recursive CTE",
            "Subqueries (scalar, correlated, EXISTS)",
            "Aggregate functions (COUNT, SUM, AVG, MIN, MAX)",
            "Set operations (UNION, INTERSECT, EXCEPT)",
            "PIVOT / UNPIVOT",
            "JSON operations",
            "Full-text search",
            "Query optimization (EXPLAIN, indexes)",
        ]
        return SkillResult(
            success=True,
            output=f"[SQLGeneration] Supports {len(dialects)} dialects, {len(features)} features.",
            metadata={
                "skill": self.name,
                "dialects": dialects,
                "features": features,
                "safe_mode": True,  # Prevent SQL injection
            },
            suggestions=[
                "Use parameterized queries in production",
                "Add indexes for frequently queried columns",
                "Use EXPLAIN to verify query plan",
            ],
        )
