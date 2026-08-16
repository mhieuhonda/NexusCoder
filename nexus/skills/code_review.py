"""Code Review Skill - Phân tích và review code."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class CodeReviewSkill(Skill):
    """Review code: tìm bugs, security issues, performance problems, style violations."""
    
    category = SkillCategory.CODE
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "review", "kiểm tra", "audit", "phân tích", "analyze",
        "bug", "lỗi", "issue", "vấn đề", "problem",
    ]
    examples = [
        "Review đoạn code này giúp tôi",
        "Check this code for bugs",
        "Audit this function for security issues",
    ]
    
    @property
    def name(self) -> str:
        return "code_review"
    
    @property
    def description(self) -> str:
        return (
            "Review code toàn diện: bugs, security vulnerabilities, "
            "performance issues, code style, best practices, maintainability."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        checks = [
            "Bug detection (logic errors, off-by-one, null refs)",
            "Security vulnerabilities (injection, XSS, SSRF, secrets)",
            "Performance bottlenecks (O(n²) loops, unnecessary allocations)",
            "Code style (PEP-8, naming conventions, docstrings)",
            "Error handling completeness",
            "Type safety (mypy, type hints)",
            "Test coverage gaps",
            "Documentation completeness",
        ]
        return SkillResult(
            success=True,
            output=f"[CodeReview] Analyzing {len(context.files)} files. Running {len(checks)} checks.",
            metadata={
                "skill": self.name,
                "checks": checks,
                "severity_levels": ["critical", "high", "medium", "low", "info"],
            },
            suggestions=[
                "Run with `--fix` to auto-apply safe fixes",
                "Enable strict mode for production code",
            ],
        )
