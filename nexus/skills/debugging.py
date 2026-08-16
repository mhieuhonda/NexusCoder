"""Debugging Skill - Debug và fix lỗi code."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class DebuggingSkill(Skill):
    """Debug code: phân tích stack trace, tìm root cause, suggest fix."""
    
    category = SkillCategory.CODE
    priority = SkillPriority.CRITICAL
    keywords: List[str] = [
        "debug", "lỗi", "error", "exception", "traceback",
        "stack trace", "fix", "sửa", "khắc phục", "crash",
        "fail", "không chạy", "broken", "không hoạt động",
    ]
    
    @property
    def name(self) -> str:
        return "debugging"
    
    @property
    def description(self) -> str:
        return (
            "Debug code: parse stack traces, identify root cause, "
            "suggest minimal fix, verify fix doesn't break other code."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        debug_steps = [
            "1. Reproduce the error consistently",
            "2. Parse stack trace / error message",
            "3. Identify root cause (not symptom)",
            "4. Propose minimal fix",
            "5. Check for related issues (same pattern elsewhere)",
            "6. Suggest regression test to prevent recurrence",
            "7. Verify fix doesn't introduce new bugs",
        ]
        return SkillResult(
            success=True,
            output=f"[Debugging] Following {len(debug_steps)}-step debugging protocol.",
            metadata={
                "skill": self.name,
                "debug_steps": debug_steps,
                "supports": [
                    "Python traceback", "JavaScript console errors",
                    "Java stack traces", "Go panics", "Rust panics",
                    "C++ segfaults", "Ruby exceptions",
                ],
            },
            suggestions=[
                "Provide full stack trace for accurate diagnosis",
                "Include input that triggered the bug",
                "Mention recent changes that may have introduced the bug",
            ],
        )
