"""Code Refactor Skill - Tái cấu trúc code."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class CodeRefactorSkill(Skill):
    """Refactor code: extract functions, simplify logic, apply design patterns."""
    
    category = SkillCategory.CODE
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "refactor", "tái cấu trúc", "cleanup", "dọn dẹp",
        "optimize", "tối ưu", "simplify", "đơn giản hóa",
        "extract", "tách", "restructure",
    ]
    
    @property
    def name(self) -> str:
        return "code_refactor"
    
    @property
    def description(self) -> str:
        return (
            "Refactor code an toàn: extract method/function, rename variables, "
            "apply design patterns, reduce complexity, eliminate duplication."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        refactorings = [
            "Extract Function / Method",
            "Extract Class",
            "Rename (variable/function/class)",
            "Inline Function / Temp",
            "Move Function / Field",
            "Replace Conditional with Polymorphism",
            "Replace Inheritance with Delegation",
            "Replace Nested Conditional with Guard Clauses",
            "Introduce Parameter Object",
            "Replace Magic Number with Symbolic Constant",
            "Decompose Conditional",
            "Consolidate Conditional Expression",
        ]
        return SkillResult(
            success=True,
            output=f"[CodeRefactor] Suggesting {len(refactorings)} refactoring patterns.",
            metadata={
                "skill": self.name,
                "available_refactorings": refactorings,
                "preserve_behavior": True,
            },
            suggestions=[
                "Always run tests after refactoring",
                "Use git branches for safe refactoring",
            ],
        )
