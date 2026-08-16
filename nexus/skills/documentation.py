"""Documentation Skill - Sinh tài liệu từ code."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class DocumentationSkill(Skill):
    """Sinh tài liệu: docstrings, README, API docs, tutorials."""
    
    category = SkillCategory.CODE
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "document", "tài liệu", "docs", "docstring", "readme",
        "api", "hướng dẫn", "tutorial", "guide", "wiki",
        "comment", "chú thích", "mô tả",
    ]
    
    @property
    def name(self) -> str:
        return "documentation"
    
    @property
    def description(self) -> str:
        return (
            "Sinh tài liệu tự động: docstrings (Google/NumPy/Sphinx), "
            "README.md, API reference, tutorials, architecture docs."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        doc_types = [
            "Docstrings (Google style)",
            "Docstrings (NumPy style)",
            "Docstrings (Sphinx reST)",
            "README.md",
            "CONTRIBUTING.md",
            "API reference (OpenAPI/Swagger)",
            "Architecture decision records (ADR)",
            "Tutorial / How-to guide",
            "Changelog",
            "Inline comments (only when necessary)",
        ]
        return SkillResult(
            success=True,
            output=f"[Documentation] Can generate {len(doc_types)} doc types.",
            metadata={
                "skill": self.name,
                "doc_types": doc_types,
                "languages": ["vi", "en", "bilingual"],
            },
            suggestions=[
                "Choose doc style based on project conventions",
                "Auto-generate API docs from type hints",
            ],
        )
