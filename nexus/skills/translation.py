"""Translation Skill - Dịch Việt-Anh song ngữ."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class TranslationSkill(Skill):
    """Dịch song ngữ Việt-Anh với context awareness."""
    
    category = SkillCategory.LANGUAGE
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "translate", "dịch", "translation", "việt", "english",
        "tiếng anh", "tiếng việt", "song ngữ",
    ]
    
    @property
    def name(self) -> str:
        return "translation"
    
    @property
    def description(self) -> str:
        return (
            "Dịch song ngữ Việt-Anh: giữ nguyên tone, context-aware, "
            "hỗ trợ technical terms, idioms, cultural nuances."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output=f"[Translation] Ready to translate: {context.prompt[:100]}",
            metadata={
                "skill": self.name,
                "supported_pairs": [
                    "vi → en", "en → vi",
                    "vi → zh", "zh → vi",
                    "vi → ja", "ja → vi",
                    "vi → ko", "ko → vi",
                    "vi → fr", "fr → vi",
                ],
                "preserves": ["tone", "formality", "technical_terms", "idioms"],
            },
            suggestions=[
                "Specify target audience (technical vs general)",
                "Indicate desired formality level",
            ],
        )
