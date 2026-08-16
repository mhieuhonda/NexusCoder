"""Code Generation Skill - Sinh code từ mô tả."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class CodeGenerationSkill(Skill):
    """Sinh code Python/JS/Go/Rust/SQL từ mô tả tự nhiên."""
    
    category = SkillCategory.CODE
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "viết", "write", "code", "function", "hàm", "class", "lớp",
        "implement", "tạo", "generate", "sinh", "snippet",
    ]
    examples = [
        "Viết hàm Python tính fibonacci",
        "Write a function to reverse a linked list",
        "Implement a binary search tree in Python",
    ]
    
    @property
    def name(self) -> str:
        return "code_generation"
    
    @property
    def description(self) -> str:
        return "Sinh code từ mô tả tự nhiên. Hỗ trợ Python, JavaScript, Go, Rust, SQL, C++, Java."
    
    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.2
        if context and context.language:
            score += 0.3
        if "```" in prompt or "def " in prompt or "function " in prompt:
            score += 0.3
        return min(1.0, score)
    
    def execute(self, context: SkillContext) -> SkillResult:
        lang = context.language or "python"
        system_prompt = (
            f"You are Nexus Coder, an expert {lang} developer. "
            f"Generate clean, production-ready code with proper error handling, "
            f"type hints, and docstrings. Follow PEP-8 / best practices."
        )
        return SkillResult(
            success=True,
            output=f"[CodeGeneration/{lang}] Ready to generate code for: {context.prompt[:200]}",
            metadata={
                "skill": self.name,
                "language": lang,
                "system_prompt": system_prompt,
                "max_tokens": context.max_tokens,
            },
            suggestions=[
                f"Specify {lang} version if needed",
                "Provide test cases for edge conditions",
                "Consider error handling strategy",
            ],
        )
