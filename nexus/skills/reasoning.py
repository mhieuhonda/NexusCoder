"""Reasoning Skill - Suy luận logic đa bước."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class ReasoningSkill(Skill):
    """Suy luận logic: deductive, inductive, abductive, causal."""
    
    category = SkillCategory.REASONING
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "reason", "suy luận", "logic", "logical", "infer",
        "suy đoán", "deduce", "inductive", "deductive",
        "causal", "nhân quả", "why", "tại sao",
        "explain", "giải thích", "analyze", "phân tích",
    ]
    
    @property
    def name(self) -> str:
        return "reasoning"
    
    @property
    def description(self) -> str:
        return (
            "Suy luận đa bước: Chain-of-Thought (CoT), Tree-of-Thought (ToT), "
            "self-consistency, reflexion, deductive/inductive/abductive reasoning."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        strategies = [
            "Chain-of-Thought (CoT) - step-by-step reasoning",
            "Tree-of-Thought (ToT) - explore multiple paths",
            "Self-Consistency - sample multiple CoTs, vote",
            "Reflexion - self-critique and revise",
            "ReAct - Reason + Act interleaved",
            "Least-to-Most - decompose complex problems",
            "Plan-and-Solve - plan first, then execute",
            "Skeleton-of-Thought - outline then expand",
        ]
        return SkillResult(
            success=True,
            output=f"[Reasoning] Using {len(strategies)} reasoning strategies.",
            metadata={
                "skill": self.name,
                "strategies": strategies,
                "max_steps": context.metadata.get("max_reasoning_steps", 10),
                "show_work": True,
            },
            suggestions=[
                "Break complex problems into smaller steps",
                "Verify each step before proceeding",
                "Consider alternative hypotheses",
            ],
        )
