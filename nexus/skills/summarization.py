"""Summarization Skill - Tóm tắt văn bản."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class SummarizationSkill(Skill):
    """Tóm tắt văn bản: extractive, abstractive, key points."""
    
    category = SkillCategory.LANGUAGE
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "summarize", "tóm tắt", "summary", "abstract",
        "brief", "ngắn gọn", "key points", "điểm chính",
        "tl;dr", "condense", "rút gọn",
    ]
    
    @property
    def name(self) -> str:
        return "summarization"
    
    @property
    def description(self) -> str:
        return (
            "Tóm tắt văn bản đa dạng: extractive (chọn câu quan trọng), "
            "abstractive (viết lại), bullet points, executive summary."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        methods = [
            "extractive (TextRank, LexRank)",
            "abstractive (seq2seq, BART, T5)",
            "key phrase extraction (RAKE, YAKE)",
            "topic modeling (LDA, BERTopic)",
            "hierarchical (document → section → paragraph)",
            "query-focused (tóm tắt theo câu hỏi)",
        ]
        return SkillResult(
            success=True,
            output=f"[Summarization] {len(methods)} methods available.",
            metadata={
                "skill": self.name,
                "methods": methods,
                "output_formats": ["paragraph", "bullets", "tldr", "executive_summary"],
                "length_options": ["short", "medium", "long"],
            },
            suggestions=[
                "Specify desired length",
                "Indicate if technical or general audience",
            ],
        )
