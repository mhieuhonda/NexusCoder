"""
Skill Registry - Đăng ký và quản lý skills
===========================================
Central registry cho tất cả skills. Hỗ trợ:
- Auto-discovery skills
- Routing prompt → best skill
- Fallback handling
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .base import Skill, SkillContext, SkillPriority


class SkillRegistry:
    """Registry quản lý tất cả skills của Nexus Coder."""
    
    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._aliases: Dict[str, str] = {}
    
    def register(self, skill: Skill, aliases: List[str] = None) -> None:
        """Đăng ký một skill mới."""
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' đã tồn tại")
        self._skills[skill.name] = skill
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = skill.name
    
    def unregister(self, name: str) -> Optional[Skill]:
        """Gỡ skill khỏi registry."""
        return self._skills.pop(name, None)
    
    def get(self, name: str) -> Optional[Skill]:
        """Lấy skill theo tên hoặc alias."""
        name = name.lower()
        if name in self._aliases:
            name = self._aliases[name]
        return self._skills.get(name)
    
    def list_skills(self) -> List[str]:
        """Danh sách tên tất cả skills."""
        return sorted(self._skills.keys())
    
    def list_by_category(self) -> Dict[str, List[str]]:
        """Group skills theo category."""
        groups = defaultdict(list)
        for name, skill in self._skills.items():
            groups[skill.category.value].append(name)
        return dict(groups)
    
    def route(self, prompt: str, context: SkillContext = None) -> Optional[Skill]:
        """Tìm skill phù hợp nhất cho prompt.
        
        Returns:
            Skill có confidence cao nhất, hoặc None nếu không có skill nào > 0.1.
        """
        scores: List[Tuple[float, str, int]] = []
        priority_weight = {
            SkillPriority.CRITICAL: 1.3,
            SkillPriority.HIGH: 1.15,
            SkillPriority.MEDIUM: 1.0,
            SkillPriority.LOW: 0.85,
        }
        for idx, (name, skill) in enumerate(self._skills.items()):
            base_score = skill.can_handle(prompt, context)
            weighted = base_score * priority_weight.get(skill.priority, 1.0)
            if weighted > 0.1:
                scores.append((weighted, name, idx))
        
        if not scores:
            return None
        
        scores.sort(key=lambda x: (-x[0], x[2]))
        return self._skills[scores[0][1]]
    
    def route_top_k(self, prompt: str, k: int = 3, context: SkillContext = None) -> List[Skill]:
        """Trả về top-k skills phù hợp nhất."""
        scored = []
        for name, skill in self._skills.items():
            score = skill.can_handle(prompt, context)
            if score > 0.1:
                scored.append((score, name))
        scored.sort(key=lambda x: -x[0])
        return [self._skills[n] for _, n in scored[:k]]
    
    def __len__(self) -> int:
        return len(self._skills)
    
    def __contains__(self, name: str) -> bool:
        return name in self._skills
    
    def __repr__(self) -> str:
        return f"<SkillRegistry: {len(self._skills)} skills>"


# =============================================================================
# Global registry singleton
# =============================================================================

_GLOBAL_REGISTRY: Optional[SkillRegistry] = None


def get_global_registry() -> SkillRegistry:
    """Lấy global registry (auto-init nếu chưa có)."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = SkillRegistry()
        _auto_register_defaults(_GLOBAL_REGISTRY)
    return _GLOBAL_REGISTRY


def _auto_register_defaults(registry: SkillRegistry) -> None:
    """Auto-register tất cả built-in skills."""
    try:
        from .code_generation import CodeGenerationSkill
        registry.register(CodeGenerationSkill(), aliases=["codegen", "write_code"])
    except ImportError:
        pass
    try:
        from .code_review import CodeReviewSkill
        registry.register(CodeReviewSkill(), aliases=["review", "review_code"])
    except ImportError:
        pass
    try:
        from .code_refactor import CodeRefactorSkill
        registry.register(CodeRefactorSkill(), aliases=["refactor"])
    except ImportError:
        pass
    try:
        from .debugging import DebuggingSkill
        registry.register(DebuggingSkill(), aliases=["debug", "fix_bug"])
    except ImportError:
        pass
    try:
        from .documentation import DocumentationSkill
        registry.register(DocumentationSkill(), aliases=["docs", "document"])
    except ImportError:
        pass
    try:
        from .testing import TestingSkill
        registry.register(TestingSkill(), aliases=["test", "unit_test"])
    except ImportError:
        pass
    try:
        from .algorithm_design import AlgorithmDesignSkill
        registry.register(AlgorithmDesignSkill(), aliases=["algo", "algorithm"])
    except ImportError:
        pass
    try:
        from .data_analysis import DataAnalysisSkill
        registry.register(DataAnalysisSkill(), aliases=["analyze", "data"])
    except ImportError:
        pass
    try:
        from .translation import TranslationSkill
        registry.register(TranslationSkill(), aliases=["translate", "trans"])
    except ImportError:
        pass
    try:
        from .summarization import SummarizationSkill
        registry.register(SummarizationSkill(), aliases=["summarize", "summary"])
    except ImportError:
        pass
    try:
        from .reasoning import ReasoningSkill
        registry.register(ReasoningSkill(), aliases=["reason", "think"])
    except ImportError:
        pass
    try:
        from .math_skill import MathSkill
        registry.register(MathSkill(), aliases=["math", "calculate"])
    except ImportError:
        pass
    try:
        from .sql_generation import SQLGenerationSkill
        registry.register(SQLGenerationSkill(), aliases=["sql", "query"])
    except ImportError:
        pass
    try:
        from .security_audit import SecurityAuditSkill
        registry.register(SecurityAuditSkill(), aliases=["security", "audit"])
    except ImportError:
        pass
    try:
        from .performance_opt import PerformanceOptimizationSkill
        registry.register(PerformanceOptimizationSkill(), aliases=["optimize", "perf"])
    except ImportError:
        pass
