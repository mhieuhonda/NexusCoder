"""
Skill Registry v0.3 - Đăng ký và quản lý skills
================================================
Central registry cho tất cả skills. Hỗ trợ:
- Auto-discovery skills (v0.3: introspects the skills/ directory)
- Routing prompt → best skill
- Fallback handling

v0.3: вместо hardcoded import list, we use dynamic discovery.
Just drop a `<name>.py` file with a `Skill` subclass and it auto-registers.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
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
            # Silently overwrite (don't raise — useful for hot reload)
            pass
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
        """Tìm skill phù hợp nhất cho prompt."""
        scores: List[Tuple[float, str, int]] = []
        priority_weight = {
            SkillPriority.CRITICAL: 1.3,
            SkillPriority.HIGH: 1.15,
            SkillPriority.MEDIUM: 1.0,
            SkillPriority.LOW: 0.85,
        }
        for idx, (name, skill) in enumerate(self._skills.items()):
            try:
                base_score = skill.can_handle(prompt, context)
            except Exception:
                base_score = 0.0
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
            try:
                score = skill.can_handle(prompt, context)
            except Exception:
                score = 0.0
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
    """Auto-register tất cả built-in skills via dynamic module discovery.

    v0.3: scans nexus/skills/*.py, imports each module, finds all `Skill`
    subclasses, instantiates them, and registers. No more manual edits needed
    when adding a new skill file.
    """
    import nexus.skills as skills_pkg
    # Known alias map for v0.3 skills (so we can preserve aliases from v0.2)
    _ALIASES: Dict[str, List[str]] = {
        "code_generation": ["codegen", "write_code"],
        "code_review": ["review", "review_code"],
        "code_refactor": ["refactor"],
        "debugging": ["debug", "fix_bug"],
        "documentation": ["docs", "document"],
        "testing": ["test", "unit_test"],
        "algorithm_design": ["algo", "algorithm"],
        "data_analysis": ["analyze", "data"],
        "translation": ["translate", "trans"],
        "summarization": ["summarize", "summary"],
        "reasoning": ["reason", "think"],
        "math_skill": ["math", "calculate"],
        "sql_generation": ["sql", "query"],
        "security_audit": ["security", "audit"],
        "performance_opt": ["optimize", "perf"],
        "graphql_skill": ["graphql"],
        "ml_metrics": ["metrics"],
        "ml_evaluation": ["evaluate"],
        "ml_training": ["train_ml"],
        "ml_inference": ["infer_ml"],
        "ml_data_preprocessing": ["preprocess"],
        "ml_feature_engineering": ["feature_eng"],
        "ml_hyperparameter_tuning": ["tune"],
        "ml_model_explainability": ["explain_ml"],
        "ml_model_selection": ["select_model"],
        "ml_data_preprocessing": ["preprocess_data"],
        "blockchain_audit": ["blockchain"],
        "system_design": ["design"],
        "cloud_deploy": ["deploy"],
        "devops_skill": ["devops"],
        "data_pipeline": ["etl"],
        "monitoring": ["monitor"],
        "logging_analytics": ["logs"],
        "ci_cd_pipeline": ["ci_cd"],
        "release_management": ["release"],
        "bug_reproduction": ["repro"],
        "caching_strategy": ["cache"],
        "statistical_analysis": ["stats"],
        "time_series_forecasting": ["forecast"],
        "anomaly_detection": ["anomaly"],
        "clustering_analysis": ["cluster"],
        "classification_automation": ["classify"],
        "knowledge_graph": ["kg"],
        "sentiment_analysis": ["sentiment"],
        "topic_modeling": ["topic"],
        "language_detection": ["detect_lang"],
        "creative_writing": ["write_creative"],
        "prompt_engineering": ["prompt"],
        "regex_master": ["regex"],
        "shell_scripting": ["bash_skill"],
        "api_design": ["api"],
        "microservices": ["microsvc"],
        "code_translation": ["transpile"],
        "code_completion": ["complete"],
        "code_explanation": ["explain"],
        "code_minification": ["minify_code"],
        "code_documentation_generation": ["gen_docs"],
        "code_duplication_detection": ["dup"],
        "code_dead_code_analysis": ["dead_code"],
        "code_complexity_analysis": ["complexity"],
        "code_dependency_analysis": ["deps"],
    }

    for module_info in pkgutil.iter_modules(skills_pkg.__path__):
        if module_info.name.startswith("_") or module_info.name in ("base", "registry"):
            continue
        module_name = f"nexus.skills.{module_info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        # Find all Skill subclasses in the module
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if not (inspect.isclass(obj) and issubclass(obj, Skill) and obj is not Skill):
                continue
            # Skip if it's imported from another module
            if obj.__module__ != module_name:
                continue
            try:
                instance = obj()
                aliases = _ALIASES.get(instance.name, [])
                registry.register(instance, aliases=aliases)
            except Exception:
                continue
