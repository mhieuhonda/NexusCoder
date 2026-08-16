"""Math Skill - Giải toán đa cấp độ."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class MathSkill(Skill):
    """Giải toán: arithmetic, algebra, calculus, statistics, linear algebra."""
    
    category = SkillCategory.REASONING
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "math", "toán", "calculate", "tính", "equation",
        "phương trình", "solve", "giải", "derivative", "đạo hàm",
        "integral", "tích phân", "matrix", "ma trận",
        "probability", "xác suất", "statistics", "thống kê",
        "algebra", "đại số", "geometry", "hình học",
    ]
    
    @property
    def name(self) -> str:
        return "math_skill"
    
    @property
    def description(self) -> str:
        return (
            "Giải toán đa cấp: arithmetic, algebra, calculus (differential/integral), "
            "linear algebra (matrix ops, eigenvalues), probability, statistics, "
            "discrete math, optimization."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        domains = {
            "arithmetic": "+, -, ×, ÷, %, ^, √, !",
            "algebra": "Linear/Quadratic/Cubic equations, systems, inequalities",
            "calculus": "Limits, derivatives, integrals, series, ODEs, PDEs",
            "linear_algebra": "Vectors, matrices, eigenvalues, SVD, PCA",
            "probability": "Bayes, distributions, expected value, variance",
            "statistics": "Hypothesis testing, regression, ANOVA, confidence intervals",
            "discrete_math": "Combinatorics, graph theory, number theory, logic",
            "optimization": "Linear/nonlinear programming, convex optimization",
            "numerical": "Root finding, interpolation, numerical integration",
        }
        return SkillResult(
            success=True,
            output=f"[MathSkill] Can solve problems in {len(domains)} math domains.",
            metadata={
                "skill": self.name,
                "domains": domains,
                "tools": ["sympy", "numpy", "scipy", "math"],
                "show_steps": True,
            },
            suggestions=[
                "Always verify answer with alternative method",
                "Show step-by-step solution",
                "Check units and dimensional analysis",
            ],
        )
