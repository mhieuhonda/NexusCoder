"""Statistical Analysis Skill - Hypothesis testing, distributions, ANOVA.

Sinh code với scipy / statsmodels: descriptive stats, t-test (one/two/paired),
ANOVA (one-way + two-way), chi-square, normality tests, distribution fitting,
và correlation analysis.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


STATS_CODE = '''"""Statistical analysis toolkit / Bộ công cụ phân tích thống kê."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats với skewness/kurtosis / Thống kê mô tả."""
    summ = df.describe()
    summ.loc["skew"] = df.skew()
    summ.loc["kurtosis"] = df.kurtosis()
    summ.loc["sem"] = df.sem()
    return summ


def normality_test(x: np.ndarray) -> Tuple[float, float, str]:
    """Shapiro-Wilk + D'Agostino K² để kiểm tra tính phân phối chuẩn."""
    if len(x) > 5000:
        # Shapiro unstable cho mẫu lớn → dùng K²
        stat, p = stats.normaltest(x)
        test = "DAgostino_K2"
    else:
        stat, p = stats.shapiro(x)
        test = "Shapiro-Wilk"
    verdict = "normal" if p > 0.05 else "non-normal"
    return float(stat), float(p), f"{test} -> {verdict} (p={p:.4f})"


def ttest_two_sample(a: np.ndarray, b: np.ndarray) -> dict:
    """Kiểm định t hai mẫu (Welch's t-test nếu variance không bằng nhau)."""
    lev_stat, lev_p = stats.levene(a, b)
    equal_var = lev_p > 0.05
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=equal_var)
    return {
        "test": "t-test (independent)",
        "equal_variance": equal_var,
        "levene_p": float(lev_p),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "reject_H0_alpha_0.05": p_val < 0.05,
    }


def paired_ttest(before: np.ndarray, after: np.ndarray) -> dict:
    t_stat, p_val = stats.ttest_rel(before, after)
    return {
        "test": "paired t-test",
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "reject_H0_alpha_0.05": p_val < 0.05,
    }


def one_way_anova(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """One-way ANOVA + Tukey HSD post-hoc."""
    groups = [g[value_col].values for _, g in df.groupby(group_col)]
    f_stat, p_val = stats.f_oneway(*groups)
    tukey = pairwise_tukeyhsd(df[value_col], df[group_col], alpha=0.05)
    return {
        "test": "one-way ANOVA",
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "reject_H0_alpha_0.05": p_val < 0.05,
        "tukey_summary": str(tukey.summary()),
    }


def two_way_anova(df: pd.DataFrame, formula: str) -> dict:
    """Two-way ANOVA bằng OLS formula, vd: 'y ~ C(A) + C(B) + C(A):C(B)'."""
    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    return {
        "test": "two-way ANOVA",
        "formula": formula,
        "table": table.to_dict(),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
    }


def chi_square(contingency_table: np.ndarray) -> dict:
    """Kiểm định chi-square cho bảng phân chéo."""
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return {
        "test": "chi-square independence",
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "reject_H0_alpha_0.05": p < 0.05,
        "cramers_v": float(np.sqrt(chi2 / (contingency_table.sum() * (min(contingency_table.shape) - 1)))),
    }


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Pearson / Spearman / Kendall correlation matrix."""
    return df.select_dtypes(include="number").corr(method=method)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    a = rng.normal(5, 2, 100)
    b = rng.normal(5.5, 2, 100)
    print(normality_test(a))
    print(ttest_two_sample(a, b))
    print(one_way_anova(
        pd.DataFrame({"y": np.concatenate([a, b, rng.normal(6, 2, 100)]),
                      "g": ["A"]*100 + ["B"]*100 + ["C"]*100}),
        "y", "g",
    ))
'''

DECISION_GUIDE = """
Statistical Test Decision Guide / Hướng dẫn chọn test
======================================================
Goal                                   | Test
---------------------------------------|------------------------------
Compare 1 sample to a known mean        | one-sample t-test
Compare means of 2 independent groups   | Welch's t-test (default, unequal var)
Compare means of 2 paired samples       | paired t-test
Compare means of 3+ independent groups  | one-way ANOVA + Tukey HSD
Compare 2 factors simultaneously        | two-way ANOVA
Test association between 2 categorical  | chi-square + Cramér's V
Test correlation between 2 continuous   | Pearson (linear) / Spearman (rank)
Test if data is normally distributed    | Shapiro-Wilk (n<5000) / D'Agostino K² (n>=5000)
Compare distributions (non-parametric)  | Mann-Whitney U / Kruskal-Wallis

Effect Size: Cohen's d (t-test), eta-squared (ANOVA), Cramér's V (chi-square)
Alpha convention: 0.05; consider Bonferroni / FDR correction for multiple tests.
"""


class StatisticalAnalysisSkill(Skill):
    """Sinh code phân tích thống kê với scipy + statsmodels."""

    category = SkillCategory.DATA
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "statistics", "statistical", "hypothesis", "p-value", "pvalue",
        "t-test", "ttest", "anova", "chi-square", "chisquare",
        "distribution", "normality", "shapiro", "correlation",
        "significance", "confidence interval",
    ]
    examples = [
        "Run a two-sample t-test between groups A and B",
        "Perform one-way ANOVA with Tukey post-hoc",
        "Check if my data is normally distributed",
    ]

    @property
    def name(self) -> str:
        return "statistical_analysis"

    @property
    def description(self) -> str:
        return (
            "Sinh toolkit thống kê: descriptive stats, t-test (Welch/paired), "
            "ANOVA (1-way + 2-way + Tukey), chi-square, normality, correlation."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.12
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        artifacts: List[Dict[str, str]] = [
            {"name": "stats_toolkit.py", "language": "python", "content": STATS_CODE},
            {"name": "TEST_SELECTION_GUIDE.md", "language": "markdown", "content": DECISION_GUIDE},
        ]

        return SkillResult(
            success=True,
            output=(
                "[statistical_analysis] Generated toolkit: descriptive stats, "
                "Welch/paired t-test, 1-way & 2-way ANOVA (+Tukey), chi-square, "
                "normality, correlation + test selection guide."
            ),
            artifacts=artifacts,
            suggestions=[
                "Always visualize distributions (histogram/QQ-plot) before parametric tests",
                "Apply Bonferroni or FDR (Benjamini-Hochberg) when running multiple comparisons",
                "Report effect sizes alongside p-values (Cohen's d / η² / Cramér's V)",
                "For small samples (n<30), prefer non-parametric (Mann-Whitney / Wilcoxon)",
                "Bootstrap (N=10000) for robust confidence intervals when assumptions fail",
            ],
            metadata={
                "skill": self.name,
                "tests_included": [
                    "shapiro-wilk", "D'Agostino K2", "Welch t-test", "paired t-test",
                    "one-way ANOVA", "Tukey HSD", "two-way ANOVA",
                    "chi-square", "Cramér's V", "Pearson/Spearman/Kendall",
                ],
                "alpha": 0.05,
                "version": self.version,
                "author": self.author,
            },
        )
