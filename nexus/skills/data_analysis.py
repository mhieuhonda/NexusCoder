"""Data Analysis Skill - Phân tích dữ liệu."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class DataAnalysisSkill(Skill):
    """Phân tích dữ liệu: EDA, statistics, visualization, insights."""
    
    category = SkillCategory.DATA
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "analyze", "phân tích", "data", "dữ liệu", "dataset",
        "statistics", "thống kê", "eda", "exploratory",
        "pandas", "numpy", "visualization", "biểu đồ",
        "insights", "pattern", "mẫu", "trend", "xu hướng",
    ]
    
    @property
    def name(self) -> str:
        return "data_analysis"
    
    @property
    def description(self) -> str:
        return (
            "Phân tích dữ liệu: EDA, descriptive/inferential statistics, "
            "data cleaning, visualization, pattern detection, insight extraction."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        analysis_steps = [
            "1. Data loading & schema inspection",
            "2. Missing value analysis & imputation",
            "3. Descriptive statistics (mean, median, std, quartiles)",
            "4. Distribution analysis (histograms, KDE)",
            "5. Correlation analysis (Pearson, Spearman)",
            "6. Outlier detection (IQR, Z-score, Isolation Forest)",
            "7. Group-by analysis & aggregations",
            "8. Time series decomposition (trend, seasonality, residual)",
            "9. Visualization (matplotlib, seaborn, plotly)",
            "10. Insight extraction & recommendations",
        ]
        return SkillResult(
            success=True,
            output=f"[DataAnalysis] {len(analysis_steps)}-step analysis pipeline.",
            metadata={
                "skill": self.name,
                "analysis_steps": analysis_steps,
                "libraries": ["pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly"],
            },
            suggestions=[
                "Always check data quality first",
                "Visualize before modeling",
                "Document assumptions",
            ],
        )
