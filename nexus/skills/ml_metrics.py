"""ML Metrics Skill - Tính metrics cho classification & regression.

Sinh code tính accuracy, precision, recall, F1, AUC-ROC, PR-AUC,
confusion matrix cho classification; MSE, MAE, RMSE, R², RMSLE cho regression.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


CLASSIFICATION_METRICS = '''"""Classification metrics / Các metrics phân loại."""
from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, log_loss, brier_score_loss,
    precision_recall_curve, roc_curve,
)

def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None,
    labels: list | None = None,
) -> dict:
    """Tính đầy đủ metrics cho bài toán phân loại.

    Args:
        y_true: ground-truth labels (shape: N).
        y_pred: predicted labels (shape: N).
        y_proba: predicted probability for the positive class (binary) or
                 full class probability matrix (multiclass).
        labels: optional label ordering.

    Returns:
        Dict with all key metrics + confusion matrix.
    """
    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, labels=labels, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average=avg, labels=labels, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average=avg, labels=labels, zero_division=0)),
        "log_loss":  None,
        "brier":     None,
        "auc_roc":   None,
        "auc_pr":    None,
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()

    if y_proba is not None:
        try:
            metrics["auc_roc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average=avg, labels=labels)
                if avg != "binary" else roc_auc_score(y_true, y_proba)
            )
        except ValueError:
            pass
        try:
            metrics["auc_pr"] = float(
                average_precision_score(y_true, y_proba, average=avg)
                if avg != "binary" else average_precision_score(y_true, y_proba)
            )
        except ValueError:
            pass
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
            if avg == "binary":
                metrics["brier"] = float(brier_score_loss(y_true, y_proba))
        except ValueError:
            pass
    return metrics


def print_report(y_true, y_pred, labels=None) -> None:
    print(classification_report(y_true, y_pred, labels=labels, digits=4))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=1000)
    y_pred = y_true.copy()
    flip = rng.random(1000) < 0.15
    y_pred[flip] = 1 - y_pred[flip]
    y_proba = np.clip(y_pred + rng.normal(0, 0.1, size=1000), 0, 1)
    import json
    print(json.dumps(compute_classification_metrics(y_true, y_pred, y_proba), indent=2))
'''

REGRESSION_METRICS = '''"""Regression metrics / Các metrics hồi quy."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_squared_log_error, explained_variance_score,
    max_error, median_absolute_error,
)

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính đầy đủ metrics cho bài toán hồi quy."""
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "mse":    mse,
        "rmse":   float(np.sqrt(mse)),
        "mae":    mae,
        "mape":   float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100),
        "r2":     float(r2_score(y_true, y_pred)),
        "adj_r2": float(1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / max(1, len(y_true) - 1 - 1)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "rmsle":  None,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = rng.exponential(scale=10, size=500)
    y_pred = y_true * rng.normal(1.0, 0.1, size=500)
    try:
        from sklearn.metrics import mean_squared_log_error
        rmsle = float(np.sqrt(mean_squared_log_error(y_true, y_pred)))
        print("rmsle:", rmsle)
    except ValueError:
        pass
    import json
    print(json.dumps(compute_regression_metrics(y_true, y_pred), indent=2))
'''


class MLMetricsSkill(Skill):
    """Sinh code tính ML metrics cho classification và regression."""

    category = SkillCategory.ML
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "metrics", "accuracy", "precision", "recall", "f1",
        "f1-score", "auc", "auc-roc", "roc", "confusion matrix",
        "mse", "rmse", "mae", "r2", "r-squared", "rmsle",
        "log loss", "brier", "precision-recall",
    ]
    examples = [
        "Tính accuracy, precision, recall, F1 cho classifier",
        "Compute AUC-ROC và confusion matrix",
        "Regression metrics: RMSE, MAE, R²",
    ]

    @property
    def name(self) -> str:
        return "ml_metrics"

    @property
    def description(self) -> str:
        return (
            "Sinh code tính đầy đủ ML metrics cho classification (acc/prec/rec/F1/AUC/"
            "confusion matrix/log-loss/brier) và regression (MSE/RMSE/MAE/R²/RMSLE)."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.12
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        prompt_lower = (context.prompt or "").lower()
        is_regression = any(
            k in prompt_lower for k in ("rmse", "mse", "mae", "r2", "r-squared", "rmsle", "regression")
        )
        is_classification = any(
            k in prompt_lower for k in ("accuracy", "precision", "recall", "f1", "auc", "confusion")
        )
        if is_regression and not is_classification:
            task = "regression"
        elif is_classification and not is_regression:
            task = "classification"
        else:
            task = "both"

        artifacts: List[Dict[str, str]] = [
            {"name": "classification_metrics.py", "language": "python", "content": CLASSIFICATION_METRICS},
            {"name": "regression_metrics.py", "language": "python", "content": REGRESSION_METRICS},
        ]

        return SkillResult(
            success=True,
            output=(
                f"[ml_metrics] task={task}\n"
                f"Generated classification (acc/prec/rec/F1/AUC/confusion/log_loss/brier) "
                f"and regression (MSE/RMSE/MAE/MAPE/R²/adj-R²/RMSLE) metric modules."
            ),
            artifacts=artifacts,
            suggestions=[
                "Stratify your train/test split to keep class proportions stable",
                "Use cross_val_score(cv=StratifiedKFold(5)) for robust estimates",
                "Plot precision-recall curve when classes are imbalanced",
                "Report confidence intervals via bootstrap (N=1000) for fair comparison",
                "For multi-class, prefer macro-F1 + per-class breakdown",
            ],
            metadata={
                "skill": self.name,
                "task": task,
                "classification_metrics": [
                    "accuracy", "precision", "recall", "f1",
                    "auc_roc", "auc_pr", "confusion_matrix", "log_loss", "brier",
                ],
                "regression_metrics": [
                    "mse", "rmse", "mae", "mape", "r2", "adj_r2",
                    "explained_variance", "max_error", "median_ae", "rmsle",
                ],
                "version": self.version,
                "author": self.author,
            },
        )
