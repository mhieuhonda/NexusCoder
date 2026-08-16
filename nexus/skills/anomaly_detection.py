"""Anomaly Detection Skill - IsolationForest / LOF / DBSCAN / statistical.

Sinh code phát hiện bất thường trong dữ liệu: Isolation Forest, Local Outlier
Factor (LOF), One-Class SVM, DBSCAN, Z-score & IQR rule, với visualisation
(PCA scatter + outlier highlight) và đánh giá (precision/recall nếu có ground truth).

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


ANOMALY_CODE = '''"""Anomaly detection toolkit / Bộ phát hiện bất thường."""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support


def iqr_outliers(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    """IQR rule / Quy tắc IQR."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    mask = (x < q1 - k * iqr) | (x > q3 + k * iqr)
    return mask


def zscore_outliers(x: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Z-score rule / Quy tắc Z-score."""
    return np.abs((x - x.mean()) / (x.std(ddof=0) + 1e-9)) > threshold


def isolation_forest(
    X: np.ndarray, contamination: float = 0.05, random_state: int = 42,
) -> Dict[str, object]:
    """Isolation Forest — mạnh với high-dimensional, không cần assumption."""
    model = IsolationForest(
        n_estimators=300, max_samples="auto",
        contamination=contamination, random_state=random_state, n_jobs=-1,
    )
    labels = model.fit_predict(X)             # 1=inlier, -1=outlier
    scores = -model.decision_function(X)     # higher = more anomalous
    return {"model": model, "labels": labels, "scores": scores}


def local_outlier_factor(X: np.ndarray, n_neighbors: int = 20) -> Dict[str, object]:
    """LOF — phát hiện anomaly dựa trên mật độ cục bộ."""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto", n_jobs=-1)
    labels = lof.fit_predict(X)
    scores = -lof.negative_outlier_factor_
    return {"labels": labels, "scores": scores}


def one_class_svm(X: np.ndarray, nu: float = 0.05) -> Dict[str, object]:
    """One-Class SVM — tốt cho novelty detection khi train chỉ có normal."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    labels = model.fit_predict(Xs)
    scores = -model.decision_function(Xs)
    return {"model": model, "scaler": scaler, "labels": labels, "scores": scores}


def dbscan_outliers(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """DBSCAN — điểm không thuộc cụm nào (-1) là anomaly."""
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(StandardScaler().fit_transform(X))
    return db.labels_ == -1


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Đánh giá khi có ground-truth (-1 = outlier, 1 = inlier)."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=-1, zero_division=0,
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def visualize(X: np.ndarray, labels: np.ndarray, title: str = "Anomalies"):
    """PCA scatter 2D với outliers highlight / Vẽ PCA 2D."""
    import matplotlib.pyplot as plt
    X2 = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    plt.figure(figsize=(8, 5))
    plt.scatter(X2[labels == 1, 0], X2[labels == 1, 1], s=8, c="steelblue", label="inlier")
    plt.scatter(X2[labels == -1, 0], X2[labels == -1, 1], s=18, c="crimson", label="anomaly")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    X = rng.normal(size=(1000, 4))
    X[:20] += rng.normal(5, 1, size=(20, 4))  # inject 20 anomalies
    res = isolation_forest(X, contamination=0.05)
    print("outliers detected:", (res["labels"] == -1).sum())
'''

MODEL_SELECTION = """
Anomaly Detection — Model Selection Guide / Hướng dẫn chọn mô hình
==================================================================
| Method            | Best For                          | Notes                              |
|-------------------|-----------------------------------|------------------------------------|
| IsolationForest   | High-D, mixed-type, scalable      | Default first choice              |
| LOF (kNN-density)  | Local anomalies, low-D            | Slow for large N (O(n²))          |
| One-Class SVM     | Novelty detection (train=normal)  | Sensitive to scaling & gamma      |
| DBSCAN            | Cluster-based anomalies           | Needs eps tuning (k-distance plot)|
| Z-score / IQR     | Univariate, explainable baseline | Fails on multi-modal distributions|
| Autoencoder       | Non-linear high-D, large data     | Requires deep-learning infra      |

Contamination: prior estimate of anomaly ratio. Tune via:
  - Domain knowledge (e.g. fraud rate = 0.1%)
  - Top-K approach: take top-k highest scores as anomalies
  - Score histogram: visual knee / elbow in distribution
"""


class AnomalyDetectionSkill(Skill):
    """Sinh toolkit phát hiện anomaly cho tabular data."""

    category = SkillCategory.ML
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "anomaly", "anomalies", "outlier", "outliers", "isolation forest",
        "isolationforest", "lof", "local outlier", "one-class svm",
        "dbscan", "z-score", "iqr", "novelty detection", "fraud",
    ]
    examples = [
        "Detect anomalies in sensor data với IsolationForest",
        "Phát hiện outlier dùng LOF",
        "Setup fraud detection pipeline",
    ]

    @property
    def name(self) -> str:
        return "anomaly_detection"

    @property
    def description(self) -> str:
        return (
            "Sinh code phát hiện bất thường: IsolationForest, LOF, One-Class SVM, "
            "DBSCAN, Z-score/IQR baseline + PCA visualization + evaluation."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.14
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        prompt_lower = (context.prompt or "").lower()
        if "lof" in prompt_lower:
            recommended = "lof"
        elif "svm" in prompt_lower:
            recommended = "one_class_svm"
        elif "dbscan" in prompt_lower:
            recommended = "dbscan"
        elif "iqr" in prompt_lower or "z-score" in prompt_lower or "zscore" in prompt_lower:
            recommended = "statistical"
        else:
            recommended = "isolation_forest"

        artifacts: List[Dict[str, str]] = [
            {"name": "anomaly_toolkit.py", "language": "python", "content": ANOMALY_CODE},
            {"name": "MODEL_SELECTION.md", "language": "markdown", "content": MODEL_SELECTION},
        ]

        return SkillResult(
            success=True,
            output=(
                f"[anomaly_detection] recommended={recommended}\n"
                f"Generated toolkit with 5 detectors + PCA viz + evaluation harness."
            ),
            artifacts=artifacts,
            suggestions=[
                "Plot score distribution to choose contamination threshold",
                "Always scale features (StandardScaler / RobustScaler) before fitting",
                "Combine unsupervised scores with rule-based features for fraud",
                "Track precision@k instead of recall when ground-truth is partial",
                "Retrain periodically — anomaly patterns drift over time",
            ],
            metadata={
                "skill": self.name,
                "recommended_model": recommended,
                "models_available": [
                    "isolation_forest", "lof", "one_class_svm",
                    "dbscan", "z_score", "iqr",
                ],
                "version": self.version,
                "author": self.author,
            },
        )
