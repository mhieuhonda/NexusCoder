"""Clustering Analysis Skill - KMeans / DBSCAN / Hierarchical pipeline.

Sinh pipeline clustering hoàn chỉnh: feature scaling, elbow + silhouette
chọn K, fit KMeans / DBSCAN / Agglomerative, đánh giá (silhouette, Davies-Bouldin,
Calinski-Harabasz), và visualization (PCA 2D scatter + dendrogram).

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


CLUSTERING_CODE = '''"""Clustering pipeline / Pipeline phân cụm."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram


def scale(X: np.ndarray) -> np.ndarray:
    """Robust scaling cho clustering / Chuẩn hóa trước khi cluster."""
    return StandardScaler().fit_transform(X)


def find_best_k(X: np.ndarray, k_range: range = range(2, 11)) -> Tuple[int, Dict[int, float]]:
    """Elbow + silhouette để chọn K / Chọn K bằng silhouette."""
    scores: Dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        scores[k] = float(silhouette_score(X, km.labels_))
    best_k = max(scores, key=scores.get)
    return best_k, scores


def kmeans_cluster(X: np.ndarray, k: int, random_state: int = 42) -> Dict[str, object]:
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return {
        "model": model, "labels": labels,
        "centroids": model.cluster_centers_,
        "inertia": float(model.inertia_),
        "silhouette": float(silhouette_score(X, labels)),
    }


def dbscan_cluster(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Dict[str, object]:
    """DBSCAN — auto-detect số cụm, đánh dấu noise (-1)."""
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    # Silhouette chỉ tính khi có ≥ 2 cụm thực sự
    sil = float(silhouette_score(X, labels)) if n_clusters >= 2 else None
    return {
        "model": model, "labels": labels,
        "n_clusters": n_clusters, "n_noise": n_noise,
        "silhouette": sil,
    }


def agglomerative_cluster(X: np.ndarray, k: int, linkage: str = "ward") -> Dict[str, object]:
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)
    return {
        "model": model, "labels": labels,
        "silhouette": float(silhouette_score(X, labels)),
    }


def gaussian_mixture_cluster(X: np.ndarray, k: int, random_state: int = 42) -> Dict[str, object]:
    """GMM — soft clustering, trả về probabilities / Phân cụm mềm."""
    model = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return {
        "model": model, "labels": labels,
        "proba": model.predict_proba(X),
        "bic": float(model.bic(X)),
        "aic": float(model.aic(X)),
        "silhouette": float(silhouette_score(X, labels)),
    }


def evaluate(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Đánh giá clustering khi không có ground-truth."""
    if len(set(labels)) < 2:
        return {"silhouette": -1.0, "davies_bouldin": float("inf"), "calinski_harabasz": 0.0}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),         # lower = better
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),  # higher = better
    }


def visualize_pca(X: np.ndarray, labels: np.ndarray, title: str = "Clusters (PCA 2D)"):
    """PCA 2D scatter tô màu theo cluster / Vẽ PCA."""
    import matplotlib.pyplot as plt
    X2 = PCA(n_components=2).fit_transform(scale(X))
    plt.figure(figsize=(8, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", s=12, alpha=0.8)
    plt.colorbar(label="cluster")
    plt.title(title)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    return plt.gcf()


def plot_dendrogram(X: np.ndarray, method: str = "ward"):
    import matplotlib.pyplot as plt
    Z = linkage(scale(X), method=method)
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="level", p=5)
    plt.title(f"Hierarchical Dendrogram ({method})")
    plt.tight_layout()
    return plt.gcf()
'''

STRATEGY_GUIDE = """
Clustering Strategy Guide / Hướng dẫn chiến lược clustering
============================================================
1. Preprocess:
   - Handle missing values & encode categoricals (OneHot / TargetEncoder).
   - Scale features (StandardScaler / RobustScaler) — clustering is distance-based.
   - For high-D: reduce first (PCA / UMAP) to combat curse of dimensionality.

2. Choose algorithm:
   - KMeans        : spherical clusters, large N, K known
   - GMM           : elliptical clusters, need soft assignment
   - DBSCAN        : arbitrary shapes, density-aware, auto-K, robust to noise
   - Agglomerative: small N, want dendrogram / hierarchy
   - HDBSCAN       : variable-density clusters (better than DBSCAN)
   - Spectral      : graph-based, non-convex clusters

3. Select K (when needed):
   - Elbow on inertia
   - Silhouette score (maximize)
   - Gap statistic
   - Davies-Bouldin (minimize) / Calinski-Harabasz (maximize)
   - Domain interpretation

4. Evaluate (unsupervised):
   - Silhouette ∈ [-1, 1] — higher = better separated
   - Davies-Bouldin — lower = better
   - Calinski-Harabasz — higher = better

5. Interpret:
   - Profile each cluster (mean per feature)
   - Visualize via PCA / t-SNE / UMAP 2D projection
"""


class ClusteringAnalysisSkill(Skill):
    """Sinh clustering pipeline (KMeans/DBSCAN/Hierarchical/GMM) + viz + eval."""

    category = SkillCategory.ML
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "cluster", "clustering", "kmeans", "k-means", "dbscan", "hdbscan",
        "hierarchical", "agglomerative", "gmm", "gaussian mixture",
        "silhouette", "segmentation", "kmeans++",
    ]
    examples = [
        "Cluster customers với KMeans",
        "Tìm số cụm tối ưu bằng silhouette",
        "DBSCAN để detect clusters có hình dạng bất kỳ",
    ]

    @property
    def name(self) -> str:
        return "clustering_analysis"

    @property
    def description(self) -> str:
        return (
            "Sinh pipeline clustering: scaling, KMeans/DBSCAN/Agglomerative/GMM, "
            "chọn K (elbow + silhouette), đánh giá (silhouette/DB/CH) + PCA viz."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.13
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        prompt_lower = (context.prompt or "").lower()
        if "dbscan" in prompt_lower or "hdbscan" in prompt_lower:
            recommended = "dbscan"
        elif "hierarchical" in prompt_lower or "agglomerative" in prompt_lower or "dendrogram" in prompt_lower:
            recommended = "agglomerative"
        elif "gmm" in prompt_lower or "gaussian mixture" in prompt_lower:
            recommended = "gmm"
        else:
            recommended = "kmeans"

        artifacts: List[Dict[str, str]] = [
            {"name": "clustering_pipeline.py", "language": "python", "content": CLUSTERING_CODE},
            {"name": "CLUSTERING_STRATEGY.md", "language": "markdown", "content": STRATEGY_GUIDE},
        ]

        return SkillResult(
            success=True,
            output=(
                f"[clustering_analysis] recommended={recommended}\n"
                f"Generated pipeline: scaling, K-selection, 4 algorithms, "
                f"3 evaluation metrics + PCA/dendrogram viz."
            ),
            artifacts=artifacts,
            suggestions=[
                "Always scale features before clustering (distance-based methods)",
                "For high-D data, try PCA / UMAP first to reduce dimensions",
                "Profile each cluster (mean per feature) to give business meaning",
                "Compare KMeans vs HDBSCAN — HDBSCAN handles variable density better",
                "Visualize with both PCA (preserve variance) AND t-SNE/UMAP (preserve locality)",
            ],
            metadata={
                "skill": self.name,
                "recommended_algorithm": recommended,
                "algorithms_available": ["kmeans", "dbscan", "agglomerative", "gmm"],
                "evaluation_metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
                "version": self.version,
                "author": self.author,
            },
        )
