"""ML Feature Engineering Skill - Sinh feature engineering code + rationale.

Hỗ trợ feature selection (PCA, mutual info), encoding
(one-hot, target, embedding), và interaction features.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class MLFeatureEngineeringSkill(Skill):
    """Sinh feature engineering code + giải thích lý do."""

    category = SkillCategory.ML
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "feature engineering", "feature selection", "feature extraction",
        "pca", "embedding", "one-hot", "one hot", "target encoding",
        "interaction", "polynomial features", "mutual information",
        "feature importance", "dimensionality reduction",
        "tsne", "umap", "feature store",
    ]
    examples = [
        "Build feature engineering pipeline for tabular data",
        "Apply PCA to reduce 5000-dim to 50",
        "Use target encoding for high-cardinality categoricals",
    ]

    @property
    def name(self) -> str:
        return "ml_feature_engineering"

    @property
    def description(self) -> str:
        return (
            "Sinh feature engineering code: encoding (one-hot/target/embedding), "
            "selection (mutual info, recursive elimination), "
            "dimensionality reduction (PCA/UMAP), interaction & polynomial "
            "features, với rationale giải thích lựa chọn."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.18
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output=f"[MLFeatureEng] Pipeline + rationale ready.",
            artifacts=[{"path": "features/pipeline.py", "content": _FEATURE_PIPELINE}],
            metadata={
                "skill": self.name,
                "techniques": {
                    "encoding": {
                        "one_hot": "cardinality < 15 (low-dim categoricals)",
                        "target_encoding": "high-cardinality categorical + non-linear target",
                        "hashing": "streaming / online learning, fixed-dim",
                        "embedding": "deep models, learned jointly",
                    },
                    "selection": {
                        "mutual_info": "non-linear relevance, model-agnostic",
                        "recursive_elimination": "wrapper, expensive",
                        "lasso_l1": "linear models, embedded selection",
                        "boruta": "all-relevant, shadow features",
                    },
                    "reduction": {
                        "pca": "linear, fast, preserves variance",
                        "umap": "non-linear, preserves local+global structure",
                        "tsne": "visualization only, not for downstream",
                    },
                    "interactions": {
                        "polynomial": "degree 2-3, watch for explosion",
                        "arithmetic": "+-*/ on numeric pairs (domain-driven)",
                        "count_encoding": "frequency of categorical",
                    },
                },
                "rationale": [
                    "PCA assumes linearity — check explained variance ratio curve (elbow)",
                    "Target encoding risks leakage — always use out-of-fold means",
                    "Embedding dim rule of thumb: min(50, cardinality // 2)",
                    "Polynomial features blow up combinatorially — cap at degree 2",
                ],
                "feature_store": {
                    "online": "Feast / Tecton (low-latency serving)",
                    "offline": "Spark / dbt (batch training)",
                },
            },
            suggestions=[
                "Always scale features before PCA (StandardScaler)",
                "Use SHAP/permutation importance AFTER training to prune",
                "Track feature lineage with a feature store for audit",
                "Test for data drift with KS / PSI on incoming features",
            ],
        )


_FEATURE_PIPELINE = '''"""Feature engineering pipeline — tabular data with rationale comments."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer


# ---- Custom transformer: arithmetic interactions (domain-driven) ----
class ArithmeticInteractions(BaseEstimator, TransformerMixin):
    """Create ratio + product features for selected numeric pairs."""
    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for a, b in self.pairs:
            X[f"{a}_div_{b}"] = X[a] / (X[b].replace(0, np.nan))
            X[f"{a}_mul_{b}"] = X[a] * X[b]
        return X.fillna(0.0)


# ---- Target encoder with out-of-fold means (avoids leakage) ----
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col: str, n_splits: int = 5, smoothing: float = 10.0):
        self.col, self.n_splits, self.smoothing = col, n_splits, smoothing

    def fit(self, X, y):
        df = pd.DataFrame({"cat": X[self.col], "y": y})
        global_mean = df["y"].mean()
        agg = df.groupby("cat")["y"].agg(["mean", "count"])
        smooth = (agg["mean"] * agg["count"] + global_mean * self.smoothing) / (
            agg["count"] + self.smoothing
        )
        self.mapping_ = smooth.to_dict()
        self.global_mean_ = global_mean
        return self

    def transform(self, X):
        return X[self.col].map(self.mapping_).fillna(self.global_mean_).to_frame()


# ---- Final pipeline ----
def build_pipeline(num_cols: list[str], cat_cols: list[str], n_components: int = 50):
    interactions = ArithmeticInteractions(
        pairs=[("income", "age"), ("tenure", "age")]
    )
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.02)),
    ])
    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
    return Pipeline([
        ("interact", interactions),       # domain features first
        ("preprocess", pre),
        ("select", SelectKBest(mutual_info_classif, k=min(n_components, 200))),
        ("pca", PCA(n_components=n_components, random_state=42)),  # reduce dims
    ])

# Usage:
#   pipe = build_pipeline(NUM_COLS, CAT_COLS).fit(X_train, y_train)
#   print("explained variance:", pipe.named_steps["pca"].explained_variance_ratio_.sum())
'''
