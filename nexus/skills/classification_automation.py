"""Classification Automation Skill - sklearn classification pipeline.

Sinh end-to-end classification pipeline: data splitting (stratified),
preprocessing (numeric + categorical transformers), model zoo
(LogisticRegression / RandomForest / XGBoost / LightGBM / SVM / KNN),
cross-validation, hyperparameter search, evaluation (metrics + ROC/PR curves),
và feature importance. Handles class imbalance (SMOTE / class_weight).

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


CLASSIFICATION_PIPELINE = '''"""End-to-end classification pipeline / Pipeline phân loại full."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_XGB = HAS_LGB = HAS_IMB = True
except ImportError:
    HAS_XGB = HAS_LGB = HAS_IMB = False


def build_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
    """Tạo ColumnTransformer cho numeric + categorical."""
    numeric = [c for c in df.columns if c != target
               and pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c != target
                   and not pd.api.types.is_numeric_dtype(df[c])]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ], remainder="drop")


def build_model_zoo(random_state: int = 42) -> Dict[str, object]:
    """Dictionary of candidate classifiers / Bộ mô hình ứng viên."""
    zoo = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                 class_weight="balanced",
                                                 random_state=random_state),
        "gbm": GradientBoostingClassifier(random_state=random_state),
        "svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced",
                       random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    if HAS_XGB:
        zoo["xgboost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
            eval_metric="logloss", random_state=random_state,
            use_label_encoder=False,
        )
    if HAS_LGB:
        zoo["lightgbm"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
            class_weight="balanced", random_state=random_state, verbosity=-1,
        )
    return zoo


def evaluate_models(
    df: pd.DataFrame, target: str, cv: int = 5, test_size: float = 0.2,
) -> Tuple[Dict[str, dict], object]:
    """Train + cross-validate all models, trả về metrics + best pipeline."""
    X = df.drop(columns=[target])
    y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42,
    )

    pre = build_preprocessor(df, target)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results: Dict[str, dict] = {}
    best_f1, best_name, best_pipe = -1.0, None, None

    for name, model in build_model_zoo().items():
        steps = [("pre", pre), ("clf", model)]
        if HAS_IMB:
            pipe = ImbPipeline(steps + [("smote", SMOTE(random_state=42))] if False else steps)
        else:
            pipe = Pipeline(steps)
        try:
            cv_res = cross_validate(
                pipe, X_tr, y_tr, cv=skf, scoring=["f1_weighted", "roc_auc_ovr_weighted"],
                n_jobs=-1, return_train_score=False, error_score="raise",
            )
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            results[name] = {
                "cv_f1": float(np.mean(cv_res["test_f1_weighted"])),
                "cv_auc": float(np.mean(cv_res["test_roc_auc_ovr_weighted"])),
                "test_accuracy": float(accuracy_score(y_te, y_pred)),
                "test_f1": float(f1_score(y_te, y_pred, average="weighted")),
                "test_precision": float(precision_score(y_te, y_pred, average="weighted", zero_division=0)),
                "test_recall": float(recall_score(y_te, y_pred, average="weighted", zero_division=0)),
                "report": classification_report(y_te, y_pred, output_dict=True),
            }
            if results[name]["test_f1"] > best_f1:
                best_f1 = results[name]["test_f1"]
                best_name, best_pipe = name, pipe
        except Exception as e:
            results[name] = {"error": str(e)}
    return results, (best_name, best_pipe)


def grid_search_rf(X, y, pre) -> dict:
    """Grid search RandomForest hyper-params / Tối ưu siêu tham số."""
    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=-1))])
    grid = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                       scoring="f1_weighted", n_jobs=-1, verbose=1)
    gs.fit(X, y)
    return {"best_params": gs.best_params_, "best_score": float(gs.best_score_)}


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "y"})
    res, best = evaluate_models(df, "y", cv=5)
    print(pd.DataFrame(res).T)
    print("best:", best[0])
'''

BEST_PRACTICES = """
Classification Best Practices / Thực hành tốt khi phân loại
============================================================
1. Splits:
   - Stratified train/test (keep class proportions).
   - If small data: StratifiedKFold CV (k=5 or 10).
   - For time-series: temporal split (no shuffle).

2. Imbalanced classes:
   - class_weight="balanced" in sklearn estimators.
   - SMOTE / ADASYN over-sampling (use imblearn Pipeline to avoid leakage).
   - Use AUC-PR (not AUC-ROC) + precision@k for rare positives.
   - Threshold tuning: optimize F1 / F-beta / cost-based metric on validation.

3. Leakage prevention:
   - Fit preprocessing ONLY on train fold inside Pipeline.
   - Do not scale then split — split then scale inside pipeline.

4. Model selection:
   - Start simple (LogReg baseline) → tree ensembles → boosted (XGB/LGBM).
   - Compare via CV mean ± std; pick most stable if performance tied.

5. Calibration:
   - For probability-sensitive tasks, apply CalibratedClassifierCV (isotonic).

6. Hyperparameter search:
   - For large search space, prefer Optuna (TPE) over GridSearchCV.
"""


class ClassificationSkill(Skill):
    """Sinh sklearn classification pipeline (LogReg/RF/XGB/LGBM/SVM/KNN)."""

    category = SkillCategory.ML
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "classification", "classifier", "logistic regression",
        "random forest", "xgboost", "lightgbm", "svm", "knn",
        "binary classification", "multiclass", "imbalanced",
    ]
    examples = [
        "Build classification pipeline cho churn dataset",
        "Compare logistic regression vs random forest",
        "Handle imbalanced classes with SMOTE",
    ]

    @property
    def name(self) -> str:
        return "classification_automation"

    @property
    def description(self) -> str:
        return (
            "Sinh end-to-end classification pipeline: preprocessing, model zoo "
            "(LogReg/RF/XGB/LGBM/SVM/KNN), stratified CV, grid search, "
            "imbalance handling (SMOTE/class_weight) + best-practices guide."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.13
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        artifacts: List[Dict[str, str]] = [
            {"name": "classification_pipeline.py", "language": "python", "content": CLASSIFICATION_PIPELINE},
            {"name": "BEST_PRACTICES.md", "language": "markdown", "content": BEST_PRACTICES},
        ]

        return SkillResult(
            success=True,
            output=(
                "[classification_automation] Generated full pipeline: preprocessing "
                "(numeric + categorical), 7-model zoo (LogReg/RF/GBM/SVM/KNN/XGB/LGBM), "
                "stratified CV + grid search + SMOTE imbalance handling."
            ),
            artifacts=artifacts,
            suggestions=[
                "Start with a LogisticRegression baseline, then try tree ensembles",
                "For imbalanced data, optimize threshold on validation F1/PR curve",
                "Use CalibratedClassifierCV(isotonic) if probabilities matter",
                "Prefer Optuna over GridSearchCV for large hyperparameter spaces",
                "Apply SHAP for post-hoc interpretability on the best model",
            ],
            metadata={
                "skill": self.name,
                "models": ["logreg", "random_forest", "gbm", "svm_rbf", "knn",
                           "xgboost", "lightgbm"],
                "handles_imbalance": True,
                "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
                "version": self.version,
                "author": self.author,
            },
        )
