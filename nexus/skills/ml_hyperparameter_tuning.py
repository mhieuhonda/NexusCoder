"""ML Hyperparameter Tuning Skill - Sinh Optuna study template.

Hỗ trợ grid / random / Bayesian (TPE, CMA-ES) search với
pruning (Median / Hyperband / Successive Halving).

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class MLHyperparameterTuningSkill(Skill):
    """Sinh Optuna study template với pruning và distributed optimization."""

    category = SkillCategory.ML
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "tune", "tuning", "hyperparameter", "hyper-parameter",
        "optuna", "grid search", "random search", "bayesian",
        "cma-es", "tpe", "pruning", "hyperband",
        "successive halving", "study", "trial",
    ]
    examples = [
        "Tune XGBoost hyperparameters with Optuna",
        "Bayesian optimization for transformer learning rate",
        "Distributed hyperparameter sweep across 8 workers",
    ]

    @property
    def name(self) -> str:
        return "ml_hyperparameter_tuning"

    @property
    def description(self) -> str:
        return (
            "Sinh Optuna study template: TPE / CMA-ES sampler, "
            "Median/Hyperband pruning, distributed optimization (RDBStorage), "
            "search spaces (int/float/categorical/loguniform), "
            "và visualization (optimization history, param importance)."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.16
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        sampler = context.metadata.get("sampler", "tpe")
        pruner = context.metadata.get("pruner", "hyperband")
        n_trials = int(context.metadata.get("n_trials", 100))

        return SkillResult(
            success=True,
            output=(
                f"[MLHPTuning] Optuna study ready: sampler={sampler}, "
                f"pruner={pruner}, n_trials={n_trials}"
            ),
            artifacts=[{"path": "tuning/optimize.py", "content": _OPTUNA_STUDY}],
            metadata={
                "skill": self.name,
                "sampler": sampler,
                "pruner": pruner,
                "n_trials": n_trials,
                "search_space": {
                    "learning_rate": "loguniform(1e-6, 1e-3)",
                    "weight_decay": "loguniform(1e-6, 1e-1)",
                    "batch_size": "categorical([16, 32, 64, 128])",
                    "warmup_ratio": "uniform(0.0, 0.1)",
                    "dropout": "uniform(0.05, 0.4)",
                    "hidden_size": "categorical([256, 512, 768, 1024])",
                    "n_layers": "int(2, 12)",
                },
                "samplers": {
                    "tpe": "default, good for most cases",
                    "cma_es": "continuous spaces, expensive but strong",
                    "grid": "small space, exhaustive, for ablation",
                    "random": "baseline, parallelizable",
                },
                "pruners": {
                    "median": "fast, default for warmup",
                    "hyperband": "best for short warmups, async",
                    "successive_halving": "synchronous, classic SHA",
                    "none": "no pruning — full trials only",
                },
                "storage": "postgresql+psycopg2://optuna:pw@db:5432/optuna",
                "parallelism": "n_workers = 8 (run multiple `study.optimize`)",
            },
            suggestions=[
                "Always run a small random search (10 trials) before Bayesian",
                "Use `study.enqueue_trial(...)` to seed known-good configs",
                "Prune with Hyperband when each trial > 5 min",
                "Visualize: optuna-dashboard, plot_optimization_history, plot_param_importances",
                "Re-run the best trial with multiple seeds to confirm stability",
            ],
        )


_OPTUNA_STUDY = '''"""Optuna study with TPE sampler + Hyperband pruner + RDB storage."""
import os
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.integration import WeightsAndBiasesCallback
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

STUDY_NAME = "xgb_auc_tuning"
STORAGE = os.getenv("OPTUNA_STORAGE", "sqlite:///optuna.db")
N_TRIALS = 100


def make_objective():
    X, y = make_classification(n_samples=8000, n_features=50, n_informative=20, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = XGBClassifier(**params, tree_method="hist", eval_metric="auc",
                              random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    return objective


def run() -> optuna.Study:
    sampler = TPESampler(seed=42, multivariate=True, group=True, n_startup_trials=10)
    pruner = HyperbandPruner(min_resource=100, max_resource=1000, reduction_factor=3)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(
        make_objective(),
        n_trials=N_TRIALS,
        n_jobs=1,
        gc_after_trial=True,
        callbacks=[WeightsAndBiasesCallback()],
    )
    print("best params:", study.best_params)
    print("best AUC:  ", study.best_value)
    return study


if __name__ == "__main__":
    run()

# Distributed: run the same script in N processes pointing at the same STORAGE.
# Dashboard:
#   optuna-dashboard sqlite:///optuna.db
'''
