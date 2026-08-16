"""ML Model Explainability Skill - SHAP / LIME / interpretation.

Sinh code tích hợp SHAP (TreeExplainer, DeepExplainer, KernelExplainer),
LIME, permutation importance, và partial dependence plots.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class MLExplainabilitySkill(Skill):
    """Sinh code giải thích model với SHAP / LIME / permutation importance."""

    category = SkillCategory.ML
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "explain", "explainability", "shap", "lime",
        "interpret", "interpretable", "feature importance",
        "permutation importance", "partial dependence", "pdp",
        "counterfactual", "attention rollout", "grad-cam",
        "model card", "fairness",
    ]
    examples = [
        "Explain XGBoost predictions with SHAP",
        "Generate SHAP summary plot for a classifier",
        "Build a model card for fairness audit",
    ]

    @property
    def name(self) -> str:
        return "ml_explainability"

    @property
    def description(self) -> str:
        return (
            "Sinh code giải thích model: SHAP (Tree/Deep/Kernel Explainer), "
            "LIME, permutation importance, partial dependence plots, "
            "Grad-CAM cho CNN, và model card template cho fairness audit."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.18
        return min(1.0, score)

    def _pick_explainer(self, prompt: str) -> str:
        p = prompt.lower()
        if any(k in p for k in ("tree", "xgboost", "lightgbm", "random forest")):
            return "shap_tree"
        if any(k in p for k in ("deep", "neural", "pytorch", "tensorflow")):
            return "shap_deep"
        if any(k in p for k in ("cnn", "image", "grad-cam", "gradcam")):
            return "grad_cam"
        if "lime" in p:
            return "lime"
        if "permutation" in p:
            return "permutation"
        return "shap_tree"

    def execute(self, context: SkillContext) -> SkillResult:
        explainer = self._pick_explainer(context.prompt)
        artifact = self._build_artifact(explainer)

        return SkillResult(
            success=True,
            output=f"[MLExplainability/{explainer}] Code ready.",
            artifacts=[artifact],
            metadata={
                "skill": self.name,
                "explainer": explainer,
                "libraries": ["shap", "lime", "interpret", "captum", "alibi"],
                "explainers": {
                    "shap_tree": "exact Shapley cho tree ensembles (fast)",
                    "shap_deep": "DeepSHAP via gradient, dùng cho NN",
                    "shap_kernel": "model-agnostic, slow, O(2^M)",
                    "lime": "local linear surrogate, fast",
                    "permutation": "model-agnostic, drop in AUC/acc",
                    "pdp": "global marginal effect of one feature",
                    "grad_cam": "spatial heatmap cho CNN",
                },
                "model_card": {
                    "sections": [
                        "Model details", "Intended use", "Training data",
                        "Evaluation data", "Metrics", "Ethical considerations",
                        "Caveats & recommendations",
                    ],
                },
            },
            suggestions=[
                "Sample a background dataset (~100 rows) for SHAP (not full data)",
                "Use TreeExplainer for XGBoost/LightGBM — exact and fast",
                "Validate explanations with domain experts before publishing",
                "Audit for fairness across protected groups (equalized odds)",
            ],
        )

    def _build_artifact(self, explainer: str) -> Dict[str, str]:
        if explainer == "shap_deep":
            return {"path": "explain/shap_deep.py", "content": _SHAP_DEEP}
        if explainer == "grad_cam":
            return {"path": "explain/grad_cam.py", "content": _GRAD_CAM}
        if explainer == "lime":
            return {"path": "explain/lime.py", "content": _LIME}
        if explainer == "permutation":
            return {"path": "explain/permutation.py", "content": _PERMUTATION}
        return {"path": "explain/shap_tree.py", "content": _SHAP_TREE}


_SHAP_TREE = '''"""SHAP TreeExplainer for XGBoost / LightGBM / RandomForest."""
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

def explain(model: XGBClassifier, X_train, X_test, sample_size: int = 100):
    """Compute SHAP values and produce summary + waterfall plots."""
    background = shap.sample(X_train, sample_size)
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test)
    # Global summary: shows feature importance + direction
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("explain/shap_summary.png", bbox_inches="tight")
    # Local: explain a single prediction
    shap.waterfall_plot(explainer(X_test.iloc[[0]])[0], show=False)
    plt.savefig("explain/shap_waterfall_0.png", bbox_inches="tight")
    return explainer.expected_value, shap_values
'''

_SHAP_DEEP = '''"""SHAP DeepExplainer for a PyTorch model."""
import torch
import shap

def explain(model, background, test_batch):
    """DeepSHAP — uses a gradient-based approximation."""
    model.eval()
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_batch)
    return shap_values
# NOTE: DeepExplainer requires PyTorch ops to be differentiable through SHAP.
# Use GradientExplainer as a fallback for unsupported layers.
'''

_GRAD_CAM = '''"""Grad-CAM heatmap for CNN image classification."""
import numpy as np
import torch
import torch.nn.functional as F
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _m, _i, out): self.activations = out
    def _bwd_hook(self, _m, _gi, grad): self.gradients = grad

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        out = self.model(x)
        self.model.zero_grad()
        out[0, class_idx].backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)         # global-avg-pool
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        return cv2.resize(cam / (cam.max() + 1e-8), (x.shape[-1], x.shape[-2]))
'''

_LIME = '''"""LIME local explanation for a tabular classifier."""
import lime
import lime.lime_tabular

def explain(model, X_train, X_test, feature_names, mode: str = "classification"):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["neg", "pos"],
        mode=mode,
        discretize_continuous=True,
        random_state=42,
    )
    exp = explainer.explain_instance(
        X_test.iloc[0].values,
        predict_fn=model.predict_proba,
        num_features=10,
    )
    exp.save_to_file("explain/lime_local_0.html")
    return exp.as_list()
'''

_PERMUTATION = '''"""Permutation importance — model-agnostic."""
import numpy as np
from sklearn.inspection import permutation_importance

def compute(model, X_val, y_val, scoring: str = "roc_auc", n_repeats: int = 10):
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=42,
        n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    return [(X_val.columns[i], result.importances_mean[i], result.importances_std[i])
            for i in order]
'''
