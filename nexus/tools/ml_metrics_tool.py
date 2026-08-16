"""
ML Metrics Tool - Tính metrics cho classification/regression.
===========================================
Pure stdlib (math + collections). Không cần sklearn/torch.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


CLASSIFICATION_METRICS = {"accuracy", "precision", "recall", "f1", "confusion_matrix", "classification_report"}
REGRESSION_METRICS = {"mse", "mae", "rmse", "r2", "mape", "msle", "explained_variance"}
TASKS = {"classification", "regression", "auto"}


class MLMetricsTool(Tool):
    """Tính ML metrics: classification + regression. Pure stdlib."""

    category = ToolCategory.ML
    safety = ToolSafety.SAFE

    @property
    def name(self) -> str:
        return "ml_metrics"

    @property
    def description(self) -> str:
        return "ML metrics: accuracy/precision/recall/f1/confusion_matrix, mse/mae/rmse/r2/mape."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "y_true": {"type": "array", "description": "Ground truth labels"},
                "y_pred": {"type": "array", "description": "Predicted labels/values"},
                "task": {
                    "type": "string",
                    "enum": sorted(TASKS),
                    "default": "auto",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Danh sách metrics cần tính (bỏ qua → tính tất cả)",
                },
                "average": {
                    "type": "string",
                    "enum": ["binary", "micro", "macro", "weighted"],
                    "default": "macro",
                },
                "positive_label": {"type": "string", "description": "Label dương (cho average=binary)"},
            },
            "required": ["y_true", "y_pred"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("y_true"):
            return "Missing required arg: y_true"
        if not args.get("y_pred"):
            return "Missing required arg: y_pred"
        if len(args["y_true"]) != len(args["y_pred"]):
            return f"Length mismatch: y_true={len(args['y_true'])} y_pred={len(args['y_pred'])}"
        task = args.get("task", "auto")
        if task not in TASKS:
            return f"Invalid task='{task}'. Supported: {sorted(TASKS)}"
        return None

    # ---- Phát hiện task / Detect task type ------------------------------

    def _detect_task(self, y_true: List[Any], y_pred: List[Any]) -> str:
        """Auto-detect: nếu labels là string hoặc có <10 giá trị unique → classification."""
        combined = y_true + y_pred
        unique = set(combined)
        # Heuristic: classification nếu ít hơn 10% unique values hoặc labels là str/bool
        if any(isinstance(v, str) for v in combined):
            return "classification"
        if any(isinstance(v, bool) for v in combined):
            return "classification"
        if len(unique) <= max(10, len(combined) // 10):
            return "classification"
        return "regression"

    # ---- Classification metrics -----------------------------------------

    def _confusion_matrix(
        self, y_true: List[Any], y_pred: List[Any], labels: Optional[List[Any]] = None
    ) -> Tuple[List[List[int]], List[Any]]:
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred), key=lambda x: str(x))
        idx = {lab: i for i, lab in enumerate(labels)}
        n = len(labels)
        matrix = [[0] * n for _ in range(n)]
        for t, p in zip(y_true, y_pred):
            if t in idx and p in idx:
                matrix[idx[t]][idx[p]] += 1
        return matrix, labels

    def _classification_metrics(
        self, y_true: List[Any], y_pred: List[Any], requested: List[str], average: str, positive_label: Optional[Any]
    ) -> Dict[str, Any]:
        matrix, labels = self._confusion_matrix(y_true, y_pred)
        n = len(labels)
        idx = {lab: i for i, lab in enumerate(labels)}

        # TP/FP/FN/TN per class / per-class counts
        per_class: Dict[Any, Dict[str, int]] = {}
        for i, lab in enumerate(labels):
            tp = matrix[i][i]
            fp = sum(matrix[r][i] for r in range(n) if r != i)
            fn = sum(matrix[i][c] for c in range(n) if c != i)
            tn = sum(matrix[r][c] for r in range(n) for c in range(n) if r != i and c != i)
            per_class[lab] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

        def safe_div(a: float, b: float) -> float:
            return a / b if b > 0 else 0.0

        precision_per = {lab: safe_div(v["tp"], v["tp"] + v["fp"]) for lab, v in per_class.items()}
        recall_per = {lab: safe_div(v["tp"], v["tp"] + v["fn"]) for lab, v in per_class.items()}
        f1_per = {
            lab: (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            for lab, (p, r) in zip(precision_per.keys(), zip(precision_per.values(), recall_per.values()))
        }

        # Aggregate theo average / aggregate by averaging strategy
        def aggregate(metric_per: Dict[Any, float]) -> float:
            if average == "binary":
                lab = positive_label if positive_label is not None else labels[-1]
                return float(metric_per.get(lab, 0.0))
            if average == "micro":
                tp_sum = sum(v["tp"] for v in per_class.values())
                fp_sum = sum(v["fp"] for v in per_class.values())
                fn_sum = sum(v["fn"] for v in per_class.values())
                p = safe_div(tp_sum, tp_sum + fp_sum)
                r = safe_div(tp_sum, tp_sum + fn_sum)
                return p if metric_per == precision_per else (r if metric_per == recall_per else safe_div(2 * p * r, p + r))
            if average == "weighted":
                total = sum(v["tp"] + v["fn"] for v in per_class.values()) or 1
                return float(sum(metric_per[lab] * (per_class[lab]["tp"] + per_class[lab]["fn"]) for lab in labels) / total)
            # macro / macro default
            return float(sum(metric_per.values()) / max(1, len(metric_per)))

        correct = sum(matrix[i][i] for i in range(n))
        total = len(y_true)
        result: Dict[str, Any] = {}

        def want(name: str) -> bool:
            return not requested or name in requested

        if want("accuracy"):
            result["accuracy"] = safe_div(correct, total)
        if want("precision"):
            result["precision"] = aggregate(precision_per)
        if want("recall"):
            result["recall"] = aggregate(recall_per)
        if want("f1"):
            result["f1"] = aggregate(f1_per)
        if want("confusion_matrix"):
            result["confusion_matrix"] = matrix
            result["labels"] = labels
        if want("classification_report"):
            result["classification_report"] = {
                "per_class": {
                    str(lab): {
                        "precision": precision_per[lab],
                        "recall": recall_per[lab],
                        "f1": f1_per[lab],
                        "support": per_class[lab]["tp"] + per_class[lab]["fn"],
                    }
                    for lab in labels
                },
                "average": average,
            }
        return result

    # ---- Regression metrics --------------------------------------------

    def _regression_metrics(self, y_true: List[float], y_pred: List[float], requested: List[str]) -> Dict[str, Any]:
        n = len(y_true)
        errors = [t - p for t, p in zip(y_true, y_pred)]
        abs_errors = [abs(e) for e in errors]
        sq_errors = [e * e for e in errors]
        mean_t = sum(y_true) / n if n else 0.0
        ss_tot = sum((t - mean_t) ** 2 for t in y_true)
        ss_res = sum(sq_errors)

        def want(name: str) -> bool:
            return not requested or name in requested

        result: Dict[str, Any] = {}
        if want("mse"):
            result["mse"] = ss_res / n if n else 0.0
        if want("mae"):
            result["mae"] = sum(abs_errors) / n if n else 0.0
        if want("rmse"):
            result["rmse"] = math.sqrt(ss_res / n) if n else 0.0
        if want("r2"):
            result["r2"] = (1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
        if want("mape"):
            # Tránh chia 0 / avoid div by zero
            denoms = [abs(t) for t in y_true if abs(t) > 1e-12]
            if denoms:
                result["mape"] = sum(abs(e / t) for t, e in zip(y_true, errors) if abs(t) > 1e-12) / len(denoms) * 100.0
            else:
                result["mape"] = float("inf")
        if want("msle"):
            try:
                result["msle"] = sum((math.log1p(max(0, p)) - math.log1p(max(0, t))) ** 2 for p, t in zip(y_pred, y_true)) / n if n else 0.0
            except Exception:
                result["msle"] = None
        if want("explained_variance"):
            var_res = sum((e - sum(errors) / n) ** 2 for e in errors) / n if n else 0.0
            var_tot = sum((t - mean_t) ** 2 for t in y_true) / n if n else 0.0
            result["explained_variance"] = (1 - var_res / var_tot) if var_tot > 0 else 1.0
        return result

    # ---- Execute --------------------------------------------------------

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        y_true_raw: List[Any] = args["y_true"]
        y_pred_raw: List[Any] = args["y_pred"]
        task = args.get("task", "auto")
        requested: List[str] = args.get("metrics", []) or []
        average = args.get("average", "macro")
        positive_label = args.get("positive_label")

        try:
            if task == "auto":
                task = self._detect_task(y_true_raw, y_pred_raw)

            if task == "classification":
                metrics = self._classification_metrics(y_true_raw, y_pred_raw, requested, average, positive_label)
                metrics["task"] = "classification"
            else:
                # Ép sang float / coerce to float
                y_true = [float(x) for x in y_true_raw]
                y_pred = [float(x) for x in y_pred_raw]
                metrics = self._regression_metrics(y_true, y_pred, requested)
                metrics["task"] = "regression"

            metrics["n_samples"] = len(y_true_raw)
            return ToolResult(
                success=True,
                output=str(metrics),
                metadata=metrics,
            )
        except (ValueError, TypeError) as e:
            return ToolResult(success=False, error=f"Invalid data: {e}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=f"Metrics compute failed: {e}", return_code=1)
