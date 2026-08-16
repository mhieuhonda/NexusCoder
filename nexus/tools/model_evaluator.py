"""
Model Evaluator Tool - Đánh giá ML model trên dataset.
===========================================
Lazy import torch / sklearn. Tải model + dataset, chạy inference,
tính metrics (reuse ml_metrics_tool logic).

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


TASKS = {"classification", "regression"}
DATASET_FORMATS = {"csv", "jsonl", "json", "npy"}


class ModelEvaluatorTool(Tool):
    """Đánh giá ML model (sklearn / PyTorch) trên dataset."""

    category = ToolCategory.ML
    safety = ToolSafety.MODERATE
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "model_evaluator"

    @property
    def description(self) -> str:
        return "Evaluate ML model trên dataset (sklearn/torch) với metrics tùy chọn."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Đường dẫn file model (.pkl/.joblib/.pt/.pth)"},
                "dataset_path": {"type": "string", "description": "Dataset file (.csv/.jsonl/.json/.npy)"},
                "task": {
                    "type": "string",
                    "enum": sorted(TASKS),
                    "default": "classification",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics cần tính (bỏ qua → tính tất cả)",
                },
                "feature_cols": {"type": "array", "items": {"type": "string"}, "description": "CSV: tên cột feature"},
                "label_col": {"type": "string", "description": "CSV: tên cột label"},
                "framework": {"type": "string", "enum": ["auto", "sklearn", "torch"], "default": "auto"},
                "batch_size": {"type": "integer", "default": 64},
            },
            "required": ["model_path", "dataset_path"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("model_path"):
            return "Missing required arg: model_path"
        if not args.get("dataset_path"):
            return "Missing required arg: dataset_path"
        task = args.get("task", "classification")
        if task not in TASKS:
            return f"Invalid task='{task}'. Supported: {sorted(TASKS)}"
        return None

    # ---- Loaders --------------------------------------------------------

    def _load_dataset(self, path: str, feature_cols: List[str], label_col: str) -> Dict[str, Any]:
        """Tải dataset từ CSV/JSON/JSONL/NPY. Trả về {X, y}."""
        ext = os.path.splitext(path)[1][1:].lower()
        if ext not in DATASET_FORMATS:
            raise ValueError(f"Unsupported dataset format: {ext}")

        if ext == "csv":
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                raise ValueError("Empty CSV")
            if not label_col:
                # Heuristic: lấy cột cuối / take last column as label
                label_col = list(rows[0].keys())[-1]
            if not feature_cols:
                feature_cols = [c for c in rows[0].keys() if c != label_col]
            X: List[List[float]] = []
            y: List[Any] = []
            for row in rows:
                try:
                    X.append([float(row[c]) for c in feature_cols])
                except (ValueError, KeyError):
                    continue
                y.append(row[label_col])
            return {"X": X, "y": y, "feature_cols": feature_cols, "label_col": label_col}

        if ext == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                X = data.get("X", [])
                y = data.get("y", data.get("labels", []))
            else:  # list of dicts
                X = [[row.get(c) for c in feature_cols] for row in data]
                y = [row.get(label_col) for row in data]
            return {"X": X, "y": y, "feature_cols": feature_cols, "label_col": label_col}

        if ext == "jsonl":
            X = []
            y = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    X.append([row.get(c) for c in feature_cols] or list(row.values())[:-1])
                    y.append(row.get(label_col) or list(row.values())[-1])
            return {"X": X, "y": y, "feature_cols": feature_cols, "label_col": label_col}

        # npy
        try:
            import numpy as np  # type: ignore
            arr = np.load(path, allow_pickle=True)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                X = arr[:, :-1].tolist()
                y = arr[:, -1].tolist()
            else:
                X = arr.tolist()
                y = []
            return {"X": X, "y": y, "feature_cols": feature_cols, "label_col": label_col}
        except ImportError:
            raise RuntimeError("numpy chưa cài để load .npy")

    # ---- Model loading --------------------------------------------------

    def _load_model(self, path: str, framework: str):
        """Tải model. Auto-detect theo extension nếu framework='auto'."""
        ext = os.path.splitext(path)[1].lower()
        if framework == "auto":
            if ext in (".pt", ".pth"):
                framework = "torch"
            else:
                framework = "sklearn"

        if framework == "torch":
            try:
                import torch  # type: ignore
            except ImportError:
                raise RuntimeError("torch chưa cài. Cài đặt: pip install torch")
            try:
                model = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                # PyTorch < 2.6 không có weights_only / older PyTorch
                model = torch.load(path, map_location="cpu")
            model.eval()
            return ("torch", model)

        # sklearn-style (pickle/joblib)
        try:
            import joblib  # type: ignore
            model = joblib.load(path)
            return ("sklearn", model)
        except ImportError:
            import pickle
            with open(path, "rb") as f:
                return ("sklearn", pickle.load(f))

    # ---- Predict --------------------------------------------------------

    def _predict(self, backend: str, model: Any, X: List[List[float]], batch_size: int) -> List[Any]:
        if backend == "torch":
            import torch  # type: ignore
            preds: List[Any] = []
            for i in range(0, len(X), batch_size):
                chunk = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
                with torch.no_grad():
                    out = model(chunk)
                # Argmax cho classification / raw output cho regression
                if out.dim() > 1 and out.shape[1] > 1:
                    preds.extend(out.argmax(dim=1).tolist())
                else:
                    preds.extend(out.squeeze(-1).tolist())
            return preds
        # sklearn predict
        return list(model.predict(X))

    # ---- Execute --------------------------------------------------------

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        model_path = args["model_path"]
        dataset_path = args["dataset_path"]
        task = args.get("task", "classification")
        metrics = args.get("metrics", []) or []
        feature_cols = args.get("feature_cols", []) or []
        label_col = args.get("label_col", "")
        framework = args.get("framework", "auto")
        batch_size = int(args.get("batch_size", 64))

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Sẽ evaluate model {model_path} trên {dataset_path} (task={task})",
                metadata={"model_path": model_path, "dataset_path": dataset_path, "task": task, "dry_run": True},
            )

        if not os.path.exists(model_path):
            return ToolResult(success=False, error=f"Model file không tồn tại: {model_path}", return_code=1)
        if not os.path.exists(dataset_path):
            return ToolResult(success=False, error=f"Dataset file không tồn tại: {dataset_path}", return_code=1)

        try:
            data = self._load_dataset(dataset_path, feature_cols, label_col)
        except Exception as e:
            return ToolResult(success=False, error=f"Load dataset failed: {e}", return_code=1)

        if not data["X"]:
            return ToolResult(success=False, error="Dataset rỗng hoặc không có features", return_code=1)

        try:
            backend, model = self._load_model(model_path, framework)
        except Exception as e:
            return ToolResult(success=False, error=f"Load model failed: {e}", return_code=1)

        try:
            preds = self._predict(backend, model, data["X"], batch_size)
        except Exception as e:
            return ToolResult(success=False, error=f"Inference failed: {e}", return_code=1)

        # Tính metrics bằng ml_metrics_tool logic (reuse internal)
        from .ml_metrics_tool import MLMetricsTool
        metrics_tool = MLMetricsTool()
        result = metrics_tool.execute(
            {
                "y_true": data["y"],
                "y_pred": preds,
                "task": task,
                "metrics": metrics,
            },
            context,
        )

        if not result.success:
            return result

        # Bổ sung metadata của evaluator / attach evaluator metadata
        result.metadata.update({
            "model_path": model_path,
            "dataset_path": dataset_path,
            "framework": backend,
            "n_samples": len(data["X"]),
            "n_features": len(data["X"][0]) if data["X"] else 0,
            "feature_cols": data["feature_cols"],
            "label_col": data["label_col"],
            "predictions": preds,
        })
        result.output = f"Evaluated {backend} model on {len(data['X'])} samples\n{result.output}"
        return result
