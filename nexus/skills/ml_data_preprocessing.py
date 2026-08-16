"""ML Data Preprocessing Skill - Sinh preprocessing pipeline.

Tạo scikit-learn Pipeline hoặc PyTorch Dataset với transform cho
tabular, text, image, time series data.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class MLDataPreprocessingSkill(Skill):
    """Sinh preprocessing pipeline: scikit-learn / PyTorch / HuggingFace."""

    category = SkillCategory.ML
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "preprocess", "preprocessing", "normalize", "normalization",
        "standardize", "tokenize", "tokenizer", "encode", "encoding",
        "clean data", "data cleaning", "impute", "scale",
        "transform", "pipeline", "feature extraction",
    ]
    examples = [
        "Preprocess tabular data with scikit-learn Pipeline",
        "Build a PyTorch dataset with tokenization for BERT",
        "Normalize image dataset for ResNet training",
    ]

    @property
    def name(self) -> str:
        return "ml_data_preprocessing"

    @property
    def description(self) -> str:
        return (
            "Sinh preprocessing pipeline cho tabular (sklearn Pipeline), "
            "text (HF tokenizer), image (torchvision transforms), "
            "và time series. Bao gồm imputation, scaling, encoding, "
            "tokenization, augmentation, và reproducibility best practices."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.15
        return min(1.0, score)

    def _detect_modality(self, prompt: str) -> str:
        p = prompt.lower()
        if any(k in p for k in ("image", "resnet", "cnn", "vision", "augment")):
            return "image"
        if any(k in p for k in ("tokeniz", "bert", "transformer", "text", "nlp")):
            return "text"
        if any(k in p for k in ("time series", "temporal", "sequence data")):
            return "timeseries"
        if any(k in p for k in ("audio", "spectrogram", "wav")):
            return "audio"
        return "tabular"

    def execute(self, context: SkillContext) -> SkillResult:
        modality = self._detect_modality(context.prompt)
        artifact = self._build_artifact(modality)

        return SkillResult(
            success=True,
            output=f"[MLDataPreprocess/{modality}] Pipeline ready.",
            artifacts=[artifact],
            metadata={
                "skill": self.name,
                "modality": modality,
                "library": {
                    "tabular": "scikit-learn",
                    "text": "transformers / tokenizers",
                    "image": "torchvision",
                    "timeseries": "scikit-learn + windowing",
                    "audio": "torchaudio",
                },
                "principles": [
                    "Fit transforms ONLY on train, apply to val/test",
                    "Cache tokenized datasets on disk to save RAM",
                    "Use .set_transform() instead of .map() for streaming",
                    "Persist the fitted pipeline with joblib for inference",
                ],
                "checks": [
                    "No label leakage (target not in features)",
                    "Train/val/test distributions similar (KS test)",
                    "No NaN/Inf after transform",
                ],
            },
            suggestions=[
                "Persist fitted pipeline: joblib.dump(pipe, 'preprocess.joblib')",
                "Add a data-leakage unit test (assert val leakage == 0)",
                "Profile with %%timeit to find I/O / CPU bottlenecks",
            ],
        )

    def _build_artifact(self, modality: str) -> Dict[str, str]:
        if modality == "text":
            return {"path": "preprocess/text_pipeline.py", "content": _TEXT_PIPELINE}
        if modality == "image":
            return {"path": "preprocess/image_pipeline.py", "content": _IMAGE_PIPELINE}
        if modality == "timeseries":
            return {"path": "preprocess/timeseries_pipeline.py", "content": _TS_PIPELINE}
        return {"path": "preprocess/tabular_pipeline.py", "content": _TABULAR_PIPELINE}


_TABULAR_PIPELINE = '''"""Tabular preprocessing — sklearn Pipeline (fit on train only)."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC = ["age", "income", "tenure"]
CATEGORICAL = ["gender", "country", "plan"]


def build_pipeline() -> Pipeline:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC),
            ("cat", Pipeline(categorical_steps), CATEGORICAL),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("pre", preprocessor)])

# Fit on train ONLY:
#   pipe = build_pipeline().fit(X_train, y_train)
#   X_train_t = pipe.transform(X_train)
#   X_val_t   = pipe.transform(X_val)
#   joblib.dump(pipe, "preprocess.joblib")
'''

_TEXT_PIPELINE = '''"""Text preprocessing — HuggingFace tokenizer + streaming cache."""
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_ID = "bert-base-uncased"
MAX_LEN = 512
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def tokenize_fn(batch: dict) -> dict:
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
    )


def build_dataset(rows: list[dict]) -> Dataset:
    ds = Dataset.from_list(rows)
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

# Cache to disk to avoid re-tokenizing:
#   ds = build_dataset(raw_rows)
#   ds.save_to_disk("data/tokenized_bert")
'''

_IMAGE_PIPELINE = '''"""Image preprocessing — torchvision transforms (train vs eval)."""
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# NOTE: ToTensor already scales uint8 [0,255] -> float [0,1] before Normalize.
'''

_TS_PIPELINE = '''"""Time series preprocessing — windowing + scaling (fit on train only)."""
import numpy as np
from sklearn.preprocessing import StandardScaler

WINDOW = 48
HORIZON = 1


def make_windows(series: np.ndarray, window: int = WINDOW, horizon: int = HORIZON):
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i : i + window])
        y.append(series[i + window : i + window + horizon])
    return np.asarray(X), np.asarray(y)


def build_train_val(series: np.ndarray, val_ratio: float = 0.2):
    split = int(len(series) * (1 - val_ratio))
    train, val = series[:split], series[split:]
    scaler = StandardScaler().fit(train.reshape(-1, 1))
    train_s = scaler.transform(train.reshape(-1, 1)).ravel()
    val_s = scaler.transform(val.reshape(-1, 1)).ravel()
    X_train, y_train = make_windows(train_s)
    X_val, y_val = make_windows(val_s)
    return X_train[..., None], y_train, X_val[..., None], y_val, scaler
'''
