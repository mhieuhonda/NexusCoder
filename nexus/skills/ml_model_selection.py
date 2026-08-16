"""ML Model Selection Skill - So sánh kiến trúc cho task type.

Trả về comparison table của các architectures phổ biến kèm
điểm mạnh/yếu và khi nào dùng.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class MLModelSelectionSkill(Skill):
    """So sánh và đề xuất kiến trúc ML/DL cho task type."""

    category = SkillCategory.ML
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "model selection", "compare models", "baseline",
        "which model", "choose model", "architecture",
        "best model", "benchmark models", "model zoo",
    ]
    examples = [
        "Which model should I use for text classification?",
        "Compare ResNet vs ViT for image classification",
        "Suggest a baseline for tabular regression",
    ]

    @property
    def name(self) -> str:
        return "ml_model_selection"

    @property
    def description(self) -> str:
        return (
            "So sánh kiến trúc ML/DL cho task type, trả về bảng comparison "
            "(điểm mạnh/yếu, params, latency, khi nào dùng) và đề xuất baseline "
            "trước khi thử complex models."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.2
        if any(kw in prompt_lower for kw in ("tabular", "image", "text", "sequence", "audio")):
            score += 0.2
        return min(1.0, score)

    def _detect_task(self, prompt: str) -> str:
        p = prompt.lower()
        if any(k in p for k in ("image", "vision", "resnet", "vit", "cnn")):
            return "image_classification"
        if any(k in p for k in ("text classification", "nlp", "bert", "transformer text")):
            return "text_classification"
        if any(k in p for k in ("ner", "token classification", "sequence labeling")):
            return "ner"
        if any(k in p for k in ("seq2seq", "translation", "summarization")):
            return "seq2seq"
        if any(k in p for k in ("time series", "forecast")):
            return "timeseries"
        if any(k in p for k in ("recommendation", "recommender")):
            return "recommendation"
        if any(k in p for k in ("anomaly", "anomalies")):
            return "anomaly"
        return "tabular"

    def execute(self, context: SkillContext) -> SkillResult:
        task = self._detect_task(context.prompt)
        table = self._comparison_table(task)

        return SkillResult(
            success=True,
            output=f"[MLModelSelection/{task}] {len(table)} candidates compared.",
            artifacts=[{"path": "model_selection/comparison.md", "content": self._render_md(task, table)}],
            metadata={
                "skill": self.name,
                "task_type": task,
                "candidates": table,
                "workflow": [
                    "1. Start with a simple baseline (logistic / GBM)",
                    "2. Establish metric + significance threshold",
                    "3. Add complexity only if delta > threshold",
                    "4. Always compare cost (params / latency) — not just accuracy",
                    "5. Re-run top 3 on multiple seeds before picking",
                ],
                "selection_criteria": [
                    "data_size (small -> GBM, large -> deep)",
                    "latency_budget (real-time -> distilled / quantized)",
                    "interpretability requirement (regulation)",
                    "compute budget (training + inference)",
                ],
            },
            suggestions=[
                "Always beat a simple baseline before reaching for transformers",
                "Cross-validate on 5 folds with the SAME folds across all models",
                "Report params + inference latency next to accuracy",
                "Try LightGBM before XGBoost — often faster and on par",
            ],
        )

    def _comparison_table(self, task: str) -> List[Dict[str, str]]:
        tables: Dict[str, List[Dict[str, str]]] = {
            "tabular": [
                {"model": "Logistic Regression", "pros": "fast, interpretable, baseline", "cons": "linear only", "params": "low", "use_when": "baseline / linearly separable"},
                {"model": "Random Forest", "pros": "robust, no scaling needed", "cons": "large memory", "params": "med", "use_when": "tabular default"},
                {"model": "LightGBM", "pros": "fast, SOTA on tabular", "cons": "tuning needed", "params": "med", "use_when": "large tabular, SOTA"},
                {"model": "XGBoost", "pros": "battle-tested, mature", "cons": "slower than LGBM", "params": "med", "use_when": "competition / Kaggle"},
                {"model": "TabNet / FT-Transformer", "pros": "end-to-end DL", "cons": "data hungry", "params": "high", "use_when": "very large data"},
            ],
            "image_classification": [
                {"model": "ResNet-50", "pros": "fast, mature, well-understood", "cons": "inductive bias", "params": "25M", "use_when": "default transfer learning"},
                {"model": "EfficientNet-B0..B7", "pros": "scaling-aware", "cons": "less ecosystem", "params": "5-66M", "use_when": "mobile / edge"},
                {"model": "ViT-Base", "pros": "SOTA at scale", "cons": "needs huge data", "params": "86M", "use_when": ">= 1M images / pretrain"},
                {"model": "ConvNeXt", "pros": "modern CNN, ViT-competitive", "cons": "newer", "params": "89M", "use_when": "want CNN speed"},
            ],
            "text_classification": [
                {"model": "Logistic + TF-IDF", "pros": "instant baseline", "cons": "no semantics", "params": "low", "use_when": "first baseline"},
                {"model": "DistilBERT", "pros": "small, fast", "cons": "lower acc", "params": "66M", "use_when": "latency budget"},
                {"model": "BERT-base", "pros": "strong default", "cons": "110M params", "params": "110M", "use_when": "general NLP"},
                {"model": "RoBERTa / DeBERTa-v3", "pros": "SOTA encoder", "cons": "heavier", "params": "110-184M", "use_when": "max encoder acc"},
                {"model": "LLM few-shot", "pros": "no training", "cons": "cost per call", "params": "7B+", "use_when": "no labeled data"},
            ],
            "seq2seq": [
                {"model": "T5-base", "pros": "unified text-to-text", "cons": "slow", "params": "220M", "use_when": "multi-task NLP"},
                {"model": "BART", "pros": "good for summarization", "cons": "encoder-decoder", "params": "140M", "use_when": "abstractive summary"},
                {"model": "NLLB / M2M100", "pros": "multilingual", "cons": "specialized", "params": "300M-1.2B", "use_when": "translation"},
                {"model": "LLM (Llama-3 / Qwen2)", "pros": "few-shot, SOTA", "cons": "expensive", "params": "8B-70B", "use_when": "SOTA gen / RAG"},
            ],
            "timeseries": [
                {"model": "ARIMA / SARIMA", "pros": "interpretable", "cons": "univariate", "params": "low", "use_when": "stable univariate"},
                {"model": "Prophet", "pros": "fast, holidays", "cons": "linear trend", "params": "low", "use_when": "business forecasting"},
                {"model": "Gradient Boosting (LGBM)", "pros": "exogenous features", "cons": "needs lags", "params": "med", "use_when": "feature-rich"},
                {"model": "N-BEATS / N-HiTS", "pros": "SOTA univariate", "cons": "DL overhead", "params": "med", "use_when": "uni forecast competition"},
                {"model": "PatchTST / TFT", "pros": "SOTA multivariate", "cons": "data hungry", "params": "high", "use_when": "multivariate, long horizon"},
            ],
        }
        return tables.get(task, tables["tabular"])

    def _render_md(self, task: str, table: List[Dict[str, str]]) -> str:
        header = "| Model | Pros | Cons | Params | Use when |\n|---|---|---|---|---|"
        rows = "\n".join(
            f"| {r['model']} | {r['pros']} | {r['cons']} | {r['params']} | {r['use_when']} |"
            for r in table
        )
        return f"# Model Selection — {task}\n\n{header}\n{rows}\n"
