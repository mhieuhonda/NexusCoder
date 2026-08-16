"""ML Evaluation Skill - Lập kế hoạch đánh giá mô hình.

Sinh evaluation plan với metrics phù hợp theo task type
(classification, regression, NLU, code generation, ...).

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class MLEvaluationSkill(Skill):
    """Lập kế hoạch đánh giá ML với metrics phù hợp task type."""

    category = SkillCategory.ML
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "evaluate", "evaluation", "benchmark", "metric",
        "perplexity", "bleu", "rouge", "meteor",
        "humaneval", "gsm8k", "mmlu", "bbh", "truthfulqa",
        "accuracy", "f1", "precision", "recall", "auc",
        "validation", "test set",
    ]
    examples = [
        "Evaluate a summarization model with ROUGE",
        "Benchmark LLM on MMLU and GSM8K",
        "Compute perplexity for language model",
    ]

    @property
    def name(self) -> str:
        return "ml_evaluation"

    @property
    def description(self) -> str:
        return (
            "Lập kế hoạch đánh giá ML: chọn metrics theo task type "
            "(classification / regression / generation / code / math), "
            "sinh harness cho standard benchmarks (MMLU, GSM8K, HumanEval, "
            "BBH, TruthfulQA) và significance testing."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.15
        return min(1.0, score)

    def _detect_task(self, prompt: str) -> str:
        p = prompt.lower()
        if any(k in p for k in ("humaneval", "code", "pass@k")):
            return "code"
        if any(k in p for k in ("gsm8k", "math", "reasoning")):
            return "math"
        if any(k in p for k in ("bleu", "rouge", "meteor", "translation")):
            return "translation"
        if any(k in p for k in ("perplexity", "language model", "lm")):
            return "lm"
        if any(k in p for k in ("summariz", "rouge")):
            return "summarization"
        if any(k in p for k in ("mmlu", "qa", "question")):
            return "qa"
        if any(k in p for k in ("auc", "f1", "classification")):
            return "classification"
        if "regression" in p:
            return "regression"
        return "classification"

    def execute(self, context: SkillContext) -> SkillResult:
        task = self._detect_task(context.prompt)
        plan = self._build_plan(task)

        return SkillResult(
            success=True,
            output=f"[MLEvaluation/{task}] Plan with {len(plan['metrics'])} metrics.",
            artifacts=[{"path": "eval/plan.md", "content": self._render_md(task, plan)}],
            metadata={
                "skill": self.name,
                "task_type": task,
                "metrics": plan["metrics"],
                "benchmarks": plan["benchmarks"],
                "tools": ["lm-evaluation-harness", "bigcode-eval", "lighteval", "sklearn"],
                "sampling": {
                    "temperature": 0.0 if task in ("code", "math", "qa") else 0.7,
                    "n_samples": 1 if task in ("code", "math") else 5,
                    "seed": 42,
                },
                "significance_test": "paired bootstrap (10000 resamples)",
            },
            suggestions=[
                "Use greedy decoding (T=0) for deterministic benchmarks like MMLU",
                "Sample 5-10 generations for pass@k on HumanEval",
                "Always include a baseline (random / heuristic) for context",
                "Stratify evaluation slices by domain / length / difficulty",
                "Report mean ± std across ≥ 3 seeds",
            ],
        )

    def _build_plan(self, task: str) -> Dict[str, List[str]]:
        plans: Dict[str, Dict[str, List[str]]] = {
            "classification": {
                "metrics": ["accuracy", "macro_f1", "precision", "recall",
                             "roc_auc", "confusion_matrix", "calibration_error"],
                "benchmarks": ["held-out test set", "cross-validation (5-fold)"],
            },
            "regression": {
                "metrics": ["mae", "mse", "rmse", "r2", "mape"],
                "benchmarks": ["held-out test set"],
            },
            "code": {
                "metrics": ["pass@1", "pass@10", "pass@100", "code_bleu", "avg_time"],
                "benchmarks": ["HumanEval", "HumanEval+", "MBPP", "APPS", "CodeContests"],
            },
            "math": {
                "metrics": ["exact_match", "maj@k", "cot_accuracy"],
                "benchmarks": ["GSM8K", "MATH", "MMLU-STEM", "AIME"],
            },
            "qa": {
                "metrics": ["exact_match", "f1", "token_f1"],
                "benchmarks": ["SQuAD", "NaturalQuestions", "TriviaQA", "MMLU"],
            },
            "translation": {
                "metrics": ["bleu", "chrf", "comet", "ter"],
                "benchmarks": ["WMT", "FLORES"],
            },
            "summarization": {
                "metrics": ["rouge1", "rouge2", "rougeL", "bertscore", "factcc"],
                "benchmarks": ["CNN/DailyMail", "XSum", "SAMSum"],
            },
            "lm": {
                "metrics": ["perplexity", "bits_per_byte", "loss"],
                "benchmarks": ["WikiText-103", "The Pile", "C4"],
            },
        }
        return plans[task]

    def _render_md(self, task: str, plan: Dict[str, List[str]]) -> str:
        metrics = "\n".join(f"- {m}" for m in plan["metrics"])
        benchmarks = "\n".join(f"- {b}" for b in plan["benchmarks"])
        return f"""# Evaluation Plan — {task}

## Metrics
{metrics}

## Benchmarks
{benchmarks}

## Protocol
- Decoding: greedy for code/math/QA; sampling T=0.7 + n=5 otherwise
- Seeds: 42, 7, 123 (mean ± std)
- Significance: paired bootstrap, 10k resamples, p < 0.05
- Report on held-out test set only; never tune on test
"""
