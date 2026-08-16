"""Sentiment Analysis Skill - TextBlob + Transformers pipelines.

Sinh code phân tích cảm xúc văn bản: TextBlob polarity/subjectivity (baseline),
VADER cho social media, HuggingFace transformers (DistilBERT SST-2) cho
production, aspect-based sentiment, và batch processing với visualization.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


TEXTBLOB_VADER = '''"""Baseline sentiment: TextBlob + VADER / Cảm xúc cơ bản."""
from __future__ import annotations
from typing import Dict, List
import pandas as pd

try:
    from textblob import TextBlob
    HAS_TB = True
except ImportError:
    HAS_TB = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False


def textblob_sentiment(text: str) -> Dict[str, float | str]:
    """Polarity ∈ [-1, 1], Subjectivity ∈ [0, 1]."""
    if not HAS_TB:
        return {"error": "textblob not installed"}
    blob = TextBlob(text)
    pol = float(blob.sentiment.polarity)
    label = "positive" if pol > 0.05 else "negative" if pol < -0.05 else "neutral"
    return {
        "polarity": pol,
        "subjectivity": float(blob.sentiment.subjectivity),
        "label": label,
        "method": "textblob",
    }


def vader_sentiment(text: str) -> Dict[str, float | str]:
    """VADER — tốt cho social media (emojis, ALL CAPS, !)."""
    if not HAS_VADER:
        return {"error": "vaderSentiment not installed"}
    s = VADER.polarity_scores(text)
    return {
        "neg": s["neg"], "neu": s["neu"], "pos": s["pos"],
        "compound": s["compound"],
        "label": "positive" if s["compound"] >= 0.05 else "negative" if s["compound"] <= -0.05 else "neutral",
        "method": "vader",
    }


def batch_sentiment(texts: List[str], method: str = "textblob") -> pd.DataFrame:
    fn = textblob_sentiment if method == "textblob" else vader_sentiment
    rows = [fn(t) for t in texts]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    samples = [
        "I absolutely love this product! Best purchase ever 😍",
        "Worst customer service ever. Refund please!!!",
        "The package arrived on Tuesday.",
    ]
    for s in samples:
        print(s)
        print(" TextBlob:", textblob_sentiment(s))
        print(" VADER   :", vader_sentiment(s))
        print()
'''

TRANSFORMERS_SENTIMENT = '''"""Production sentiment with transformers / Phân tích cảm xúc bằng transformers."""
from __future__ import annotations
from typing import Dict, List
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline,
)


def load_sentiment_pipeline(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    """Load HF pipeline / Tải pipeline từ HuggingFace."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tok,
                     device=0 if torch.cuda.is_available() else -1,
                     top_k=None,  # return all classes
    )


def predict(pipe, texts: List[str], batch_size: int = 32) -> pd.DataFrame:
    """Batch predict với probability / Dự đoán theo lô."""
    preds = pipe(texts, batch_size=batch_size, truncation=True, max_length=512)
    # preds: List[List[{label, score}]]
    rows = []
    for p in preds:
        row = {entry["label"].lower(): float(entry["score"]) for entry in p}
        row["label"] = max(row, key=row.get) if row else "unknown"
        row["confidence"] = row.get(row["label"], 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def aspect_based_sentiment(pipe, text: str, aspects: List[str]) -> Dict[str, Dict]:
    """Aspect-based: tách câu theo aspect → sentiment cho từng aspect."""
    import re
    # Simple sentence splitter / Tách câu đơn giản
    sentences = re.split(r"(?<=[.!?])\\s+", text)
    result: Dict[str, Dict] = {}
    for aspect in aspects:
        relevant = [s for s in sentences if aspect.lower() in s.lower()]
        if not relevant:
            result[aspect] = {"label": "not_mentioned", "score": 0.0}
            continue
        joined = " ".join(relevant)
        p = pipe(joined, truncation=True, max_length=512)[0]
        result[aspect] = {"label": p["label"].lower(), "score": float(p["score"])}
    return result


if __name__ == "__main__":
    pipe = load_sentiment_pipeline()
    samples = [
        "The battery life is amazing but the screen is mediocre.",
        "Loved the camera, hated the customer service.",
    ]
    df = predict(pipe, samples)
    print(df)
    print("Aspect-based:")
    print(aspect_based_sentiment(pipe, samples[1], ["camera", "service"]))
'''

VIZ_GUIDE = """
Sentiment Visualization Guide / Hướng dẫn trực quan hóa
=========================================================
1. Histogram of polarity scores — detect skew & bimodality.
2. Time-series of average sentiment (rolling 7-day mean) — trend monitoring.
3. Wordcloud per sentiment class — discover drivers of positive/negative.
4. Stacked bar of sentiment % per category — compare products/topics.
5. Heatmap of aspect × sentiment — aspect-based dashboard.
6. Sankey diagram: source → aspect → sentiment — flow of opinions.

Class Imbalance Tips:
  - If labels are imbalanced, use stratified split + class_weight.
  - For Twitter/social data, fine-tune on a domain dataset (e.g. TweetEval).
  - Calibrate probabilities (isotonic) before using as confidence thresholds.
"""


class SentimentAnalysisSkill(Skill):
    """Sinh sentiment analysis pipeline (TextBlob/VADER/Transformers)."""

    category = SkillCategory.LANGUAGE
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "sentiment", "polarity", "opinion", "opinion mining",
        "positive", "negative", "neutral", "vader", "textblob",
        "transformers", "emotion", "aspect-based",
    ]
    examples = [
        "Phân tích cảm xúc comment khách hàng",
        "Setup sentiment với DistilBERT",
        "Aspect-based sentiment cho review sản phẩm",
    ]

    @property
    def name(self) -> str:
        return "sentiment_analysis"

    @property
    def description(self) -> str:
        return (
            "Sinh pipeline phân tích cảm xúc: TextBlob/VADER baseline, HuggingFace "
            "transformers (DistilBERT) cho production, aspect-based sentiment, "
            "batch processing + visualization guide."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.14
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        prompt_lower = (context.prompt or "").lower()
        if "aspect" in prompt_lower:
            recommended = "aspect_based"
        elif "transformer" in prompt_lower or "bert" in prompt_lower or "huggingface" in prompt_lower:
            recommended = "transformers"
        elif "vader" in prompt_lower:
            recommended = "vader"
        else:
            recommended = "textblob"

        artifacts: List[Dict[str, str]] = [
            {"name": "baseline_sentiment.py", "language": "python", "content": TEXTBLOB_VADER},
            {"name": "transformers_sentiment.py", "language": "python", "content": TRANSFORMERS_SENTIMENT},
            {"name": "VIZ_GUIDE.md", "language": "markdown", "content": VIZ_GUIDE},
        ]

        return SkillResult(
            success=True,
            output=(
                f"[sentiment_analysis] recommended={recommended}\n"
                f"Generated TextBlob+VADER baseline, DistilBERT transformers pipeline, "
                f"aspect-based sentiment, and visualization guide."
            ),
            artifacts=artifacts,
            suggestions=[
                "Use VADER for social media (handles emoji, ALL CAPS, !)",
                "Fine-tune on domain data for production accuracy",
                "Calibrate probabilities (isotonic) before thresholding",
                "Cache embeddings if running batch on >1M texts",
                "Track sentiment over time with rolling 7-day mean for trend detection",
            ],
            metadata={
                "skill": self.name,
                "recommended_model": recommended,
                "models_available": ["textblob", "vader", "distilbert-sst2", "aspect_based"],
                "languages_supported": ["en"],
                "version": self.version,
                "author": self.author,
            },
        )
