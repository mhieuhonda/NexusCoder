"""Quality Filter - Lọc low-quality samples."""
from __future__ import annotations

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Quality metrics của một sample."""
    length: int
    word_count: int
    avg_word_length: float
    unique_word_ratio: float
    has_code: bool
    has_urls: bool
    has_special_chars: bool
    repetition_score: float
    quality_score: float
    passed: bool


class QualityFilter:
    """Filter samples dựa trên quality heuristics.
    
    Criteria:
    - Length: min 50, max 100k chars
    - Word count: min 10
    - Unique word ratio: > 0.3
    - Repetition score: < 0.5
    - No excessive special chars
    - No obvious spam/garbage
    
    Usage:
        qf = QualityFilter()
        if qf.filter(sample):
            keep_sample(sample)
    """
    
    # Patterns indicating low quality
    SPAM_PATTERNS = [
        r"click\s+here",
        r"buy\s+now",
        r"free\s+download",
        r"limited\s+time\s+offer",
        r"\$\$\$",
        r"viagra|casino|lottery",
    ]
    SPAM_RE = re.compile("|".join(SPAM_PATTERNS), re.IGNORECASE)
    
    # Code indicators
    CODE_PATTERNS = [
        r"```", r"def\s+\w+\s*\(", r"function\s+\w+\s*\(",
        r"class\s+\w+", r"import\s+\w+", r"from\s+\w+\s+import",
        r"console\.log", r"print\s*\(", r"return\s+",
    ]
    CODE_RE = re.compile("|".join(CODE_PATTERNS))
    
    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 100000,
        min_words: int = 10,
        min_unique_ratio: float = 0.3,
        max_repetition: float = 0.5,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.min_unique_ratio = min_unique_ratio
        self.max_repetition = max_repetition
    
    def compute_metrics(self, text: str) -> QualityMetrics:
        """Compute quality metrics."""
        if not text:
            return QualityMetrics(0, 0, 0, 0, False, False, False, 1.0, 0.0, False)
        
        length = len(text)
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return QualityMetrics(length, 0, 0, 0, False, False, False, 1.0, 0.0, False)
        
        avg_word_length = sum(len(w) for w in words) / word_count
        unique_words = set(w.lower() for w in words)
        unique_ratio = len(unique_words) / word_count
        
        has_code = bool(self.CODE_RE.search(text))
        has_urls = bool(re.search(r"https?://\S+", text))
        has_special = bool(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text))
        
        # Repetition: check if any 10-word sequence repeats more than 3 times
        # v0.4 fix: range(word_count - 9) so the last window (words[-10:]) is included.
        repetition_score = 0.0
        if word_count > 30:
            sequences = {}
            for i in range(word_count - 9):
                seq = " ".join(words[i:i+10]).lower()
                sequences[seq] = sequences.get(seq, 0) + 1
            max_repeat = max(sequences.values()) if sequences else 0
            repetition_score = min(1.0, max_repeat / 5)
        
        # Compute overall quality score
        score = 0.5
        if self.min_length <= length <= self.max_length:
            score += 0.1
        if word_count >= self.min_words:
            score += 0.1
        if unique_ratio >= self.min_unique_ratio:
            score += 0.1
        if repetition_score <= self.max_repetition:
            score += 0.1
        if not has_special:
            score += 0.05
        if has_code:
            score += 0.05  # Code samples are valuable
        if not self.SPAM_RE.search(text):
            score += 0.05
        
        score = min(1.0, score)
        passed = score >= 0.6
        
        return QualityMetrics(
            length=length,
            word_count=word_count,
            avg_word_length=avg_word_length,
            unique_word_ratio=unique_ratio,
            has_code=has_code,
            has_urls=has_urls,
            has_special_chars=has_special,
            repetition_score=repetition_score,
            quality_score=score,
            passed=passed,
        )
    
    def filter(self, sample: Dict[str, Any]) -> bool:
        """Return True if sample passes quality filter."""
        text = sample.get("text", "")
        metrics = self.compute_metrics(text)
        return metrics.passed
    
    def process(self, samples):
        """Filter iterator of samples."""
        for sample in samples:
            if self.filter(sample):
                # Attach metrics to metadata
                metrics = self.compute_metrics(sample.get("text", ""))
                sample = dict(sample)
                sample["metadata"] = sample.get("metadata", {})
                sample["metadata"]["quality"] = {
                    "score": metrics.quality_score,
                    "length": metrics.length,
                    "word_count": metrics.word_count,
                    "has_code": metrics.has_code,
                }
                yield sample
