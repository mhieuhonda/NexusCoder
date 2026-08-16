"""
Language Identification Processor for Nexus Coder v0.3
=====================================================
Identifies the language of each text sample and filters mislabeled ones.

Uses a fast heuristic-based detector (no external deps). Optionally uses
`langdetect` if available for higher accuracy on ambiguous samples.

Languages of interest:
  - "vi" (Vietnamese)
  - "en" (English)
  - "code" (programming code — detected via shebang, def/class, etc.)

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import re
from typing import Dict, Optional


# Regex patterns for code detection
_CODE_PATTERNS = [
    r"^\s*(def|class|import|from|package|func|fn|func|public|private|func)\s+\w+",
    r"^\s*#!\s*/",                                # shebang
    r"^\s*(#include|#define|#ifndef)\s+",        # C/C++ preprocessor
    r"^\s*(echo|set|export|alias)\s+",            # shell
    r"\b(function|return|if|else|for|while|var|let|const)\b.*\{",
]

_CODE_REGEX = re.compile("|".join(_CODE_PATTERNS), re.MULTILINE)

# Vietnamese character ranges (combining diacritics + tone marks)
_VI_CHARS = set("ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵđ")

# Common English stopwords
_EN_STOP = {
    "the", "and", "is", "are", "of", "to", "in", "that", "it", "with",
    "for", "as", "on", "at", "by", "be", "this", "an", "or", "from",
}


def detect_language(text: str, sample_size: int = 2000) -> Dict[str, float]:
    """Detect language of `text`. Returns dict {lang: confidence}.

    Returns the highest-confidence language as {"lang": "vi"/"en"/"code", "confidence": float}.
    """
    if not text or not text.strip():
        return {"lang": "unknown", "confidence": 0.0}

    sample = text[:sample_size]

    # Code detection (highest priority — code often contains natural language too)
    if _CODE_REGEX.search(sample):
        # Check if code dominates (>50% lines look like code)
        code_lines = sum(1 for line in sample.split("\n") if _CODE_REGEX.match(line))
        total_lines = max(1, len(sample.split("\n")))
        if code_lines / total_lines > 0.3:
            return {"lang": "code", "confidence": min(0.95, 0.5 + code_lines / total_lines / 2)}

    # Vietnamese: count chars with diacritics
    vi_chars = sum(1 for c in sample if c in _VI_CHARS)
    if vi_chars >= 5:
        # Definitely Vietnamese if there are many tone marks
        confidence = min(0.99, 0.5 + vi_chars / max(1, len(sample)) * 10)
        return {"lang": "vi", "confidence": confidence}

    # Try langdetect if available
    try:
        from langdetect import detect_langs
        results = detect_langs(sample)
        if results:
            top = results[0]
            lang = top.lang
            conf = float(top.prob)
            if lang == "vi":
                return {"lang": "vi", "confidence": conf}
            if lang == "en":
                return {"lang": "en", "confidence": conf}
            return {"lang": lang, "confidence": conf}
    except ImportError:
        pass
    except Exception:
        pass

    # Heuristic English: count common stopwords
    words = re.findall(r"\b[a-z]{2,}\b", sample.lower())
    if not words:
        return {"lang": "unknown", "confidence": 0.0}
    en_count = sum(1 for w in words if w in _EN_STOP)
    en_ratio = en_count / len(words)
    if en_ratio > 0.05:
        return {"lang": "en", "confidence": min(0.9, en_ratio * 5)}

    return {"lang": "unknown", "confidence": 0.0}


class LanguageIdProcessor:
    """Filter / tag samples by detected language.

    Usage:
        processor = LanguageIdProcessor(min_confidence=0.85, allowed={"vi", "en", "code"})
        for sample in stream:
            if processor.keep(sample["text"]):
                ...
    """

    def __init__(
        self,
        min_confidence: float = 0.85,
        allowed_languages: Optional[set] = None,
    ):
        self.min_confidence = min_confidence
        self.allowed_languages = allowed_languages or {"vi", "en", "code"}

    def keep(self, text: str) -> bool:
        """Return True if sample should be kept."""
        result = detect_language(text)
        if result["lang"] not in self.allowed_languages:
            return False
        return result["confidence"] >= self.min_confidence

    def tag(self, sample: Dict) -> Dict:
        """Add 'lang' and 'lang_confidence' fields to sample dict."""
        result = detect_language(sample.get("text", sample.get("content", "")))
        sample["lang"] = result["lang"]
        sample["lang_confidence"] = result["confidence"]
        return sample

    def batch_filter(self, samples):
        """Yield only samples that pass the filter."""
        for s in samples:
            text = s.get("text", s.get("content", ""))
            if self.keep(text):
                yield s


__all__ = ["detect_language", "LanguageIdProcessor"]
