"""Safety Filters - Lọc nội dung không an toàn."""
from __future__ import annotations

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FilterResult:
    """Kết quả filter."""
    passed: bool
    score: float  # 0.0 = unsafe, 1.0 = safe
    reason: Optional[str] = None
    categories: List[str] = None


class ContentFilter:
    """Filter nội dung toxic / harmful."""
    
    HARMFUL_PATTERNS = [
        # Violence
        (r"\b(kill|murder|assassinate|execute)\s+(someone|him|her|them|people)\b", "violence"),
        (r"\bbomb\s+(recipe|how\s+to\s+make)\b", "violence"),
        # Hate speech patterns
        (r"\b(racial|ethnic)\s+slur\b", "hate_speech"),
        # Self-harm
        (r"\b(suicide|self-harm)\s+(method|how\s+to)\b", "self_harm"),
        # Illegal
        (r"\b(drug|cocaine|heroin)\s+(recipe|manufacture|synthesize)\b", "illegal"),
        (r"\bchild\s+exploitation\b", "illegal"),
    ]
    
    def __init__(self):
        self._compiled = [(re.compile(p, re.IGNORECASE), cat) for p, cat in self.HARMFUL_PATTERNS]
    
    def check(self, text: str) -> FilterResult:
        """Check text for harmful content."""
        if not text:
            return FilterResult(passed=True, score=1.0)
        
        matched_categories = []
        for pattern, category in self._compiled:
            if pattern.search(text):
                matched_categories.append(category)
        
        if matched_categories:
            return FilterResult(
                passed=False,
                score=0.0,
                reason=f"Harmful content detected: {', '.join(matched_categories)}",
                categories=matched_categories,
            )
        
        return FilterResult(passed=True, score=1.0)


class PIIFilter:
    """Detect and mask PII (Personally Identifiable Information)."""
    
    PII_PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"\b\+?[\d\s\-\(\)]{10,15}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
        "ip": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        "api_key": re.compile(r"\b(?:sk-|pk-|ghp_|gho_|github_pat_)[A-Za-z0-9]{20,}\b"),
    }
    
    def check(self, text: str) -> FilterResult:
        """Check for PII."""
        if not text:
            return FilterResult(passed=True, score=1.0)
        
        found = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            if pattern.search(text):
                found.append(pii_type)
        
        if found:
            return FilterResult(
                passed=False,
                score=0.3,
                reason=f"PII detected: {', '.join(found)}",
                categories=found,
            )
        
        return FilterResult(passed=True, score=1.0)
    
    def mask(self, text: str) -> str:
        """Mask PII in text."""
        for pii_type, pattern in self.PII_PATTERNS.items():
            text = pattern.sub(f"[{pii_type.upper()}_REDACTED]", text)
        return text


class SafetyFilter:
    """Composite safety filter."""
    
    def __init__(self, enable_content: bool = True, enable_pii: bool = True):
        self.content_filter = ContentFilter() if enable_content else None
        self.pii_filter = PIIFilter() if enable_pii else None
    
    def check(self, text: str) -> FilterResult:
        """Run all filters."""
        results = []
        if self.content_filter:
            results.append(("content", self.content_filter.check(text)))
        if self.pii_filter:
            results.append(("pii", self.pii_filter.check(text)))
        
        if not results:
            return FilterResult(passed=True, score=1.0)
        
        # Aggregate: passed only if ALL pass
        all_passed = all(r.passed for _, r in results)
        min_score = min(r.score for _, r in results)
        
        if all_passed:
            return FilterResult(passed=True, score=min_score)
        
        reasons = [f"{name}: {r.reason}" for name, r in results if not r.passed]
        categories = []
        for _, r in results:
            if r.categories:
                categories.extend(r.categories)
        
        return FilterResult(
            passed=False,
            score=min_score,
            reason="; ".join(reasons),
            categories=categories,
        )
    
    def sanitize(self, text: str) -> Tuple[str, FilterResult]:
        """Sanitize text: check + mask PII."""
        result = self.check(text)
        if not result.passed and self.pii_filter:
            # Try masking PII
            masked = self.pii_filter.mask(text)
            recheck = self.check(masked)
            if recheck.passed:
                return masked, recheck
        return text, result
