"""Text Cleaner - Làm sạch text data."""
from __future__ import annotations

import re
import html
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class CleanerConfig:
    """Config cho TextCleaner."""
    remove_html: bool = True
    remove_urls: bool = False
    remove_emojis: bool = False
    normalize_whitespace: bool = True
    normalize_unicode: bool = True
    remove_control_chars: bool = True
    min_length: int = 50
    max_length: int = 100000
    fix_encoding: bool = True


class TextCleaner:
    """Làm sạch text data cho training.
    
    Usage:
        cleaner = TextCleaner()
        cleaned = cleaner.clean("some messy text...")
    """
    
    # Common patterns
    HTML_TAG_RE = re.compile(r"<[^>]+>")
    URL_RE = re.compile(r"https?://\S+|www\.\S+")
    MULTI_SPACE_RE = re.compile(r"[ \t]+")
    MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
    EMOJI_RE = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    
    def __init__(self, config: CleanerConfig = None):
        self.config = config or CleanerConfig()
    
    def clean(self, text: str) -> str:
        """Clean a single text."""
        if not text or not isinstance(text, str):
            return ""
        
        cfg = self.config
        
        # Fix encoding issues
        if cfg.fix_encoding:
            text = text.replace("\ufeff", "").replace("\u200b", "")
        
        # Normalize unicode
        if cfg.normalize_unicode:
            import unicodedata
            text = unicodedata.normalize("NFC", text)
        
        # Remove control characters
        if cfg.remove_control_chars:
            text = self.CONTROL_CHARS_RE.sub("", text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        if cfg.remove_html:
            text = self.HTML_TAG_RE.sub(" ", text)
        
        # Remove URLs
        if cfg.remove_urls:
            text = self.URL_RE.sub("[URL]", text)
        
        # Remove emojis
        if cfg.remove_emojis:
            text = self.EMOJI_RE.sub("", text)
        
        # Normalize whitespace
        if cfg.normalize_whitespace:
            text = self.MULTI_SPACE_RE.sub(" ", text)
            text = self.MULTI_NEWLINE_RE.sub("\n\n", text)
            text = text.strip()
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean multiple texts."""
        return [self.clean(t) for t in texts]
    
    def filter(self, text: str) -> bool:
        """Return True if text passes quality filters."""
        if not text:
            return False
        if len(text) < self.config.min_length:
            return False
        if len(text) > self.config.max_length:
            return False
        # Check ratio of printable chars
        non_print = sum(1 for c in text if not c.isprintable() and c not in "\n\r\t")
        if non_print / len(text) > 0.05:
            return False
        # Check word repetition (low diversity)
        words = text.split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False
        return True
    
    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a sample dict (in-place safe)."""
        sample = dict(sample)
        if "text" in sample:
            cleaned = self.clean(sample["text"])
            if not self.filter(cleaned):
                return None  # Filter out
            sample["text"] = cleaned
            sample["metadata"] = sample.get("metadata", {})
            sample["metadata"]["cleaned"] = True
            sample["metadata"]["cleaned_length"] = len(cleaned)
        return sample
