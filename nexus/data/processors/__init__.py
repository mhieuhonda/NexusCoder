"""Data processors package (v0.3 expanded).

v0.2: TextCleaner, Deduplicator, QualityFilter, CodeFormatter
v0.3: + LanguageIdProcessor, CodeQualityProcessor
"""
from .cleaner import TextCleaner
from .deduplicator import Deduplicator
from .quality_filter import QualityFilter
from .code_formatter import CodeFormatter

# v0.3 NEW
try:
    from .language_id import LanguageIdProcessor
except ImportError:
    LanguageIdProcessor = None  # type: ignore

try:
    from .code_quality import CodeQualityProcessor
except ImportError:
    CodeQualityProcessor = None  # type: ignore


__all__ = [
    "TextCleaner",
    "Deduplicator",
    "QualityFilter",
    "CodeFormatter",
    # v0.3 NEW
    "LanguageIdProcessor",
    "CodeQualityProcessor",
]
