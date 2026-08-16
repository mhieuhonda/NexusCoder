"""
Nexus Data Module - v0.2 NEW
============================
Pipeline thu thập và xử lý training data.

Sources:
- GitHubCollector: Code từ public GitHub repos
- HuggingFaceCollector: Datasets từ HuggingFace Hub
- ArxivCollector: Scientific papers
- WikipediaCollector: General knowledge
- StackOverflowCollector: Q&A pairs

Processors:
- TextCleaner: Làm sạch text
- CodeFormatter: Format code samples
- Deduplicator: Loại bỏ duplicates (MinHash)
- QualityFilter: Lọc low-quality samples
"""

from .collectors.github_collector import GitHubCollector
from .collectors.huggingface_collector import HuggingFaceCollector
from .collectors.arxiv_collector import ArxivCollector
from .collectors.wikipedia_collector import WikipediaCollector
from .collectors.stackoverflow_collector import StackOverflowCollector
from .processors.cleaner import TextCleaner
from .processors.deduplicator import Deduplicator
from .processors.quality_filter import QualityFilter
from .processors.code_formatter import CodeFormatter
from .curriculum import CurriculumLearning

__all__ = [
    "GitHubCollector",
    "HuggingFaceCollector",
    "ArxivCollector",
    "WikipediaCollector",
    "StackOverflowCollector",
    "TextCleaner",
    "Deduplicator",
    "QualityFilter",
    "CodeFormatter",
    "CurriculumLearning",
]
