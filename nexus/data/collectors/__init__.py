"""Data collectors package."""
from .github_collector import GitHubCollector
from .huggingface_collector import HuggingFaceCollector
from .arxiv_collector import ArxivCollector
from .wikipedia_collector import WikipediaCollector
from .stackoverflow_collector import StackOverflowCollector

__all__ = [
    "GitHubCollector",
    "HuggingFaceCollector",
    "ArxivCollector",
    "WikipediaCollector",
    "StackOverflowCollector",
]
