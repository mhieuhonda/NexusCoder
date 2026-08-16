"""Data collectors package (v0.3 expanded).

v0.2: GitHub, HuggingFace, arXiv, Wikipedia, StackOverflow
v0.3: + The-Stack, StarCoder2-data, Python-Alpaca
"""
from .github_collector import GitHubCollector
from .huggingface_collector import HuggingFaceCollector
from .arxiv_collector import ArxivCollector
from .wikipedia_collector import WikipediaCollector
from .stackoverflow_collector import StackOverflowCollector

# v0.3 NEW
try:
    from .the_stack_collector import TheStackCollector
except ImportError:
    TheStackCollector = None  # type: ignore

try:
    from .starcoder2_collector import StarCoder2Collector
except ImportError:
    StarCoder2Collector = None  # type: ignore

try:
    from .python_alpaca_collector import PythonAlpacaCollector
except ImportError:
    PythonAlpacaCollector = None  # type: ignore


__all__ = [
    "GitHubCollector",
    "HuggingFaceCollector",
    "ArxivCollector",
    "WikipediaCollector",
    "StackOverflowCollector",
    # v0.3 NEW
    "TheStackCollector",
    "StarCoder2Collector",
    "PythonAlpacaCollector",
]
