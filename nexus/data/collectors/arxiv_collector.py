"""
Arxiv Collector - Thu thập scientific papers từ arXiv
======================================================
"""
from __future__ import annotations

import os
import logging
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Iterator, Any
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Thông tin một arXiv paper."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    pdf_url: str


class ArxivCollector:
    """Collect papers từ arXiv API.
    
    Usage:
        collector = ArxivCollector()
        papers = collector.search("transformer attention", max_results=100)
        for paper in papers:
            print(paper.title)
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    CATEGORIES = [
        "cs.CL",  # Computation and Language (NLP)
        "cs.LG",  # Machine Learning
        "cs.AI",  # Artificial Intelligence
        "cs.SE",  # Software Engineering
        "cs.PL",  # Programming Languages
        "cs.CV",  # Computer Vision
        "stat.ML",  # Statistics - Machine Learning
    ]
    
    def __init__(self, delay: float = 3.0):
        """Args:
            delay: Seconds between API calls (arXiv rate limit: 1 req per 3s)
        """
        self.delay = delay
        self._last_request = 0.0
    
    def search(
        self,
        query: str,
        max_results: int = 100,
        category: Optional[str] = None,
        sort_by: str = "relevance",
    ) -> List[ArxivPaper]:
        """Search arXiv papers.
        
        Args:
            query: Search query
            max_results: Max papers to return
            category: Filter by arXiv category (e.g. "cs.CL")
            sort_by: "relevance", "lastUpdatedDate", "submittedDate"
        """
        self._rate_limit()
        
        params = {
            "search_query": self._build_query(query, category),
            "start": 0,
            "max_results": min(max_results, 2000),
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        
        url = f"{self.BASE_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NexusCoder-Collector/0.2"})
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read().decode()
            
            return self._parse_response(xml_data)
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []
    
    def _build_query(self, query: str, category: Optional[str]) -> str:
        """Build arXiv query string (URL-encoded for safety)."""
        # v0.4 fix: use urllib.parse.quote so special chars in query don't break URL.
        import urllib.parse
        parts = []
        if query:
            q = urllib.parse.quote(query, safe='')
            parts.append(f'(abs:"{q}" OR ti:"{q}")')
        if category:
            parts.append(f"cat:{category}")
        return " AND ".join(parts) if parts else "all:*"

    def _parse_response(self, xml_data: str) -> List[ArxivPaper]:
        """Parse arXiv API XML response."""
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        papers = []
        try:
            root = ET.fromstring(xml_data)
            for entry in root.findall("atom:entry", ns):
                # v0.4 fix: None-safe access for each field
                id_el = entry.find("atom:id", ns)
                arxiv_id = (
                    id_el.text.split("/")[-1]
                    if id_el is not None and id_el.text
                    else ""
                )

                title_el = entry.find("atom:title", ns)
                title = (
                    title_el.text.strip().replace("\n", " ")
                    if title_el is not None and title_el.text
                    else ""
                )

                summary_el = entry.find("atom:summary", ns)
                abstract = (
                    summary_el.text.strip().replace("\n", " ")
                    if summary_el is not None and summary_el.text
                    else ""
                )

                published_el = entry.find("atom:published", ns)
                published = (
                    published_el.text
                    if published_el is not None and published_el.text
                    else ""
                )

                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        authors.append(name.text)
                
                categories = []
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href")
                
                # Get categories
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term")
                    if term:
                        categories.append(term)
                
                papers.append(ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    categories=categories,
                    published=published,
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                ))
        except Exception as e:
            logger.error(f"Parse error: {e}")
        
        return papers
    
    def _rate_limit(self) -> None:
        """Enforce rate limit."""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()
    
    def collect(self, queries: List[str], max_per_query: int = 100) -> Iterator[Dict[str, Any]]:
        """Collect papers from multiple queries, yield as text samples."""
        for query in queries:
            papers = self.search(query, max_results=max_per_query)
            for paper in papers:
                yield {
                    "text": f"Title: {paper.title}\n\nAuthors: {', '.join(paper.authors)}\n\nAbstract: {paper.abstract}",
                    "source": "arxiv",
                    "language": "en",
                    "metadata": {
                        "arxiv_id": paper.arxiv_id,
                        "categories": paper.categories,
                        "published": paper.published,
                    },
                }


# Curated search queries for ML/CS topics
CURATED_QUERIES = [
    "transformer architecture",
    "mixture of experts",
    "large language model",
    "attention mechanism",
    "code generation",
    "program synthesis",
    "neural machine translation",
    "retrieval augmented generation",
    "instruction tuning",
    "reinforcement learning human feedback",
    "chain of thought reasoning",
    "prompt engineering",
    "fine-tuning language model",
    "quantization neural network",
    "knowledge distillation",
    "multi-agent systems",
    "tool use language model",
    "code completion",
    "static analysis",
    "program verification",
]
