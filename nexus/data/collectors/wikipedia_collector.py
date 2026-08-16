"""
Wikipedia Collector - Thu thập dữ liệu từ Wikipedia
====================================================
"""
from __future__ import annotations

import logging
import urllib.request
import urllib.parse
import json
from typing import List, Dict, Optional, Iterator, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WikiArticle:
    """Một Wikipedia article."""
    title: str
    content: str
    url: str
    language: str
    categories: List[str]


class WikipediaCollector:
    """Collect articles từ Wikipedia API.
    
    Supports Vietnamese (vi) and English (en) Wikipedia.
    """
    
    BASE_URLS = {
        "vi": "https://vi.wikipedia.org/w/api.php",
        "en": "https://en.wikipedia.org/w/api.php",
    }
    
    RANDOM_TOPICS = {
        "vi": [
            "Trí tuệ nhân tạo", "Học máy", "Mạng nơ-ron nhân tạo",
            "Python (ngôn ngữ lập trình)", "JavaScript", "Linux",
            "Cơ sở dữ liệu", "Thuật toán", "Cấu trúc dữ liệu",
            "Lập trình hướng đối tượng", "API", "JSON", "Git",
            "Hệ điều hành", "Máy học sâu", "Xử lý ngôn ngữ tự nhiên",
            "Học sâu", "Big data", "Điện toán đám mây",
        ],
        "en": [
            "Artificial intelligence", "Machine learning", "Neural network",
            "Python (programming language)", "JavaScript", "Linux",
            "Database", "Algorithm", "Data structure",
            "Object-oriented programming", "API", "JSON", "Git",
            "Operating system", "Deep learning", "Natural language processing",
            "Big data", "Cloud computing", "Transformer (deep learning model)",
            "Large language model", "GPT-4", "BERT (language model)",
        ],
    }
    
    def __init__(self, language: str = "vi"):
        self.language = language
        self.base_url = self.BASE_URLS.get(language, self.BASE_URLS["en"])
    
    def get_article(self, title: str) -> Optional[WikiArticle]:
        """Lấy nội dung một Wikipedia article theo title."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|categories",
            "exintro": "false",
            "explaintext": "true",
            "cllimit": "10",
            "format": "json",
            "redirects": "1",
        }
        
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NexusCoder-Collector/0.2"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
            
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None
            
            page = list(pages.values())[0]
            if "missing" in page:
                return None
            
            content = page.get("extract", "")
            if not content or len(content) < 100:
                return None
            
            categories = []
            for cat in page.get("categories", []):
                categories.append(cat["title"].replace("Category:", ""))
            
            title_resolved = page.get("title", title)
            url_resolved = f"https://{self.language}.wikipedia.org/wiki/{urllib.parse.quote(title_resolved.replace(' ', '_'))}"
            
            return WikiArticle(
                title=title_resolved,
                content=content,
                url=url_resolved,
                language=self.language,
                categories=categories,
            )
        except Exception as e:
            logger.error(f"Wikipedia fetch failed for '{title}': {e}")
            return None
    
    def collect(
        self,
        topics: Optional[List[str]] = None,
        max_per_topic: int = 1,
    ) -> Iterator[Dict[str, Any]]:
        """Collect articles, yield as text samples."""
        topics = topics or self.RANDOM_TOPICS.get(self.language, self.RANDOM_TOPICS["en"])
        
        for topic in topics:
            article = self.get_article(topic)
            if article:
                yield {
                    "text": f"# {article.title}\n\n{article.content}",
                    "source": f"wikipedia_{self.language}",
                    "language": self.language,
                    "metadata": {
                        "title": article.title,
                        "url": article.url,
                        "categories": article.categories,
                    },
                }
    
    def collect_random(self, count: int = 100) -> Iterator[Dict[str, Any]]:
        """Collect random articles via Wikipedia API."""
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": "0",  # Main namespace
            "rnlimit": str(count),
            "format": "json",
        }
        
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NexusCoder-Collector/0.2"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
            
            for item in data.get("query", {}).get("random", []):
                article = self.get_article(item["title"])
                if article:
                    yield {
                        "text": f"# {article.title}\n\n{article.content}",
                        "source": f"wikipedia_{self.language}_random",
                        "language": self.language,
                        "metadata": {
                            "title": article.title,
                            "url": article.url,
                        },
                    }
        except Exception as e:
            logger.error(f"Wikipedia random failed: {e}")
