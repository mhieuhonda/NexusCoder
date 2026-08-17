"""
StackOverflow Collector - Thu thập Q&A từ StackOverflow
=========================================================
"""
from __future__ import annotations

import logging
import urllib.request
import urllib.parse
import json
import time
from typing import List, Dict, Optional, Iterator, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SOQuestion:
    """Một StackOverflow question."""
    question_id: int
    title: str
    body: str
    tags: List[str]
    score: int
    answer_count: int
    accepted_answer_id: Optional[int] = None
    answers: List[Dict] = None


# v0.4 fix: expose at module level (was inside the class, broke `from ... import CURATED_TAGS`)
CURATED_TAGS = [
    "python", "javascript", "java", "c#", "php", "android",
    "html", "jquery", "c++", "css", "ios", "mysql",
    "sql", "node.js", "reactjs", "ruby-on-rails", "vue.js",
    "typescript", "docker", "git", "go", "rust",
    "machine-learning", "deep-learning", "pytorch", "tensorflow",
    "pandas", "numpy", "regex", "algorithm", "data-structures",
    "unit-testing", "debugging", "performance", "security",
]


class StackOverflowCollector:
    """Collect Q&A từ StackOverflow API.

    StackOverflow API: 10000 requests/day without key, 50000 with key.
    Rate limit: 30 requests/second.
    """

    BASE_URL = "https://api.stackexchange.com/2.3"

    # Backward-compat alias (deprecation: prefer module-level CURATED_TAGS)
    CURATED_TAGS = CURATED_TAGS
    
    def __init__(
        self,
        key: Optional[str] = None,
        access_token: Optional[str] = None,
        page_size: int = 100,
    ):
        self.key = key
        self.access_token = access_token
        self.page_size = min(page_size, 100)
    
    def search(
        self,
        tag: str,
        max_results: int = 500,
        min_score: int = 5,
        sort: str = "votes",
    ) -> List[SOQuestion]:
        """Search questions by tag.
        
        Args:
            tag: Tag to filter (e.g. "python")
            max_results: Max questions to return
            min_score: Minimum question score
            sort: "votes", "creation", "activity"
        """
        questions = []
        page = 1
        
        while len(questions) < max_results and page <= 50:  # API limit: 50 pages
            params = {
                "order": "desc",
                "sort": sort,
                "tagged": tag,
                "site": "stackoverflow",
                "pagesize": str(self.page_size),
                "page": str(page),
                "filter": "withbody",  # Include body
                "min": str(min_score),
            }
            if self.key:
                params["key"] = self.key
            if self.access_token:
                params["access_token"] = self.access_token
            
            url = f"{self.BASE_URL}/questions?{urllib.parse.urlencode(params)}"
            
            try:
                req = urllib.request.Request(url, headers={
                    "Accept-Encoding": "gzip",
                    "User-Agent": "NexusCoder-Collector/0.2",
                })
                with urllib.request.urlopen(req, timeout=30) as response:
                    # Handle gzip
                    if response.headers.get("Content-Encoding") == "gzip":
                        import gzip
                        data = json.loads(gzip.decompress(response.read()).decode())
                    else:
                        data = json.loads(response.read().decode())
                
                items = data.get("items", [])
                if not items:
                    break
                
                for item in items:
                    questions.append(SOQuestion(
                        question_id=item["question_id"],
                        title=item["title"],
                        body=item.get("body", ""),
                        tags=item.get("tags", []),
                        score=item.get("score", 0),
                        answer_count=item.get("answer_count", 0),
                        accepted_answer_id=item.get("accepted_answer_id"),
                    ))
                
                # Check if more pages
                if not data.get("has_more", False):
                    break
                
                # Backoff if needed
                if data.get("backoff"):
                    time.sleep(data["backoff"])
                
                page += 1
                time.sleep(0.5)  # Polite delay
                
            except Exception as e:
                logger.error(f"SO search failed: {e}")
                break
        
        return questions[:max_results]
    
    def get_answers(self, question_ids: List[int]) -> Dict[int, List[Dict]]:
        """Get answers for multiple questions."""
        if not question_ids:
            return {}
        
        ids_str = ";".join(str(qid) for qid in question_ids[:100])  # Max 100 ids
        params = {
            "order": "desc",
            "sort": "votes",
            "site": "stackoverflow",
            "filter": "withbody",
        }
        if self.key:
            params["key"] = self.key
        
        url = f"{self.BASE_URL}/questions/{ids_str}/answers?{urllib.parse.urlencode(params)}"
        
        try:
            req = urllib.request.Request(url, headers={
                "Accept-Encoding": "gzip",
                "User-Agent": "NexusCoder-Collector/0.2",
            })
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    data = json.loads(gzip.decompress(response.read()).decode())
                else:
                    data = json.loads(response.read().decode())
            
            answers_by_q = {}
            for ans in data.get("items", []):
                qid = ans["question_id"]
                if qid not in answers_by_q:
                    answers_by_q[qid] = []
                answers_by_q[qid].append({
                    "answer_id": ans["answer_id"],
                    "body": ans.get("body", ""),
                    "score": ans.get("score", 0),
                    "is_accepted": ans.get("is_accepted", False),
                })
            
            return answers_by_q
        except Exception as e:
            logger.error(f"SO get_answers failed: {e}")
            return {}
    
    def collect(
        self,
        tags: Optional[List[str]] = None,
        max_per_tag: int = 100,
        include_answers: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """Collect Q&A pairs as training samples.
        
        Yields:
            Dict with keys: text (formatted Q&A), source, language, metadata
        """
        tags = tags or self.CURATED_TAGS[:10]
        
        for tag in tags:
            logger.info(f"Collecting SO tag: {tag}")
            questions = self.search(tag, max_results=max_per_tag)
            
            if include_answers and questions:
                qids = [q.question_id for q in questions if q.accepted_answer_id]
                answers_by_q = self.get_answers(qids)
            else:
                answers_by_q = {}
            
            for q in questions:
                # Format as Q&A pair
                answer_text = ""
                if q.question_id in answers_by_q:
                    accepted = [a for a in answers_by_q[q.question_id] if a["is_accepted"]]
                    if accepted:
                        answer_text = accepted[0]["body"]
                    elif answers_by_q[q.question_id]:
                        answer_text = answers_by_q[q.question_id][0]["body"]
                
                if not answer_text:
                    continue
                
                # Strip HTML tags (simple)
                import re
                q_body_clean = re.sub(r"<[^>]+>", "", q.body)
                a_clean = re.sub(r"<[^>]+>", "", answer_text)
                
                text = (
                    f"Question: {q.title}\n\n"
                    f"Tags: {', '.join(q.tags)}\n\n"
                    f"{q_body_clean}\n\n"
                    f"Answer:\n{a_clean}"
                )
                
                yield {
                    "text": text,
                    "source": "stackoverflow",
                    "language": "en",
                    "metadata": {
                        "question_id": q.question_id,
                        "tags": q.tags,
                        "score": q.score,
                        "title": q.title,
                    },
                }
