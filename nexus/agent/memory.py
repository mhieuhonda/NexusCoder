"""Memory System - Quản lý lịch sử hội thoại."""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Message:
    """Một message trong hội thoại."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """Quản lý lịch sử hội thoại với sliding window.
    
    Features:
    - Lưu trữ messages
    - Sliding window (giữ N messages gần nhất)
    - Summarization (khi đầy, summarize cũ)
    - Importance scoring
    - Search trong history
    
    Usage:
        memory = ConversationMemory(max_messages=50)
        memory.add(role="user", content="Hello")
        memory.add(role="assistant", content="Hi there!")
        history = memory.get_history()
    """
    
    def __init__(
        self,
        max_messages: int = 50,
        max_tokens: int = 4000,
        summarize_threshold: float = 0.8,
    ):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summarize_threshold = summarize_threshold
        self._messages: List[Message] = []
        self._summary: Optional[str] = None
        self._importance_scores: List[float] = []
    
    def add(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> None:
        """Add a message to memory."""
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._messages.append(msg)
        self._importance_scores.append(importance)
        
        # Trigger summarization if threshold reached
        if len(self._messages) >= self.max_messages * self.summarize_threshold:
            self._compress()
    
    def get_history(
        self,
        last_n: Optional[int] = None,
        include_summary: bool = True,
    ) -> List[Dict[str, str]]:
        """Get conversation history.
        
        Args:
            last_n: Only return last N messages (None = all)
            include_summary: Include previous summary if available
        
        Returns:
            List of {"role": ..., "content": ...}
        """
        history = []
        if include_summary and self._summary:
            history.append({
                "role": "system",
                "content": f"[Previous conversation summary]: {self._summary}",
            })
        
        messages = self._messages[-last_n:] if last_n else self._messages
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content,
            })
        
        return history
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Search in memory for relevant messages."""
        query_lower = query.lower()
        scored = []
        for msg, score in zip(self._messages, self._importance_scores):
            content_lower = msg.content.lower()
            # Simple keyword matching
            matches = sum(1 for word in query_lower.split() if word in content_lower)
            if matches > 0:
                relevance = matches / max(len(query_lower.split()), 1)
                scored.append((relevance * score, msg))
        
        scored.sort(key=lambda x: -x[0])
        return [
            {"role": m.role, "content": m.content}
            for _, m in scored[:limit]
        ]
    
    def clear(self) -> None:
        """Clear all memory."""
        self._messages.clear()
        self._importance_scores.clear()
        self._summary = None
    
    def _compress(self) -> None:
        """Compress old messages into summary."""
        # Keep recent messages, summarize older ones
        keep_count = self.max_messages // 2
        old_messages = self._messages[:-keep_count]
        old_scores = self._importance_scores[:-keep_count]
        
        # Build summary (simple: concatenate key points)
        summary_parts = []
        for msg in old_messages:
            if msg.role == "user":
                summary_parts.append(f"User asked: {msg.content[:100]}")
            elif msg.role == "assistant":
                summary_parts.append(f"Assistant replied: {msg.content[:100]}")
        
        new_summary = " | ".join(summary_parts[-10:])  # Last 10 interactions
        
        if self._summary:
            self._summary = f"{self._summary} | {new_summary}"
        else:
            self._summary = new_summary
        
        # Truncate summary if too long
        if len(self._summary) > 2000:
            self._summary = self._summary[-2000:]
        
        # Keep only recent messages
        self._messages = self._messages[-keep_count:]
        self._importance_scores = self._importance_scores[-keep_count:]
    
    def stats(self) -> Dict[str, Any]:
        """Get memory stats."""
        total_chars = sum(len(m.content) for m in self._messages)
        return {
            "message_count": len(self._messages),
            "max_messages": self.max_messages,
            "total_chars": total_chars,
            "has_summary": self._summary is not None,
            "summary_length": len(self._summary) if self._summary else 0,
        }
    
    def save(self, path: str) -> None:
        """Save memory to file."""
        data = {
            "messages": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp, "metadata": m.metadata}
                for m in self._messages
            ],
            "summary": self._summary,
            "max_messages": self.max_messages,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str) -> None:
        """Load memory from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._messages = [
            Message(
                role=m["role"],
                content=m["content"],
                timestamp=m.get("timestamp", ""),
                metadata=m.get("metadata", {}),
            )
            for m in data.get("messages", [])
        ]
        self._summary = data.get("summary")
        self.max_messages = data.get("max_messages", self.max_messages)
