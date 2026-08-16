"""
Skill Base Class - Nền tảng cho tất cả skills
==============================================
Định nghĩa interface chung cho mọi skill trong Nexus Coder.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class SkillCategory(str, Enum):
    """Phân loại skills theo domain."""
    CODE = "code"
    REASONING = "reasoning"
    LANGUAGE = "language"
    DATA = "data"
    DEVOPS = "devops"
    SECURITY = "security"


class SkillPriority(str, Enum):
    """Độ ưu tiên khi nhiều skills match."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SkillContext:
    """Context passed to skill khi execute.
    
    Attributes:
        prompt: Câu lệnh từ user
        language: Ngôn ngữ lập trình (nếu có)
        files: Danh sách file liên quan
        history: Lịch sử hội thoại
        metadata: Extra metadata
        max_tokens: Giới hạn output
        temperature: Sampling temperature
    """
    prompt: str = ""
    language: Optional[str] = None
    files: List[str] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class SkillResult:
    """Kết quả trả về từ skill.
    
    Attributes:
        success: Có thành công không
        output: Output text
        artifacts: Files/code được tạo
        suggestions: Gợi ý tiếp theo
        error: Thông báo lỗi nếu có
        metadata: Extra metadata
    """
    success: bool = True
    output: str = ""
    artifacts: List[Dict[str, str]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    """Base class cho mọi skill trong Nexus Coder.
    
    Mỗi skill phải implement:
    - name: Tên định danh duy nhất
    - description: Mô tả ngắn
    - execute: Hàm chính thực thi skill
    - can_handle: Kiểm tra xem skill có xử lý được prompt không
    """
    
    category: SkillCategory = SkillCategory.CODE
    priority: SkillPriority = SkillPriority.MEDIUM
    keywords: List[str] = []
    examples: List[str] = []
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tên duy nhất của skill (snake_case)."""
        ...
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Mô tả ngắn gọn skill làm gì."""
        ...
    
    @property
    def version(self) -> str:
        return "0.2.0"
    
    @property
    def author(self) -> str:
        return "Hieu Louis"
    
    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        """Trả về confidence score [0.0, 1.0] cho prompt này.
        
        Default implementation: match keywords.
        Override để implement logic phức tạp hơn.
        """
        if not prompt:
            return 0.0
        prompt_lower = prompt.lower()
        if not self.keywords:
            return 0.1
        matches = sum(1 for kw in self.keywords if kw.lower() in prompt_lower)
        return min(1.0, matches / max(1, len(self.keywords)) * 2)
    
    @abstractmethod
    def execute(self, context: SkillContext) -> SkillResult:
        """Thực thi skill với context đã cho."""
        ...
    
    def get_system_prompt(self) -> str:
        """System prompt đặc thù cho skill (dùng khi gọi LLM)."""
        return f"You are using the {self.name} skill. {self.description}"
    
    def __repr__(self) -> str:
        return f"<Skill {self.name} (priority={self.priority.value})>"
