"""
Tool Base Class - Nền tảng cho tất cả tools
============================================
Định nghĩa interface chung cho mọi tool trong Nexus Coder.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolSafety(str, Enum):
    """Mức độ an toàn của tool."""
    SAFE = "safe"            # Read-only, no side effects
    MODERATE = "moderate"    # Writes to local files
    DANGEROUS = "dangerous"  # Executes commands, network ops
    DESTRUCTIVE = "destructive"  # Can delete data, requires confirmation


class ToolCategory(str, Enum):
    """Phân loại tools."""
    FILE = "file"
    EXEC = "exec"
    WEB = "web"
    CODE = "code"
    MATH = "math"
    PARSER = "parser"
    SYSTEM = "system"
    NETWORK = "network"
    CRYPTO = "crypto"
    DATA = "data"


@dataclass
class ToolContext:
    """Context cho tool execution.
    
    Attributes:
        working_dir: Thư mục làm việc
        timeout: Timeout seconds
        env: Environment variables
        sandbox: Có chạy trong sandbox không
        user_id: ID của user (cho audit)
        dry_run: Chỉ simulate, không thực sự chạy
    """
    working_dir: str = "."
    timeout: int = 30
    env: Dict[str, str] = field(default_factory=dict)
    sandbox: bool = True
    user_id: Optional[str] = None
    dry_run: bool = False


@dataclass
class ToolResult:
    """Kết quả trả về từ tool.
    
    Attributes:
        success: Có thành công không
        output: Output text (stdout)
        error: Error output (stderr)
        return_code: Exit code (nếu có)
        artifacts: Files được tạo/sửa
        metadata: Extra metadata
        duration: Thời gian thực thi (seconds)
    """
    success: bool = True
    output: str = ""
    error: Optional[str] = None
    return_code: int = 0
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0


class Tool(ABC):
    """Base class cho mọi tool trong Nexus Coder.
    
    Mỗi tool phải implement:
    - name: Tên định danh duy nhất
    - description: Mô tả ngắn
    - execute: Hàm chính thực thi tool
    - validate_args: Validate arguments trước khi chạy
    """
    
    category: ToolCategory = ToolCategory.FILE
    safety: ToolSafety = ToolSafety.SAFE
    requires_confirmation: bool = False
    timeout: int = 30
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tên duy nhất của tool (snake_case)."""
        ...
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Mô tả ngắn gọn tool làm gì."""
        ...
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """JSON schema cho parameters."""
        return {}
    
    @property
    def version(self) -> str:
        return "0.2.0"
    
    @property
    def author(self) -> str:
        return "Hieu Louis"
    
    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate args. Trả về error message nếu invalid, None nếu OK."""
        return None
    
    @abstractmethod
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        """Thực thi tool với args và context."""
        ...
    
    def __repr__(self) -> str:
        return f"<Tool {self.name} (safety={self.safety.value})>"
