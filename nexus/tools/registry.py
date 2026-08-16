"""
Tool Registry - Đăng ký và quản lý tools
=========================================
Central registry cho tất cả tools. Hỗ trợ:
- Auto-discovery tools
- Safety-gated execution
- Audit logging
"""
from __future__ import annotations

from typing import Dict, List, Optional, Callable
import time
import json
import os

from .base import Tool, ToolContext, ToolResult, ToolSafety


class ToolRegistry:
    """Registry quản lý tất cả tools của Nexus Coder."""
    
    def __init__(self, audit_log: Optional[str] = None):
        self._tools: Dict[str, Tool] = {}
        self._aliases: Dict[str, str] = {}
        self._hooks_pre: List[Callable] = []
        self._hooks_post: List[Callable] = []
        self.audit_log = audit_log
        if audit_log:
            os.makedirs(os.path.dirname(audit_log) or ".", exist_ok=True)
    
    def register(self, tool: Tool, aliases: List[str] = None) -> None:
        """Đăng ký một tool mới."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' đã tồn tại")
        self._tools[tool.name] = tool
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = tool.name
    
    def unregister(self, name: str) -> Optional[Tool]:
        """Gỡ tool khỏi registry."""
        return self._tools.pop(name, None)
    
    def get(self, name: str) -> Optional[Tool]:
        """Lấy tool theo tên hoặc alias."""
        name = name.lower()
        if name in self._aliases:
            name = self._aliases[name]
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """Danh sách tên tất cả tools."""
        return sorted(self._tools.keys())
    
    def list_by_category(self) -> Dict[str, List[str]]:
        """Group tools theo category."""
        groups: Dict[str, List[str]] = {}
        for name, tool in self._tools.items():
            cat = tool.category.value
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(name)
        return groups
    
    def list_by_safety(self) -> Dict[str, List[str]]:
        """Group tools theo safety level."""
        groups: Dict[str, List[str]] = {}
        for name, tool in self._tools.items():
            s = tool.safety.value
            if s not in groups:
                groups[s] = []
            groups[s].append(name)
        return groups
    
    def execute(self, name: str, args: Dict, context: ToolContext = None) -> ToolResult:
        """Execute tool theo tên với safety checks và audit."""
        if context is None:
            context = ToolContext()
        
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found",
                return_code=127,
            )
        
        # Validate args
        err = tool.validate_args(args)
        if err:
            return ToolResult(success=False, error=err, return_code=2)
        
        # Safety gate
        if tool.safety in (ToolSafety.DANGEROUS, ToolSafety.DESTRUCTIVE):
            if context.dry_run:
                return ToolResult(
                    success=True,
                    output=f"[DRY RUN] Would execute {tool.name} with args: {args}",
                    metadata={"dry_run": True, "tool": tool.name},
                )
        
        # Pre-hooks
        for hook in self._hooks_pre:
            try:
                hook(tool, args, context)
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Pre-hook blocked execution: {e}",
                    return_code=126,
                )
        
        # Execute
        start = time.time()
        try:
            result = tool.execute(args, context)
        except Exception as e:
            result = ToolResult(
                success=False,
                error=f"Tool execution failed: {e}",
                return_code=1,
            )
        result.duration = time.time() - start
        
        # Post-hooks
        for hook in self._hooks_post:
            try:
                hook(tool, args, context, result)
            except Exception:
                pass  # Post-hook errors shouldn't fail execution
        
        # Audit log
        self._audit(tool, args, context, result)
        
        return result
    
    def _audit(self, tool: Tool, args: Dict, context: ToolContext, result: ToolResult) -> None:
        """Ghi audit log."""
        if not self.audit_log:
            return
        entry = {
            "timestamp": time.time(),
            "tool": tool.name,
            "safety": tool.safety.value,
            "args": args,
            "working_dir": context.working_dir,
            "user_id": context.user_id,
            "success": result.success,
            "return_code": result.return_code,
            "duration": result.duration,
        }
        try:
            with open(self.audit_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass
    
    def add_pre_hook(self, hook: Callable) -> None:
        """Add pre-execution hook (e.g., rate limiter)."""
        self._hooks_pre.append(hook)
    
    def add_post_hook(self, hook: Callable) -> None:
        """Add post-execution hook (e.g., metrics collector)."""
        self._hooks_post.append(hook)
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __repr__(self) -> str:
        return f"<ToolRegistry: {len(self._tools)} tools>"


# =============================================================================
# Global registry singleton
# =============================================================================

_GLOBAL_REGISTRY: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """Lấy global registry (auto-init nếu chưa có)."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ToolRegistry(audit_log="./logs/tool_audit.jsonl")
        _auto_register_defaults(_GLOBAL_REGISTRY)
    return _GLOBAL_REGISTRY


def _auto_register_defaults(registry: ToolRegistry) -> None:
    """Auto-register tất cả built-in tools."""
    try:
        from .file_ops import FileReadTool, FileWriteTool, FileListTool, FileDeleteTool
        registry.register(FileReadTool(), aliases=["read", "cat"])
        registry.register(FileWriteTool(), aliases=["write", "save"])
        registry.register(FileListTool(), aliases=["ls", "list"])
        registry.register(FileDeleteTool(), aliases=["rm", "delete"])
    except ImportError:
        pass
    try:
        from .shell import ShellExecTool
        registry.register(ShellExecTool(), aliases=["bash", "exec", "sh"])
    except ImportError:
        pass
    try:
        from .python_exec import PythonExecTool
        registry.register(PythonExecTool(), aliases=["py", "python"])
    except ImportError:
        pass
    try:
        from .git_ops import GitTool
        registry.register(GitTool(), aliases=["git"])
    except ImportError:
        pass
    try:
        from .web_tools import HTTPRequestTool, WebFetchTool, WebSearchTool
        registry.register(HTTPRequestTool(), aliases=["http", "curl"])
        registry.register(WebFetchTool(), aliases=["fetch", "scrape"])
        registry.register(WebSearchTool(), aliases=["search", "google"])
    except ImportError:
        pass
    try:
        from .code_tools import CodeSearchTool, CodeLintTool, CodeFormatTool
        registry.register(CodeSearchTool(), aliases=["grep", "rg"])
        registry.register(CodeLintTool(), aliases=["lint"])
        registry.register(CodeFormatTool(), aliases=["format", "fmt"])
    except ImportError:
        pass
    try:
        from .calculator import CalculatorTool
        registry.register(CalculatorTool(), aliases=["calc", "math"])
    except ImportError:
        pass
    try:
        from .parsers import JSONParserTool, YAMLParserTool, CSVParserTool
        registry.register(JSONParserTool(), aliases=["json"])
        registry.register(YAMLParserTool(), aliases=["yaml"])
        registry.register(CSVParserTool(), aliases=["csv"])
    except ImportError:
        pass
    try:
        from .search import RegexSearchTool
        registry.register(RegexSearchTool(), aliases=["regex"])
    except ImportError:
        pass
    try:
        from .archive import ArchiveTool
        registry.register(ArchiveTool(), aliases=["zip", "tar"])
    except ImportError:
        pass
    try:
        from .crypto import HashTool, EncryptTool
        registry.register(HashTool(), aliases=["hash", "sha"])
        registry.register(EncryptTool(), aliases=["encrypt", "aes"])
    except ImportError:
        pass
    try:
        from .datetime_tool import DateTimeTool
        registry.register(DateTimeTool(), aliases=["time", "date"])
    except ImportError:
        pass
    try:
        from .network import DNSTool, PingTool
        registry.register(DNSTool(), aliases=["dns"])
        registry.register(PingTool(), aliases=["ping"])
    except ImportError:
        pass
