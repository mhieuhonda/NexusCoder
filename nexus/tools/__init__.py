"""
Nexus Tools Module - v0.2 NEW
=============================
Hệ thống Tools cho Nexus Coder Agent.

Tools là các hàm executable mà Agent có thể gọi để tương tác với môi trường:
- file_ops: read/write/list files
- shell: execute bash commands (sandboxed)
- python_exec: execute Python code
- git_ops: git operations
- web_tools: HTTP requests, web search, fetch
- code_tools: code search, lint, format
- calculator: math operations
- parsers: JSON/YAML/TOML/CSV parsing
- search: regex search in files
- archive: zip/tar operations
- crypto: hash, encrypt/decrypt
- datetime: time/date operations
- network: DNS, ping, port check
- system: system info, env vars
- converters: unit conversions
"""

from .base import Tool, ToolResult, ToolContext
from .registry import ToolRegistry, get_global_registry

__all__ = [
    "Tool",
    "ToolResult",
    "ToolContext",
    "ToolRegistry",
    "get_global_registry",
]
