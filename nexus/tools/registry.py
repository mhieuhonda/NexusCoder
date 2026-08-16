"""
Tool Registry v0.3 - Đăng ký và quản lý tools
=============================================
Central registry cho tất cả tools. Hỗ trợ:
- Auto-discovery tools (v0.3: dynamic introspection of tools/)
- Safety-gated execution
- Audit logging

v0.3: dynamic discovery — drop a `<name>.py` with a `Tool` subclass and
it auto-registers. No more manual edits when adding a new tool file.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
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
            pass  # silently overwrite for hot reload
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
        return sorted(self._tools.keys())

    def list_by_category(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for name, tool in self._tools.items():
            cat = tool.category.value
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(name)
        return groups

    def list_by_safety(self) -> Dict[str, List[str]]:
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
                pass

        # Audit log
        self._audit(tool, args, context, result)
        return result

    def _audit(self, tool: Tool, args: Dict, context: ToolContext, result: ToolResult) -> None:
        if not self.audit_log:
            return
        entry = {
            "timestamp": time.time(),
            "tool": tool.name,
            "safety": tool.safety.value,
            "args": _safe_repr(args),
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
        self._hooks_pre.append(hook)

    def add_post_hook(self, hook: Callable) -> None:
        self._hooks_post.append(hook)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"<ToolRegistry: {len(self._tools)} tools>"


def _safe_repr(obj, max_len: int = 5000) -> str:
    """Convert args to safe string for audit (truncate long values)."""
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
        return s[:max_len] + ("…truncated" if len(s) > max_len else "")
    except Exception:
        return repr(obj)[:max_len]


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
    """Auto-register tất cả built-in tools via dynamic module discovery.

    v0.3: scans nexus/tools/*.py, imports each module, finds all `Tool`
    subclasses, instantiates them, and registers. No more manual edits
    needed when adding a new tool file.
    """
    import nexus.tools as tools_pkg

    # Known alias map (preserves v0.2 aliases for backward compat)
    _ALIASES: Dict[str, List[str]] = {
        "file_read": ["read", "cat"],
        "file_write": ["write", "save"],
        "file_list": ["ls", "list"],
        "file_delete": ["rm", "delete"],
        "shell_exec": ["bash", "exec", "sh"],
        "python_exec": ["py", "python"],
        "git": ["git"],
        "http_request": ["http", "curl"],
        "web_fetch": ["fetch", "scrape"],
        "web_search": ["search", "google"],
        "code_search": ["grep", "rg"],
        "code_lint": ["lint"],
        "code_format": ["format", "fmt"],
        "calculator": ["calc", "math"],
        "json_parse": ["json"],
        "yaml_parse": ["yaml"],
        "csv_parse": ["csv"],
        "regex_search": ["regex"],
        "archive": ["zip", "tar"],
        "hash": ["sha"],
        "encrypt": ["aes"],
        "datetime": ["time", "date"],
        "dns_lookup": ["dns"],
        "ping": ["ping"],
        # v0.3 new tools (selected aliases)
        "docker": ["docker"],
        "kubectl": ["k8s"],
        "terraform": ["tf"],
        "ansible": ["ans"],
        "aws": ["aws_cli"],
        "gcloud": ["gcp"],
        "azure": ["az"],
        "ssh": ["ssh_exec"],
        "scp": [],
        "rsync": [],
        "systemd": ["svc"],
        "crontab": ["cron"],
        "sql_runner": ["sql"],
        "sql_formatter": [],
        "sql_migrator": ["migrate"],
        "postgres": ["pg"],
        "mysql": [],
        "sqlite": [],
        "redis": [],
        "mongo": ["mongodb"],
        "elasticsearch": ["es"],
        "kafka": [],
        "rabbitmq": ["amqp"],
        "graphql_client": ["gql"],
        "websocket": ["ws"],
        "grpc": ["grpc_call"],
        "url_shortener": ["shorten"],
        "dns_query": ["dig"],
        "traceroute": ["trace"],
        "port_scanner": ["scan"],
        "ssl_checker": ["ssl"],
        "ssl_generator": ["gen_ssl"],
        "cert_checker": ["cert"],
        "web_scraper": ["scrape_adv"],
        "web_crawler": ["crawl"],
        "web_auth": ["auth"],
        "code_ast": ["ast"],
        "code_complexity": ["cc"],
        "code_dependency": ["deps_code"],
        "code_metrics": ["loc"],
        "code_smells": ["smells"],
        "code_formatter_advanced": ["fmt_adv"],
        "code_minifier": ["minify"],
        "code_transpiler": ["transpile"],
        "code_runner": ["run"],
        "code_tester": ["test"],
        "code_compiler": ["compile"],
        "code_profiler": ["profile"],
        "code_coverage": ["coverage"],
        "jwt": ["jwt_decode"],
        "oauth": ["oauth2"],
        "api_key_validator": ["check_key"],
        "markdown_converter": ["md"],
        "pdf_generator": ["pdf"],
        "image_processor": ["image"],
        "statistics": ["stats_tool"],
        "linear_algebra": ["linalg"],
        "probability": ["prob"],
        "ml_metrics": ["ml_eval"],
        "model_evaluator": ["eval_model"],
        "benchmark_runner": ["benchmark"],
        "log_analyzer": ["logs"],
    }

    for module_info in pkgutil.iter_modules(tools_pkg.__path__):
        if module_info.name.startswith("_") or module_info.name in ("base", "registry"):
            continue
        module_name = f"nexus.tools.{module_info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        # Find all Tool subclasses in the module
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if not (inspect.isclass(obj) and issubclass(obj, Tool) and obj is not Tool):
                continue
            if obj.__module__ != module_name:
                continue
            try:
                instance = obj()
                aliases = _ALIASES.get(instance.name, [])
                registry.register(instance, aliases=aliases)
            except Exception:
                continue
