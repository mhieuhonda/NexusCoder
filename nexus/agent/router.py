"""Tool Router - Phát hiện và route tool calls từ user input."""
from __future__ import annotations

import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Một tool call được detect."""
    name: str
    args: Dict[str, Any]
    raw: str  # Original text that triggered the call


class ToolRouter:
    """Phát hiện tool calls trong user input và route chúng.
    
    Detects patterns like:
    - "read file /path/to/file"
    - "execute: ls -la"
    - "@tool file_read path=/tmp/test.txt"
    - JSON: {"tool": "file_read", "args": {"path": "/tmp/test.txt"}}
    
    Usage:
        router = ToolRouter(tool_registry)
        calls = router.detect_tool_calls(user_input)
        for call in calls:
            result = tool_registry.execute(call["name"], call["args"], ctx)
    """
    
    # Natural language patterns
    NL_PATTERNS = [
        # (regex, tool_name, arg_extractor)
        (r"read\s+(?:file\s+)?[`'\"]?([^\s`'\"]+)[`'\"]?", "file_read", lambda m: {"path": m.group(1)}),
        (r"(?:write|save)\s+(?:file\s+)?[`'\"]?([^\s`'\"]+)[`'\"]?\s*(?:with|containing|:)?\s*(.*)", "file_write", lambda m: {"path": m.group(1), "content": m.group(2) or ""}),
        (r"(?:list|ls)\s+(?:files?\s+)?(?:in\s+)?[`'\"]?([^\s`'\"]+)[`'\"]?", "file_list", lambda m: {"path": m.group(1)}),
        (r"(?:run|execute|exec)\s*[:`]?\s*(.+)", "shell_exec", lambda m: {"command": m.group(1).strip("`'\" ")}),
        (r"(?:search|grep)\s+(?:for\s+)?[`'\"]?([^`'\"]+)[`'\"]?\s*(?:in\s+)?([^\s]*)?", "regex_search", lambda m: {"pattern": m.group(1), "path": m.group(2) or "."}),
        (r"(?:http\s+)?(?:get|post|put|delete)\s+([^\s]+)", "http_request", lambda m: {"url": m.group(1), "method": "GET" if "get" in m.group(0).lower() else "POST"}),
        (r"fetch\s+([^\s]+)", "web_fetch", lambda m: {"url": m.group(1)}),
        (r"search\s+(?:web\s+)?(?:for\s+)?[`'\"]?([^`'\"]+)[`'\"]?", "web_search", lambda m: {"query": m.group(1)}),
        (r"(?:git\s+)?(status|log|diff|add|commit|push|pull|branch)\b\s*(.*)", "git_ops", lambda m: {"command": m.group(1) + (" " + m.group(2) if m.group(2) else "")}),
    ]
    
    def __init__(self, tool_registry=None):
        self.tool_registry = tool_registry
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), name, extractor)
            for p, name, extractor in self.NL_PATTERNS
        ]
    
    def detect_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Detect tool calls in text.
        
        Returns:
            List of {"name": ..., "args": ...}
        """
        if not text:
            return []
        
        calls = []
        
        # Check JSON format first
        json_calls = self._detect_json_calls(text)
        calls.extend(json_calls)
        
        # Check @tool format
        at_calls = self._detect_at_calls(text)
        calls.extend(at_calls)
        
        # Check natural language patterns
        nl_calls = self._detect_nl_calls(text)
        calls.extend(nl_calls)
        
        # Filter by available tools if registry provided
        if self.tool_registry:
            calls = [c for c in calls if c["name"] in self.tool_registry]
        
        # Deduplicate
        seen = set()
        unique = []
        for c in calls:
            key = (c["name"], json.dumps(c.get("args", {}), sort_keys=True))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        
        return unique
    
    def _detect_json_calls(self, text: str) -> List[Dict[str, Any]]:
        """Detect JSON-format tool calls."""
        calls = []
        # Find JSON blocks
        json_pattern = re.compile(r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}', re.DOTALL)
        for match in json_pattern.finditer(text):
            try:
                data = json.loads(match.group(0))
                if "tool" in data:
                    calls.append({
                        "name": data["tool"],
                        "args": data.get("args", {}),
                        "raw": match.group(0),
                    })
            except json.JSONDecodeError:
                continue
        return calls
    
    def _detect_at_calls(self, text: str) -> List[Dict[str, Any]]:
        """Detect @tool format calls."""
        calls = []
        # Pattern: @tool_name arg1=val1 arg2=val2
        at_pattern = re.compile(r'@(\w+)\s+([^\n]+)')
        for match in at_pattern.finditer(text):
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            
            # Parse args (key=value pairs or positional)
            args = {}
            # Try key=value
            kv_pattern = re.compile(r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))')
            kv_matches = kv_pattern.findall(args_str)
            if kv_matches:
                for k, v1, v2, v3 in kv_matches:
                    args[k] = v1 or v2 or v3
            else:
                # Positional - just take as "input"
                args["input"] = args_str
            
            calls.append({
                "name": tool_name,
                "args": args,
                "raw": match.group(0),
            })
        return calls
    
    def _detect_nl_calls(self, text: str) -> List[Dict[str, Any]]:
        """Detect natural language tool calls."""
        calls = []
        for pattern, tool_name, extractor in self._compiled_patterns:
            for match in pattern.finditer(text):
                try:
                    args = extractor(match)
                    if args:
                        calls.append({
                            "name": tool_name,
                            "args": args,
                            "raw": match.group(0),
                        })
                except (IndexError, AttributeError):
                    continue
        return calls
    
    def format_tool_help(self) -> str:
        """Generate help text for available tools."""
        if not self.tool_registry:
            return "No tools available"
        
        lines = ["Available tools:"]
        by_cat = self.tool_registry.list_by_category()
        for cat, tools in sorted(by_cat.items()):
            lines.append(f"\n[{cat.upper()}]")
            for t in tools:
                tool = self.tool_registry.get(t)
                lines.append(f"  {t}: {tool.description}")
        return "\n".join(lines)
