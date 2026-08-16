"""Regex Search Tool - search files với regex."""
from __future__ import annotations

import os
import re
from typing import Dict, Any

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class RegexSearchTool(Tool):
    """Search regex trong files."""
    category = ToolCategory.CODE
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "regex_search"
    
    @property
    def description(self) -> str:
        return "Search regex trong files. Hỗ trợ groups, case-insensitive, multiline."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "file_glob": {"type": "string", "default": "*"},
                "flags": {"type": "string", "default": ""},
                "max_matches": {"type": "integer", "default": 100},
            },
            "required": ["pattern", "path"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        import fnmatch
        
        pattern = args["pattern"]
        path = args["path"]
        file_glob = args.get("file_glob", "*")
        flags_str = args.get("flags", "")
        max_matches = args.get("max_matches", 100)
        
        flags = 0
        if "i" in flags_str: flags |= re.IGNORECASE
        if "m" in flags_str: flags |= re.MULTILINE
        if "s" in flags_str: flags |= re.DOTALL
        if "x" in flags_str: flags |= re.VERBOSE
        
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(success=False, error=f"Invalid regex: {e}", return_code=2)
        
        full_path = path if os.path.isabs(path) else os.path.join(context.working_dir, path)
        
        matches = []
        files_scanned = 0
        
        if os.path.isfile(full_path):
            files_to_scan = [full_path]
        else:
            files_to_scan = []
            for root, dirs, files in os.walk(full_path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in (
                    "venv", "__pycache__", "node_modules", ".git", "dist", "build",
                )]
                for fname in files:
                    if fnmatch.fnmatch(fname, file_glob):
                        files_to_scan.append(os.path.join(root, fname))
        
        for fpath in files_to_scan:
            files_scanned += 1
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                for m in regex.finditer(content):
                    line_num = content[:m.start()].count("\n") + 1
                    matches.append({
                        "file": fpath,
                        "line": line_num,
                        "match": m.group(0),
                        "groups": m.groups(),
                    })
                    if len(matches) >= max_matches:
                        break
            except Exception:
                continue
            if len(matches) >= max_matches:
                break
        
        output_lines = []
        for m in matches:
            output_lines.append(f"{m['file']}:{m['line']}: {m['match']}")
        
        return ToolResult(
            success=True,
            output="\n".join(output_lines) if output_lines else "No matches",
            metadata={
                "total_matches": len(matches),
                "files_scanned": files_scanned,
                "truncated": len(matches) >= max_matches,
            },
        )
