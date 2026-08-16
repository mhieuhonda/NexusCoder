"""Parser Tools - JSON/YAML/CSV/TOML parsing."""
from __future__ import annotations

import json
import csv
import io
from typing import Dict, Any

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class JSONParserTool(Tool):
    """Parse JSON string hoặc file."""
    category = ToolCategory.PARSER
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "json_parse"
    
    @property
    def description(self) -> str:
        return "Parse JSON string/file. Hỗ trợ pretty-print, validate, query (jq-like)."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "JSON string hoặc path to .json file"},
                "query": {"type": "string", "description": "Dot-path query (e.g. 'a.b.0.c')"},
                "pretty": {"type": "boolean", "default": True},
            },
            "required": ["input"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        import os
        
        inp = args["input"]
        query = args.get("query")
        pretty = args.get("pretty", True)
        
        # Try as file path first
        if os.path.exists(inp) and inp.endswith(".json"):
            with open(inp, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Parse as string
            try:
                data = json.loads(inp)
            except json.JSONDecodeError as e:
                return ToolResult(success=False, error=f"Invalid JSON: {e}", return_code=1)
        
        # Apply query
        if query:
            for key in query.split("."):
                if isinstance(data, list):
                    try:
                        data = data[int(key)]
                    except (ValueError, IndexError):
                        return ToolResult(success=False, error=f"Invalid index: {key}", return_code=1)
                elif isinstance(data, dict):
                    if key not in data:
                        return ToolResult(success=False, error=f"Key not found: {key}", return_code=1)
                    data = data[key]
                else:
                    return ToolResult(success=False, error=f"Cannot index into {type(data).__name__}", return_code=1)
        
        if pretty:
            output = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        else:
            output = json.dumps(data, ensure_ascii=False, default=str)
        
        return ToolResult(
            success=True,
            output=output,
            metadata={"type": type(data).__name__, "query": query},
        )


class YAMLParserTool(Tool):
    """Parse YAML string hoặc file."""
    category = ToolCategory.PARSER
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "yaml_parse"
    
    @property
    def description(self) -> str:
        return "Parse YAML string/file. Requires PyYAML."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string"},
                "pretty": {"type": "boolean", "default": True},
            },
            "required": ["input"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            import yaml
        except ImportError:
            return ToolResult(
                success=False,
                error="PyYAML not installed. Run: pip install pyyaml",
                return_code=1,
            )
        
        import os
        inp = args["input"]
        pretty = args.get("pretty", True)
        
        if os.path.exists(inp) and (inp.endswith(".yaml") or inp.endswith(".yml")):
            with open(inp, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            try:
                data = yaml.safe_load(inp)
            except yaml.YAMLError as e:
                return ToolResult(success=False, error=f"Invalid YAML: {e}", return_code=1)
        
        output = json.dumps(data, indent=2, ensure_ascii=False, default=str) if pretty else str(data)
        return ToolResult(
            success=True,
            output=output,
            metadata={"type": type(data).__name__},
        )


class CSVParserTool(Tool):
    """Parse CSV file/string."""
    category = ToolCategory.PARSER
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "csv_parse"
    
    @property
    def description(self) -> str:
        return "Parse CSV. Trả về headers + rows. Hỗ trợ custom delimiter."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string"},
                "delimiter": {"type": "string", "default": ","},
                "has_header": {"type": "boolean", "default": True},
                "max_rows": {"type": "integer", "default": 100},
            },
            "required": ["input"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        import os
        inp = args["input"]
        delimiter = args.get("delimiter", ",")
        has_header = args.get("has_header", True)
        max_rows = args.get("max_rows", 100)
        
        if os.path.exists(inp):
            with open(inp, "r", encoding="utf-8", newline="") as f:
                content = f.read()
        else:
            content = inp
        
        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        rows = list(reader)[:max_rows + 1]
        
        if not rows:
            return ToolResult(success=True, output="(empty CSV)", metadata={"rows": 0})
        
        if has_header:
            headers = rows[0]
            data_rows = rows[1:]
            output_lines = [" | ".join(headers), "-" * 40]
            for row in data_rows[:max_rows]:
                output_lines.append(" | ".join(row))
            metadata = {"headers": headers, "rows": len(data_rows), "truncated": len(data_rows) >= max_rows}
        else:
            output_lines = [f"Row {i}: " + " | ".join(row) for i, row in enumerate(rows[:max_rows])]
            metadata = {"rows": len(rows), "truncated": len(rows) >= max_rows}
        
        return ToolResult(
            success=True,
            output="\n".join(output_lines),
            metadata=metadata,
        )
