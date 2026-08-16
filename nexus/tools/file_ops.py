"""File Operations Tools - read/write/list/delete files."""
from __future__ import annotations

import os
import glob
from typing import Dict, Any, List
from pathlib import Path

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class FileReadTool(Tool):
    """Đọc nội dung file."""
    category = ToolCategory.FILE
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "file_read"
    
    @property
    def description(self) -> str:
        return "Đọc nội dung file text. Hỗ trợ offset/limit cho file lớn."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Đường dẫn file"},
                "offset": {"type": "integer", "description": "Dòng bắt đầu (0-indexed)", "default": 0},
                "limit": {"type": "integer", "description": "Số dòng tối đa", "default": 2000},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["path"],
        }
    
    def validate_args(self, args: Dict[str, Any]) -> Any:
        path = args.get("path")
        if not path:
            return "Missing required arg: path"
        full_path = os.path.join(os.getcwd(), path) if not os.path.isabs(path) else path
        if not os.path.exists(full_path):
            return f"File not found: {path}"
        return None
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path = args["path"]
        offset = args.get("offset", 0)
        limit = args.get("limit", 2000)
        encoding = args.get("encoding", "utf-8")
        
        full_path = path if os.path.isabs(path) else os.path.join(context.working_dir, path)
        
        try:
            with open(full_path, "r", encoding=encoding) as f:
                lines = f.readlines()
            total_lines = len(lines)
            selected = lines[offset:offset + limit]
            content = "".join(selected)
            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "path": full_path,
                    "total_lines": total_lines,
                    "shown_lines": len(selected),
                    "offset": offset,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)


class FileWriteTool(Tool):
    """Ghi nội dung vào file (overwrite hoặc append)."""
    category = ToolCategory.FILE
    safety = ToolSafety.MODERATE
    requires_confirmation = True
    
    @property
    def name(self) -> str:
        return "file_write"
    
    @property
    def description(self) -> str:
        return "Ghi content vào file. Hỗ trợ append mode."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean", "default": False},
                "create_dirs": {"type": "boolean", "default": True},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["path", "content"],
        }
    
    def validate_args(self, args: Dict[str, Any]) -> Any:
        if not args.get("path"):
            return "Missing required arg: path"
        if "content" not in args:
            return "Missing required arg: content"
        return None
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path = args["path"]
        content = args["content"]
        append = args.get("append", False)
        create_dirs = args.get("create_dirs", True)
        encoding = args.get("encoding", "utf-8")
        
        full_path = path if os.path.isabs(path) else os.path.join(context.working_dir, path)
        
        try:
            if create_dirs:
                os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
            mode = "a" if append else "w"
            with open(full_path, mode, encoding=encoding) as f:
                f.write(content)
            return ToolResult(
                success=True,
                output=f"Wrote {len(content)} chars to {full_path}",
                artifacts=[full_path],
                metadata={"bytes": len(content.encode(encoding)), "append": append},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)


class FileListTool(Tool):
    """Liệt kê files trong thư mục."""
    category = ToolCategory.FILE
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "file_list"
    
    @property
    def description(self) -> str:
        return "Liệt kê files trong thư mục với glob pattern."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string", "default": "*"},
                "recursive": {"type": "boolean", "default": False},
                "include_hidden": {"type": "boolean", "default": False},
            },
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path = args.get("path", ".")
        pattern = args.get("pattern", "*")
        recursive = args.get("recursive", False)
        include_hidden = args.get("include_hidden", False)
        
        full_path = path if os.path.isabs(path) else os.path.join(context.working_dir, path)
        
        if recursive:
            glob_pattern = os.path.join(full_path, "**", pattern)
            files = glob.glob(glob_pattern, recursive=True)
        else:
            glob_pattern = os.path.join(full_path, pattern)
            files = glob.glob(glob_pattern)
        
        if not include_hidden:
            files = [f for f in files if not os.path.basename(f).startswith(".")]
        
        files_sorted = sorted(files)
        output_lines = []
        for f in files_sorted:
            try:
                if os.path.isdir(f):
                    output_lines.append(f"  📁 {f}/")
                else:
                    size = os.path.getsize(f)
                    output_lines.append(f"  📄 {f} ({size} bytes)")
            except OSError:
                output_lines.append(f"  ❓ {f}")
        
        return ToolResult(
            success=True,
            output="\n".join(output_lines) or "(empty)",
            metadata={"count": len(files_sorted), "path": full_path},
        )


class FileDeleteTool(Tool):
    """Xóa file hoặc thư mục."""
    category = ToolCategory.FILE
    safety = ToolSafety.DESTRUCTIVE
    requires_confirmation = True
    
    @property
    def name(self) -> str:
        return "file_delete"
    
    @property
    def description(self) -> str:
        return "Xóa file hoặc thư mục. CẢNH BÁO: không thể undo!"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean", "default": False},
            },
            "required": ["path"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        import shutil
        path = args["path"]
        recursive = args.get("recursive", False)
        
        full_path = path if os.path.isabs(path) else os.path.join(context.working_dir, path)
        
        if not os.path.exists(full_path):
            return ToolResult(success=False, error=f"Not found: {full_path}", return_code=1)
        
        try:
            if os.path.isdir(full_path):
                if not recursive:
                    return ToolResult(
                        success=False,
                        error="Is a directory. Use recursive=True to delete.",
                        return_code=1,
                    )
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
            return ToolResult(
                success=True,
                output=f"Deleted: {full_path}",
                metadata={"deleted_path": full_path},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
