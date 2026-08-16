"""Archive Tool - zip/tar operations."""
from __future__ import annotations

import os
import zipfile
import tarfile
from typing import Dict, Any

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class ArchiveTool(Tool):
    """Tạo/giải nén zip/tar archives."""
    category = ToolCategory.FILE
    safety = ToolSafety.MODERATE
    
    @property
    def name(self) -> str:
        return "archive"
    
    @property
    def description(self) -> str:
        return "Tạo/giải nén ZIP/TAR/GZ archives."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "extract", "list"]},
                "archive_path": {"type": "string"},
                "source_path": {"type": "string", "description": "For create: dir to compress; for extract: target dir"},
                "format": {"type": "string", "enum": ["zip", "tar", "tar.gz", "tar.bz2"], "default": "zip"},
            },
            "required": ["action", "archive_path"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        action = args["action"]
        archive_path = args["archive_path"]
        source = args.get("source_path")
        fmt = args.get("format", "zip")
        
        full_archive = archive_path if os.path.isabs(archive_path) else os.path.join(context.working_dir, archive_path)
        
        try:
            if action == "create":
                if not source:
                    return ToolResult(success=False, error="source_path required for create", return_code=2)
                full_source = source if os.path.isabs(source) else os.path.join(context.working_dir, source)
                if not os.path.exists(full_source):
                    return ToolResult(success=False, error=f"Source not found: {full_source}", return_code=1)
                
                if fmt == "zip":
                    with zipfile.ZipFile(full_archive, "w", zipfile.ZIP_DEFLATED) as zf:
                        if os.path.isfile(full_source):
                            zf.write(full_source, os.path.basename(full_source))
                        else:
                            for root, dirs, files in os.walk(full_source):
                                for fname in files:
                                    fpath = os.path.join(root, fname)
                                    arcname = os.path.relpath(fpath, full_source)
                                    zf.write(fpath, arcname)
                else:
                    mode = {"tar": "w", "tar.gz": "w:gz", "tar.bz2": "w:bz2"}[fmt]
                    with tarfile.open(full_archive, mode) as tf:
                        tf.add(full_source, arcname=os.path.basename(full_source))
                
                size = os.path.getsize(full_archive)
                return ToolResult(
                    success=True,
                    output=f"Created {full_archive} ({size} bytes)",
                    artifacts=[full_archive],
                    metadata={"action": "create", "format": fmt, "size": size},
                )
            
            elif action == "extract":
                if not source:
                    source = os.path.dirname(full_archive) or "."
                full_target = source if os.path.isabs(source) else os.path.join(context.working_dir, source)
                os.makedirs(full_target, exist_ok=True)
                
                if full_archive.endswith(".zip"):
                    with zipfile.ZipFile(full_archive, "r") as zf:
                        zf.extractall(full_target)
                        names = zf.namelist()
                else:
                    with tarfile.open(full_archive, "r:*") as tf:
                        tf.extractall(full_target)
                        names = tf.getnames()
                
                return ToolResult(
                    success=True,
                    output=f"Extracted {len(names)} files to {full_target}",
                    metadata={"action": "extract", "files": names[:20], "total": len(names)},
                )
            
            elif action == "list":
                if full_archive.endswith(".zip"):
                    with zipfile.ZipFile(full_archive, "r") as zf:
                        names = zf.namelist()
                else:
                    with tarfile.open(full_archive, "r:*") as tf:
                        names = tf.getnames()
                
                output = "\n".join(names[:100])
                if len(names) > 100:
                    output += f"\n... and {len(names) - 100} more"
                
                return ToolResult(
                    success=True,
                    output=output or "(empty archive)",
                    metadata={"total_files": len(names)},
                )
            
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}", return_code=2)
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
