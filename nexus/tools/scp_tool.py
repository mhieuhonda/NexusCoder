"""
SCP Tool - Copy files over SSH via `scp` CLI.
Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import Dict, Any, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class SCPTool(Tool):
    """Copy files to/from remote host via `scp`."""
    category = ToolCategory.NETWORK
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "scp"

    @property
    def description(self) -> str:
        return (
            "Copy files qua SSH bằng `scp`. Hỗ trợ upload (local→remote), "
            "download (remote→local), recursive, key auth, port. Có dry_run."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Remote host/IP"},
                "user": {"type": "string", "description": "SSH user"},
                "port": {"type": "integer", "default": 22},
                "direction": {
                    "type": "string",
                    "enum": ["upload", "download"],
                    "description": "upload: local→remote; download: remote→local",
                },
                "local_path": {"type": "string", "description": "Local file/dir path"},
                "remote_path": {"type": "string", "description": "Remote file/dir path"},
                "recursive": {"type": "boolean", "default": False, "description": "-r for directories"},
                "key_filename": {"type": "string", "description": "Private key path (-i)"},
                "preserve": {"type": "boolean", "default": False, "description": "-p preserve attrs"},
                "compress": {"type": "boolean", "default": False, "description": "-C compression"},
                "timeout": {"type": "integer", "default": 120, "description": "ConnectTimeout in seconds"},
            },
            "required": ["host", "user", "direction", "local_path", "remote_path"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        for k in ("host", "user", "direction", "local_path", "remote_path"):
            if not args.get(k):
                return f"Missing required arg: {k}"
        if args["direction"] not in {"upload", "download"}:
            return "direction must be 'upload' or 'download'"
        return None

    def _build_command(self, args: Dict[str, Any]) -> List[str]:
        cmd: List[str] = [
            "scp",
            "-P", str(args.get("port", 22)),
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={args.get('timeout', 120)}",
            "-o", "BatchMode=yes",
        ]
        if args.get("recursive"):
            cmd.append("-r")
        if args.get("preserve"):
            cmd.append("-p")
        if args.get("compress"):
            cmd.append("-C")
        if args.get("key_filename"):
            cmd += ["-i", args["key_filename"]]

        user_host = f"{args['user']}@{args['host']}"
        local = args["local_path"]
        remote = args["remote_path"]
        if args["direction"] == "upload":
            cmd += [local, f"{user_host}:{remote}"]
        else:  # download
            cmd += [f"{user_host}:{remote}", local]
        return cmd

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cmd = self._build_command(args)

        # Dry-run simulation
        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would execute: {' '.join(shlex.quote(c) for c in cmd)}",
                metadata={
                    "dry_run": True,
                    "command": cmd,
                    "direction": args["direction"],
                    "host": args["host"],
                },
            )

        # Kiểm tra local path tồn tại (upload) hoặc thư mục cha (download)
        if args["direction"] == "upload":
            if not os.path.exists(args["local_path"]):
                return ToolResult(
                    success=False,
                    error=f"Local path not found: {args['local_path']}",
                    return_code=2,
                )
        else:
            parent = os.path.dirname(os.path.abspath(args["local_path"]))
            if not os.path.isdir(parent):
                return ToolResult(
                    success=False,
                    error=f"Local parent dir not found: {parent}",
                    return_code=2,
                )

        env = dict(os.environ)
        env.update(context.env)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=context.timeout,
                check=False,
            )
            return ToolResult(
                success=(result.returncode == 0),
                output=result.stdout,
                error=result.stderr or None,
                return_code=result.returncode,
                artifacts=[args["local_path"]],
                metadata={
                    "command": cmd,
                    "direction": args["direction"],
                    "host": args["host"],
                    "user": args["user"],
                    "local_path": args["local_path"],
                    "remote_path": args["remote_path"],
                    "dry_run": False,
                },
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="scp CLI not found. Cài đặt openssh-client.",
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"scp timed out after {context.timeout}s",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
