"""
Rsync Tool - Sync files/dirs via `rsync` CLI.
Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import Dict, Any, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class RsyncTool(Tool):
    """Sync files/dirs giữa local và remote dùng `rsync`."""
    category = ToolCategory.NETWORK
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "rsync"

    @property
    def description(self) -> str:
        return (
            "Sync files/dirs với rsync. Hỗ trợ -a (archive), -v (verbose), "
            "-z (compress), --delete, --exclude, --dry-run (rsync native), "
            "remote ssh. Có context.dry_run simulation."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path (local hoặc user@host:path)"},
                "destination": {"type": "string", "description": "Destination path (local hoặc user@host:path)"},
                "archive": {"type": "boolean", "default": True, "description": "-a archive mode"},
                "verbose": {"type": "boolean", "default": True, "description": "-v"},
                "compress": {"type": "boolean", "default": True, "description": "-z"},
                "delete": {"type": "boolean", "default": False, "description": "--delete extraneous files"},
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "--exclude patterns",
                },
                "include": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "--include patterns (applied before exclude)",
                },
                "checksum": {"type": "boolean", "default": False, "description": "-c skip based on checksum"},
                "progress": {"type": "boolean", "default": False, "description": "--info=progress2"},
                "ssh_key": {"type": "string", "description": "SSH key path (-e 'ssh -i KEY')"},
                "ssh_port": {"type": "integer", "default": 22, "description": "SSH port (-p trong ssh)"},
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Raw rsync flags",
                },
            },
            "required": ["source", "destination"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("source"):
            return "Missing required arg: source"
        if not args.get("destination"):
            return "Missing required arg: destination"
        return None

    def _build_command(self, args: Dict[str, Any]) -> List[str]:
        cmd: List[str] = ["rsync"]
        if args.get("archive", True):
            cmd.append("-a")
        if args.get("verbose", True):
            cmd.append("-v")
        if args.get("compress", True):
            cmd.append("-z")
        if args.get("checksum"):
            cmd.append("-c")
        if args.get("delete"):
            cmd.append("--delete")
        if args.get("progress"):
            cmd.append("--info=progress2")
        # include trước exclude (rsync áp dụng theo thứ tự)
        for pat in (args.get("include") or []):
            cmd += ["--include", pat]
        for pat in (args.get("exclude") or []):
            cmd += ["--exclude", pat]
        # SSH remote
        is_remote = ":" in args["source"] or ":" in args["destination"]
        if is_remote:
            ssh_parts = ["ssh"]
            if args.get("ssh_key"):
                ssh_parts += ["-i", args["ssh_key"]]
            ssh_parts += ["-p", str(args.get("ssh_port", 22))]
            ssh_parts += ["-o", "StrictHostKeyChecking=accept-new"]
            cmd += ["-e", " ".join(ssh_parts)]
        # raw extras
        cmd += list(args.get("extra_args") or [])
        cmd += [args["source"], args["destination"]]
        return cmd

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cmd = self._build_command(args)

        # Dry-run simulation
        if context.dry_run:
            sim_cmd = list(cmd)
            sim_cmd.insert(1, "--dry-run")
            return ToolResult(
                success=True,
                output=f"[dry-run] Would execute: {' '.join(shlex.quote(c) for c in sim_cmd)}",
                metadata={
                    "dry_run": True,
                    "command": sim_cmd,
                    "source": args["source"],
                    "destination": args["destination"],
                },
            )

        # Kiểm tra source local tồn tại (nếu source không phải remote)
        if ":" not in args["source"]:
            if not os.path.exists(args["source"]):
                return ToolResult(
                    success=False,
                    error=f"Source not found: {args['source']}",
                    return_code=2,
                )
        # Tạo dest dir local nếu cần
        if ":" not in args["destination"]:
            dest_dir = os.path.dirname(args["destination"].rstrip("/"))
            if dest_dir and not os.path.isdir(dest_dir):
                try:
                    os.makedirs(dest_dir, exist_ok=True)
                except OSError as e:
                    return ToolResult(
                        success=False,
                        error=f"Cannot create dest dir {dest_dir}: {e}",
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
                artifacts=[args["source"], args["destination"]],
                metadata={
                    "command": cmd,
                    "source": args["source"],
                    "destination": args["destination"],
                    "delete": args.get("delete", False),
                    "dry_run": False,
                },
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="rsync CLI not found. Cài đặt rsync.",
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"rsync timed out after {context.timeout}s",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
