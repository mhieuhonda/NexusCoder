"""
Docker Tool - Wrap docker CLI for container/image management.
Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import Dict, Any, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Operations cho phép // Allowed operations
DOCKER_OPERATIONS = {
    "build", "run", "ps", "images", "stop", "rm", "rmi",
    "exec", "logs", "compose",
}

# Read-only ops (không cần dry_run simulate) // read-only ops
READONLY_OPS = {"ps", "images", "logs"}

# State-changing ops // state-changing ops (cần confirmation & dry_run)
WRITE_OPS = {"build", "run", "stop", "rm", "rmi", "exec", "compose"}


class DockerTool(Tool):
    """Wrap `docker` CLI: build/run/ps/images/stop/rm/rmi/exec/logs/compose."""
    category = ToolCategory.DEVOPS
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "docker"

    @property
    def description(self) -> str:
        return (
            "Wrap docker CLI: build, run, ps, images, stop, rm, rmi, exec, "
            "logs, compose up/down. Có dry_run support."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": sorted(DOCKER_OPERATIONS),
                    "description": "Docker operation",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tham số bổ sung / extra CLI args",
                },
                "image": {"type": "string", "description": "Image name (build/run/rmi)"},
                "container": {"type": "string", "description": "Container name/id (stop/rm/exec/logs)"},
                "file": {"type": "string", "description": "Dockerfile path (build) hoặc compose file (compose)"},
                "tag": {"type": "string", "description": "Tag (build)"},
                "command": {"type": "string", "description": "Command (run/exec/compose)"},
            },
            "required": ["operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        op = args.get("operation")
        if not op:
            return "Missing required arg: operation"
        if op not in DOCKER_OPERATIONS:
            return f"Unsupported operation: {op}"
        # Validate required contextual args
        if op in {"stop", "rm", "exec", "logs"} and not args.get("container"):
            return f"Operation '{op}' requires 'container' arg"
        if op in {"run", "build", "rmi"} and not (args.get("image") or args.get("file")):
            return f"Operation '{op}' requires 'image' or 'file' arg"
        return None

    def _build_command(self, args: Dict[str, Any]) -> List[str]:
        """Build docker CLI argv list."""
        op = args["operation"]
        extra: List[str] = list(args.get("args", []) or [])
        if op == "build":
            cmd = ["docker", "build"]
            if args.get("file"):
                cmd += ["-f", args["file"]]
            if args.get("tag"):
                cmd += ["-t", args["tag"]]
            return cmd + extra + [args.get("file") or "."]
        if op == "run":
            cmd = ["docker", "run", "-d"]
            return cmd + [args.get("image")] + extra + (
                shlex.split(args["command"]) if args.get("command") else []
            )
        if op == "ps":
            return ["docker", "ps"] + extra
        if op == "images":
            return ["docker", "images"] + extra
        if op == "stop":
            return ["docker", "stop"] + [args["container"]] + extra
        if op == "rm":
            return ["docker", "rm", "-f"] + [args["container"]] + extra
        if op == "rmi":
            return ["docker", "rmi"] + [args.get("image")] + extra
        if op == "exec":
            cmd = ["docker", "exec"]
            return cmd + [args["container"]] + (
                shlex.split(args["command"]) if args.get("command") else extra
            )
        if op == "logs":
            return ["docker", "logs", "--tail", "500"] + [args["container"]] + extra
        if op == "compose":
            sub = args.get("command", "up -d")
            cmd = ["docker", "compose"]
            if args.get("file"):
                cmd += ["-f", args["file"]]
            return cmd + shlex.split(sub)
        return ["docker"]

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cmd = self._build_command(args)
        op = args["operation"]
        cwd = args.get("file") and os.path.dirname(args["file"]) or context.working_dir
        # Resolve file dir nếu là compose/build
        if op in {"compose", "build"} and args.get("file"):
            cwd = os.path.dirname(os.path.abspath(args["file"])) or context.working_dir

        # Dry-run: simulate // dry-run simulation
        if context.dry_run and op in WRITE_OPS:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would execute: {' '.join(cmd)}",
                metadata={
                    "dry_run": True,
                    "command": cmd,
                    "operation": op,
                    "cwd": cwd,
                },
            )

        env = dict(os.environ)
        env.update(context.env)
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=context.timeout,
                check=False,
            )
            return ToolResult(
                success=(result.returncode == 0),
                output=result.stdout,
                error=result.stderr or None,
                return_code=result.returncode,
                metadata={
                    "operation": op,
                    "command": cmd,
                    "cwd": cwd,
                    "dry_run": False,
                },
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="docker CLI not found. Install Docker hoặc cấu hình PATH.",
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Docker command timed out after {context.timeout}s",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
