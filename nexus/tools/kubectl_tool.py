"""
Kubectl Tool - Wrap kubectl CLI for Kubernetes management.
Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import Dict, Any, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Các operation được hỗ trợ // Supported kubectl operations
KUBECTL_OPERATIONS = {
    "get", "apply", "delete", "describe", "logs",
    "exec", "port-forward",
}

# Read-only ops // read-only operations
READONLY_OPS = {"get", "describe", "logs"}

# Write/destructive ops (cần confirmation + dry_run) // mutating ops
WRITE_OPS = {"apply", "delete", "exec", "port-forward"}


class KubectlTool(Tool):
    """Wrap `kubectl` CLI: get/apply/delete/describe/logs/exec/port-forward."""
    category = ToolCategory.DEVOPS
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "kubectl"

    @property
    def description(self) -> str:
        return (
            "Wrap kubectl CLI: get, apply, delete, describe, logs, exec, "
            "port-forward. Hỗ trợ dry-run và namespace."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": sorted(KUBECTL_OPERATIONS),
                },
                "resource": {
                    "type": "string",
                    "description": "Resource type (pods, svc, deploy, ...)",
                },
                "name": {"type": "string", "description": "Resource name"},
                "namespace": {"type": "string", "default": "default"},
                "filename": {"type": "string", "description": "YAML file (apply/delete -f)"},
                "command": {"type": "string", "description": "Command for exec/logs"},
                "tail": {"type": "integer", "default": 200, "description": "Log lines"},
                "port": {"type": "string", "description": "Port-forward spec e.g. 8080:80"},
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tham số bổ sung",
                },
            },
            "required": ["operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        op = args.get("operation")
        if not op:
            return "Missing required arg: operation"
        if op not in KUBECTL_OPERATIONS:
            return f"Unsupported operation: {op}"
        if op in {"delete", "describe"} and not (args.get("resource") or args.get("filename")):
            return f"Operation '{op}' requires 'resource' or 'filename'"
        if op == "exec" and not args.get("command"):
            return "Operation 'exec' requires 'command' arg"
        if op == "port-forward" and not args.get("port"):
            return "Operation 'port-forward' requires 'port' arg"
        if op == "apply" and not args.get("filename"):
            return "Operation 'apply' requires 'filename' arg"
        return None

    def _build_command(self, args: Dict[str, Any]) -> List[str]:
        op = args["operation"]
        ns = args.get("namespace")
        cmd = ["kubectl"]
        if ns:
            cmd += ["-n", ns]
        extra: List[str] = list(args.get("extra_args", []) or [])

        if op == "get":
            cmd += ["get", args.get("resource", "pods")]
            if args.get("name"):
                cmd.append(args["name"])
            return cmd + extra
        if op == "apply":
            return cmd + ["apply", "-f", args["filename"]] + extra
        if op == "delete":
            cmd += ["delete"]
            if args.get("filename"):
                cmd += ["-f", args["filename"]]
            else:
                cmd.append(args["resource"])
                if args.get("name"):
                    cmd.append(args["name"])
            return cmd + extra
        if op == "describe":
            cmd += ["describe", args["resource"]]
            if args.get("name"):
                cmd.append(args["name"])
            return cmd + extra
        if op == "logs":
            cmd += ["logs", "--tail", str(args.get("tail", 200))]
            return cmd + [args["resource"], args.get("name", "")] + extra
        if op == "exec":
            cmd += ["exec", args["resource"], args.get("name", ""), "--"]
            return cmd + shlex.split(args["command"]) + extra
        if op == "port-forward":
            cmd += ["port-forward", args["resource"], args.get("name", "")]
            return cmd + [args["port"]] + extra
        return cmd

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cmd = self._build_command(args)
        op = args["operation"]
        # Clean empty tokens (logs/exec when no resource name)
        cmd = [c for c in cmd if c != ""]

        # Dry-run simulation
        if context.dry_run and op in WRITE_OPS:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would execute: {' '.join(cmd)}",
                metadata={"dry_run": True, "command": cmd, "operation": op},
            )

        env = dict(os.environ)
        env.update(context.env)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=context.timeout,
                env=env,
                check=False,
            )
            return ToolResult(
                success=(result.returncode == 0),
                output=result.stdout,
                error=result.stderr or None,
                return_code=result.returncode,
                metadata={
                    "operation": op,
                    "namespace": args.get("namespace", "default"),
                    "command": cmd,
                    "dry_run": False,
                },
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="kubectl CLI not found. Install kubectl.",
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"kubectl timed out after {context.timeout}s",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
