"""
Systemd Tool - Manage systemd services via `systemctl`.
Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, Any, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Operations được hỗ trợ // Supported systemctl operations
SYSTEMD_OPERATIONS = {
    "start", "stop", "restart", "reload", "status",
    "enable", "disable", "is-active", "is-enabled",
    "list-units", "list-unit-files", "daemon-reload",
    "show", "cat", "edit",
}

# Read-only ops
READONLY_OPS = {"status", "is-active", "is-enabled", "list-units", "list-unit-files", "show", "cat"}

# Mutating ops (cần confirmation + dry_run)
WRITE_OPS = {"start", "stop", "restart", "reload", "enable", "disable", "daemon-reload", "edit"}


class SystemdTool(Tool):
    """Manage systemd services: start/stop/restart/status/enable/disable."""
    category = ToolCategory.SYSTEM
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "systemd"

    @property
    def description(self) -> str:
        return (
            "Wrap systemctl: start/stop/restart/reload/status/enable/disable, "
            "list-units, daemon-reload, show, cat. Hỗ trợ --user mode và dry_run."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": sorted(SYSTEMD_OPERATIONS),
                },
                "service": {"type": "string", "description": "Service unit name (e.g. nginx.service)"},
                "user_mode": {"type": "boolean", "default": False, "description": "--user scope"},
                "no_block": {"type": "boolean", "default": False, "description": "--no-block (fire and forget)"},
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra systemctl flags",
                },
            },
            "required": ["operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        op = args.get("operation")
        if not op:
            return "Missing required arg: operation"
        if op not in SYSTEMD_OPERATIONS:
            return f"Unsupported operation: {op}"
        no_service_required = {"list-units", "list-unit-files", "daemon-reload"}
        if op not in no_service_required and not args.get("service"):
            return f"Operation '{op}' requires 'service' arg"
        return None

    def _build_command(self, args: Dict[str, Any]) -> List[str]:
        cmd: List[str] = ["systemctl"]
        if args.get("user_mode"):
            cmd.append("--user")
        if args.get("no_block") and args["operation"] in {"start", "stop", "restart", "reload"}:
            cmd.append("--no-block")
        cmd.append(args["operation"])
        if args.get("service"):
            cmd.append(args["service"])
        cmd += list(args.get("extra_args") or [])
        return cmd

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        op = args["operation"]
        cmd = self._build_command(args)

        # Dry-run simulation (state-changing ops)
        if context.dry_run and op in WRITE_OPS:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would execute: {' '.join(cmd)}",
                metadata={
                    "dry_run": True,
                    "command": cmd,
                    "operation": op,
                    "service": args.get("service"),
                },
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
            success = result.returncode == 0
            # status/is-active/is-enabled: trả về non-zero nếu không active — vẫn success từ góc nhìn tool
            metadata: Dict[str, Any] = {
                "operation": op,
                "service": args.get("service"),
                "command": cmd,
                "user_mode": args.get("user_mode", False),
                "dry_run": False,
            }
            if op == "status":
                metadata["active"] = "active (running)" in result.stdout
            elif op == "is-active":
                metadata["active"] = result.stdout.strip() == "active"
            elif op == "is-enabled":
                metadata["enabled"] = result.stdout.strip() == "enabled"
            return ToolResult(
                success=success,
                output=result.stdout,
                error=result.stderr or None,
                return_code=result.returncode,
                metadata=metadata,
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="systemctl not found. Hệ thống không dùng systemd?",
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"systemctl {op} timed out after {context.timeout}s",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
