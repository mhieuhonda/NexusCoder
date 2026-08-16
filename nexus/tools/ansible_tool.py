"""
Ansible Tool - Wrap ansible-playbook & ansible ad-hoc commands.
Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import Dict, Any, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Operation được hỗ trợ
ANSIBLE_OPERATIONS = {
    "playbook", "ping", "command", "shell", "copy", "setup",
    "facts", "module",
}

# Read-only operations
READONLY_OPS = {"ping", "setup", "facts"}

# Mutating operations (cần confirmation + dry_run)
WRITE_OPS = {"playbook", "command", "shell", "copy", "module"}


class AnsibleTool(Tool):
    """Wrap `ansible-playbook` và `ansible` ad-hoc commands."""
    category = ToolCategory.DEVOPS
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "ansible"

    @property
    def description(self) -> str:
        return (
            "Wrap ansible CLI: run playbooks, ad-hoc commands (ping, command, "
            "shell, copy, setup/facts). Hỗ trợ inventory, become, check mode."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": sorted(ANSIBLE_OPERATIONS),
                },
                "playbook": {"type": "string", "description": "Path to playbook YAML"},
                "inventory": {"type": "string", "description": "Inventory file hoặc comma-separated hosts"},
                "host_pattern": {"type": "string", "description": "Target host pattern (ad-hoc)", "default": "all"},
                "module": {"type": "string", "description": "Ansible module name (module operation)"},
                "args": {"type": "string", "description": "Module args e.g. 'cmd=uptime'"},
                "become": {"type": "boolean", "default": False, "description": "sudo/privilege escalation"},
                "become_user": {"type": "string", "default": "root"},
                "check_mode": {"type": "boolean", "default": False, "description": "--check (dry-run mode)"},
                "extra_vars": {
                    "type": "object",
                    "description": "-e key=value",
                },
                "tags": {"type": "array", "items": {"type": "string"}, "description": "--tags"},
                "verbose": {"type": "integer", "default": 0, "description": "-v level (0-4)"},
            },
            "required": ["operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        op = args.get("operation")
        if not op:
            return "Missing required arg: operation"
        if op not in ANSIBLE_OPERATIONS:
            return f"Unsupported operation: {op}"
        if op == "playbook" and not args.get("playbook"):
            return "Operation 'playbook' requires 'playbook' arg"
        if op == "module" and not args.get("module"):
            return "Operation 'module' requires 'module' arg"
        return None

    def _common_flags(self, args: Dict[str, Any]) -> List[str]:
        flags: List[str] = []
        if args.get("inventory"):
            flags += ["-i", args["inventory"]]
        if args.get("become"):
            flags.append("--become")
            if args.get("become_user"):
                flags += ["--become-user", args["become_user"]]
        if args.get("check_mode"):
            flags.append("--check")
        if args.get("extra_vars"):
            for k, v in args["extra_vars"].items():
                flags += ["-e", f"{k}={v}"]
        if args.get("tags"):
            flags += ["--tags", ",".join(args["tags"])]
        v = int(args.get("verbose", 0))
        if v > 0:
            flags.append("-" + "v" * min(v, 4))
        return flags

    def _build_command(self, args: Dict[str, Any]) -> List[str]:
        op = args["operation"]
        common = self._common_flags(args)

        if op == "playbook":
            return ["ansible-playbook"] + common + [args["playbook"]]
        if op == "ping":
            return ["ansible", args.get("host_pattern", "all"), "-m", "ping"] + common
        if op in {"command", "shell"}:
            mod = "command" if op == "command" else "shell"
            return ["ansible", args.get("host_pattern", "all"), "-m", mod] + (
                ["-a", args["args"]] if args.get("args") else []
            ) + common
        if op == "copy":
            return ["ansible", args.get("host_pattern", "all"), "-m", "copy"] + (
                ["-a", args["args"]] if args.get("args") else []
            ) + common
        if op in {"setup", "facts"}:
            return ["ansible", args.get("host_pattern", "all"), "-m", "setup"] + common
        if op == "module":
            cmd = ["ansible", args.get("host_pattern", "all"), "-m", args["module"]]
            if args.get("args"):
                cmd += ["-a", args["args"]]
            return cmd + common
        return ["ansible"]

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        op = args["operation"]
        cmd = self._build_command(args)
        cwd = context.working_dir

        # Dry-run simulation (trừ khi --check đã được set)
        if context.dry_run and op in WRITE_OPS and not args.get("check_mode"):
            sim_cmd = list(cmd)
            # chèn --check để simulate
            if op == "playbook":
                sim_cmd.insert(1, "--check")
            return ToolResult(
                success=True,
                output=f"[dry-run] Would execute: {' '.join(sim_cmd)}",
                metadata={
                    "dry_run": True,
                    "command": sim_cmd,
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
                error="ansible CLI not found. Cài đặt Ansible.",
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"ansible timed out after {context.timeout}s",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
