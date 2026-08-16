"""
Azure CLI Tool - Wrap `az` CLI for VM, Storage, AKS, Functions.
Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, Any, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Service được hỗ trợ // Supported Azure service groups
AZURE_SERVICES = {
    "vm", "storage", "aks", "functionapp", "webapp",
    "group", "account", "network", "sql", "cosmosdb",
    "keyvault", "servicebus", "eventgrid", "loganalytics",
    "appconfig", "containerapp",
}

# Read-only ops
READONLY_OPS = {"list", "show", "get", "view"}

# Write ops
WRITE_OPS = {"create", "delete", "update", "start", "stop", "restart", "deploy", "run", "invoke", "set"}


class AzureCliTool(Tool):
    """Wrap `az` CLI cho VM/Storage/AKS/Functions/App Service."""
    category = ToolCategory.CLOUD
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "azure_cli"

    @property
    def description(self) -> str:
        return (
            "Wrap az CLI: vm (create/list/start/stop), storage (blob/container), "
            "aks (create/get-credentials), functionapp, webapp, group. "
            "Hỗ trợ --subscription, --resource-group, --output, dry_run."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "enum": sorted(AZURE_SERVICES),
                },
                "operation": {"type": "string", "description": "CLI operation e.g. create, list, show"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Positional args (name, ...)",
                },
                "options": {
                    "type": "object",
                    "description": "--key value flags",
                },
                "subscription": {"type": "string", "description": "--subscription"},
                "resource_group": {"type": "string", "description": "--resource-group"},
                "output_format": {
                    "type": "string",
                    "enum": ["json", "yaml", "table", "tsv", "none"],
                    "default": "json",
                },
                "query": {"type": "string", "description": "--query JMESPath"},
            },
            "required": ["service", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        svc = args.get("service")
        if not svc:
            return "Missing required arg: service"
        if not args.get("operation"):
            return "Missing required arg: operation"
        if svc not in AZURE_SERVICES:
            return f"Unsupported service: {svc}. Supported: {sorted(AZURE_SERVICES)}"
        return None

    def _is_write_op(self, op: str) -> bool:
        op_lower = op.lower()
        for w in WRITE_OPS:
            if w in op_lower:
                return True
        return False

    def _build_command(self, args: Dict[str, Any]) -> List[str]:
        cmd: List[str] = ["az", args["service"], args["operation"]]
        for a in (args.get("args") or []):
            cmd.append(str(a))
        if args.get("subscription"):
            cmd += ["--subscription", args["subscription"]]
        if args.get("resource_group"):
            cmd += ["--resource-group", args["resource_group"]]
        if args.get("query"):
            cmd += ["--query", args["query"]]
        cmd += ["--output", args.get("output_format", "json")]
        for k, v in (args.get("options") or {}).items():
            if isinstance(v, bool):
                if v:
                    cmd.append(f"--{k}")
            elif isinstance(v, (list, tuple)):
                for item in v:
                    cmd += [f"--{k}", str(item)]
            else:
                cmd += [f"--{k}", str(v)]
        return cmd

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cmd = self._build_command(args)
        op = args["operation"]
        is_write = self._is_write_op(op)

        # Dry-run simulation
        if context.dry_run and is_write:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would execute: {' '.join(cmd)}",
                metadata={
                    "dry_run": True,
                    "command": cmd,
                    "service": args["service"],
                    "operation": op,
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
            return ToolResult(
                success=(result.returncode == 0),
                output=result.stdout,
                error=result.stderr or None,
                return_code=result.returncode,
                metadata={
                    "service": args["service"],
                    "operation": op,
                    "command": cmd,
                    "subscription": args.get("subscription"),
                    "resource_group": args.get("resource_group"),
                    "dry_run": False,
                },
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="az CLI not found. Cài đặt Azure CLI.",
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"az command timed out after {context.timeout}s",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
