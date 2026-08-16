"""
Gcloud CLI Tool - Wrap `gcloud` CLI for GCE, GCS, Cloud Run, GKE.
Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, Any, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Service được hỗ trợ // Supported GCP services
GCLOUD_SERVICES = {
    "compute", "storage", "run", "container", "functions",
    "app", "sql", "iam", "projects", "auth", "config",
    "pubsub", "bigquery", "logging", "monitoring", "secrets",
}

# Read-only ops
READONLY_OPS = {"list", "describe", "get-credentials", "print-config", "view"}

# Write ops
WRITE_OPS = {"create", "delete", "update", "deploy", "set", "add", "remove", "start", "stop", "run", "call"}


class GcloudCliTool(Tool):
    """Wrap `gcloud` CLI cho GCE/GCS/Cloud Run/GKE/Functions."""
    category = ToolCategory.CLOUD
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "gcloud_cli"

    @property
    def description(self) -> str:
        return (
            "Wrap gcloud CLI: compute (instances/disks), storage (ls/cp/rm), "
            "run (deploy/services), container (clusters), functions, app, sql. "
            "Hỗ trợ --project, --region, --quiet, dry_run."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "enum": sorted(GCLOUD_SERVICES),
                },
                "operation": {"type": "string", "description": "CLI operation group e.g. instances, services, ls, deploy"},
                "subcommand": {"type": "string", "description": "Specific subcommand e.g. list, describe, create"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Positional args",
                },
                "options": {
                    "type": "object",
                    "description": "--key value flags",
                },
                "project": {"type": "string", "description": "--project"},
                "region": {"type": "string", "description": "--region"},
                "zone": {"type": "string", "description": "--zone"},
                "quiet": {"type": "boolean", "default": False, "description": "--quiet (no prompts)"},
                "format": {
                    "type": "string",
                    "enum": ["json", "yaml", "text", "table", "csv"],
                    "default": "json",
                },
            },
            "required": ["service", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        svc = args.get("service")
        if not svc:
            return "Missing required arg: service"
        if not args.get("operation"):
            return "Missing required arg: operation"
        if svc not in GCLOUD_SERVICES:
            return f"Unsupported service: {svc}. Supported: {sorted(GCLOUD_SERVICES)}"
        return None

    def _is_write_op(self, sub: Optional[str]) -> bool:
        if not sub:
            return False
        sub_lower = sub.lower()
        for w in WRITE_OPS:
            if w in sub_lower:
                return True
        return False

    def _build_command(self, args: Dict[str, Any]) -> List[str]:
        cmd: List[str] = ["gcloud", args["service"], args["operation"]]
        if args.get("subcommand"):
            cmd.append(args["subcommand"])
        for a in (args.get("args") or []):
            cmd.append(str(a))
        if args.get("project"):
            cmd += ["--project", args["project"]]
        if args.get("region"):
            cmd += ["--region", args["region"]]
        if args.get("zone"):
            cmd += ["--zone", args["zone"]]
        if args.get("quiet"):
            cmd.append("--quiet")
        cmd += ["--format", args.get("format", "json")]
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
        is_write = self._is_write_op(args.get("subcommand")) or args["operation"] in WRITE_OPS

        # Dry-run simulation
        if context.dry_run and is_write:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would execute: {' '.join(cmd)}",
                metadata={
                    "dry_run": True,
                    "command": cmd,
                    "service": args["service"],
                    "operation": args["operation"],
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
                    "operation": args["operation"],
                    "subcommand": args.get("subcommand"),
                    "command": cmd,
                    "project": args.get("project"),
                    "region": args.get("region"),
                    "zone": args.get("zone"),
                    "dry_run": False,
                },
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="gcloud CLI not found. Cài đặt Google Cloud SDK.",
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"gcloud command timed out after {context.timeout}s",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
