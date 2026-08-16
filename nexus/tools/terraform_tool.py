"""
Terraform Tool - Wrap terraform CLI for IaC workflows.
Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, Any, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Operation được hỗ trợ // Supported operations
TF_OPERATIONS = {
    "init", "plan", "apply", "destroy", "validate",
    "fmt", "output", "refresh", "show", "state",
}

# Read-only ops
READONLY_OPS = {"validate", "fmt", "output", "show"}

# Mutating ops (cần confirmation + dry_run) // mutating operations
WRITE_OPS = {"init", "plan", "apply", "destroy", "refresh", "state"}


class TerraformTool(Tool):
    """Wrap `terraform` CLI: init/plan/apply/destroy/validate/fmt/output."""
    category = ToolCategory.CLOUD
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "terraform"

    @property
    def description(self) -> str:
        return (
            "Wrap terraform CLI: init, plan, apply, destroy, validate, fmt, "
            "output. Hỗ trợ -var, -var-file, -target và dry_run."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": sorted(TF_OPERATIONS),
                },
                "working_dir": {
                    "type": "string",
                    "description": "Thư mục chứa .tf files",
                },
                "var_file": {"type": "string", "description": "-var-file path"},
                "vars": {
                    "type": "object",
                    "description": "Biến -var key=value",
                },
                "target": {"type": "string", "description": "-target resource address"},
                "auto_approve": {"type": "boolean", "default": False, "description": "-auto-approve (apply/destroy)"},
                "plan_file": {"type": "string", "description": "Saved plan file path"},
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
        if op not in TF_OPERATIONS:
            return f"Unsupported operation: {op}"
        return None

    def _build_command(self, args: Dict[str, Any]) -> List[str]:
        op = args["operation"]
        cmd = ["terraform", op]
        extra: List[str] = list(args.get("extra_args", []) or [])

        if op in {"init", "plan", "apply", "destroy", "refresh"}:
            cmd.append("-input=false")
            if op == "init":
                cmd.append("-upgrade=false")
            if op in {"plan", "apply", "destroy"} and args.get("var_file"):
                cmd += ["-var-file", args["var_file"]]
            if op in {"plan", "apply", "destroy"} and args.get("vars"):
                for k, v in args["vars"].items():
                    cmd += ["-var", f"{k}={v}"]
            if args.get("target"):
                cmd += ["-target", args["target"]]
            if op in {"apply", "destroy"} and args.get("auto_approve"):
                cmd.append("-auto-approve")
            if op == "plan" and args.get("plan_file"):
                cmd += ["-out", args["plan_file"]]
            if op == "apply" and args.get("plan_file"):
                cmd.append(args["plan_file"])
        elif op == "fmt":
            cmd.append("-recursive")
        elif op == "output":
            if args.get("target"):
                cmd.append(args["target"])
            cmd.append("-json")
        elif op == "show":
            if args.get("plan_file"):
                cmd.append(args["plan_file"])
        elif op == "state":
            # require subcommand via extra_args
            pass
        return cmd + extra

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        op = args["operation"]
        cmd = self._build_command(args)
        cwd = args.get("working_dir") or context.working_dir

        # Dry-run simulation
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
        # Tự động_YES cho plan // auto-yes plan in non-interactive
        if op == "plan":
            env.setdefault("TF_INPUT", "0")
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
                error="terraform CLI not found. Cài đặt Terraform.",
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"terraform {op} timed out after {context.timeout}s",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
