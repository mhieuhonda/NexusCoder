"""
AWS CLI Tool - Wrap `aws` CLI for S3, EC2, Lambda, IAM operations.
Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import Dict, Any, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Service được hỗ trợ // Supported AWS services
AWS_SERVICES = {
    "s3", "ec2", "lambda", "iam", "sts", "dynamodb",
    "rds", "cloudformation", "ecs", "ecr", "sns", "sqs",
    "cloudwatch", "logs", "ssm", "secretsmanager",
}

# Read-only ops (không cần confirmation)
READONLY_OPS = {"ls", "get", "describe", "list", "cat", "head", "whoami", "get-caller-identity"}

# Write ops (cần confirmation + dry_run)
WRITE_OPS = {"cp", "mv", "rm", "sync", "create", "delete", "put", "update", "invoke", "run"}


class AWSCliTool(Tool):
    """Wrap `aws` CLI cho S3/EC2/Lambda/IAM và các AWS service khác."""
    category = ToolCategory.CLOUD
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "aws_cli"

    @property
    def description(self) -> str:
        return (
            "Wrap aws CLI: S3 (ls/cp/rm/sync), EC2 (describe-instances, start/stop), "
            "Lambda (invoke/list), IAM, STS, và các service khác. Hỗ trợ --profile, --region, dry_run."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "enum": sorted(AWS_SERVICES),
                    "description": "AWS service name (s3, ec2, lambda, ...)",
                },
                "operation": {"type": "string", "description": "CLI operation e.g. ls, cp, describe-instances"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Positional args (bucket/key, instance-ids, ...)",
                },
                "options": {
                    "type": "object",
                    "description": "--key value pairs",
                },
                "profile": {"type": "string", "description": "AWS profile (--profile)"},
                "region": {"type": "string", "description": "AWS region (--region)"},
                "output_format": {
                    "type": "string",
                    "enum": ["json", "yaml", "text", "table"],
                    "default": "json",
                },
            },
            "required": ["service", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        svc = args.get("service")
        op = args.get("operation")
        if not svc:
            return "Missing required arg: service"
        if not op:
            return "Missing required arg: operation"
        if svc not in AWS_SERVICES:
            return f"Unsupported service: {svc}. Supported: {sorted(AWS_SERVICES)}"
        return None

    def _is_write_op(self, op: str) -> bool:
        op_lower = op.lower()
        for w in WRITE_OPS:
            if w in op_lower:
                return True
        return False

    def _build_command(self, args: Dict[str, Any]) -> List[str]:
        cmd: List[str] = ["aws"]
        if args.get("profile"):
            cmd += ["--profile", args["profile"]]
        if args.get("region"):
            cmd += ["--region", args["region"]]
        cmd += ["--output", args.get("output_format", "json")]
        cmd += [args["service"], args["operation"]]
        # Positional args
        for a in (args.get("args") or []):
            cmd.append(str(a))
        # --key value options
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
                    "profile": args.get("profile"),
                    "region": args.get("region"),
                    "dry_run": False,
                },
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="aws CLI not found. Cài đặt AWS CLI v2.",
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"aws command timed out after {context.timeout}s",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
