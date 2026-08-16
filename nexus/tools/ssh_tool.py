"""
SSH Tool - Execute commands over SSH (paramiko preferred, subprocess fallback).
Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import Dict, Any, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class SSHTool(Tool):
    """Execute commands over SSH. Ưu tiên paramiko, fallback subprocess ssh."""
    category = ToolCategory.NETWORK
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "ssh"

    @property
    def description(self) -> str:
        return (
            "Execute commands over SSH. Dùng paramiko nếu có (lazy import), "
            "nếu không fallback sang `ssh` CLI. Hỗ trợ key/password auth, port, timeout."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Hostname hoặc IP"},
                "user": {"type": "string", "description": "SSH user"},
                "port": {"type": "integer", "default": 22},
                "command": {"type": "string", "description": "Remote command to run"},
                "key_filename": {"type": "string", "description": "Private key path"},
                "password": {"type": "string", "description": "Password (không khuyến khích, dùng key nếu có)"},
                "timeout": {"type": "integer", "default": 30, "description": "Connection/command timeout (s)"},
            },
            "required": ["host", "user", "command"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("host"):
            return "Missing required arg: host"
        if not args.get("user"):
            return "Missing required arg: user"
        if not args.get("command"):
            return "Missing required arg: command"
        if args.get("port") and not isinstance(args["port"], int):
            return "port must be integer"
        if not args.get("key_filename") and not args.get("password"):
            return "Either key_filename or password is required"
        return None

    def _execute_paramiko(
        self, args: Dict[str, Any], context: ToolContext
    ) -> Optional[ToolResult]:
        """Run via paramiko. Trả về None nếu paramiko không có sẵn."""
        try:
            import paramiko  # type: ignore  # lazy import
        except ImportError:
            return None

        host = args["host"]
        user = args["user"]
        port = int(args.get("port", 22))
        cmd = args["command"]
        timeout = int(args.get("timeout", context.timeout))
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            connect_kwargs: Dict[str, Any] = {
                "hostname": host,
                "username": user,
                "port": port,
                "timeout": timeout,
                "banner_timeout": timeout,
            }
            if args.get("key_filename"):
                connect_kwargs["key_filename"] = args["key_filename"]
            elif args.get("password"):
                connect_kwargs["password"] = args["password"]
            client.connect(**connect_kwargs)
            stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            rc = stdout.channel.recv_exit_status()
            return ToolResult(
                success=(rc == 0),
                output=out,
                error=err or None,
                return_code=rc,
                metadata={
                    "backend": "paramiko",
                    "host": host,
                    "user": user,
                    "port": port,
                    "command": cmd,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"paramiko error: {e}",
                return_code=1,
                metadata={"backend": "paramiko", "host": host},
            )
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _execute_subprocess(
        self, args: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Fallback: dùng `ssh` CLI via subprocess."""
        host = args["host"]
        user = args["user"]
        port = int(args.get("port", 22))
        cmd = args["command"]
        timeout = int(args.get("timeout", context.timeout))

        ssh_cmd: List[str] = [
            "ssh",
            "-p", str(port),
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={timeout}",
            "-o", "BatchMode=yes",
        ]
        if args.get("key_filename"):
            ssh_cmd += ["-i", args["key_filename"]]
        ssh_cmd.append(f"{user}@{host}")
        ssh_cmd.append(cmd)

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would execute: {' '.join(shlex.quote(c) for c in ssh_cmd)}",
                metadata={
                    "dry_run": True,
                    "backend": "subprocess",
                    "command": ssh_cmd,
                },
            )

        env = dict(os.environ)
        env.update(context.env)
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
                check=False,
            )
            return ToolResult(
                success=(result.returncode == 0),
                output=result.stdout,
                error=result.stderr or None,
                return_code=result.returncode,
                metadata={
                    "backend": "subprocess",
                    "host": host,
                    "user": user,
                    "port": port,
                    "command": ssh_cmd,
                },
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="ssh CLI not found AND paramiko not installed.",
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"ssh timed out after {timeout}s",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        # Dry-run check trước (áp dụng cả 2 backends)
        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would SSH to {args['user']}@{args['host']} and run: {args['command']}",
                metadata={
                    "dry_run": True,
                    "host": args["host"],
                    "user": args["user"],
                    "command": args["command"],
                },
            )
        # Thử paramiko trước, fallback subprocess
        result = self._execute_paramiko(args, context)
        if result is not None:
            return result
        return self._execute_subprocess(args, context)
