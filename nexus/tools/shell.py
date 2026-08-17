"""Shell Execution Tool - chạy bash commands (sandboxed)."""
from __future__ import annotations

import os
import subprocess
from typing import Dict, Any, List

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Commands cấm vì nguy hiểm (require explicit allowlist)
BLOCKED_COMMANDS = {
    "rm -rf /", "mkfs", "dd if=/dev/zero", ":(){:|:&};:",
    "chmod -R 777 /", "shutdown", "reboot", "halt",
    "init 0", "init 6",
}


class ShellExecTool(Tool):
    """Execute shell commands với sandboxing."""
    category = ToolCategory.EXEC
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True
    
    @property
    def name(self) -> str:
        return "shell_exec"
    
    @property
    def description(self) -> str:
        return (
            "Execute shell command (bash). Output được capture. "
            "Có timeout và blocked commands để chống dangerous ops."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "cwd": {"type": "string", "description": "Working directory"},
                "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds"},
                "env": {"type": "object", "description": "Extra env vars"},
            },
            "required": ["command"],
        }
    
    def validate_args(self, args: Dict[str, Any]) -> Any:
        cmd = args.get("command", "")
        if not cmd or not cmd.strip():
            return "Empty command"
        # Check blocked commands
        for blocked in BLOCKED_COMMANDS:
            if blocked in cmd:
                return f"Blocked command pattern: {blocked}"
        return None
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cmd = args["command"]
        cwd = args.get("cwd") or context.working_dir
        timeout = args.get("timeout", context.timeout)
        extra_env = args.get("env", {})

        env = os.environ.copy()
        env.update(extra_env)
        env.update(context.env)
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return ToolResult(
                success=(result.returncode == 0),
                output=result.stdout,
                error=result.stderr if result.stderr else None,
                return_code=result.returncode,
                metadata={
                    "command": cmd,
                    "cwd": cwd,
                    "timeout": timeout,
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout}s",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
