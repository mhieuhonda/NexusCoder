"""Git Operations Tool - git commands."""
from __future__ import annotations

import subprocess
from typing import Dict, Any

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


SAFE_GIT_COMMANDS = {
    "status", "log", "diff", "show", "branch", "tag",
    "ls-files", "ls-tree", "blame", "shortlog", "describe",
    "rev-parse", "config --get", "remote -v", "stash list",
}

MODERATE_GIT_COMMANDS = {
    "add", "commit", "fetch", "pull", "merge", "rebase",
    "stash", "checkout -b", "switch -c", "tag -a",
}

DANGEROUS_GIT_COMMANDS = {
    "push", "reset --hard", "clean -fd", "push --force",
    "branch -D", "tag -d", "rebase -i",
}


class GitTool(Tool):
    """Execute git commands với safety checks."""
    category = ToolCategory.FILE
    safety = ToolSafety.DANGEROUS  # default for safety
    
    @property
    def name(self) -> str:
        return "git_ops"
    
    @property
    def description(self) -> str:
        return (
            "Execute git commands. Auto-classify safety: "
            "read-only (status, log, diff) = SAFE, "
            "writes (commit, merge) = MODERATE, "
            "destructive (push, reset --hard) = DANGEROUS."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Git command (e.g. 'status', 'add .', 'commit -m \"msg\"')"},
                "repo": {"type": "string", "description": "Path to git repo (default: cwd)"},
            },
            "required": ["command"],
        }
    
    def validate_args(self, args: Dict[str, Any]) -> Any:
        cmd = args.get("command", "").strip()
        if not cmd:
            return "Empty git command"
        # Check for dangerous patterns
        for danger in DANGEROUS_GIT_COMMANDS:
            if danger in cmd:
                args["_safety_override"] = "dangerous"
                break
        return None
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        cmd = args["command"]
        repo = args.get("repo") or context.working_dir
        
        full_cmd = f"git -C {repo} {cmd}"
        
        try:
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=context.timeout,
                check=False,
            )
            return ToolResult(
                success=(result.returncode == 0),
                output=result.stdout,
                error=result.stderr if result.stderr else None,
                return_code=result.returncode,
                metadata={"git_command": cmd, "repo": repo},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Git command timed out",
                return_code=124,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
