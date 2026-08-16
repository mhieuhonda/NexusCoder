"""
Crontab Tool - Read/write user crontab via `crontab` CLI.
Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, Any, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# Operations được hỗ trợ
CRONTAB_OPERATIONS = {"list", "add", "remove", "replace", "show"}

# Read-only ops
READONLY_OPS = {"list", "show"}

# Write ops (cần confirmation + dry_run)
WRITE_OPS = {"add", "remove", "replace"}


class CrontabTool(Tool):
    """Đọc/ghi user crontab via `crontab` CLI."""
    category = ToolCategory.SYSTEM
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "crontab"

    @property
    def description(self) -> str:
        return (
            "Read/write user crontab: list (`crontab -l`), add entry, "
            "remove all (`crontab -r`), replace from file/text (`crontab -`). "
            "Có dry_run và backup trước khi ghi."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": sorted(CRONTAB_OPERATIONS),
                },
                "entry": {
                    "type": "string",
                    "description": "Cron entry to add (e.g. '0 3 * * * /opt/backup.sh')",
                },
                "content": {
                    "type": "string",
                    "description": "Full crontab content for 'replace'",
                },
                "user": {"type": "string", "description": "Target user (-u). Cần root."},
                "backup": {"type": "boolean", "default": True, "description": "Backup current crontab trước khi ghi"},
            },
            "required": ["operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        op = args.get("operation")
        if not op:
            return "Missing required arg: operation"
        if op not in CRONTAB_OPERATIONS:
            return f"Unsupported operation: {op}"
        if op == "add" and not args.get("entry"):
            return "Operation 'add' requires 'entry' arg"
        if op == "replace" and not args.get("content"):
            return "Operation 'replace' requires 'content' arg"
        return None

    def _list_current(self, user: Optional[str], timeout: int) -> tuple:
        """Return (stdout, returncode). Trả về chuỗi rỗng nếu chưa có crontab."""
        cmd = ["crontab", "-l"]
        if user:
            cmd = ["crontab", "-u", user, "-l"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return result.stdout, result.returncode
        except FileNotFoundError:
            raise
        except subprocess.TimeoutExpired:
            raise

    def _write_crontab(
        self,
        content: str,
        user: Optional[str],
        timeout: int,
        env: Dict[str, str],
    ) -> subprocess.CompletedProcess:
        cmd = ["crontab", "-"]
        if user:
            cmd = ["crontab", "-u", user, "-"]
        return subprocess.run(
            cmd,
            input=content,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        op = args["operation"]
        user = args.get("user")
        env = dict(os.environ)
        env.update(context.env)

        # ---- list / show ----
        if op in {"list", "show"}:
            try:
                stdout, rc = self._list_current(user, context.timeout)
                if rc != 0 and "no crontab for" not in (stdout + "").lower():
                    return ToolResult(
                        success=False,
                        error=stdout or "crontab -l failed",
                        return_code=rc,
                    )
                return ToolResult(
                    success=True,
                    output=stdout or "(no crontab)",
                    return_code=0,
                    metadata={
                        "operation": op,
                        "user": user,
                        "empty": not stdout.strip(),
                    },
                )
            except FileNotFoundError:
                return ToolResult(
                    success=False,
                    error="crontab CLI not found.",
                    return_code=127,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(success=False, error="crontab -l timed out", return_code=124)

        # ---- Dry-run for write ops ----
        if context.dry_run:
            preview: str
            if op == "add":
                preview = f"[dry-run] Would add entry: {args['entry']}"
            elif op == "remove":
                preview = "[dry-run] Would remove all crontab entries (crontab -r)"
            else:  # replace
                preview = f"[dry-run] Would replace crontab with:\n{args['content']}"
            return ToolResult(
                success=True,
                output=preview,
                metadata={
                    "dry_run": True,
                    "operation": op,
                    "user": user,
                },
            )

        # ---- Read current for backup/append ----
        try:
            current, _ = self._list_current(user, context.timeout)
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="crontab CLI not found.",
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="crontab -l timed out", return_code=124)

        backup_path: Optional[str] = None
        if args.get("backup", True) and current.strip():
            backup_path = os.path.join(
                context.working_dir,
                f".crontab.backup.{os.getpid()}.txt",
            )
            try:
                with open(backup_path, "w", encoding="utf-8") as f:
                    f.write(current)
            except OSError as e:
                return ToolResult(
                    success=False,
                    error=f"Backup failed: {e}",
                    return_code=1,
                )

        # ---- Compose new content ----
        if op == "add":
            new_content = current.rstrip() + "\n" + args["entry"] + "\n"
        elif op == "remove":
            new_content = ""
        else:  # replace
            new_content = args["content"] + (
                "" if args["content"].endswith("\n") else "\n"
            )

        # ---- Write ----
        try:
            result = self._write_crontab(new_content, user, context.timeout, env)
            return ToolResult(
                success=(result.returncode == 0),
                output=new_content,
                error=result.stderr or None,
                return_code=result.returncode,
                artifacts=[backup_path] if backup_path else [],
                metadata={
                    "operation": op,
                    "user": user,
                    "backup_path": backup_path,
                    "entry_count": len([l for l in new_content.splitlines() if l.strip() and not l.strip().startswith("#")]),
                    "dry_run": False,
                },
            )
        except FileNotFoundError:
            return ToolResult(success=False, error="crontab CLI not found.", return_code=127)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="crontab - timed out", return_code=124)
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
