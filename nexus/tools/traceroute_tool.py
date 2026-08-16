"""
Traceroute Tool - Trace network path tới một host.
Author: Hieu Louis (2026)

Wraps `traceroute` (Linux/Mac) hoặc `tracert` (Windows) subprocess.
Capture stdout/stderr, respect timeout, không cần deps ngoài.
"""
from __future__ import annotations

import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class TracerouteTool(Tool):
    """Run traceroute (Linux/Mac) hoặc tracert (Windows)."""

    category = ToolCategory.NETWORK
    safety = ToolSafety.SAFE

    # Regex đơn giản để parse từng hop line // simple parser for hop lines
    _HOP_RE = re.compile(r"^\s*(\d+)\s+(.*)$")

    @property
    def name(self) -> str:
        return "traceroute"

    @property
    def description(self) -> str:
        return (
            "Traceroute: hiển thị đường đi mạng từ máy này tới host đích "
            "(mỗi hop = 1 router, kèm RTT). Dùng lệnh `traceroute` "
            "(Linux/Mac) hoặc `tracert` (Windows)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Hostname hoặc IP đích"},
                "max_hops": {"type": "integer", "default": 30, "description": "Số hop tối đa"},
                "timeout": {"type": "integer", "default": 5, "description": "Timeout mỗi hop (giây)"},
                "queries": {"type": "integer", "default": 3, "description": "Số probe mỗi hop"},
                "no_resolve": {
                    "type": "boolean",
                    "default": False,
                    "description": "Không resolve IP → hostname (chỉ IP)",
                },
            },
            "required": ["host"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("host"):
            return "Missing required arg: host"
        max_hops = int(args.get("max_hops", 30))
        if not (1 <= max_hops <= 64):
            return "max_hops phải nằm trong [1, 64]"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        host: str = str(args["host"]).strip()
        max_hops: int = int(args.get("max_hops", 30))
        per_hop_timeout: int = int(args.get("timeout", 5))
        queries: int = int(args.get("queries", 3))
        no_resolve: bool = bool(args.get("no_resolve", False))

        # Total timeout: hops * per_hop_timeout * queries + buffer
        # Tổng timeout cho subprocess không được vượt context.timeout
        total_budget = max_hops * per_hop_timeout * queries + 10
        subproc_timeout = min(total_budget, max(context.timeout, 30))

        # Detect platform & build cmd // platform-aware command
        is_windows = sys.platform == "win32"
        if is_windows:
            cmd: List[str] = ["tracert", "-d" if no_resolve else "", "-h", str(max_hops),
                              "-w", str(per_hop_timeout * 1000), host]
            cmd = [c for c in cmd if c]  # drop empty
        else:
            cmd = ["traceroute"]
            if no_resolve:
                cmd.append("-n")
            cmd += ["-m", str(max_hops), "-w", str(per_hop_timeout),
                    "-q", str(queries), host]

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would run: {' '.join(cmd)}",
                metadata={
                    "dry_run": True,
                    "host": host,
                    "max_hops": max_hops,
                    "command": cmd,
                },
            )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=subproc_timeout,
                check=False,
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error=(
                    f"Không tìm thấy lệnh {'tracert' if is_windows else 'traceroute'}"
                ),
                return_code=127,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Traceroute timeout sau {subproc_timeout}s",
                return_code=124,
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=str(e), return_code=1)

        # Parse hops // parse hops
        hops = self._parse_hops(result.stdout)
        return ToolResult(
            success=(result.returncode == 0),
            output=result.stdout.strip(),
            error=(result.stderr.strip() if result.stderr else None),
            return_code=result.returncode,
            metadata={
                "host": host,
                "max_hops": max_hops,
                "hops_parsed": hops,
                "hop_count": len(hops),
                "platform": "windows" if is_windows else "unix",
                "reached_target": bool(hops) and host in hops[-1].get("hosts", []),
            },
        )

    def _parse_hops(self, text: str) -> List[Dict[str, Any]]:
        """Parse traceroute output thành structured hops."""
        hops: List[Dict[str, Any]] = []
        for line in text.splitlines():
            m = self._HOP_RE.match(line)
            if not m:
                continue
            hop_num = int(m.group(1))
            rest = m.group(2).strip()
            # Tách RTT và hostname/IP // split tokens
            rtts: List[str] = []
            hosts: List[str] = []
            for tok in rest.split():
                # RTT looks like 12.345 ms hoặc * hoặc "3000.000ms!"
                if tok in ("*", "* * *", ""):
                    rtts.append("*")
                elif re.match(r"^\d+(\.\d+)?\s*ms$", tok) or re.match(r"^\d+(\.\d+)?ms!?$", tok):
                    rtts.append(tok.replace("ms", "").replace("!", ""))
                else:
                    hosts.append(tok)
            hops.append({
                "hop": hop_num,
                "hosts": hosts,
                "rtts": rtts,
            })
        return hops
