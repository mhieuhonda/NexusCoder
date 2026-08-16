"""Network Tools - DNS, ping, port check."""
from __future__ import annotations

import socket
import subprocess
from typing import Dict, Any

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class DNSTool(Tool):
    """DNS lookup: A, AAAA, MX, NS, CNAME, TXT records."""
    category = ToolCategory.NETWORK
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "dns_lookup"
    
    @property
    def description(self) -> str:
        return "DNS lookup: resolve domain → IP (A/AAAA), MX, NS, CNAME, TXT records."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "record_type": {
                    "type": "string",
                    "enum": ["A", "AAAA", "MX", "NS", "CNAME", "TXT", "SOA", "ANY"],
                    "default": "A",
                },
            },
            "required": ["domain"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        domain = args["domain"]
        record_type = args.get("record_type", "A")
        
        try:
            if record_type == "A":
                results = socket.getaddrinfo(domain, None, socket.AF_INET)
                ips = list(set(r[4][0] for r in results))
                return ToolResult(
                    success=True,
                    output=f"A records for {domain}:\n" + "\n".join(f"  {ip}" for ip in ips),
                    metadata={"domain": domain, "records": ips},
                )
            elif record_type == "AAAA":
                results = socket.getaddrinfo(domain, None, socket.AF_INET6)
                ips = list(set(r[4][0] for r in results))
                return ToolResult(
                    success=True,
                    output=f"AAAA records for {domain}:\n" + "\n".join(f"  {ip}" for ip in ips),
                    metadata={"domain": domain, "records": ips},
                )
            else:
                # Use dig if available
                try:
                    result = subprocess.run(
                        ["dig", "+short", domain, record_type],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    if result.returncode == 0:
                        records = [r.strip() for r in result.stdout.splitlines() if r.strip()]
                        return ToolResult(
                            success=True,
                            output=f"{record_type} records for {domain}:\n" + "\n".join(f"  {r}" for r in records),
                            metadata={"domain": domain, "records": records},
                        )
                    else:
                        return ToolResult(
                            success=False,
                            error=f"dig failed: {result.stderr}",
                            return_code=1,
                        )
                except FileNotFoundError:
                    return ToolResult(
                        success=False,
                        error="dig not installed. Only A/AAAA supported via socket.",
                        return_code=1,
                    )
        except socket.gaierror as e:
            return ToolResult(success=False, error=f"DNS resolution failed: {e}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)


class PingTool(Tool):
    """Ping host để check connectivity."""
    category = ToolCategory.NETWORK
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "ping"
    
    @property
    def description(self) -> str:
        return "Ping host để kiểm tra connectivity và đo latency."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "count": {"type": "integer", "default": 4},
                "timeout": {"type": "integer", "default": 5},
            },
            "required": ["host"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        host = args["host"]
        count = args.get("count", 4)
        
        # Detect OS for ping flag
        import sys
        flag = "-n" if sys.platform == "win32" else "-c"
        wait_flag = "-w" if sys.platform == "win32" else "-W"
        
        try:
            result = subprocess.run(
                ["ping", flag, str(count), wait_flag, str(args.get("timeout", 5)), host],
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
                metadata={"host": host, "count": count},
            )
        except FileNotFoundError:
            return ToolResult(success=False, error="ping command not found", return_code=1)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="Ping timed out", return_code=124)
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
