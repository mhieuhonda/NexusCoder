"""
DNS Query Tool - Truy vấn DNS nâng cao với dnspython.
Author: Hieu Louis (2026)

Lazy import `dnspython`. Nếu không có, fallback sang stdlib `socket`
cho A/AAAA records (chỉ basic resolution).
"""
from __future__ import annotations

import socket
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class DNSQueryTool(Tool):
    """Truy vấn DNS nâng cao: A, AAAA, MX, NS, TXT, CNAME, SOA, SRV."""

    category = ToolCategory.NETWORK
    safety = ToolSafety.SAFE

    SUPPORTED_TYPES = ("A", "AAAA", "MX", "NS", "TXT", "CNAME", "SOA", "SRV")

    @property
    def name(self) -> str:
        return "dns_query"

    @property
    def description(self) -> str:
        return (
            "Truy vấn DNS nâng cao (A, AAAA, MX, NS, TXT, CNAME, SOA, SRV) "
            "qua dnspython. Hỗ trợ custom nameserver và DNS-over-TLS tuỳ chọn. "
            "Fallback sang socket.getaddrinfo cho A/AAAA nếu thiếu deps."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Tên miền cần truy vấn"},
                "record_type": {
                    "type": "string",
                    "enum": list(self.SUPPORTED_TYPES),
                    "default": "A",
                    "description": "Loại record DNS",
                },
                "nameserver": {
                    "type": "string",
                    "description": "Custom nameserver (vd 8.8.8.8). Bỏ trống = system default.",
                },
                "timeout": {"type": "integer", "default": 10, "description": "Timeout (giây)"},
            },
            "required": ["domain"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("domain"):
            return "Missing required arg: domain"
        rtype = str(args.get("record_type", "A")).upper()
        if rtype not in self.SUPPORTED_TYPES:
            return f"record_type phải là {self.SUPPORTED_TYPES}, nhận được '{rtype}'"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        domain: str = str(args["domain"]).strip().rstrip(".")
        rtype: str = str(args.get("record_type", "A")).upper()
        nameserver: Optional[str] = args.get("nameserver")
        timeout = int(args.get("timeout") or 10)

        # Thử dnspython // try dnspython
        try:
            import dns.resolver  # type: ignore
            import dns.exception  # type: ignore
        except ImportError:
            # Fallback A/AAAA via socket // stdlib fallback
            if rtype in ("A", "AAAA"):
                return self._fallback_socket(domain, rtype, timeout)
            return ToolResult(
                success=False,
                error=(
                    "dnspython chưa cài. Chỉ hỗ trợ A/AAAA qua socket. "
                    "Cài: pip install dnspython"
                ),
                return_code=1,
            )

        try:
            resolver = dns.resolver.Resolver()
            resolver.timeout = timeout
            resolver.lifetime = timeout + 5
            if nameserver:
                resolver.nameservers = [nameserver]

            answer = resolver.resolve(domain, rtype)
            records: List[Dict[str, Any]] = []
            for rr in answer:
                records.append(self._format_rr(rr, rtype))

            ttl = None
            try:
                ttl = answer.rrset.ttl if answer.rrset else None  # type: ignore[union-attr]
            except Exception:
                pass

            pretty = "\n".join(f"  {r['value']}" for r in records)
            return ToolResult(
                success=True,
                output=f"{rtype} records for {domain}:\n{pretty}",
                metadata={
                    "domain": domain,
                    "record_type": rtype,
                    "nameserver": nameserver or "system",
                    "ttl": ttl,
                    "records": records,
                    "count": len(records),
                },
            )
        except dns.resolver.NXDOMAIN:
            return ToolResult(
                success=False,
                error=f"NXDOMAIN: {domain} không tồn tại",
                return_code=1,
                metadata={"domain": domain, "record_type": rtype},
            )
        except dns.resolver.NoAnswer:
            return ToolResult(
                success=True,
                output=f"Không có {rtype} record cho {domain}",
                metadata={"domain": domain, "record_type": rtype, "records": [], "count": 0},
            )
        except dns.exception.Timeout:
            return ToolResult(
                success=False,
                error=f"DNS query timeout sau {timeout}s",
                return_code=124,
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=str(e), return_code=1)

    def _format_rr(self, rr: Any, rtype: str) -> Dict[str, Any]:
        """Format 1 record thành dict chuẩn // normalize RR."""
        try:
            if rtype == "MX":
                return {"value": str(rr.exchange).rstrip("."), "preference": int(rr.preference)}
            if rtype == "SRV":
                return {
                    "value": str(rr.target).rstrip("."),
                    "priority": int(rr.priority),
                    "weight": int(rr.weight),
                    "port": int(rr.port),
                }
            if rtype in ("SOA",):
                return {
                    "mname": str(rr.mname).rstrip("."),
                    "rname": str(rr.rname).rstrip("."),
                    "serial": int(rr.serial),
                    "refresh": int(rr.refresh),
                    "retry": int(rr.retry),
                    "expire": int(rr.expire),
                    "minimum": int(rr.minimum),
                }
            if rtype == "TXT":
                txt = b"".join(s for s in rr.strings)
                return {"value": txt.decode("utf-8", errors="replace")}
            return {"value": str(rr).rstrip(".")}
        except Exception as e:  # noqa: BLE001
            return {"value": str(rr), "format_error": str(e)}

    def _fallback_socket(self, domain: str, rtype: str, timeout: int) -> ToolResult:
        """Fallback cho A/AAAA khi thiếu dnspython // socket-based fallback."""
        try:
            socket.setdefaulttimeout(timeout)
            family = socket.AF_INET6 if rtype == "AAAA" else socket.AF_INET
            results = socket.getaddrinfo(domain, None, family)
            ips = list(dict.fromkeys(r[4][0] for r in results))
            pretty = "\n".join(f"  {ip}" for ip in ips)
            return ToolResult(
                success=True,
                output=f"{rtype} records for {domain}:\n{pretty}",
                metadata={
                    "domain": domain,
                    "record_type": rtype,
                    "records": [{"value": ip} for ip in ips],
                    "count": len(ips),
                    "fallback": "socket.getaddrinfo",
                },
            )
        except socket.gaierror as e:
            return ToolResult(
                success=False,
                error=f"DNS resolution failed: {e}",
                return_code=1,
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=str(e), return_code=1)
