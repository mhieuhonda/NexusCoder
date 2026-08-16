"""
Port Scanner Tool - Quét port mở trên một host.
Author: Hieu Louis (2026)

⚠️ ETHICS WARNING: Chỉ quét systems mà bạn sở hữu hoặc có quyền quét.
Quét host không được phép có thể vi phạm luật (Computer Fraud & Abuse Act,
Cybercrime Convention, luật Tội phạm mạng VN 2015/2018).

Dùng stdlib `socket` non-blocking connect với timeout per-port.
Không cần deps ngoài.
"""
from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class PortScannerTool(Tool):
    """Quét port mở trên một host qua TCP connect().

    ⚠️ Chỉ sử dụng trên systems mà bạn sở hữu hoặc có quyền rõ ràng.
    """

    category = ToolCategory.SECURITY
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True
    timeout = 60

    # Giới hạn để tránh lạm dụng // hard caps for safety
    MAX_PORTS_PER_SCAN = 65535
    MAX_WORKERS = 200

    @property
    def name(self) -> str:
        return "port_scanner"

    @property
    def description(self) -> str:
        return (
            "Quét TCP port trên một host. Hỗ trợ list port (vd [22,80,443]) "
            "hoặc range string '1-1024'. Non-blocking connect với timeout. "
            "⚠️ Chỉ dùng cho host bạn sở hữu/có quyền."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Hostname hoặc IP đích"},
                "ports": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "integer"}},
                        {"type": "string", "description": "vd '1-1024' hoặc '80,443,8080'"},
                    ],
                    "description": "Danh sách port hoặc range string",
                },
                "timeout_per_port": {"type": "integer", "default": 1, "description": "Timeout cho mỗi port (giây)"},
                "workers": {"type": "integer", "default": 50, "description": "Số thread scan song song"},
            },
            "required": ["host", "ports"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("host"):
            return "Missing required arg: host"
        if not args.get("ports"):
            return "Missing required arg: ports"
        ports = self._parse_ports(args["ports"])
        if not ports:
            return "Không parse được ports (phải là list[int] hoặc '1-1024' hoặc '80,443')"
        if len(ports) > self.MAX_PORTS_PER_SCAN:
            return f"Quá nhiều ports (>{self.MAX_PORTS_PER_SCAN})"
        workers = int(args.get("workers", 50))
        if not (1 <= workers <= self.MAX_WORKERS):
            return f"workers phải nằm trong [1, {self.MAX_WORKERS}]"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        host: str = str(args["host"]).strip()
        ports: List[int] = self._parse_ports(args["ports"])
        per_port_timeout: float = float(args.get("timeout_per_port", 1))
        workers = min(int(args.get("workers", 50)), self.MAX_WORKERS, max(1, len(ports)))

        if context.dry_run:
            return ToolResult(
                success=True,
                output=(
                    f"[dry-run] Would scan {len(ports)} ports on {host} "
                    f"(workers={workers}, timeout={per_port_timeout}s)"
                ),
                metadata={
                    "dry_run": True,
                    "host": host,
                    "port_count": len(ports),
                    "port_range": [min(ports), max(ports)],
                },
            )

        # Resolve host trước // resolve first to fail fast
        try:
            target_ip = socket.gethostbyname(host)
        except socket.gaierror as e:
            return ToolResult(
                success=False,
                error=f"Cannot resolve host '{host}': {e}",
                return_code=1,
            )

        open_ports: List[Dict[str, Any]] = []
        closed_count = 0
        error_count = 0

        def _scan(p: int) -> Tuple[int, bool, Optional[str]]:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(per_port_timeout)
                    rc = s.connect_ex((target_ip, p))
                    if rc == 0:
                        # Try banner grab (best-effort) // banner grab
                        banner = ""
                        try:
                            s.settimeout(0.5)
                            data = s.recv(128)
                            banner = data.decode("utf-8", errors="replace").strip()
                        except Exception:
                            pass
                        return p, True, banner or None
                    return p, False, None
            except socket.timeout:
                return p, False, None
            except Exception as e:  # noqa: BLE001
                return p, False, str(e)

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_scan, p): p for p in ports}
                for fut in as_completed(futures):
                    port, is_open, info = fut.result()
                    if is_open:
                        open_ports.append({
                            "port": port,
                            "state": "open",
                            "banner": info,
                            "service": self._guess_service(port),
                        })
                    elif info is None:
                        closed_count += 1
                    else:
                        error_count += 1
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=str(e), return_code=1)

        # Sort mở theo port // sort open ports
        open_ports.sort(key=lambda x: x["port"])

        # Format output
        lines: List[str] = [f"Scan result for {host} ({target_ip}):"]
        lines.append(f"  Scanned: {len(ports)} ports | Open: {len(open_ports)} | "
                     f"Closed: {closed_count} | Error: {error_count}")
        if open_ports:
            lines.append("  Open ports:")
            for op in open_ports:
                banner = f" — {op['banner']!r}" if op["banner"] else ""
                lines.append(f"    {op['port']}/tcp  {op['service']}{banner}")
        output = "\n".join(lines)

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "host": host,
                "ip": target_ip,
                "ports_scanned": len(ports),
                "open_ports": [p["port"] for p in open_ports],
                "open_count": len(open_ports),
                "closed_count": closed_count,
                "error_count": error_count,
                "open_details": open_ports,
            },
        )

    def _parse_ports(self, ports: Union[List[int], str]) -> List[int]:
        """Parse list[int] hoặc '1-1024' hoặc '80,443,8080' → sorted unique list."""
        if isinstance(ports, list):
            return sorted(set(int(p) for p in ports if 0 < int(p) <= 65535))
        if isinstance(ports, str):
            out: List[int] = []
            for part in ports.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    lo_i, hi_i = int(lo), int(hi)
                    if lo_i > hi_i:
                        lo_i, hi_i = hi_i, lo_i
                    out.extend(range(lo_i, hi_i + 1))
                else:
                    out.append(int(part))
            return sorted(set(p for p in out if 0 < p <= 65535))
        return []

    # Well-known service map // IANA-ish common services
    _SERVICES = {
        21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
        80: "http", 110: "pop3", 143: "imap", 443: "https", 465: "smtps",
        587: "submission", 993: "imaps", 995: "pop3s", 3306: "mysql",
        3389: "rdp", 5432: "postgresql", 6379: "redis", 8080: "http-alt",
        8443: "https-alt", 9200: "elasticsearch", 27017: "mongodb",
        9092: "kafka", 5672: "amqp", 1883: "mqtt", 6443: "k8s-api",
        10250: "k8s-kubelet", 2375: "docker", 50051: "grpc",
    }

    def _guess_service(self, port: int) -> str:
        return self._SERVICES.get(port, "unknown")
