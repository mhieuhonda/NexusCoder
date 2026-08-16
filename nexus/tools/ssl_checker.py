"""
SSL Checker Tool - Kiểm tra SSL/TLS certificate của một host:port.
Author: Hieu Louis (2026)

Dùng stdlib `ssl` + `socket`. Trả về issuer/subject/expiry/SAN/days_until_expiry.
Không cần deps ngoài.
"""
from __future__ import annotations

import socket
import ssl
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class SSLCheckerTool(Tool):
    """Check SSL certificate cho host:port (mặc định 443)."""

    category = ToolCategory.SECURITY
    safety = ToolSafety.SAFE

    @property
    def name(self) -> str:
        return "ssl_checker"

    @property
    def description(self) -> str:
        return (
            "Kiểm tra SSL/TLS certificate của host:port. Trả về subject, "
            "issuer, not_before, not_after, days_until_expiry, SAN list, "
            "signature algorithm. Dùng stdlib ssl + socket."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Hostname cần kiểm tra"},
                "port": {"type": "integer", "default": 443, "description": "Port TLS"},
                "timeout": {"type": "integer", "default": 10, "description": "Timeout (giây)"},
                "verify": {
                    "type": "boolean",
                    "default": False,
                    "description": "Có verify chain trust không (mặc định False để vẫn xem được cert ngay cả khi expired).",
                },
            },
            "required": ["host"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("host"):
            return "Missing required arg: host"
        port = int(args.get("port", 443))
        if not (1 <= port <= 65535):
            return "port phải nằm trong [1, 65535]"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        host: str = str(args["host"]).strip()
        port: int = int(args.get("port", 443))
        timeout = int(args.get("timeout") or 10)
        verify: bool = bool(args.get("verify", False))

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would check SSL for {host}:{port}",
                metadata={"dry_run": True, "host": host, "port": port},
            )

        # Tạo SSLContext // build SSL context
        ctx = ssl.create_default_context()
        if not verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        try:
            with socket.create_connection((host, port), timeout=timeout) as sock:
                with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                    cert_bin = ssock.getpeercert(binary_form=True)
                    peer_cert_dict = ssock.getpeercert()  # dict nếu CERT_NONE thì {}
                    cipher = ssock.cipher()
                    version = ssock.version()
        except socket.timeout:
            return ToolResult(
                success=False,
                error=f"Connection timeout tới {host}:{port} sau {timeout}s",
                return_code=124,
            )
        except socket.gaierror as e:
            return ToolResult(
                success=False,
                error=f"DNS resolve failed: {e}",
                return_code=1,
            )
        except ssl.SSLError as e:
            return ToolResult(
                success=False,
                error=f"SSL error: {e}",
                return_code=1,
                metadata={"host": host, "port": port},
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=str(e), return_code=1)

        if not cert_bin:
            return ToolResult(
                success=False,
                error=f"Không lấy được certificate từ {host}:{port}",
                return_code=1,
            )

        # Parse cert dict (nếu verify=True) hoặc dùng cryptography để parse DER
        # Khi CERT_NONE, peer_cert_dict sẽ rỗng; cần parse binary.
        info = self._parse_cert(cert_bin, peer_cert_dict, host)

        return ToolResult(
            success=True,
            output=self._format_report(host, port, info, version, cipher),
            metadata={
                "host": host,
                "port": port,
                "subject": info.get("subject"),
                "issuer": info.get("issuer"),
                "not_before": info.get("not_before"),
                "not_after": info.get("not_after"),
                "days_until_expiry": info.get("days_until_expiry"),
                "expired": info.get("expired"),
                "san": info.get("san", []),
                "signature_algorithm": info.get("signature_algorithm"),
                "serial_number": info.get("serial_number"),
                "tls_version": version,
                "cipher": list(cipher) if cipher else None,
                "verified_chain": verify,
            },
        )

    def _parse_cert(
        self, cert_bin: bytes, peer_cert_dict: Dict[str, Any], host: str
    ) -> Dict[str, Any]:
        """Parse certificate. Ưu tiên cryptography (nếu có), fallback dict."""
        # Try cryptography lib // prefer cryptography lib for full details
        try:
            from cryptography import x509  # type: ignore
            from cryptography.hazmat.backends import default_backend  # type: ignore
            cert = x509.load_der_x509_certificate(cert_bin, default_backend())
            subject = cert.subject.rfc4514_string()
            issuer = cert.issuer.rfc4514_string()
            not_before = cert.not_valid_before_utc if hasattr(cert, "not_valid_before_utc") else cert.not_valid_before
            not_after = cert.not_valid_after_utc if hasattr(cert, "not_valid_after_utc") else cert.not_valid_after
            not_before_iso = not_before.isoformat() if not_before else None
            not_after_iso = not_after.isoformat() if not_after else None
            san_list: List[str] = []
            try:
                ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
                san_list = ext.value.get_values_for_type(x509.DNSName)
            except x509.ExtensionNotFound:
                pass
            return {
                "subject": subject,
                "issuer": issuer,
                "not_before": not_before_iso,
                "not_after": not_after_iso,
                "days_until_expiry": self._days_until(not_after_iso),
                "expired": self._is_expired(not_after_iso),
                "san": san_list,
                "signature_algorithm": cert.signature_algorithm_oid._name if hasattr(cert, "signature_algorithm_oid") else str(cert.signature_hash_algorithm),
                "serial_number": format(cert.serial_number, "x"),
            }
        except ImportError:
            pass

        # Fallback: peer_cert dict (chỉ khi verify=True)
        if peer_cert_dict:
            subject = dict(x[0] for x in peer_cert_dict.get("subject", []))
            issuer = dict(x[0] for x in peer_cert_dict.get("issuer", []))
            not_before = peer_cert_dict.get("notBefore")
            not_after = peer_cert_dict.get("notAfter")
            san_list = [v for sub in peer_cert_dict.get("subjectAltName", []) for v in (sub[1],) if sub[0] == "DNS"]
            return {
                "subject": subject,
                "issuer": issuer,
                "not_before": not_before,
                "not_after": not_after,
                "days_until_expiry": self._days_until(not_after),
                "expired": self._is_expired(not_after),
                "san": san_list,
                "signature_algorithm": None,
                "serial_number": peer_cert_dict.get("serialNumber"),
            }

        return {"san": [], "subject": host, "issuer": "unknown"}

    @staticmethod
    def _parse_asn1_date(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            # Python ssl module format: "May 25 23:59:59 2025 GMT"
            return datetime.strptime(s, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        try:
            # ISO format
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None

    def _days_until(self, not_after: Optional[str]) -> Optional[int]:
        dt = self._parse_asn1_date(not_after)
        if not dt:
            return None
        return (dt - datetime.now(timezone.utc)).days

    def _is_expired(self, not_after: Optional[str]) -> Optional[bool]:
        days = self._days_until(not_after)
        return None if days is None else days < 0

    def _format_report(
        self, host: str, port: int, info: Dict[str, Any],
        tls_version: Optional[str], cipher: Optional[tuple],
    ) -> str:
        days = info.get("days_until_expiry")
        expired = info.get("expired")
        status = "EXPIRED" if expired else (f"{days}d left" if days is not None else "?")
        lines = [
            f"SSL/TLS report for {host}:{port}",
            f"  Subject            : {info.get('subject')}",
            f"  Issuer             : {info.get('issuer')}",
            f"  Not before         : {info.get('not_before')}",
            f"  Not after          : {info.get('not_after')}",
            f"  Days until expiry  : {status}",
            f"  Serial             : {info.get('serial_number')}",
            f"  Signature algorithm: {info.get('signature_algorithm')}",
            f"  TLS version        : {tls_version}",
            f"  Cipher             : {cipher[0] if cipher else 'n/a'}",
        ]
        san = info.get("san") or []
        lines.append(f"  SAN ({len(san)}):")
        for s in san:
            lines.append(f"    - {s}")
        return "\n".join(lines)
