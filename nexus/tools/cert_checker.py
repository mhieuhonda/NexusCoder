"""
Certificate Checker Tool - Parse & inspect X.509 certificate files.
Author: Hieu Louis (2026)

Lazy import `cryptography`. Nếu không có, fallback sang `openssl x509`
subprocess. Hỗ trợ PEM và DER.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class CertCheckerTool(Tool):
    """Parse và inspect X.509 certificate files (.pem, .crt, .der)."""

    category = ToolCategory.SECURITY
    safety = ToolSafety.SAFE

    @property
    def name(self) -> str:
        return "cert_checker"

    @property
    def description(self) -> str:
        return (
            "Parse & inspect X.509 certificate (.pem/.crt/.der). Trả về "
            "subject, issuer, validity, SAN, signature algorithm, serial. "
            "Auto-detect format (PEM/DER) nếu arg `format` không chỉ định."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Đường dẫn file cert (.pem/.crt/.cer/.der)"},
                "format": {
                    "type": "string",
                    "enum": ["pem", "der", "auto"],
                    "default": "auto",
                    "description": "Format cert. auto = tự detect.",
                },
            },
            "required": ["path"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        path = str(args.get("path", "")).strip()
        if not path:
            return "Missing required arg: path"
        if not os.path.exists(path):
            return f"File không tồn tại: {path}"
        if not os.path.isfile(path):
            return f"Path không phải file: {path}"
        fmt = str(args.get("format", "auto"))
        if fmt not in ("pem", "der", "auto"):
            return f"format phải là pem/der/auto, nhận được '{fmt}'"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path: str = str(args["path"])
        fmt: str = str(args.get("format", "auto"))

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Would inspect cert {path} (format={fmt})",
                metadata={"dry_run": True, "path": path, "format": fmt},
            )

        # Read bytes
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as e:
            return ToolResult(success=False, error=f"Cannot read file: {e}", return_code=1)

        # Auto-detect format
        if fmt == "auto":
            fmt = self._detect_format(data, path)
        if fmt not in ("pem", "der"):
            return ToolResult(
                success=False,
                error=f"Cannot detect format. Hint: pass `format` pem/der.",
                return_code=1,
            )

        # Try cryptography lib first
        try:
            from cryptography import x509  # type: ignore
            from cryptography.hazmat.backends import default_backend  # type: ignore
            cert_obj = self._load_with_cryptography(x509, default_backend, data, fmt)
            info = self._cert_to_dict_cryptography(cert_obj)
            return ToolResult(
                success=True,
                output=self._format_report(info, path, fmt, backend="cryptography"),
                metadata={**info, "path": path, "format": fmt, "backend": "cryptography"},
            )
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            # Nếu cryptography có nhưng parse lỗi → thử openssl fallback
            last_err = str(e)
        else:
            last_err = None

        # Fallback: openssl CLI
        info, err = self._parse_with_openssl(path, fmt, context.timeout or 15)
        if info is not None:
            return ToolResult(
                success=True,
                output=self._format_report(info, path, fmt, backend="openssl CLI"),
                metadata={**info, "path": path, "format": fmt, "backend": "openssl"},
            )
        return ToolResult(
            success=False,
            error=(
                f"Không parse được cert. cryptography không khả dụng hoặc "
                f"lỗi parse ({last_err}); openssl CLI fallback cũng thất bại: {err}"
            ),
            return_code=1,
        )

    # ---------- Format detection ----------
    @staticmethod
    def _detect_format(data: bytes, path: str) -> str:
        """PEM nếu có '-----BEGIN CERTIFICATE-----', ngược lại DER."""
        head = data[:64].lstrip()
        if head.startswith(b"-----BEGIN"):
            return "pem"
        ext = os.path.splitext(path)[1].lower()
        if ext in (".der", ".cer"):
            return "der"
        if ext in (".pem", ".crt"):
            return "pem"
        # Heuristic: DER thường bắt đầu bằng 0x30 (ASN.1 SEQUENCE)
        if data[:1] == b"\x30":
            return "der"
        return "pem"  # default

    # ---------- cryptography lib ----------
    @staticmethod
    def _load_with_cryptography(
        x509_module: Any, backend: Any, data: bytes, fmt: str
    ) -> Any:
        if fmt == "pem":
            return x509_module.load_pem_x509_certificate(data, backend)
        return x509_module.load_der_x509_certificate(data, backend)

    def _cert_to_dict_cryptography(self, cert: Any) -> Dict[str, Any]:
        from cryptography import x509 as _x509  # type: ignore
        from cryptography.x509.oid import ExtensionOID  # type: ignore

        not_before = getattr(cert, "not_valid_before_utc", None) or cert.not_valid_before
        not_after = getattr(cert, "not_valid_after_utc", None) or cert.not_valid_after

        san_list: List[str] = []
        try:
            ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            san = ext.value
            san_list = san.get_values_for_type(_x509.DNSName)
            try:
                for ip in san.get_values_for_type(_x509.IPAddress):
                    san_list.append(str(ip))
            except Exception:
                pass
        except _x509.ExtensionNotFound:
            pass

        sig_algo = None
        try:
            sig_algo = cert.signature_algorithm_oid._name
        except Exception:
            sig_algo = str(cert.signature_hash_algorithm)

        return {
            "subject": cert.subject.rfc4514_string(),
            "issuer": cert.issuer.rfc4514_string(),
            "not_before": not_before.isoformat() if not_before else None,
            "not_after": not_after.isoformat() if not_after else None,
            "san": san_list,
            "signature_algorithm": sig_algo,
            "serial_number": format(cert.serial_number, "x"),
            "version": str(cert.version),
            "is_ca": self._is_ca(cert, _x509, ExtensionOID),
        }

    @staticmethod
    def _is_ca(cert: Any, x509_module: Any, ExtensionOID: Any) -> bool:
        try:
            ext = cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS)
            return bool(ext.value.ca)
        except Exception:
            return False

    # ---------- openssl CLI fallback ----------
    def _parse_with_openssl(
        self, path: str, fmt: str, timeout: int
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        cmd = ["openssl", "x509", "-in", path, "-noout", "-text"]
        if fmt == "der":
            cmd.insert(2, "-inform")
            cmd.insert(3, "DER")
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )
        except FileNotFoundError:
            return None, "openssl CLI không có trên PATH"
        except subprocess.TimeoutExpired:
            return None, "openssl timeout"
        except Exception as e:  # noqa: BLE001
            return None, str(e)

        if res.returncode != 0:
            return None, res.stderr.strip() or "openssl parse error"

        text = res.stdout
        info: Dict[str, Any] = {
            "subject": self._grep_field(text, "Subject:"),
            "issuer": self._grep_field(text, "Issuer:"),
            "not_before": self._grep_field(text, "Not Before:"),
            "not_after": self._grep_field(text, "Not After:"),
            "signature_algorithm": self._grep_field(text, "Signature Algorithm:") or None,
            "serial_number": self._grep_field(text, "Serial Number:"),
            "san": [],
            "version": self._grep_field(text, "Version:"),
            "is_ca": False,
        }
        # Try SAN extraction
        san_line = self._grep_field(text, "DNS:")
        if san_line:
            info["san"] = [s.strip() for s in san_line.split(",") if s.strip().startswith("DNS:")]
            info["san"] = [s.replace("DNS:", "").strip() for s in info["san"]]
        info["is_ca"] = "CA:TRUE" in text
        return info, None

    @staticmethod
    def _grep_field(text: str, key: str) -> Optional[str]:
        for line in text.splitlines():
            if key in line:
                return line.split(":", 1)[-1].strip() or None
        return None

    # ---------- Report ----------
    @staticmethod
    def _format_report(info: Dict[str, Any], path: str, fmt: str, backend: str) -> str:
        san = info.get("san") or []
        lines = [
            f"Certificate inspection ({fmt.upper()}, backend={backend}):",
            f"  Path     : {path}",
            f"  Subject  : {info.get('subject')}",
            f"  Issuer   : {info.get('issuer')}",
            f"  Not before: {info.get('not_before')}",
            f"  Not after : {info.get('not_after')}",
            f"  Serial   : {info.get('serial_number')}",
            f"  Version  : {info.get('version')}",
            f"  Sig algo : {info.get('signature_algorithm')}",
            f"  Is CA    : {info.get('is_ca')}",
            f"  SAN ({len(san)}):",
        ]
        for s in san:
            lines.append(f"    - {s}")
        return "\n".join(lines)
