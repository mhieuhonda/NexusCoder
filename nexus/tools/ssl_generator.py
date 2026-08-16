"""
SSL Generator Tool - Tạo self-signed SSL certificate (X.509).
Author: Hieu Louis (2026)

Dùng `cryptography` (lazy import). Sinh cặp key + self-signed cert PEM.
⚠️ Chỉ dùng cho dev/test. KHÔNG dùng cho production trust chains.
"""
from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class SSLGeneratorTool(Tool):
    """Generate self-signed X.509 certificate + RSA private key.

    Outputs:
      - <output_dir>/<common_name>.crt  (certificate PEM)
      - <output_dir>/<common_name>.key  (private key PEM, không mật khẩu)
    """

    category = ToolCategory.SECURITY
    safety = ToolSafety.DANGEROUS
    requires_confirmation = True
    timeout = 60

    @property
    def name(self) -> str:
        return "ssl_generator"

    @property
    def description(self) -> str:
        return (
            "Sinh self-signed X.509 certificate + RSA private key (PEM). "
            "Hỗ trợ SAN list, organisation, country, validity_days, key_size. "
            "⚠️ Chỉ dùng cho dev/test, không production."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "common_name": {"type": "string", "description": "Common Name (CN), vd example.com"},
                "organization": {"type": "string", "default": "NexusCoder", "description": "Organization (O)"},
                "country": {"type": "string", "default": "VN", "description": "Country code 2 ký tự (C)"},
                "validity_days": {"type": "integer", "default": 365, "description": "Số ngày hiệu lực"},
                "key_size": {"type": "integer", "default": 2048, "description": "RSA key size (1024/2048/4096)"},
                "san": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Subject Alternative Names (DNS hoặc IP)",
                },
                "output_dir": {
                    "type": "string",
                    "default": ".",
                    "description": "Thư mục lưu file PEM (mặc định = cwd)",
                },
            },
            "required": ["common_name"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        cn = str(args.get("common_name", "")).strip()
        if not cn:
            return "Missing required arg: common_name"
        if len(cn) > 64:
            return "common_name quá dài (max 64 ký tự)"
        key_size = int(args.get("key_size", 2048))
        if key_size not in (1024, 2048, 3072, 4096):
            return f"key_size phải là 1024/2048/3072/4096, nhận được {key_size}"
        validity = int(args.get("validity_days", 365))
        if not (1 <= validity <= 3650):
            return "validity_days phải nằm trong [1, 3650]"
        country = str(args.get("country", "VN"))
        if len(country) != 2 or not country.isalpha():
            return "country phải là mã 2 ký tự alpha (vd VN, US)"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            from cryptography import x509  # type: ignore
            from cryptography.x509.oid import NameOID  # type: ignore
            from cryptography.hazmat.primitives import hashes, serialization  # type: ignore
            from cryptography.hazmat.primitives.asymmetric import rsa  # type: ignore
            from cryptography.hazmat.backends import default_backend  # type: ignore
        except ImportError:
            return ToolResult(
                success=False,
                error="cryptography chưa cài. Cài: pip install cryptography",
                return_code=1,
            )

        cn: str = str(args["common_name"]).strip()
        org: str = str(args.get("organization", "NexusCoder"))
        country: str = str(args.get("country", "VN")).upper()
        validity_days: int = int(args.get("validity_days", 365))
        key_size: int = int(args.get("key_size", 2048))
        san_list: List[str] = args.get("san") or []
        output_dir: str = str(args.get("output_dir") or context.working_dir or ".")

        # Safe filename from CN // safe filename from CN
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in cn).strip("._-") or "cert"
        cert_path = os.path.join(output_dir, f"{safe_name}.crt")
        key_path = os.path.join(output_dir, f"{safe_name}.key")

        if context.dry_run:
            return ToolResult(
                success=True,
                output=(
                    f"[dry-run] Would generate self-signed cert CN={cn} "
                    f"({key_size}-bit RSA, {validity_days}d) → {cert_path} + {key_path}"
                ),
                metadata={
                    "dry_run": True,
                    "common_name": cn,
                    "key_size": key_size,
                    "validity_days": validity_days,
                    "san": san_list,
                    "cert_path": cert_path,
                    "key_path": key_path,
                },
            )

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                success=False,
                error=f"Cannot create output_dir {output_dir}: {e}",
                return_code=1,
            )

        try:
            # 1. Generate RSA private key // generate key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend(),
            )

            # 2. Build subject/issuer (self-signed = subject == issuer)
            name_attrs = [
                x509.NameAttribute(NameOID.COMMON_NAME, cn),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, org),
                x509.NameAttribute(NameOID.COUNTRY_NAME, country),
            ]
            name = x509.Name(name_attrs)
            issuer = name  # self-signed

            # 3. Build SAN list (filter DNS/IP)
            san_entries: List[Any] = []
            for s in san_list:
                s = str(s).strip()
                if not s:
                    continue
                # Detect IP // detect IP literals
                import ipaddress
                try:
                    ipaddress.ip_address(s)
                    san_entries.append(x509.IPAddress(ipaddress.ip_address(s)))
                    continue
                except ValueError:
                    san_entries.append(x509.DNSName(s))

            now = datetime.datetime.now(datetime.timezone.utc)
            builder = (
                x509.CertificateBuilder()
                .subject_name(name)
                .issuer_name(issuer)
                .public_key(private_key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - datetime.timedelta(minutes=1))
                .not_valid_after(now + datetime.timedelta(days=validity_days))
                .add_extension(
                    x509.BasicConstraints(ca=False, path_length=None),
                    critical=True,
                )
                .add_extension(
                    x509.KeyUsage(
                        digital_signature=True,
                        key_encipherment=True,
                        content_commitment=False,
                        data_encipherment=False,
                        key_agreement=False,
                        key_cert_sign=False,
                        crl_sign=False,
                        encipher_only=False,
                        decipher_only=False,
                    ),
                    critical=True,
                )
                .add_extension(
                    x509.ExtendedKeyUsage([
                        x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                        x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    ]),
                    critical=False,
                )
            )
            if san_entries:
                builder = builder.add_extension(
                    x509.SubjectAlternativeName(san_entries),
                    critical=False,
                )
            certificate = builder.sign(
                private_key=private_key,
                algorithm=hashes.SHA256(),
                backend=default_backend(),
            )

            # 4. Serialize to PEM // serialize PEM
            cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
            key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )

            # 5. Write files with strict perms
            with open(key_path, "wb") as f:
                f.write(key_pem)
            try:
                os.chmod(key_path, 0o600)
            except OSError:
                pass

            with open(cert_path, "wb") as f:
                f.write(cert_pem)
            try:
                os.chmod(cert_path, 0o644)
            except OSError:
                pass

            return ToolResult(
                success=True,
                output=(
                    f"Generated self-signed cert:\n"
                    f"  CN={cn}, O={org}, C={country}\n"
                    f"  Key: RSA {key_size}-bit\n"
                    f"  Validity: {validity_days} days\n"
                    f"  SAN: {san_list or '(none)'}\n"
                    f"  Cert → {cert_path}\n"
                    f"  Key  → {key_path}"
                ),
                artifacts=[cert_path, key_path],
                metadata={
                    "common_name": cn,
                    "organization": org,
                    "country": country,
                    "key_size": key_size,
                    "validity_days": validity_days,
                    "san": san_list,
                    "serial_number": format(certificate.serial_number, "x"),
                    "cert_path": cert_path,
                    "key_path": key_path,
                },
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=str(e), return_code=1)
