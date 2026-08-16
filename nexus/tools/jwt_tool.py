"""
JWT Tool - Decode / Encode / Verify JSON Web Tokens.
===========================================
Tool thao tác JWT: decode payload, encode token, verify chữ ký.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import base64
import json
import hmac
import hashlib
import time
from typing import Any, Dict, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


def _b64url_decode(seg: str) -> bytes:
    """Giải mã base64url (thêm padding nếu thiếu). / Decode base64url with padding fix."""
    pad = "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg + pad)


def _b64url_encode(raw: bytes) -> str:
    """Mã hóa base64url (bỏ padding). / Encode bytes to base64url without padding."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


# Algorithms được hỗ trợ cho HMAC (không cần lib ngoài). / HMAC algorithms supported w/o external deps.
_HMAC_ALGS: Dict[str, str] = {
    "HS256": "sha256",
    "HS384": "sha384",
    "HS512": "sha512",
}


class JWTTool(Tool):
    """Decode / Encode / Verify JWT (HS256/384/512 + RS256 via PyJWT)."""

    category = ToolCategory.CRYPTO
    safety = ToolSafety.SAFE

    @property
    def name(self) -> str:
        return "jwt"

    @property
    def description(self) -> str:
        return "Decode / encode / verify JWT tokens. Hỗ trợ HS256/384/512 (stdlib) + RS256 (PyJWT)."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "token": {"type": "string", "description": "JWT token (cho decode/verify)"},
                "action": {
                    "type": "string",
                    "enum": ["decode", "encode", "verify"],
                    "default": "decode",
                },
                "secret": {"type": "string", "description": "Secret key (cho HMAC)"},
                "algorithm": {
                    "type": "string",
                    "enum": ["HS256", "HS384", "HS512", "RS256", "none"],
                    "default": "HS256",
                },
                "payload": {
                    "type": "object",
                    "description": "Payload dict (cho encode)",
                },
            },
            "required": ["action"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        action = args.get("action", "decode")
        if action in ("decode", "verify") and not args.get("token"):
            return f"Missing required arg: token (cho action='{action}')"
        if action == "encode" and not args.get("payload"):
            return "Missing required arg: payload (cho action='encode')"
        if action in ("encode", "verify") and not args.get("secret") and args.get("algorithm", "HS256") != "none":
            return f"Missing required arg: secret (cho action='{action}')"
        return None

    # ---- Hành động chính / Main actions ---------------------------------

    def _decode(self, token: str) -> ToolResult:
        """Decode JWT mà không cần verify chữ ký (fallback thủ công)."""
        try:
            import jwt  # type: ignore
            unverified = jwt.decode(token, options={"verify_signature": False})
            header = jwt.get_unverified_header(token)
            return ToolResult(
                success=True,
                output=json.dumps({"header": header, "payload": unverified}, ensure_ascii=False, indent=2),
                metadata={"header": header, "payload": unverified, "verified": False},
            )
        except ImportError:
            pass
        except Exception:
            # Nếu PyJWT raise lỗi thì fallback về parse thủ công / fall back to manual parse
            pass

        try:
            parts = token.split(".")
            if len(parts) < 2:
                return ToolResult(success=False, error="Invalid JWT format (cần ít nhất 2 phần)", return_code=1)
            header = json.loads(_b64url_decode(parts[0]))
            payload = json.loads(_b64url_decode(parts[1]))
            return ToolResult(
                success=True,
                output=json.dumps({"header": header, "payload": payload}, ensure_ascii=False, indent=2),
                metadata={"header": header, "payload": payload, "verified": False},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Decode failed: {e}", return_code=1)

    def _encode(self, payload: Dict[str, Any], secret: str, algorithm: str) -> ToolResult:
        """Encode JWT — thử PyJWT trước, fallback HMAC thủ công."""
        try:
            import jwt  # type: ignore
            token = jwt.encode(payload, secret, algorithm=algorithm)
            return ToolResult(
                success=True,
                output=token,
                metadata={"algorithm": algorithm, "verified": True},
            )
        except ImportError:
            pass

        # Fallback HMAC thủ công / Manual HMAC fallback
        if algorithm not in _HMAC_ALGS:
            return ToolResult(
                success=False,
                error=f"Algorithm '{algorithm}' cần PyJWT: pip install PyJWT",
                return_code=1,
            )
        header = {"alg": algorithm, "typ": "JWT"}
        h_b64 = _b64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode())
        p_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
        signing_input = f"{h_b64}.{p_b64}".encode("ascii")
        sig = hmac.new(secret.encode(), signing_input, getattr(hashlib, _HMAC_ALGS[algorithm])).digest()
        token = f"{h_b64}.{p_b64}.{_b64url_encode(sig)}"
        return ToolResult(
            success=True,
            output=token,
            metadata={"algorithm": algorithm, "verified": True, "backend": "stdlib-hmac"},
        )

    def _verify(self, token: str, secret: str, algorithm: str) -> ToolResult:
        """Verify chữ ký JWT."""
        try:
            import jwt  # type: ignore
            payload = jwt.decode(token, secret, algorithms=[algorithm])
            return ToolResult(
                success=True,
                output="Signature valid",
                metadata={"payload": payload, "verified": True, "backend": "PyJWT"},
            )
        except ImportError:
            pass
        except Exception as e:
            return ToolResult(success=False, error=f"Verify failed (PyJWT): {e}", return_code=1)

        if algorithm not in _HMAC_ALGS:
            return ToolResult(
                success=False,
                error=f"Algorithm '{algorithm}' cần PyJWT: pip install PyJWT",
                return_code=1,
            )
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return ToolResult(success=False, error="Invalid JWT format (cần đúng 3 phần)", return_code=1)
            header = json.loads(_b64url_decode(parts[0]))
            if header.get("alg") != algorithm:
                return ToolResult(
                    success=False,
                    error=f"Algorithm mismatch: header={header.get('alg')} expected={algorithm}",
                    return_code=1,
                )
            signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
            expected_sig = hmac.new(secret.encode(), signing_input, getattr(hashlib, _HMAC_ALGS[algorithm])).digest()
            actual_sig = _b64url_decode(parts[2])
            if not hmac.compare_digest(expected_sig, actual_sig):
                return ToolResult(success=False, error="Invalid signature", return_code=1)
            payload = json.loads(_b64url_decode(parts[1]))
            # Kiểm tra exp / check expiry
            if isinstance(payload.get("exp"), (int, float)) and payload["exp"] < time.time():
                return ToolResult(success=False, error="Token expired", return_code=1)
            return ToolResult(
                success=True,
                output="Signature valid",
                metadata={"payload": payload, "verified": True, "backend": "stdlib-hmac"},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Verify failed: {e}", return_code=1)

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        action = args.get("action", "decode")
        algorithm = args.get("algorithm", "HS256")

        if action == "decode":
            return self._decode(args["token"])
        if action == "encode":
            return self._encode(args["payload"], args.get("secret", ""), algorithm)
        if action == "verify":
            return self._verify(args["token"], args.get("secret", ""), algorithm)
        return ToolResult(success=False, error=f"Unknown action: {action}", return_code=1)
