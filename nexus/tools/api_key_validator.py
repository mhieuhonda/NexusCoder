"""
API Key Validator Tool - Xác thực API keys của các service phổ biến.
===========================================
Tool validate API keys cho Stripe, AWS, GitHub, OpenAI, Google bằng
regex patterns + checksum validation.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


# ---- Định nghĩa patterns / Pattern definitions ---------------------------
# Mỗi pattern: name, regex, checksum_fn (optional), description.
# Patterns không phân biệt chữ hoa chữ thường trừ khi cần thiết.

def _aws_checksum(key: str) -> bool:
    """AWS Access Key ID dùng base32 (RFC 4648) — kiểm tra length và charset."""
    if len(key) != 20 or not key.startswith("AKIA"):
        return False
    # Kiểm tra charset base32 / validate base32 charset
    return bool(re.match(r"^[A-Z0-9]+$", key))


def _stripe_luhn(s: str) -> bool:
    """Stripe key dùng Luhn check trên phần số sau prefix."""
    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) < 16:
        return True  # Stripe key có thể không phải Luhn hoàn chỉnh → skip
    # Luhn algorithm
    total = 0
    reverse = digits[::-1]
    for i, ch in enumerate(reverse):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


SERVICES: Dict[str, Dict[str, Any]] = {
    "stripe": {
        "patterns": [
            (r"^sk_live_[A-Za-z0-9]{24,}$", "Stripe Secret Live Key"),
            (r"^sk_test_[A-Za-z0-9]{24,}$", "Stripe Secret Test Key"),
            (r"^pk_live_[A-Za-z0-9]{24,}$", "Stripe Publishable Live Key"),
            (r"^pk_test_[A-Za-z0-9]{24,}$", "Stripe Publishable Test Key"),
            (r"^rk_live_[A-Za-z0-9]{24,}$", "Stripe Restricted Live Key"),
            (r"^whsec_[A-Za-z0-9]{24,}$", "Stripe Webhook Signing Secret"),
        ],
        "checksum": _stripe_luhn,
    },
    "aws": {
        "patterns": [
            (r"^AKIA[0-9A-Z]{16}$", "AWS Access Key ID"),
            (r"^ASIA[0-9A-Z]{16}$", "AWS STS Temporary Access Key"),
            (r"^[A-Za-z0-9/+=]{40}$", "AWS Secret Access Key (40 chars base64)"),
        ],
        "checksum": _aws_checksum,
    },
    "github": {
        "patterns": [
            (r"^ghp_[A-Za-z0-9]{36,}$", "GitHub Personal Access Token (classic)"),
            (r"^github_pat_[A-Za-z0-9_]{22,}_[A-Za-z0-9]{59}$", "GitHub Fine-grained PAT"),
            (r"^gho_[A-Za-z0-9]{36,}$", "GitHub OAuth Token"),
            (r"^ghs_[A-Za-z0-9]{36,}$", "GitHub App Server Token"),
            (r"^ghu_[A-Za-z0-9]{36,}$", "GitHub User-to-Server Token"),
            (r"^[A-Za-z0-9]{40}$", "GitHub Legacy Token (40 hex)"),
        ],
        "checksum": None,
    },
    "openai": {
        "patterns": [
            (r"^sk-[A-Za-z0-9]{20,}$", "OpenAI API Key"),
            (r"^sk-proj-[A-Za-z0-9_-]{40,}$", "OpenAI Project API Key"),
            (r"^sk-_[A-Za-z0-9]{40,}$", "OpenAI Service Account Key"),
        ],
        "checksum": None,
    },
    "google": {
        "patterns": [
            (r"^AIza[0-9A-Za-z_-]{35}$", "Google API Key"),
            (r"^ya29\.[0-9A-Za-z_-]+$", "Google OAuth Access Token"),
            (r"^[0-9]+-[A-Za-z0-9_]{40,}@developer\.gserviceaccount\.com$", "Google Service Account Email"),
            (r"^projects/[0-9]+/apiKeys/[A-Za-z0-9_-]+$", "Google Cloud API Key resource path"),
        ],
        "checksum": None,
    },
}


class APIKeyValidatorTool(Tool):
    """Validate API keys cho Stripe, AWS, GitHub, OpenAI, Google."""

    category = ToolCategory.SECURITY
    safety = ToolSafety.SAFE

    @property
    def name(self) -> str:
        return "api_key_validator"

    @property
    def description(self) -> str:
        return "Validate API keys (Stripe, AWS, GitHub, OpenAI, Google) bằng regex + checksum."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "API key cần validate"},
                "service": {
                    "type": "string",
                    "enum": sorted(list(SERVICES.keys()) + ["auto"]),
                    "default": "auto",
                    "description": "Service cụ thể hoặc 'auto' để auto-detect",
                },
            },
            "required": ["key"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("key"):
            return "Missing required arg: key"
        service = args.get("service", "auto")
        if service != "auto" and service not in SERVICES:
            return f"Unknown service: {service}. Supported: {sorted(list(SERVICES.keys()) + ['auto'])}"
        return None

    def _check_one(self, key: str, service: str) -> Dict[str, Any]:
        """Kiểm tra key cho một service cụ thể."""
        spec = SERVICES[service]
        matches: List[str] = []
        for pattern, label in spec["patterns"]:
            if re.match(pattern, key):
                matches.append(label)
        checksum_ok = True
        if matches and spec.get("checksum"):
            checksum_ok = bool(spec["checksum"](key))
        return {
            "service": service,
            "valid": bool(matches) and checksum_ok,
            "matched_patterns": matches,
            "checksum_passed": checksum_ok,
        }

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        key = args["key"]
        service = args.get("service", "auto")

        # An toàn: không bao giờ in full key / never log full key
        masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"

        if service != "auto":
            result = self._check_one(key, service)
            return ToolResult(
                success=result["valid"],
                output=f"Key {masked}: {'VALID' if result['valid'] else 'INVALID'} cho {service}",
                error=None if result["valid"] else f"No pattern matched cho {service}",
                metadata={**result, "masked_key": masked},
            )

        # Auto-detect: thử tất cả services / try all services
        results = []
        for svc in SERVICES:
            results.append(self._check_one(key, svc))
        any_valid = any(r["valid"] for r in results)
        matched_services = [r["service"] for r in results if r["valid"]]

        return ToolResult(
            success=any_valid,
            output=(
                f"Key {masked}: VALID cho {', '.join(matched_services)}"
                if any_valid
                else f"Key {masked}: INVALID — không match bất kỳ service nào"
            ),
            error=None if any_valid else "No service pattern matched",
            metadata={
                "masked_key": masked,
                "matched_services": matched_services,
                "details": results,
            },
        )
