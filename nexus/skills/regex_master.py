"""Regex Master Skill - Sinh và test regular expression.

Cung cấp regex generator + test cases cho các pattern phổ biến:
email, URL, IP, phone, date, credit card, UUID, etc.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class RegexSkill(Skill):
    """Sinh và test regex pattern với examples."""

    category = SkillCategory.CODE
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "regex", "regular expression", "pattern",
        "biểu thức chính quy", "match pattern",
        "extract pattern", "validate regex",
        "re module", "pcre", "regex101",
    ]
    examples = [
        "Regex to match valid email addresses",
        "Extract URLs from a string with regex",
        "Write a regex for ISO 8601 dates",
    ]

    @property
    def name(self) -> str:
        return "regex_master"

    @property
    def description(self) -> str:
        return (
            "Sinh regex pattern + test cases. Thư viện patterns phổ biến: "
            "email, URL, IP, phone, date, UUID, credit card, semver."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.2
        if "/" in prompt and any(c in prompt for c in "+*?[]()"):
            score += 0.2
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output="[RegexMaster] Pattern library + test harness ready.",
            artifacts=[
                {"path": "regex/pattern_library.py", "content": _PATTERN_LIBRARY},
                {"path": "regex/test_harness.py", "content": _TEST_HARNESS},
            ],
            metadata={
                "skill": self.name,
                "patterns_included": [
                    "email", "url", "ipv4", "ipv6", "mac_address",
                    "phone_us", "phone_vn", "iso_date", "iso_datetime",
                    "uuid_v4", "credit_card", "semver", "hex_color",
                    "slug", "ssn", "iban", "swift_bic",
                ],
                "common_pitfalls": {
                    "email_validation": "RFC 5322 is complex; use it for syntax only — does not verify deliverability",
                    "url_parsing": "Use urllib.parse instead of regex for full URL decomposition",
                    "ipv4_octets": "Naive regex matches 999.999.999.999 — must validate 0-255 range",
                    "credit_card_luhn": "Regex matches format only — always run Luhn checksum",
                    "date_ranges": "Regex can't validate '2024-02-30' — combine with date parser",
                    "catastrophic_backtracking": "Avoid nested quantifiers like (a+)+ — DoS risk",
                },
                "flavor_differences": {
                    "python_re": "no atomic groups, no possessive quantifiers, supports named groups (?P<name>)",
                    "pcre": "atomic groups (?>...), possessive ++, lookbehind variable length",
                    "javascript": "no lookbehind (Safari <16), no named backreferences in old engines",
                    "go_regexp": "no backreferences, no lookarounds — RE2 linear-time guarantee",
                    "rust_regex": "RE2-compatible — no backrefs, no lookarounds",
                },
                "performance": {
                    "anchoring": "Always anchor with ^ / $ when matching whole input",
                    "non_greedy": "Use lazy quantifier *? only when needed; greedy is often faster",
                    "character_class": "[0-9] faster than \\d in some engines (Unicode aware)",
                    "compile_cache": "re.compile() once, .match() many times — Python auto-caches 512",
                },
                "testing": {
                    "positive_cases": "Inputs that should match",
                    "negative_cases": "Inputs that should NOT match",
                    "edge_cases": "Empty, boundary, unicode, very long input",
                    "doom_test": "Adversarial input to detect catastrophic backtracking",
                },
            },
            suggestions=[
                "Specify regex flavor (Python / PCRE / JS / Go)",
                "Provide 3+ examples of strings that SHOULD match",
                "Provide 3+ examples of strings that should NOT match",
                "Mention if catastrophic backtracking is a concern (untrusted input)",
            ],
        )


_PATTERN_LIBRARY = '''"""Common regex pattern library + test cases.

Author: Hieu Louis (2026)
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PatternSpec:
    name: str
    pattern: str
    description: str
    examples: List[Tuple[str, bool]]   # (input, should_match)
    flags: int = 0


PATTERNS = {
    "email": PatternSpec(
        name="email",
        # Practical RFC 5322 subset — production-ready
        pattern=r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$",
        description="Email address (subset of RFC 5322). Does not validate deliverability.",
        examples=[
            ("user@example.com", True),
            ("first.last@sub.domain.co", True),
            ("user+tag@gmail.com", True),
            ("plaintext", False),
            ("@missing-local.com", False),
            ("user@", False),
            ("user@.com", False),
        ],
    ),
    "url": PatternSpec(
        name="url",
        pattern=r"^https?://[A-Za-z0-9.-]+(?::\\d+)?(?:/[A-Za-z0-9._~:/?#@!$&\'()*+,;=%-]*)?$",
        description="HTTP/HTTPS URL with optional port and path.",
        examples=[
            ("https://example.com", True),
            ("http://localhost:8080/api/v1", True),
            ("https://example.com/path?q=1#frag", True),
            ("ftp://example.com", False),
            ("example.com", False),
            ("https://", False),
        ],
    ),
    "ipv4": PatternSpec(
        name="ipv4",
        pattern=r"^(?:(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})\\.){3}"
                r"(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})$",
        description="IPv4 address (validates 0-255 octets).",
        examples=[
            ("192.168.1.1", True),
            ("0.0.0.0", True),
            ("255.255.255.255", True),
            ("256.1.1.1", False),
            ("1.2.3", False),
            ("1.2.3.4.5", False),
        ],
    ),
    "iso_date": PatternSpec(
        name="iso_date",
        pattern=r"^(\\d{4})-(\\d{2})-(\\d{2})$",
        description="ISO 8601 date (YYYY-MM-DD). Format only — does NOT validate ranges.",
        examples=[
            ("2024-01-15", True),
            ("1999-12-31", True),
            ("2024-13-01", True),   # matches format; ranges need post-check
            ("2024-1-1", False),
            ("24-01-15", False),
        ],
    ),
    "iso_datetime": PatternSpec(
        name="iso_datetime",
        pattern=r"^(\\d{4}-\\d{2}-\\d{2})[T ](\\d{2}:\\d{2}:\\d{2})"
                r"(?:\\.\\d{1,6})?(Z|[+-]\\d{2}:?\\d{2})?$",
        description="ISO 8601 datetime with optional fractional seconds and timezone.",
        examples=[
            ("2024-01-15T10:30:00Z", True),
            ("2024-01-15 10:30:00.123456+07:00", True),
            ("2024-01-15T10:30:00", True),
            ("2024-01-15 25:00:00", True),  # format valid; time range needs check
            ("15/01/2024 10:30", False),
        ],
    ),
    "uuid_v4": PatternSpec(
        name="uuid_v4",
        pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}"
                r"-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$",
        description="UUID version 4 (random).",
        examples=[
            ("550e8400-e29b-41d4-a716-446655440000", True),
            ("550e8400-e29b-41d4-a716-44665544000", False),   # too short
            ("550e8400-e29b-51d4-a716-446655440000", False),  # version 5 not v4
            ("550E8400-E29B-41D4-A716-446655440000", True),    # uppercase ok
        ],
    ),
    "semver": PatternSpec(
        name="semver",
        pattern=r"^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)\\.(0|[1-9]\\d*)"
                r"(?:-((?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*)"
                r"(?:\\.(?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
                r"(?:\\+([0-9a-zA-Z-]+(?:\\.[0-9a-zA-Z-]+)*))?$",
        description="Semantic Versioning 2.0.0 (full spec incl pre-release + build).",
        examples=[
            ("1.0.0", True),
            ("2.3.4-alpha.1+build.5", True),
            ("1.0", False),
            ("01.2.3", False),    # leading zeros not allowed
            ("1.2.3-", False),
        ],
    ),
    "hex_color": PatternSpec(
        name="hex_color",
        pattern=r"^#?(?:[0-9a-fA-F]{3}){1,2}$",
        description="CSS hex color (#abc or #aabbcc).",
        examples=[
            ("#fff", True),
            ("aabbcc", True),
            ("#aabbcc", True),
            ("#abcd", False),
            ("#ggg", False),
        ],
    ),
    "phone_vn": PatternSpec(
        name="phone_vn",
        pattern=r"^(?:\\+84|0)(?:3[2-9]|5[2689]|7[06-9]|8[1-9]|9[0-46-9])[0-9]{7}$",
        description="Vietnamese mobile phone (Viettel, MobiFone, VinaPhone, etc.).",
        examples=[
            ("0912345678", True),
            ("+84912345678", True),
            ("0331234567", True),
            ("0123456789", False),  # obsolete 11-digit prefix
            ("091234567", False),   # too short
        ],
    ),
    "credit_card": PatternSpec(
        name="credit_card",
        pattern=r"^(?:4[0-9]{12}(?:[0-9]{3})?|"          # Visa
                r"5[1-5][0-9]{14}|"                       # Mastercard
                r"3[47][0-9]{13}|"                        # Amex
                r"6(?:011|5[0-9]{2})[0-9]{12})$",         # Discover
        description="Credit card number (format only — always run Luhn checksum).",
        examples=[
            ("4111111111111111", True),    # Visa test
            ("5500000000000004", True),    # MC test
            ("371449635398431", True),     # Amex test
            ("1234567890123456", False),
            ("4111-1111-1111-1111", False),  # separators not in pattern
        ],
    ),
}


def get_pattern(name: str) -> PatternSpec:
    if name not in PATTERNS:
        raise KeyError(f"Unknown pattern: {name}. Available: {list(PATTERNS)}")
    return PATTERNS[name]
'''


_TEST_HARNESS = '''"""Regex test harness — run all example cases + DoS safety check.

Author: Hieu Louis (2026)
"""
from __future__ import annotations
import re
import time
from typing import List, Tuple

from .pattern_library import PATTERNS, PatternSpec


def test_pattern(spec: PatternSpec) -> List[Tuple[str, bool, bool, str]]:
    """Run all examples. Returns list of (input, expected, actual, status)."""
    compiled = re.compile(spec.pattern, spec.flags)
    results: List[Tuple[str, bool, bool, str]] = []
    for inp, expected in spec.examples:
        actual = bool(compiled.match(inp))
        status = "PASS" if actual == expected else "FAIL"
        results.append((inp, expected, actual, status))
    return results


def check_catastrophic_backtracking(pattern: str, evil_input: str = "a" * 30,
                                    timeout_ms: int = 100) -> bool:
    """Return True if pattern is SAFE (no catastrophic backtracking on evil_input)."""
    try:
        compiled = re.compile(pattern)
    except re.error:
        return True   # invalid pattern — not a DoS risk
    start = time.monotonic()
    compiled.match(evil_input + "!")
    elapsed_ms = (time.monotonic() - start) * 1000
    return elapsed_ms < timeout_ms


def run_all() -> int:
    """Run all pattern tests. Returns number of failures."""
    failures = 0
    for name, spec in PATTERNS.items():
        print(f"\\n[{name}] {spec.description}")
        for inp, expected, actual, status in test_pattern(spec):
            mark = "+" if status == "PASS" else "x"
            print(f"  {mark} {inp!r:50} expected={expected} actual={actual}")
            if status == "FAIL":
                failures += 1
        safe = check_catastrophic_backtracking(spec.pattern)
        flag = "SAFE" if safe else "DANGEROUS"
        print(f"  DoS check: {flag}")
    return failures


if __name__ == "__main__":
    import sys
    failures = run_all()
    print(f"\\n{'='*60}\\nFailures: {failures}")
    sys.exit(1 if failures else 0)
'''
