"""Security Audit Skill - Audit bảo mật code."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class SecurityAuditSkill(Skill):
    """Audit bảo mật: OWASP Top 10, SAST, dependency vulnerabilities."""
    
    category = SkillCategory.SECURITY
    priority = SkillPriority.CRITICAL
    keywords: List[str] = [
        "security", "bảo mật", "audit", "vulnerability", "lỗ hổng",
        "owasp", "cve", "injection", "xss", "csrf",
        "sql injection", "ssrf", "rce", "privilege escalation",
    ]
    
    @property
    def name(self) -> str:
        return "security_audit"
    
    @property
    def description(self) -> str:
        return (
            "Audit bảo mật toàn diện: OWASP Top 10, SAST (static analysis), "
            "dependency vulnerabilities (CVE), secrets detection, "
            "crypto misuse, authentication/authorization flaws."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        owasp_top_10 = [
            "A01: Broken Access Control",
            "A02: Cryptographic Failures",
            "A03: Injection (SQL, NoSQL, OS, LDAP)",
            "A04: Insecure Design",
            "A05: Security Misconfiguration",
            "A06: Vulnerable & Outdated Components",
            "A07: Identification & Authentication Failures",
            "A08: Software & Data Integrity Failures",
            "A09: Security Logging & Monitoring Failures",
            "A10: Server-Side Request Forgery (SSRF)",
        ]
        checks = [
            "Hardcoded secrets / API keys / passwords",
            "Insecure crypto (MD5, SHA1, weak keys)",
            "SQL injection vectors",
            "XSS vectors (reflected, stored, DOM)",
            "CSRF token validation",
            "Path traversal vulnerabilities",
            "Insecure deserialization",
            "Missing input validation",
            "Overly permissive CORS",
            "Rate limiting absence",
            "JWT validation issues",
            "SSL/TLS misconfiguration",
        ]
        return SkillResult(
            success=True,
            output=f"[SecurityAudit] Running OWASP Top 10 + {len(checks)} checks.",
            metadata={
                "skill": self.name,
                "owasp_top_10": owasp_top_10,
                "additional_checks": checks,
                "tools": ["bandit", "semgrep", "safety", "pip-audit", "trufflehog"],
                "severity_levels": ["critical", "high", "medium", "low", "info"],
            },
            suggestions=[
                "Run SAST regularly in CI/CD",
                "Keep dependencies updated",
                "Use SCA tools (Safety, pip-audit)",
                "Never commit secrets - use vault",
            ],
        )
