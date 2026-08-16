"""Guardrails - Bảo vệ model khỏi misuse."""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class GuardrailAction(str, Enum):
    """Action khi guardrail trigger."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REDACT = "redact"


@dataclass
class Guardrail:
    """Một guardrail rule."""
    name: str
    description: str
    check_fn: Callable[[str], bool]
    action: GuardrailAction = GuardrailAction.BLOCK
    message: str = ""
    
    def evaluate(self, text: str) -> Dict[str, Any]:
        triggered = self.check_fn(text)
        return {
            "name": self.name,
            "triggered": triggered,
            "action": self.action.value if triggered else GuardrailAction.ALLOW.value,
            "message": self.message if triggered else "",
        }


class GuardrailManager:
    """Quản lý nhiều guardrails.
    
    Usage:
        mgr = GuardrailManager()
        mgr.add(Guardrail(
            name="no_secrets",
            description="Block API keys",
            check_fn=lambda t: "sk-" in t or "ghp_" in t,
            action=GuardrailAction.BLOCK,
            message="API keys are not allowed",
        ))
        result = mgr.check(user_input)
        if not result["allowed"]:
            print(result["message"])
    """
    
    def __init__(self):
        self._guardrails: List[Guardrail] = []
    
    def add(self, guardrail: Guardrail) -> None:
        """Add a guardrail."""
        self._guardrails.append(guardrail)
    
    def remove(self, name: str) -> Optional[Guardrail]:
        """Remove a guardrail by name."""
        for i, g in enumerate(self._guardrails):
            if g.name == name:
                return self._guardrails.pop(i)
        return None
    
    def check(self, text: str) -> Dict[str, Any]:
        """Check text against all guardrails.
        
        Returns:
            Dict with:
                - allowed: bool
                - triggered: List of triggered guardrail names
                - action: overall action (most restrictive)
                - message: combined messages
        """
        triggered = []
        actions = []
        messages = []
        
        for g in self._guardrails:
            result = g.evaluate(text)
            if result["triggered"]:
                triggered.append(g.name)
                actions.append(g.action)
                if g.message:
                    messages.append(g.message)
        
        # Most restrictive action
        action_priority = {
            GuardrailAction.BLOCK: 4,
            GuardrailAction.REDACT: 3,
            GuardrailAction.WARN: 2,
            GuardrailAction.ALLOW: 1,
        }
        
        if not actions:
            overall_action = GuardrailAction.ALLOW
        else:
            overall_action = max(actions, key=lambda a: action_priority.get(a, 0))
        
        return {
            "allowed": overall_action in (GuardrailAction.ALLOW, GuardrailAction.WARN),
            "triggered": triggered,
            "action": overall_action.value,
            "message": "; ".join(messages) if messages else "",
        }
    
    def list_guardrails(self) -> List[Dict[str, str]]:
        """List all registered guardrails."""
        return [
            {
                "name": g.name,
                "description": g.description,
                "action": g.action.value,
            }
            for g in self._guardrails
        ]


def get_default_guardrails() -> GuardrailManager:
    """Get default guardrail configuration."""
    mgr = GuardrailManager()
    
    # No secrets
    mgr.add(Guardrail(
        name="no_api_keys",
        description="Block obvious API keys and tokens",
        check_fn=lambda t: any(s in t for s in ["sk-", "ghp_", "gho_", "github_pat_", "AKIA"]),
        action=GuardrailAction.BLOCK,
        message="API keys/tokens are not allowed in input",
    ))
    
    # No PII
    mgr.add(Guardrail(
        name="no_pii",
        description="Warn on PII (email, phone, SSN)",
        check_fn=lambda t: any(c in t for c in ["@", "ssn", "social security"]),
        action=GuardrailAction.WARN,
        message="PII detected - please remove personal information",
    ))
    
    # No harmful content
    mgr.add(Guardrail(
        name="no_harmful",
        description="Block harmful content (violence, illegal)",
        check_fn=lambda t: any(w in t.lower() for w in ["bomb recipe", "kill tutorial", "drug manufacture"]),
        action=GuardrailAction.BLOCK,
        message="Harmful content is not allowed",
    ))
    
    # Length limit
    mgr.add(Guardrail(
        name="length_limit",
        description="Warn on very long inputs",
        check_fn=lambda t: len(t) > 50000,
        action=GuardrailAction.WARN,
        message="Input is very long, may be truncated",
    ))
    
    return mgr
