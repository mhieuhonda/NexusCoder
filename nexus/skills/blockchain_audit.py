"""Blockchain Audit Skill - Smart contract audit checklist + vulnerability patterns.

Hỗ trợ Solidity / EVM chains. Sinh audit checklist, common
vulnerability patterns (reentrancy, overflow, ...), và security
best practices (OpenZeppelin, slither, mythril).

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class BlockchainAuditSkill(Skill):
    """Audit smart contracts: checklist + common vulnerabilities + tooling."""

    category = SkillCategory.BLOCKCHAIN
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "solidity", "smart contract", "audit", "erc20", "erc721",
        "erc1155", "web3", "ethereum", "evm", "foundry", "hardhat",
        "reentrancy", "overflow", "underflow", "governance",
        "defi", "flash loan", "proxy", "upgradeable",
    ]
    examples = [
        "Audit this ERC20 contract for reentrancy",
        "Check governance contract for known vulnerabilities",
        "Run slither + mythril on my Solidity code",
    ]

    @property
    def name(self) -> str:
        return "blockchain_audit"

    @property
    def description(self) -> str:
        return (
            "Audit smart contracts (Solidity / EVM): checklist toàn diện, "
            "common vulnerability patterns (reentrancy, integer overflow, "
            "access control, oracle manipulation, flash loan attacks), "
            "và static analysis tooling (Slither, Mythril, Echidna, Foundry fuzz)."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.15
        if any(k in prompt_lower for k in (".sol", "pragma solidity", "contract ", "function ")) and "solidity" in prompt_lower:
            score += 0.3
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output=f"[BlockchainAudit] {len(_VULNS)} vulnerability patterns + checklist ready.",
            artifacts=[{"path": "audit/checklist.md", "content": self._render_checklist()}],
            metadata={
                "skill": self.name,
                "audit_phases": [
                    "1. Manual review (logic, access control)",
                    "2. Static analysis (Slither, Mythril)",
                    "3. Fuzz + invariant testing (Echidna, Foundry)",
                    "4. Formal verification (Certora, Halmos) — optional",
                    "5. Gas optimization review",
                    "6. Cross-check against SWC registry",
                ],
                "vulnerabilities": _VULNS,
                "tools": {
                    "static": ["slither", "mythril", "solhint", "securify"],
                    "fuzz": ["echidna", "foundry fuzz", "medusa"],
                    "formal": ["certora", "halmos", "keccak"],
                    "monitoring": ["forta", "openzeppelin defender"],
                },
                "standards": ["SWC Registry", "OpenZeppelin Contracts", "EIP-20/721/1155/4626"],
                "severity": ["critical", "high", "medium", "low", "info"],
            },
            suggestions=[
                "Use OpenZeppelin's SafeERC20 + ReentrancyGuard — never roll your own",
                "Run slither in CI on every PR: slither . --exclude-dependencies",
                "Add invariant tests with Foundry (testFuzz_* and invariant_*)",
                "Get a third-party audit before mainnet — never self-audit for production",
                "Time-lock + multisig on governance (>= 48h timelock)",
            ],
        )

    def _render_checklist(self) -> str:
        lines = ["# Smart Contract Audit Checklist", ""]
        for v in _VULNS:
            lines.append(f"## {v['id']}: {v['name']}  (severity: {v['severity']})")
            lines.append(f"**Description:** {v['description']}")
            lines.append(f"**Mitigation:** {v['mitigation']}")
            lines.append("")
        return "\n".join(lines)


_VULNS: List[Dict[str, str]] = [
    {
        "id": "SWC-107",
        "name": "Reentrancy",
        "severity": "critical",
        "description": (
            "External call to untrusted contract re-enters the function "
            "before state is updated, draining funds."
        ),
        "mitigation": (
            "Use Checks-Effects-Interactions pattern + ReentrancyGuard. "
            "Pull payments over push. Use OpenZeppelin's nonReentrant."
        ),
    },
    {
        "id": "SWC-101",
        "name": "Integer Overflow / Underflow",
        "severity": "high",
        "description": "Arithmetic wraps around (pre-0.8 Solidity).",
        "mitigation": "Use Solidity >= 0.8 (built-in overflow checks) or SafeMath.",
    },
    {
        "id": "SWC-105",
        "name": "Unauthorized Access / Missing Access Control",
        "severity": "critical",
        "description": "Functions callable by anyone (mint, withdraw, pause).",
        "mitigation": "Use onlyRole / ownable / access control. Prefer RBAC.",
    },
    {
        "id": "SWC-116",
        "name": "Block Timestamp Manipulation",
        "severity": "medium",
        "description": "Miners can tweak block.timestamp within ~15s.",
        "mitigation": "Don't use block.timestamp for strict randomness or critical logic.",
    },
    {
        "id": "SWC-114",
        "name": "Transaction Order Dependence (Front-running)",
        "severity": "high",
        "description": "Adversary sees mempool tx and front-runs.",
        "mitigation": "Commit-reveal scheme, slippage tolerance, MEV-protected routers.",
    },
    {
        "id": "SWC-113",
        "name": "DoS via Block Gas Limit / Unbounded Loop",
        "severity": "high",
        "description": "Loop over dynamic array grows past block gas limit -> permanent DoS.",
        "mitigation": "Cap iterations; split into batches; avoid storing rewards in growing arrays.",
    },
    {
        "id": "ORACLE",
        "name": "Oracle Manipulation / Flash Loan Attack",
        "severity": "critical",
        "description": "Single-DEX price oracle spoofable via flash loans.",
        "mitigation": "Use TWAP (Uniswap V3), Chainlink aggregators, or median of multiple sources.",
    },
    {
        "id": "PROXY",
        "name": "Upgradeable Proxy Storage Collision",
        "severity": "high",
        "description": "Logic contract state vars collide with proxy admin slot.",
        "mitigation": "Use EIP-1967 transparent / UUPS proxies with storage gaps; OpenZeppelin upgrades plugin.",
    },
    {
        "id": "GOV",
        "name": "Governance / Flash Loan Voting",
        "severity": "high",
        "description": "Attacker borrows tokens, votes, repays in one tx.",
        "mitigation": "Snapshot voting + lock-up periods (e.g., veToken).",
    },
    {
        "id": "GAS",
        "name": "Gas Griefing / Unbounded Refund",
        "severity": "medium",
        "description": "Recipient contract's fallback blocks ether transfer.",
        "mitigation": "Use .call with value + checks-effects-interactions; CEI pattern.",
    },
]
