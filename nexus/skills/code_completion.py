"""Code Completion Skill - Hoàn thành code kiểu Copilot.

Cung cấp chiến lược completion: context-aware, type-aware,
multi-line completion, Fill-In-the-Middle (FIM), và example artifact.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class CodeCompletionSkill(Skill):
    """Hoàn thành code dựa trên context (prefix + suffix + imports)."""

    category = SkillCategory.CODE
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "complete", "autocomplete", "copilot", "snippet",
        "hoàn thành", "tự động hoàn thành", "fill in",
        "infill", "continue code", "next line",
        "intellisense", "suggest code", "complete this",
    ]
    examples = [
        "Complete this function: def factorial(n):",
        "Autocomplete the boilerplate for a FastAPI route",
        "Copilot-style complete this React component",
    ]

    @property
    def name(self) -> str:
        return "code_completion"

    @property
    def description(self) -> str:
        return (
            "Hoàn thành code kiểu Copilot: line, block, function-level. "
            "Hỗ trợ FIM (Fill-In-the-Middle), context-aware, type-aware."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.18
        # Phát hiện dangling code / detect dangling code markers
        dangling_markers = ["def ", "function ", "class ", "func ", "fn ", "=>", "{"]
        if any(m in prompt for m in dangling_markers) and not prompt.rstrip().endswith((";", "}")):
            score += 0.2
        # Cursor markers
        if "<|cursor|>" in prompt or "<CURSOR>" in prompt or "[[cursor]]" in prompt:
            score += 0.4
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        lang = context.language or "python"
        return SkillResult(
            success=True,
            output=(
                f"[CodeCompletion/{lang}] FIM-style completion ready. "
                f"Producing prefix-suffix-aware suggestion."
            ),
            artifacts=[
                {"path": "completion/example_completion.txt", "content": _EXAMPLE_COMPLETION},
                {"path": "completion/strategy.md", "content": _COMPLETION_STRATEGY},
            ],
            metadata={
                "skill": self.name,
                "language": lang,
                "modes": {
                    "line": "single line, no newline insertion",
                    "block": "multi-line, balanced brackets",
                    "function": "complete function body from signature",
                    "file": "scaffold entire file from description",
                },
                "fim_format": {
                    "prompt_template": "<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>",
                    "note": "FIM tokens let the model leverage suffix context for mid-line completion",
                },
                "context_window_strategy": {
                    "imports": "always include (1k tokens)",
                    "type_defs": "include if referenced in prefix",
                    "same_file_functions": "top-K by retrieval over embeddings",
                    "recent_edits": "include if within 50 lines of cursor",
                    "git_diff": "include hunk headers for stylistic consistency",
                },
                "ranking_features": [
                    "BM25 against project symbols",
                    "embedding cosine similarity",
                    "tree-sitter scope awareness",
                    "type compatibility (mypy/pyright)",
                    "indentation match",
                ],
                "safety": {
                    "secrets_filter": "block completion containing API keys / passwords",
                    "license_check": "flag verbatim copies of GPL code (>20 token match)",
                    "syntax_check": "reject if tree-sitter parse fails",
                },
            },
            suggestions=[
                "Place cursor marker <|cursor|> exactly where completion should start",
                "Provide 3-5 lines of prefix context for best results",
                "Specify language and language version explicitly",
                "For multi-line completion, indicate desired length (e.g. ~10 lines)",
            ],
        )


_EXAMPLE_COMPLETION = '''# Example FIM-style completion (language: python)

# --- Prefix ---
# def quicksort(arr: list[int]) -> list[int]:
#     """Sort arr via quicksort, return new list."""
#     if len(arr) <= 1:
#         return arr
#     pivot = arr[len(arr) // 2]
# <|cursor|>
# --- Suffix ---
#     return arr

# --- Suggested completion ---
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Confidence: 0.92 | Type-checked: OK | Style: matches PEP-8
'''


_COMPLETION_STRATEGY = """# Code Completion Strategy

## 1. Context Assembly
- Collect: imports, type definitions, surrounding scope, recent edits.
- Rank candidate context by BM25 + embedding similarity + scope (tree-sitter).

## 2. FIM (Fill-In-the-Middle)
- Use prefix + suffix tokens to complete mid-line code.
- Critical for partial-line edits, parameter lists, and conditional branches.

## 3. Candidate Generation
- Generate K=4 candidates (temperature=0.2 for code).
- Nucleus sampling (top_p=0.95) + repetition penalty 1.1.

## 4. Ranking & Filtering
- syntax_valid (tree-sitter parse) — must pass
- type_check (pyright/mypy for Python) — boost score
- indentation_match (cursor column) — boost score
- secrets_filter — drop candidate
- license_check — flag if verbatim match > 20 tokens

## 5. Post-processing
- Trim trailing whitespace.
- Balance unbalanced brackets if mode=block.
- Re-indent to match cursor.
- Strip duplicate leading lines already present in prefix.

## 6. Telemetry (opt-in)
- Log acceptance/rejection, edit distance, latency.
- DO NOT log source code itself, only anonymized metrics.
"""
