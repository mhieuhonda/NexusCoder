"""Code Explanation Skill - Giải thích code từng bước.

Framework explain: mục đích, interface, luồng điều khiển, dữ liệu,
edge cases, độ phức tạp, và potential pitfalls.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class CodeExplanationSkill(Skill):
    """Giải thích code tự nhiên từng bước cho developer."""

    category = SkillCategory.CODE
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "explain", "giải thích", "what does this code", "walk through",
        "walk me through", "describe code", "how does this work",
        "hiểu code", "phân tích code", "break down",
        "what is this function doing", "comment code",
    ]
    examples = [
        "Explain this Python decorator step by step",
        "What does this recursive function do?",
        "Walk me through this SQL query",
    ]

    @property
    def name(self) -> str:
        return "code_explanation"

    @property
    def description(self) -> str:
        return (
            "Giải thích code tự nhiên: mục đích, luồng điều khiển, "
            "biến đổi dữ liệu, edge cases, độ phức tạp, và pitfalls."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.15
        if "```" in prompt or "def " in prompt or "function " in prompt:
            score += 0.2
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output="[CodeExplanation] Step-by-step explanation framework ready.",
            artifacts=[
                {"path": "explanation/framework.md", "content": _EXPLANATION_FRAMEWORK},
                {"path": "explanation/example.md", "content": _EXAMPLE_EXPLANATION},
            ],
            metadata={
                "skill": self.name,
                "explanation_levels": [
                    "ELI5 (giải thích như mới học code)",
                    "junior dev (giải thích từng dòng)",
                    "senior dev (focus kiến trúc + trade-offs)",
                    "expert (focus correctness + perf characteristics)",
                ],
                "framework_steps": [
                    "1. One-sentence summary (mục đích)",
                    "2. Inputs / outputs / side effects",
                    "3. Step-by-step walkthrough (line hoặc block)",
                    "4. Data flow diagram (text-based)",
                    "5. Edge cases & error handling",
                    "6. Time/space complexity",
                    "7. Pitfalls / code smells / suggestions",
                ],
                "diagram_styles": ["ascii", "mermaid sequence", "mermaid flowchart"],
                "audience_tuning": {
                    "eli5": "Use analogies, no jargon, 1 concept per paragraph",
                    "junior": "Explain syntax, link to docs, define jargon",
                    "senior": "Skip basics, focus on architecture & trade-offs",
                    "expert": "Focus on correctness, perf, alternatives",
                },
            },
            suggestions=[
                "Specify audience level (ELI5 / junior / senior / expert)",
                "Provide code in fenced block for accurate line references",
                "Ask for specific aspect (complexity, correctness, security)",
            ],
        )


_EXPLANATION_FRAMEWORK = """# Code Explanation Framework

## Level 0: One-Sentence Summary
> "This code does X by Y."

## Level 1: Interface Contract
- **Inputs**: parameters, types, constraints
- **Outputs**: return type, side effects, exceptions
- **Preconditions**: what must be true before calling
- **Postconditions**: what is guaranteed after return

## Level 2: Step-by-Step Walkthrough
For each block:
1. **What** is being done (one sentence)
2. **Why** it's done this way (motivation)
3. **How** it interacts with prior/next blocks

## Level 3: Data Flow
```
input -> [transform 1] -> [filter] -> [aggregate] -> output
```

## Level 4: Edge Cases & Error Handling
- Null / undefined / empty inputs
- Boundary conditions (0, 1, max_int, negative)
- Concurrency / reentrancy
- Resource exhaustion (memory, file handles)

## Level 5: Complexity
- Time: O(?) - best / average / worst
- Space: O(?) - auxiliary vs total
- Practical: cache misses, branch prediction

## Level 6: Pitfalls & Suggestions
- Code smells (long method, deep nesting, magic numbers)
- Common bugs (off-by-one, race conditions)
- Refactor opportunities (extract method, replace conditional with polymorphism)
"""


_EXAMPLE_EXPLANATION = '''# Example Explanation: Binary Search

## Code
```python
def binary_search(arr: list[int], target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

## Summary
Binary search finds `target` in `arr` (already sorted ascending), returning its index or -1.

## Interface
- **Inputs**: sorted list `arr`, int `target`
- **Output**: index of `target` in `arr`, or -1 if not found
- **Precondition**: `arr` sorted ascending
- **Postcondition**: returned index i satisfies `arr[i] == target`, or i == -1

## Walkthrough
1. `lo=0, hi=len(arr)-1`: initialize search bounds.
2. `while lo <= hi`: loop until search space empty.
3. `mid = (lo + hi) // 2`: pick middle index.
   - Note: risk of overflow in C — Python ints are arbitrary precision so safe.
4. `arr[mid] == target`: hit, return `mid`.
5. `arr[mid] < target`: target in right half, move `lo` past `mid`.
6. `arr[mid] > target`: target in left half, move `hi` before `mid`.
7. `return -1`: search space exhausted, not found.

## Complexity
- Time: O(log n) - halve search space each iteration.
- Space: O(1) - only three variables.

## Pitfalls
- Integer overflow in `mid = (lo + hi) // 2` in C/Java. Use `lo + (hi - lo) // 2`.
- Input MUST be sorted; precondition not enforced.
- Returns first-found index, not necessarily the leftmost duplicate.

## Suggestions
- Add `is_sorted` assertion for debug builds.
- Use `bisect_left` from stdlib for leftmost match.
'''
