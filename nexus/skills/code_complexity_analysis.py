"""Code Complexity Analysis Skill - Phân tích độ phức tạp.

Tính Cyclomatic (McCabe) và Cognitive Complexity (SonarSource),
với example calculation cho từng loại.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class CodeComplexitySkill(Skill):
    """Tính cyclomatic + cognitive complexity, suggest refactors."""

    category = SkillCategory.CODE
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "cyclomatic complexity", "cognitive complexity", "complexity",
        "mccabe", "code complexity", "độ phức tạp",
        "function complexity", "branch complexity",
        "too complex", "complex function",
    ]
    examples = [
        "Calculate cyclomatic complexity of this function",
        "Why is this function rated 'complex' by SonarQube?",
        "Reduce cognitive complexity of this method",
    ]

    @property
    def name(self) -> str:
        return "code_complexity"

    @property
    def description(self) -> str:
        return (
            "Tính cyclomatic (McCabe) + cognitive (SonarSource) complexity. "
            "Suggest refactors: extract method, guard clauses, polymorphism."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.18
        if "def " in prompt or "function " in prompt:
            score += 0.1
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output="[CodeComplexity] McCabe + cognitive complexity calculator ready.",
            artifacts=[
                {"path": "complexity/calculator.py", "content": _COMPLEXITY_CALCULATOR},
                {"path": "complexity/example.md", "content": _EXAMPLE_CALCULATION},
            ],
            metadata={
                "skill": self.name,
                "metrics": {
                    "cyclomatic": {
                        "definition": "M = E - N + 2P (Edges - Nodes + 2*Connected Components)",
                        "shortcut": "M = decision_points + 1",
                        "decision_points": ["if", "elif", "for", "while", "except", "and", "or",
                                            "ternary", "case/default"],
                        "thresholds": {
                            "low": "<= 5",
                            "moderate": "6 - 10",
                            "high": "11 - 20",
                            "very_high": "21 - 50",
                            "untestable": "> 50",
                        },
                    },
                    "cognitive": {
                        "definition": "SonarSource metric — penalizes nesting + recursion + breaks",
                        "increments": [
                            "+1 per if/else/for/while/except/case",
                            "+1 per nesting level (compound cost)",
                            "+1 per boolean op (and/or/not)",
                            "+1 per jump (break/continue/return inside loop)",
                            "+1 per recursion (caller == callee)",
                            "+1 per goto-like pattern",
                        ],
                        "thresholds": {
                            "low": "<= 5",
                            "moderate": "6 - 10",
                            "high": "11 - 20",
                            "very_high": "21 - 30",
                            "untestable": "> 30",
                        },
                    },
                    "halstead": "Difficulty / Effort / Volume (rarely used in practice)",
                    "npath": "Number of independent paths — exponential in branches",
                },
                "refactor_patterns": [
                    "Extract Method (split large function)",
                    "Replace Conditional with Polymorphism (if-elif ladder -> strategy)",
                    "Decompose Conditional (long boolean expr -> named predicate)",
                    "Guard Clauses (early return replaces nested if-else)",
                    "Replace Nested Conditionals with State/Strategy",
                    "Compose Method (sequence of intention-revealing calls)",
                ],
                "tooling": {
                    "python": "radon cc (cyclomatic), radon mi (maintainability), xenon (CI)",
                    "javascript": "escomplex, typhonjs-escomplex",
                    "java": "PMD, SonarQube",
                    "go": "gocyclo (cyclomatic only)",
                    "rust": "rust-code-analysis (both metrics)",
                },
                "ci_thresholds": {
                    "block_pr": "cyclomatic > 15 OR cognitive > 20",
                    "warn": "cyclomatic > 10 OR cognitive > 15",
                    "trend": "Track average per file; fail regression > 10%",
                },
            },
            suggestions=[
                "Specify which metric (cyclomatic / cognitive / both)",
                "Provide code in fenced block for accurate analysis",
                "Ask for refactor suggestions if complexity > threshold",
            ],
        )


_COMPLEXITY_CALCULATOR = '''"""Cyclomatic + Cognitive Complexity calculator.

Author: Hieu Louis (2026)
"""
from __future__ import annotations
import ast
from dataclasses import dataclass


@dataclass
class ComplexityResult:
    cyclomatic: int
    cognitive: int
    decision_points: int
    nesting_max: int
    rating: str            # "low" | "moderate" | "high" | "very_high" | "untestable"


def analyze(func: ast.FunctionDef) -> ComplexityResult:
    visitor = _ComplexityVisitor(func.name)
    visitor.visit(func)
    cyclo = visitor.decision_points + 1
    cognitive = visitor.cognitive
    nesting_max = visitor.max_nesting
    rating = _rate(cyclo, cognitive)
    return ComplexityResult(
        cyclomatic=cyclo,
        cognitive=cognitive,
        decision_points=visitor.decision_points,
        nesting_max=nesting_max,
        rating=rating,
    )


# Cyclomatic: count decision points
# Cognitive: SonarSource algorithm (penalize nesting + recursion + jumps)


class _ComplexityVisitor(ast.NodeVisitor):
    DECISION_NODES = (
        ast.If, ast.For, ast.AsyncFor, ast.While,
        ast.ExceptHandler, ast.BoolOp, ast.IfExp,
    )

    def __init__(self, func_name: str) -> None:
        self.func_name = func_name
        self.decision_points = 0
        self.cognitive = 0
        self.nesting = 0
        self.max_nesting = 0
        self.in_loop = False

    def _visit_decision(self, node):
        self.decision_points += 1
        self.cognitive += self.nesting + 1
        self.nesting += 1
        self.max_nesting = max(self.max_nesting, self.nesting)
        self.generic_visit(node)
        self.nesting -= 1

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Each additional operand in `and`/`or` is +1
        self.decision_points += max(0, len(node.values) - 1)
        self.cognitive += max(0, len(node.values) - 1)
        self.generic_visit(node)

    visit_If = _visit_decision
    visit_For = _visit_decision
    visit_AsyncFor = _visit_decision
    visit_While = _visit_decision
    visit_ExceptHandler = _visit_decision

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.decision_points += 1
        self.cognitive += 1
        self.generic_visit(node)

    def visit_Break(self, node: ast.Break) -> None:
        if self.in_loop:
            self.cognitive += 1
        self.generic_visit(node)

    def visit_Continue(self, node: ast.Continue) -> None:
        if self.in_loop:
            self.cognitive += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.func_name:
            self.cognitive += 1   # recursion penalty
        else:
            self._visit_decision(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def _rate(cyclo: int, cognitive: int) -> str:
    if cyclo <= 5 and cognitive <= 5:
        return "low"
    if cyclo <= 10 and cognitive <= 10:
        return "moderate"
    if cyclo <= 20 and cognitive <= 20:
        return "high"
    if cyclo <= 50 and cognitive <= 30:
        return "very_high"
    return "untestable"
'''


_EXAMPLE_CALCULATION = '''# Example: Cyclomatic + Cognitive Complexity Calculation

## Sample Code
```python
def process(items, flag):
    result = []
    for item in items:                       # cyclomatic +1, cognitive +1
        if item.is_valid and flag:           # cyclomatic +1 (if) +1 (and), cognitive +2 (nested) +1 (and)
            if item.priority > 5:             # cyclomatic +1, cognitive +3 (doubly nested)
                result.append(item)
            else:
                continue                     # cognitive +1 (jump in loop)
        elif item.is_optional:               # cyclomatic +1 (elif), cognitive +2
            result.append(item)
    return result
```

## Cyclomatic Complexity (McCabe)
Decision points counted:
- `for` ... 1
- `if` ... 1
- `and` ... 1
- `if` (nested) ... 1
- `elif` ... 1
Total decision_points = 5

`M = decision_points + 1 = 6`

Rating: **moderate**

## Cognitive Complexity (SonarSource)
- `for` at nesting 0: +1 (nesting 0 + base 1)
- `if` at nesting 1: +2 (nesting 1 + base 1)
- `and` operand: +1
- nested `if` at nesting 2: +3 (nesting 2 + base 1)
- `continue` (jump in loop): +1
- `elif` at nesting 1: +2 (nesting 1 + base 1)
Total cognitive = 1 + 2 + 1 + 3 + 1 + 2 = **10**

Rating: **moderate** (close to high boundary 11)

## Refactor Suggestions
1. **Extract Method**: pull nested `if item.priority > 5` into `_should_include(item)`.
2. **Guard Clause**: replace `elif` with early `continue` to flatten structure.
3. **Replace Conditional with Strategy** if `flag`/`priority` combos grow.

## Refactored (target: cyclo <= 4, cognitive <= 5)
```python
def process(items, flag):
    return [it for it in items if _should_keep(it, flag)]

def _should_keep(item, flag):
    if not (item.is_valid and flag):
        return item.is_optional
    return item.priority > 5
```
- `process`: cyclo=1, cognitive=1
- `_should_keep`: cyclo=2, cognitive=3
'''
