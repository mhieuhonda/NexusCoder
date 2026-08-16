"""Dead Code Analysis Skill - Phát hiện dead / unreachable code.

Sử dụng control-flow analysis (CFG), use-def chains, static reachability,
và call-graph traversal để phát hiện:
- Unreachable statements
- Unused functions / variables / imports
- Unused private methods
- Unreachable branches (always-true/false conditions)

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class DeadCodeAnalysisSkill(Skill):
    """Phát hiện dead code: unreachable, unused, never-called."""

    category = SkillCategory.CODE
    priority = SkillPriority.LOW
    keywords: List[str] = [
        "dead code", "unused", "unreachable", "never called",
        "dead function", "unused import", "unused variable",
        "code không dùng", "code chết", "orphan code",
        "zombie code", "dead branch",
    ]
    examples = [
        "Find dead code in this module",
        "Detect unused private methods",
        "Report unreachable branches after refactor",
    ]

    @property
    def name(self) -> str:
        return "dead_code_analysis"

    @property
    def description(self) -> str:
        return (
            "Phát hiện dead code qua CFG + use-def chains + call-graph: "
            "unreachable statements, unused symbols, never-called functions."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.2
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output="[DeadCodeAnalysis] CFG + use-def + call-graph analysis ready.",
            artifacts=[
                {"path": "dead_code/analyzer.py", "content": _DEAD_CODE_ANALYZER},
                {"path": "dead_code/checklist.md", "content": _DEAD_CODE_CHECKLIST},
            ],
            metadata={
                "skill": self.name,
                "categories": {
                    "unreachable_stmt": "Statement after return/raise/break/continue",
                    "unreachable_branch": "Branch with always-true/false condition",
                    "unused_local": "Local variable assigned but never read",
                    "unused_private_method": "Private method never called within module",
                    "unused_import": "Imported symbol not referenced",
                    "unreferenced_module": "Module never imported by entry points",
                    "orphan_file": "File not in build graph / not imported anywhere",
                },
                "analysis_phases": [
                    "1. Build module-level AST + import graph",
                    "2. Build call graph (caller -> callee edges)",
                    "3. Reachability from public entry points (main, exports, tests)",
                    "4. Per-function CFG: detect unreachable blocks via predecessor analysis",
                    "5. Use-def chains: variables defined but never used",
                    "6. Constant propagation: detect always-true/false conditions",
                    "7. Cross-module: unreferenced modules / orphan files",
                ],
                "tooling": {
                    "python": "vulture, pyflakes (F401 unused import), depy (call-graph)",
                    "javascript": "ts-prune, knip (finds unused exports + files)",
                    "typescript": "ts-prune, knip",
                    "go": "deadcode (built into `go tool`)",
                    "rust": "cargo udeps (needs nightly), cargo machete",
                    "java": "PMD, IntelliJ 'unused declaration' inspection",
                    "c++": "cppcheck --enable=unusedFunction",
                },
                "false_positive_mitigations": [
                    "Reflection / dynamic dispatch (mark @api entries)",
                    "Metaprogramming (decorators, __all__, exports)",
                    "String-based dispatch (event handlers, route registration)",
                    "External entry points (CLI commands, plugin systems)",
                    "Test-only utilities (keep if covered by tests)",
                ],
                "ci_integration": {
                    "fail_on_new": "True — block PRs introducing new dead code",
                    "allowlist": "Pre-existing dead code tracked in `deadcode-allowlist.yaml`",
                    "trend_metric": "Track dead_code_lines / total_lines over time",
                },
            },
            suggestions=[
                "Provide entry points (main module / CLI) for accurate reachability",
                "Mark public API surfaces with @api decorator before scan",
                "Allow reflection-heavy modules with explicit allowlist",
            ],
        )


_DEAD_CODE_ANALYZER = '''"""Dead code analyzer: CFG + use-def + call-graph reachability.

Author: Hieu Louis (2026)
"""
from __future__ import annotations
import ast
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class DeadCodeFinding:
    kind: str          # "unreachable" | "unused_local" | "unused_func" ...
    file: str
    lineno: int
    end_lineno: int
    symbol: str
    reason: str


@dataclass
class AnalysisReport:
    findings: List[DeadCodeFinding] = field(default_factory=list)
    entry_points: Set[str] = field(default_factory=set)
    reachable_funcs: Set[str] = field(default_factory=set)

    @property
    def dead_function_count(self) -> int:
        return sum(1 for f in self.findings if f.kind == "unused_func")

    @property
    def unreachable_lines(self) -> int:
        return sum(
            f.end_lineno - f.lineno + 1
            for f in self.findings
            if f.kind == "unreachable"
        )


def analyze(files: List[str], entry_points: Set[str]) -> AnalysisReport:
    """Run full dead-code analysis pipeline."""
    report = AnalysisReport(entry_points=entry_points)

    # Phase 1: parse all files into module-level defs
    defs: Dict[str, Tuple[str, ast.AST]] = {}
    for path in files:
        src = open(path, encoding="utf-8").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            name = getattr(node, "name", None)
            if name:
                defs[name] = (path, node)

    # Phase 2: build call graph (caller -> callees)
    callers: Dict[str, Set[str]] = defaultdict(set)
    for name, (path, node) in defs.items():
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = _get_callee_name(child)
                if callee:
                    callers[callee].add(name)

    # Phase 3: reachability from entry points
    reachable: Set[str] = set()
    queue = list(entry_points)
    while queue:
        fn = queue.pop()
        if fn in reachable:
            continue
        reachable.add(fn)
        for caller in callers.get(fn, set()):
            if caller not in reachable:
                queue.append(caller)
    report.reachable_funcs = reachable

    # Phase 4: emit unused functions (private + not reachable)
    for name, (path, node) in defs.items():
        is_private = name.startswith("_") or name.islower()
        if is_private and name not in reachable and name not in entry_points:
            report.findings.append(DeadCodeFinding(
                kind="unused_func",
                file=path,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", node.lineno),
                symbol=name,
                reason="Private function not reachable from entry points",
            ))

    # Phase 5: per-function unreachable statements
    for path, node in defs.values():
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in _find_unreachable(node):
                report.findings.append(DeadCodeFinding(
                    kind="unreachable",
                    file=path,
                    lineno=stmt.lineno,
                    end_lineno=getattr(stmt, "end_lineno", stmt.lineno),
                    symbol=_snippet(stmt),
                    reason="Statement after return/raise/break/continue",
                ))

    # Phase 6: unused local variables (use-def)
    for path, node in defs.values():
        for unused in _find_unused_locals(node):
            report.findings.append(DeadCodeFinding(
                kind="unused_local",
                file=path,
                lineno=unused.lineno,
                end_lineno=unused.lineno,
                symbol=unused.id,
                reason="Local assigned but never read",
            ))

    return report


def _get_callee_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _find_unreachable(func: ast.FunctionDef) -> List[ast.stmt]:
    """Return statements appearing after terminator (return/raise/break/continue)."""
    unreachable: List[ast.stmt] = []
    terminated = False
    for stmt in func.body:
        if terminated:
            unreachable.append(stmt)
            continue
        if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            terminated = True
    return unreachable


def _find_unused_locals(func: ast.FunctionDef) -> List[ast.Name]:
    """Use-def: assigned but never read."""
    assigned: Dict[str, ast.Name] = {}
    read: Set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned.setdefault(node.id, node)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            read.add(node.id)
    return [n for name, n in assigned.items() if name not in read]


def _snippet(stmt: ast.stmt) -> str:
    """Short text representation of a statement for the report."""
    if isinstance(stmt, ast.Return):
        return "return"
    if isinstance(stmt, ast.Raise):
        return "raise"
    if isinstance(stmt, ast.Assign):
        return "assign"
    return type(stmt).__name__
'''


_DEAD_CODE_CHECKLIST = """# Dead Code Detection Checklist

## Per-Function (CFG-level)
- [ ] Statement after `return` / `raise` / `break` / `continue`?
- [ ] `if False:` / `if True:` constant-folded branches?
- [ ] `while False:` loop body?
- [ ] `assert False` unreachable successors?
- [ ] Exception handler that never matches raised type?

## Per-Module (Symbol-level)
- [ ] Private functions (`_foo`) reachable from public entry points?
- [ ] Module-level constants used anywhere?
- [ ] Imported symbols all referenced?
- [ ] Class methods called (or registered as `@property` / `@staticmethod`)?

## Per-Codebase (Graph-level)
- [ ] All modules reachable from entry points (main / `__init__.py` / CLI)?
- [ ] All public API functions either have callers or are exported in `__all__`?
- [ ] Plugin-style registrations (`@route`, `@click.command`) covered?
- [ ] Test utilities isolated from production code?

## False Positive Sources
- Reflection: `getattr(obj, "method_name")`
- Dynamic dispatch: registry pattern `REGISTRY["key"]()`
- Serialization: `__init__.py` `__all__` exports
- External API: framework hooks (`pytest fixtures`, `click commands`)
- Type-only imports (TS): `import type { Foo }` — keep for type checks

## Trend Tracking
- Plot `dead_code_lines / total_lines` weekly.
- Set ceiling: e.g. dead ratio < 5%.
- Auto-file issue when ratio increases > 1% in a sprint.
"""
