"""Code Dependency Analysis Skill - Phân tích dependency graph.

Extract import graph, build dependency tree, detect circular dependencies,
compute fan-in / fan-out, và suggest module boundaries.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class CodeDependencySkill(Skill):
    """Trích dependency graph, phát hiện cycle, tính fan-in/out."""

    category = SkillCategory.CODE
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "dependency", "dependencies", "import", "dependency tree",
        "depgraph", "import graph", "circular import",
        "module dependency", "fan in", "fan out",
        "sự phụ thuộc", "đồ thị phụ thuộc",
    ]
    examples = [
        "Show the dependency graph of this package",
        "Find circular imports in the codebase",
        "Which modules have the highest fan-in?",
    ]

    @property
    def name(self) -> str:
        return "code_dependency"

    @property
    def description(self) -> str:
        return (
            "Trích dependency graph: imports, call graph, fan-in/fan-out, "
            "circular dependency detection, suggest module boundaries."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.15
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output="[CodeDependency] Import graph extraction + cycle detection ready.",
            artifacts=[
                {"path": "dependency/graph_extractor.py", "content": _GRAPH_EXTRACTOR},
                {"path": "dependency/report_template.md", "content": _REPORT_TEMPLATE},
            ],
            metadata={
                "skill": self.name,
                "graph_types": {
                    "import_graph": "module -> set of imported modules (static)",
                    "call_graph": "function -> set of called functions (intra-procedural)",
                    "type_graph": "class -> set of referenced types",
                    "runtime_graph": "actual module loads (instrumented, e.g. sys.modules diff)",
                },
                "metrics": {
                    "fan_in": "Number of modules depending on this one",
                    "fan_out": "Number of modules this one depends on",
                    "instability": "I = fan_out / (fan_in + fan_out) — 0 = stable, 1 = unstable",
                    "abstractness": "A = abstract_classes / total_classes (per module)",
                    "distance_main_seq": "D = |A + I - 1| — 0 is on the main sequence (good)",
                },
                "cycle_detection": [
                    "Tarjan SCC (Strongly Connected Components) — O(V+E)",
                    "DFS with color marking (white/gray/black) — simpler, O(V+E)",
                    "Johnson's algorithm for enumerating ALL elementary cycles",
                ],
                "visualization": {
                    "graphviz": "dot -Tsvg deps.dot -o deps.svg",
                    "mermaid": "graph TD; A-->B; B-->C;",
                    "d3": "force-directed layout for interactive exploration",
                    "cytoscape": "for large graphs (10k+ nodes)",
                },
                "tooling": {
                    "python": "pydeps, snakefood, pyreverse (built-in with pylint)",
                    "javascript": "madge (CLI + lib, supports circular detection)",
                    "typescript": "madge, dependency-cruiser (rules-based)",
                    "go": "go mod graph, goda (rich analysis)",
                    "rust": "cargo tree, cargo-deny (license/advisory)",
                    "java": "Maven Enforcer (ban-circular-dependencies), JDeps",
                },
                "refactor_targets": [
                    "God module: fan_in + fan_out both very high",
                    "Cycle: A->B->C->A — break with Dependency Inversion (interface in shared module)",
                    "Leaky abstraction: low-level module imported by high-level (SOLID violation)",
                    "Dead module: zero fan_in (orphan)",
                ],
            },
            suggestions=[
                "Provide package root or list of files to scan",
                "Specify output format: dot / mermaid / json",
                "Run cycle detection if refactoring is planned",
            ],
        )


_GRAPH_EXTRACTOR = '''"""Dependency graph extractor for Python modules.

Builds import graph, detects cycles (Tarjan SCC), computes fan-in/out.

Author: Hieu Louis (2026)
"""
from __future__ import annotations
import ast
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class DependencyGraph:
    edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    modules: Set[str] = field(default_factory=set)

    def add_edge(self, src: str, dst: str) -> None:
        if src != dst:
            self.edges[src].add(dst)
        self.modules.add(src)
        self.modules.add(dst)

    def fan_out(self, module: str) -> int:
        return len(self.edges.get(module, set()))

    def fan_in(self, module: str) -> int:
        return sum(1 for src, dsts in self.edges.items() if module in dsts)

    def instability(self, module: str) -> float:
        fin, fout = self.fan_in(module), self.fan_out(module)
        total = fin + fout
        return fout / total if total else 0.0


def build_import_graph(root: str, package_name: str) -> DependencyGraph:
    """Walk directory, parse each .py, extract import edges."""
    graph = DependencyGraph()
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            module = _path_to_module(os.path.relpath(path, root), package_name)
            src = open(path, encoding="utf-8").read()
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                for dep in _extract_imports(node, package_name):
                    graph.add_edge(module, dep)
    return graph


def _extract_imports(node: ast.AST, package_name: str) -> List[str]:
    """Return list of imported module dotted names (only local package)."""
    deps: List[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.startswith(package_name):
                deps.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        if node.module and node.module.startswith(package_name):
            deps.append(node.module)
    return deps


def _path_to_module(rel_path: str, package_name: str) -> str:
    parts = rel_path.replace(os.sep, ".").removesuffix(".py")
    if parts.endswith(".__init__"):
        parts = parts.removesuffix(".__init__")
    return f"{package_name}.{parts}" if parts else package_name


def find_cycles(graph: DependencyGraph) -> List[List[str]]:
    """Tarjan SCC algorithm — returns list of strongly connected components
    of size >= 2 (these contain cycles)."""
    index_counter = [0]
    stack: List[str] = []
    lowlink: Dict[str, int] = {}
    index: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    sccs: List[List[str]] = []

    def strongconnect(node: str) -> None:
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True
        for succ in graph.edges.get(node, set()):
            if succ not in index:
                strongconnect(succ)
                lowlink[node] = min(lowlink[node], lowlink[succ])
            elif on_stack.get(succ):
                lowlink[node] = min(lowlink[node], index[succ])
        if lowlink[node] == index[node]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == node:
                    break
            if len(comp) >= 2:
                sccs.append(comp)

    for m in graph.modules:
        if m not in index:
            strongconnect(m)
    return sccs


def hotspots(graph: DependencyGraph, top_k: int = 10) -> List[Tuple[str, int, int, float]]:
    """Return top-K modules by instability — refactor candidates."""
    rows = [
        (m, graph.fan_in(m), graph.fan_out(m), graph.instability(m))
        for m in graph.modules
    ]
    return sorted(rows, key=lambda r: r[3], reverse=True)[:top_k]
'''


_REPORT_TEMPLATE = '''# Dependency Analysis Report

## Summary
- Modules analyzed: <N>
- Total edges: <E>
- Cycles detected: <C>
- Orphan modules (fan_in=0): <O>

## Graph Visualization
```dot
digraph deps {
    rankdir=LR;
    node [shape=box];
    "pkg.api" -> "pkg.service";
    "pkg.service" -> "pkg.repo";
    "pkg.repo" -> "pkg.models";
    "pkg.api" -> "pkg.models";   // shortcut — consider removing
}
```

## Cycle Report
```
Cycle #1 (length 3):
  pkg.a -> pkg.b -> pkg.c -> pkg.a

Suggested fix: extract shared interface into pkg.interfaces,
invert dependency: pkg.a depends on pkg.interfaces, pkg.c implements it.
```

## Instability Hotspots (top 10)
| Module        | Fan-in | Fan-out | Instability | Notes                  |
|---------------|--------|---------|-------------|------------------------|
| pkg.api       | 0      | 8       | 1.00        | entry point — OK       |
| pkg.utils     | 14     | 2       | 0.13        | god module — review   |
| pkg.models    | 22     | 1       | 0.04        | stable foundation — OK |

## Action Items
- [ ] Break cycle in pkg.a / pkg.b / pkg.c via interface extraction
- [ ] Split pkg.utils (high fan-in + high fan-out = god module)
- [ ] Verify pkg.orphan is truly dead (run dead-code skill)
'''
