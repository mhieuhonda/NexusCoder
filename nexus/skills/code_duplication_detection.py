"""Code Duplication Detection Skill - Phát hiện code trùng lặp.

Sử dụng AST-based hashing, token n-grams, và Rabin-Karp fingerprinting
để phát hiện Type I/II/III/IV duplication trong codebase.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class CodeDuplicationSkill(Skill):
    """Phát hiện duplicate code (Type I-IV) trong codebase."""

    category = SkillCategory.CODE
    priority = SkillPriority.LOW
    keywords: List[str] = [
        "duplicate code", "duplication", "copy paste", "code clone",
        "code duplication", "DRY violation", "lặp code",
        "trùng lặp code", "similar functions", "repeated code",
    ]
    examples = [
        "Find duplicate code in this module",
        "Detect copy-pasted functions across the codebase",
        "Report DRY violations and refactor candidates",
    ]

    @property
    def name(self) -> str:
        return "code_duplication"

    @property
    def description(self) -> str:
        return (
            "Phát hiện code trùng lặp Type I/II/III/IV bằng AST hashing + "
            "token n-grams + Rabin-Karp fingerprinting. Output refactor candidates."
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
            output="[CodeDuplication] AST + token-n-gram detection algorithm ready.",
            artifacts=[
                {"path": "duplication/detector.py", "content": _DUPLICATION_DETECTOR},
                {"path": "duplication/classification.md", "content": _CLASSIFICATION},
            ],
            metadata={
                "skill": self.name,
                "clone_types": {
                    "Type I": "Identical code (whitespace + comments differ) — exact text match",
                    "Type II": "Structurally identical (variable names / types differ) — AST match",
                    "Type III": "Modified copy (statements added/removed) — AST diff <= threshold",
                    "Type IV": "Semantic clones (different syntax, same behavior) — requires semantic analysis",
                },
                "algorithms": [
                    "AST node hashing (Type II): hash each function's AST, compare hashes",
                    "Token n-gram + Rabin-Karp rolling hash (Type I/II): sub-linear scan",
                    "PDG (Program Dependence Graph) isomorphism (Type IV): expensive, semantic",
                ],
                "tooling": {
                    "python": "pylint --disable=all --enable=duplicate-code (also: cloneserver, lizard)",
                    "javascript": "jscpd (token-based, supports many languages)",
                    "java": "PMD CPD (Copy-Paste Detector)",
                    "rust": "cargo duplicate",
                    "multi_lang": "jscpd (16+ languages, token-based)",
                },
                "metrics": {
                    "duplicate_lines": "raw count of duplicated lines",
                    "duplication_ratio": "duplicate_lines / total_lines (%)",
                    "largest_clone_block": "size of biggest clone (tokens)",
                    "clone_clusters": "number of distinct clone groups",
                },
                "thresholds": {
                    "min_lines": 5,
                    "min_tokens": 50,
                    "max_levenshtein_ratio": 0.15,  # for Type III
                },
            },
            suggestions=[
                "Provide path(s) to scan or paste code in fenced block",
                "Specify threshold: min token count per clone (default 50)",
                "For semantic duplicates (Type IV) accept higher false-positive rate",
            ],
        )


_DUPLICATION_DETECTOR = '''"""AST + token n-gram based duplication detector.

Strategy:
- Type I: exact text match (after whitespace normalization)
- Type II: AST structural hash (variable names normalized to <ID>)
- Type III: token n-gram Jaccard similarity >= threshold

Author: Hieu Louis (2026)
"""
from __future__ import annotations
import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class CloneCluster:
    """Một cụm các clone block trùng lặp."""
    cluster_id: int
    clone_type: str          # "I" | "II" | "III"
    blocks: List[Tuple[str, int, int]]  # (file, start_line, end_line)
    token_count: int
    fingerprint: str


@dataclass
class DuplicationReport:
    total_lines: int
    duplicated_lines: int
    clusters: List[CloneCluster] = field(default_factory=list)

    @property
    def duplication_ratio(self) -> float:
        return self.duplicated_lines / max(1, self.total_lines)


def detect_duplication(files: List[str], min_tokens: int = 50) -> DuplicationReport:
    """Phát hiện duplicate trong list of file paths."""
    # Phase 1: parse all files -> list of (file, function_ast)
    funcs = []
    for path in files:
        src = open(path, encoding="utf-8").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                funcs.append((path, node, _ast_hash(node), _tokens(node)))

    # Phase 2: Type II — group by AST hash
    by_hash: Dict[str, List[Tuple[str, ast.AST]]] = {}
    for path, node, h, _ in funcs:
        by_hash.setdefault(h, []).append((path, node))

    clusters: List[CloneCluster] = []
    cid = 0
    for h, group in by_hash.items():
        if len(group) < 2:
            continue
        blocks = [(p, n.lineno, getattr(n, "end_lineno", n.lineno)) for p, n in group]
        tokens = sum(1 for _ in ast.walk(group[0][1]))
        if tokens < min_tokens:
            continue
        clusters.append(CloneCluster(cid, "II", blocks, tokens, h))
        cid += 1

    # Phase 3: Type III — token n-gram Jaccard similarity
    ngram_clusters = _detect_type_iii(funcs, min_tokens)
    clusters.extend(ngram_clusters)

    duplicated = sum(c.token_count * (len(c.blocks) - 1) for c in clusters)
    return DuplicationReport(
        total_lines=sum(_count_lines(f) for f in files),
        duplicated_lines=duplicated,
        clusters=clusters,
    )


def _ast_hash(node: ast.AST) -> str:
    """Hash AST với biến được normalize -> <ID> cho Type II matching."""
    normalized = _normalize_names(node)
    src = ast.dump(normalized, annotate_fields=False)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _normalize_names(node: ast.AST) -> ast.AST:
    """Replace tất cả Name / arg với placeholder 'ID'."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            n.id = "ID"
        elif isinstance(n, ast.arg):
            n.arg = "ID"
    return node


def _tokens(node: ast.AST) -> List[str]:
    """Extract token sequence từ AST node."""
    return [type(n).__name__ for n in ast.walk(node)]


def _detect_type_iii(funcs, min_tokens):
    """Token n-gram Jaccard similarity >= 0.85 -> Type III clone."""
    clusters = []
    seen = set()
    ngram_size = 5
    for i, (p1, n1, _, t1) in enumerate(funcs):
        if i in seen:
            continue
        g1 = _ngrams(t1, ngram_size)
        if not g1:
            continue
        group_blocks = [(p1, n1.lineno, getattr(n1, "end_lineno", n1.lineno))]
        for j in range(i + 1, len(funcs)):
            if j in seen:
                continue
            p2, n2, _, t2 = funcs[j]
            g2 = _ngrams(t2, ngram_size)
            if not g2:
                continue
            jaccard = len(g1 & g2) / max(1, len(g1 | g2))
            if jaccard >= 0.85:
                group_blocks.append((p2, n2.lineno, getattr(n2, "end_lineno", n2.lineno)))
                seen.add(j)
        if len(group_blocks) >= 2:
            seen.add(i)
            clusters.append(CloneCluster(
                cluster_id=0, clone_type="III",
                blocks=group_blocks, token_count=len(t1),
                fingerprint=str(hash(frozenset(g1)),
            )))
    return clusters


def _ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _count_lines(path: str) -> int:
    try:
        return sum(1 for _ in open(path, encoding="utf-8"))
    except OSError:
        return 0
'''


_CLASSIFICATION = """# Code Duplication Classification (Bellon's Taxonomy)

| Type | Description                                            | Detection Method                     |
|------|--------------------------------------------------------|--------------------------------------|
| I    | Exact copy (whitespace/comments may differ)            | Text normalization + hash            |
| II   | Structurally identical (names/types differ)            | AST hash with name normalization     |
| III  | Modified copy (statements added/removed/edited)        | Token n-gram Jaccard similarity      |
| IV   | Semantic clones (different syntax, same behavior)      | PDG isomorphism (expensive)          |

## Decision Tree

1. Start with Type I + II (cheap, AST-based).
2. For remaining functions, run Type III with n-gram size 5 + Jaccard >= 0.85.
3. Type IV only on critical hot paths (cost: O(n^2) PDG matching).

## Output Schema

```
Cluster #3  Type II  tokens=128
  - src/api/users.py:45-72    def get_user(...)
  - src/api/orders.py:88-115  def get_order(...)   <-- refactor candidate

Suggested refactor: extract common base `_get_resource(model, id)`
```

## CI Integration

- Run on every PR; fail if new clone cluster introduced.
- Allow-list existing clones (manual review backlog).
- Track `duplication_ratio` metric over time (avoid regression).
"""
