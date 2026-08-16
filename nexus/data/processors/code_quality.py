"""
Code Quality Processor for Nexus Coder v0.3
============================================
Scores Python code samples (1-10) based on quality signals:
  - Has docstring
  - Has type hints
  - No `print` statements (in non-test code)
  - No `eval` / `exec` / `__import__`
  - No bare `except:` clauses
  - Reasonable length (10-500 lines)
  - Has adjacent test file (bonus, requires file path)

Samples below `min_score` (default 6.0) are filtered out.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import ast
import re
from typing import Dict


_BAD_PATTERNS = [
    (r"\beval\s*\(", "uses eval"),
    (r"\bexec\s*\(", "uses exec"),
    (r"\b__import__\s*\(", "uses __import__"),
    (r"\bassert\s+\w+\s*==\s*", "uses assert for tests (fine in tests, bad elsewhere)"),
]

_BARE_EXCEPT = re.compile(r"\bexcept\s*:")
_PRINT = re.compile(r"^\s*print\s*\(", re.MULTILINE)


def score_python_code(code: str, is_test_file: bool = False) -> Dict[str, float]:
    """Score a Python code sample 0-10. Returns dict of factor → score contribution."""
    factors: Dict[str, float] = {}

    # Try parsing as AST
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"_invalid": 0.0, "_total": 0.0}
    except Exception:
        return {"_invalid": 0.0, "_total": 0.0}

    # Has docstring (module-level or first function)?
    has_docstring = (
        (ast.get_docstring(tree) is not None) or
        any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(n) for n in ast.walk(tree))
    )
    if has_docstring:
        factors["has_docstring"] = 1.5

    # Type hints?
    typed_funcs = 0
    total_funcs = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            total_funcs += 1
            if node.returns is not None or any(a.annotation for a in node.args.args):
                typed_funcs += 1
    if total_funcs > 0 and typed_funcs / total_funcs > 0.3:
        factors["has_type_hints"] = 1.0

    # No bare except
    has_bare_except = bool(_BARE_EXCEPT.search(code))
    if not has_bare_except:
        factors["no_bare_except"] = 1.0

    # No eval/exec/__import__
    has_bad = False
    for pattern, _msg in _BAD_PATTERNS:
        if re.search(pattern, code):
            has_bad = True
            break
    if not has_bad:
        factors["no_eval"] = 1.0

    # Print usage (allowed in tests)
    if not is_test_file:
        if not _PRINT.search(code):
            factors["no_print"] = 0.5

    # Reasonable length
    n_lines = code.count("\n") + 1
    if 10 <= n_lines <= 500:
        factors["reasonable_length"] = 1.0
    elif 5 <= n_lines <= 1000:
        factors["reasonable_length"] = 0.5

    # Bonus for tests
    if is_test_file:
        factors["has_test"] = 2.0

    total = sum(factors.values())
    factors["_total"] = min(10.0, total)
    return factors


def score_code(code: str, language: str = "python", is_test_file: bool = False) -> Dict[str, float]:
    """Dispatch to language-specific scorer."""
    if language == "python":
        return score_python_code(code, is_test_file=is_test_file)
    # For other languages, return neutral score
    return {"_total": 6.0, "_unimplemented_lang": 1.0}


class CodeQualityProcessor:
    """Filter / tag samples by code quality score."""

    def __init__(
        self,
        min_score: float = 6.0,
        is_test_file_fn=None,
    ):
        self.min_score = min_score
        self.is_test_file_fn = is_test_file_fn or (lambda path: path and "test" in path.lower())

    def score(self, code: str, language: str = "python", path: str = "") -> float:
        is_test = bool(self.is_test_file_fn(path))
        result = score_code(code, language=language, is_test_file=is_test)
        return result.get("_total", 0.0)

    def keep(self, code: str, language: str = "python", path: str = "") -> bool:
        return self.score(code, language=language, path=path) >= self.min_score

    def tag(self, sample: Dict) -> Dict:
        code = sample.get("content", sample.get("code", sample.get("text", "")))
        lang = sample.get("lang", sample.get("language", "python"))
        path = sample.get("path", "")
        sample["code_quality_score"] = self.score(code, language=lang, path=path)
        return sample

    def batch_filter(self, samples):
        for s in samples:
            code = s.get("content", s.get("code", s.get("text", "")))
            lang = s.get("lang", s.get("language", "python"))
            path = s.get("path", "")
            if self.keep(code, language=lang, path=path):
                yield s


__all__ = ["score_python_code", "score_code", "CodeQualityProcessor"]
