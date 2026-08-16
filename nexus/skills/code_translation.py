"""Code Translation Skill - Dịch code giữa các ngôn ngữ lập trình.

Hỗ trợ transpile Python <-> JS/TS, Go, Rust, Java, C++, C# với chiến lược
mapping idioms, thư viện tương đương, và xử lý khác biệt type system.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class CodeTranslationSkill(Skill):
    """Dịch code từ ngôn ngữ này sang ngôn ngữ khác, giữ nguyên ngữ nghĩa."""

    category = SkillCategory.CODE
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "translate", "dịch", "convert", "chuyển", "transpile",
        "port", "porting", "python to", "python sang",
        "javascript to", "go to", "rust to", "java to",
        "to python", "to javascript", "to go", "to rust",
        "convert python", "migrate code",
    ]
    examples = [
        "Convert Python script to Go",
        "Translate this JavaScript function to Rust",
        "Port Java code to Python idiomatic style",
    ]

    @property
    def name(self) -> str:
        return "code_translation"

    @property
    def description(self) -> str:
        return (
            "Dịch code giữa các ngôn ngữ: Python, JS/TS, Go, Rust, Java, C++, C#. "
            "Transpile + idiomatic rewrite + thư viện tương đương."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.15
        # Phát hiện cặp "X to Y" pattern / detect "X to Y" pattern
        langs = ("python", "javascript", "js", "typescript", "ts", "go",
                 "rust", "java", "c++", "c#")
        for src in langs:
            for dst in langs:
                if src != dst and f"{src} to {dst}" in prompt_lower:
                    score += 0.25
                    break
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output=(
                "[CodeTranslation] Transpilation strategy ready. "
                "Mapping idioms, library equivalents, and type system differences."
            ),
            artifacts=[
                {"path": "translation/language_mapping.md", "content": _LANGUAGE_MAPPING},
                {"path": "translation/transpile_strategy.py", "content": _TRANSPILE_STRATEGY},
            ],
            metadata={
                "skill": self.name,
                "supported_pairs": [
                    "python<->javascript", "python<->go", "python<->rust",
                    "python<->java", "javascript<->typescript",
                    "java<->c#", "c++<->rust",
                ],
                "translation_phases": [
                    "1. Lexical + syntactic parse (AST) của source",
                    "2. Type inference nếu đích statically typed",
                    "3. Idiom mapping (list comprehension -> for-loop, etc.)",
                    "4. Library substitution (requests -> fetch/http, numpy -> ndarray)",
                    "5. Error/exception model translation (try/except <-> Result/Option)",
                    "6. Concurrency primitive mapping (asyncio <-> tokio <-> goroutines)",
                    "7. Idiomatic rewrite + formatting (ruff / gofmt / rustfmt)",
                    "8. Generate equivalence tests (golden file) để verify semantic",
                ],
                "idiom_examples": {
                    "list_comprehension": "Python: [f(x) for x in xs]  ->  Rust: xs.iter().map(f).collect()",
                    "dict_default": "Python: d.get(k, default)  ->  Go: if v, ok := d[k]; !ok { ... }",
                    "optional_chaining": "JS: a?.b?.c  ->  Rust: a.and_then(|x| x.b).and_then(|y| y.c)",
                    "decorator": "Python @cache  ->  Java @Cacheable annotation",
                    "exception_to_result": "Python try/except  ->  Rust Result<T, E> + ? operator",
                    "generator": "Python yield  ->  Go channel + goroutine",
                },
                "library_equivalents": {
                    "http_client": {
                        "python": "httpx/requests",
                        "js": "fetch/axios",
                        "go": "net/http",
                        "rust": "reqwest",
                    },
                    "json": {
                        "python": "json",
                        "js": "JSON",
                        "go": "encoding/json",
                        "rust": "serde_json",
                    },
                    "async": {
                        "python": "asyncio",
                        "js": "Promise/async-await",
                        "go": "goroutines+channels",
                        "rust": "tokio",
                    },
                    "testing": {
                        "python": "pytest",
                        "js": "jest/vitest",
                        "go": "testing",
                        "rust": "cargo test",
                    },
                },
            },
            suggestions=[
                "Provide source code in a fenced block để dịch chính xác",
                "Specify target language version (e.g. Python 3.12 vs 3.8)",
                "Indicate whether to keep exact behavior or rewrite idiomatically",
                "Run golden tests to verify semantic equivalence post-translation",
            ],
        )


_LANGUAGE_MAPPING = '''# Language Mapping Reference

| Construct        | Python              | JavaScript         | Go                  | Rust                |
|------------------|---------------------|---------------------|---------------------|---------------------|
| Variable         | x = 1               | let x = 1           | x := 1              | let x = 1           |
| Constant         | X = 1 (convention)  | const x = 1         | const x = 1         | const X: i32 = 1    |
| Function         | def f(x): ...       | function f(x){}     | func f(x T){}       | fn f(x: T) {}       |
| Lambda           | lambda x: x+1       | x => x+1            | func(x T)T{return} | `\\|x\\| x+1`         |
| List/Array       | [1,2,3]             | [1,2,3]             | []int{1,2,3}        | vec![1,2,3]         |
| Dict/Map         | {"a":1}             | {a:1}               | map[string]int      | HashMap::new()      |
| Optional         | x=None              | x=undefined/null    | *T + ok pattern     | Option<T>           |
| Error            | raise ValueError    | throw new Error     | return err          | return Err(...)      |
| Class            | class C:           | class C {}          | struct + methods    | struct + impl       |
| Interface        | (duck typing)       | (structural)         | interface           | trait               |
| Generics         | (runtime)           | (TS only)            | [T any]             | <T>                 |
| Async            | async def / await   | async/await         | go func(){}         | async fn / .await   |

## Type System Differences

| Concern            | Python          | JS/TS                  | Go                   | Rust                |
|--------------------|-----------------|------------------------|----------------------|---------------------|
| Null safety        | None checks     | undefined/null         | nil checks + zero    | Option<T>           |
| Type inference     | runtime         | TS compile-time        | yes                  | yes (strong)        |
| Numeric overflow   | arbitrary int   | IEEE-754 number        | wraps (math/checked) | wraps (checked_*)   |
| Strings            | unicode         | UTF-16                 | UTF-8 byte slice     | UTF-8 str           |
| Memory mgmt        | GC (refcount)   | GC                     | GC                   | ownership (no GC)   |
'''


_TRANSPILE_STRATEGY = '''"""Transpilation Strategy Template.

7-step pipeline để dịch code giữ nguyên ngữ nghĩa.
Author: Hieu Louis (2026)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class TranslationConfig:
    source_lang: str
    target_lang: str
    source_version: str = "latest"
    target_version: str = "latest"
    mode: str = "idiomatic"        # "idiomatic" | "literal" | "conservative"
    preserve_comments: bool = True
    rewrite_stdlib_calls: bool = True
    emit_tests: bool = True


@dataclass
class TranslationReport:
    warnings: List[str]
    manual_review_required: List[str]
    library_substitutions: Dict[str, str]
    confidence: float   # 0.0 - 1.0


def translate(source: str, config: TranslationConfig) -> Tuple[str, TranslationReport]:
    """7-step transpilation pipeline.

    Returns: (translated_code, report)
    """
    # 1. Parse source -> AST
    ast = _parse(source, config.source_lang)

    # 2. Type inference (best-effort) if target is statically typed
    if config.target_lang in {"go", "rust", "java", "c++", "c#"}:
        ast = _infer_types(ast)

    # 3. Idiom mapping (list comprehension -> map/filter, etc.)
    ast = _map_idioms(ast, config)

    # 4. Library substitution (requests -> httpx/fetch/reqwest)
    if config.rewrite_stdlib_calls:
        ast = _rewrite_stdlib(ast, config)

    # 5. Error model translation (try/except -> Result/Option or try/catch)
    ast = _translate_error_model(ast, config)

    # 6. Code generation
    code = _generate(ast, config.target_lang)

    # 7. Format (ruff / gofmt / rustfmt / prettier)
    formatted = _format(code, config.target_lang)

    report = TranslationReport(
        warnings=[],
        manual_review_required=[],
        library_substitutions={},
        confidence=0.85,
    )
    return formatted, report


def _parse(src: str, lang: str):
    """Stub: dispatch to tree-sitter / libcst / babel / etc."""
    ...


def _infer_types(ast):
    """Stub: optional type inference for statically-typed targets."""
    return ast


def _map_idioms(ast, cfg):
    """Stub: rewrite idioms (comprehension -> map/filter, etc.)."""
    return ast


def _rewrite_stdlib(ast, cfg):
    """Stub: replace source-stdlib calls with target-stdlib equivalents."""
    return ast


def _translate_error_model(ast, cfg):
    """Stub: convert try/except to Result/Option or try/catch."""
    return ast


def _generate(ast, lang: str) -> str:
    """Stub: emit code from AST."""
    return ""


def _format(code: str, lang: str) -> str:
    """Stub: invoke language formatter (ruff / gofmt / rustfmt / prettier)."""
    return code
'''
