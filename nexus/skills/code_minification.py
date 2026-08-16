"""Code Minification Skill - Minify JS/CSS/HTML/JSON.

Strategy: remove whitespace + comments, mangle identifiers, collapse dead
code, tree-shake unused exports. Sử dụng tool phù hợp per language.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class CodeMinificationSkill(Skill):
    """Minify code: JS, CSS, HTML, JSON. Giảm size, giữ semantic."""

    category = SkillCategory.CODE
    priority = SkillPriority.LOW
    keywords: List[str] = [
        "minify", "minification", "compress code", "uglify",
        "terser", "cssnano", "html minify", "nén code",
        "shrink", "reduce size", "bundle size", "tree shake",
    ]
    examples = [
        "Minify this JavaScript file",
        "Compress this CSS to production size",
        "Uglify this JS preserving function names",
    ]

    @property
    def name(self) -> str:
        return "code_minification"

    @property
    def description(self) -> str:
        return (
            "Minify JS/CSS/HTML/JSON: remove comments + whitespace, "
            "mangle identifiers, collapse dead code, tree-shake unused exports."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.18
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        lang = (context.language or "javascript").lower()
        return SkillResult(
            success=True,
            output=f"[CodeMinification/{lang}] Minification strategy + toolchain ready.",
            artifacts=[
                {"path": "minify/strategy.md", "content": _MINIFY_STRATEGY},
                {"path": "minify/example.txt", "content": _EXAMPLE_MINIFIED_JS},
            ],
            metadata={
                "skill": self.name,
                "language": lang,
                "toolchain": {
                    "javascript": "terser --compress --mangle",
                    "typescript": "tsc + terser (or esbuild)",
                    "css": "cssnano / lightningcss (Rust, fastest)",
                    "html": "html-minifier-terser",
                    "json": "jq -c (lossless)",
                    "python": "pyminifier (limited) — prefer zipapp + bytecode-only distribution",
                },
                "techniques": [
                    "Whitespace removal (spaces, newlines, indentation)",
                    "Comment stripping (// and /* */ and <!-- -->)",
                    "Identifier mangling (shorter names: myVar -> a)",
                    "Dead code elimination (unreachable statements)",
                    "Tree shaking (drop unused exports)",
                    "Constant folding (2+3 -> 5)",
                    "Property mangling (only when --mangle-props)",
                    "Hex/octal/unicode escape compression",
                    "Boolean shortcut (true -> !0, false -> !1)",
                ],
                "trade_offs": {
                    "size_vs_debuggability": "mangled names break stack traces — ship sourcemaps to Sentry",
                    "size_vs_startup": "esbuild may produce slightly larger bundle but parses faster",
                    "compression_vs_safety": "property mangling risky with bracket access",
                },
                "best_practices": [
                    "Always emit sourcemaps (.map) and upload to error tracker",
                    "Measure gzipped + brotli sizes, not raw bytes",
                    "Use same minifier across build matrix to keep sourcemaps consistent",
                    "Cache bust with content-hash filenames",
                ],
            },
            suggestions=[
                "Specify if source maps should be emitted",
                "Indicate if identifier mangling is safe (no eval, no bracket access)",
                "Check bundle budget (e.g. < 200 KB gzipped initial)",
            ],
        )


_MINIFY_STRATEGY = """# Minification Strategy

## Per-Language Pipeline

### JavaScript / TypeScript
```bash
# terser CLI
terser input.js \\\\
  --compress passes=2,drop_console=true,drop_debugger=true \\\\
  --mangle toplevel \\\\
  --source-map url='out.js.map' \\\\
  --output out.js
```

### CSS
```bash
# lightningcss (Rust, fastest)
lightningcss --minify --bundle --targets 'defaults' input.css -o out.css
```

### HTML
```bash
html-minifier-terser \\\\
  --collapse-whitespace --remove-comments \\\\
  --minify-css true --minify-js true \\\\
  input.html -o out.html
```

### JSON (config files)
```bash
jq -c . input.json > out.min.json
```

## Bundle Budget
- JS initial: < 200 KB gzipped
- CSS initial: < 50 KB gzipped
- Per-route lazy chunk: < 50 KB gzipped

## Pitfalls
- Mangled property names break `obj['dynamicProp']` access.
- Drop `console.log` only in production — keep in staging for tracing.
- Inline `<script>` minified by HTML minifier may conflict with CSP nonces.
"""


_EXAMPLE_MINIFIED_JS = """// ---- Before ----
function calculateTotal(items, taxRate) {
  // Sum item prices
  let subtotal = 0;
  for (let i = 0; i < items.length; i++) {
    subtotal += items[i].price * items[i].quantity;
  }
  const tax = subtotal * taxRate;
  return subtotal + tax;
}

// ---- After (terser --compress --mangle) ----
function calculateTotal(a,b){let c=0;for(let d=0;d<a.length;d++)c+=a[d].price*a[d].quantity;return c+c*b}

// Size: 274 -> 119 bytes (-57%). Gzipped: 158 -> 96 bytes (-39%).
"""
