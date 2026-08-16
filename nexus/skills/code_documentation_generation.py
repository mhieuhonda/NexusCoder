"""Code Documentation Skill - Sinh docstring/comment tự động.

Hỗ trợ Python (Google/NumPy/Sphinx), JS (JSDoc), TS (TSDoc), Go (godoc),
Rust (rustdoc), Java (Javadoc), với template per style.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class CodeDocumentationSkill(Skill):
    """Sinh docstring và comment cho function/class/module."""

    category = SkillCategory.DOCUMENTATION
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "docstring", "document function", "document class",
        "jsdoc", "javadoc", "godoc", "rustdoc", "tsdoc",
        "generate docs", "documentation", "tài liệu",
        "viết docstring", "comment code", "annotate",
    ]
    examples = [
        "Generate Google-style docstring for this Python function",
        "Write JSDoc for this JavaScript function",
        "Document all public methods of this class",
    ]

    @property
    def name(self) -> str:
        return "code_documentation"

    @property
    def description(self) -> str:
        return (
            "Sinh docstring/comment cho Python (Google/NumPy/Sphinx), "
            "JS (JSDoc), TS (TSDoc), Go (godoc), Rust (rustdoc), Java (Javadoc)."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.16
        if "def " in prompt or "function " in prompt or "func " in prompt:
            score += 0.15
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        lang = (context.language or "python").lower()
        return SkillResult(
            success=True,
            output=f"[CodeDocumentation/{lang}] Docstring templates ready.",
            artifacts=[
                {"path": "docs/templates.md", "content": _DOCSTRING_TEMPLATES},
                {"path": "docs/jsdoc_template.md", "content": _JSDOC_TEMPLATE},
                {"path": "docs/strategy.md", "content": _DOC_STRATEGY},
            ],
            metadata={
                "skill": self.name,
                "language": lang,
                "styles": {
                    "python": ["google", "numpy", "sphinx", "rest"],
                    "javascript": ["jsdoc"],
                    "typescript": ["tsdoc (typedoc)"],
                    "go": ["godoc (no annotations)"],
                    "rust": ["rustdoc markdown"],
                    "java": ["javadoc"],
                    "kotlin": ["kdoc"],
                },
                "extraction_targets": [
                    "purpose (first sentence)",
                    "parameters (name, type, meaning, default, constraints)",
                    "return value (type, meaning, conditions)",
                    "raises/throws (exception types + when)",
                    "examples (doctest-runnable when possible)",
                    "side effects",
                    "deprecated + replacement",
                    "see also",
                ],
                "tooling": {
                    "python": "Sphinx + autodoc + napoleon + intersphinx",
                    "js": "TypeDoc (TS) / JSDoc (JS)",
                    "go": "godoc / pkg.go.dev",
                    "rust": "cargo doc",
                    "java": "Javadoc + Maven Javadoc plugin",
                },
                "validation": [
                    "doctest for Python examples",
                    "mypy/pyright on type annotations",
                    "lint: every public symbol has docs (CI check)",
                ],
            },
            suggestions=[
                "Pick style explicitly: 'google' / 'numpy' / 'sphinx' for Python",
                "Ask for doctest-runnable examples when applicable",
                "Document exceptions explicitly even if not raised directly",
            ],
        )


_DOCSTRING_TEMPLATES = '''# Python Docstring Templates

## Google style

```python
def compute_discount(cart, customer_tier, coupon=None):
    """Compute discount for a cart.

    Applies tiered discount rules based on cart total and customer tier.
    Discount is capped at 40% for retail customers.

    Args:
        cart: List of (sku, unit_price, quantity) tuples. Must be non-empty.
        customer_tier: One of "bronze", "silver", "gold". Case-insensitive.
        coupon: Optional coupon code. None for no coupon.

    Returns:
        Tuple of (discount_amount, final_total). discount_amount in
        [0, cart_subtotal]. final_total is non-negative.

    Raises:
        ValueError: If cart is empty or customer_tier is unknown.
        CouponExpiredError: If coupon code is past its expiry date.

    Examples:
        >>> compute_discount([("A1", 100, 2)], "gold")
        (20.0, 180.0)
    """
    ...
```

## NumPy style

```python
def compute_discount(cart, customer_tier, coupon=None):
    """Compute discount for a cart.

    Applies tiered discount rules based on cart total and customer tier.

    Parameters
    ----------
    cart : list[tuple[str, float, int]]
        List of (sku, unit_price, quantity) tuples. Must be non-empty.
    customer_tier : {"bronze", "silver", "gold"}
        Customer loyalty tier. Case-insensitive.
    coupon : str, optional
        Optional coupon code. None for no coupon.

    Returns
    -------
    tuple[float, float]
        (discount_amount, final_total). discount_amount in [0, subtotal].

    Raises
    ------
    ValueError
        If cart is empty or customer_tier is unknown.
    CouponExpiredError
        If coupon code is past its expiry date.

    Examples
    --------
    >>> compute_discount([("A1", 100, 2)], "gold")
    (20.0, 180.0)
    """
    ...
```

## Sphinx (reST) style

```python
def compute_discount(cart, customer_tier, coupon=None):
    """Compute discount for a cart.

    Applies tiered discount rules based on cart total and customer tier.

    :param cart: List of (sku, unit_price, quantity) tuples. Must be non-empty.
    :type cart: list[tuple[str, float, int]]
    :param customer_tier: One of "bronze", "silver", "gold". Case-insensitive.
    :type customer_tier: str
    :param coupon: Optional coupon code. None for no coupon.
    :type coupon: str | None
    :returns: (discount_amount, final_total).
    :rtype: tuple[float, float]
    :raises ValueError: If cart is empty or customer_tier is unknown.
    :raises CouponExpiredError: If coupon code is past expiry.

    Example::

        >>> compute_discount([("A1", 100, 2)], "gold")
        (20.0, 180.0)
    """
    ...
```
'''


_JSDOC_TEMPLATE = '''# JSDoc / TSDoc Template

```javascript
/**
 * Compute discount for a cart.
 *
 * Applies tiered discount rules based on cart total and customer tier.
 * Discount is capped at 40% for retail customers.
 *
 * @param {Array<{sku: string, unitPrice: number, quantity: number}>} cart
 *   List of cart items. Must be non-empty.
 * @param {"bronze" | "silver" | "gold"} customerTier
 *   Customer loyalty tier. Case-insensitive.
 * @param {string | null} [coupon=null]
 *   Optional coupon code. Pass null for no coupon.
 * @returns {{discountAmount: number, finalTotal: number}}
 *   Discount amount (0 <= d <= subtotal) and final total.
 * @throws {TypeError} If cart is empty.
 * @throws {CouponExpiredError} If coupon code is past expiry.
 *
 * @example
 * const { discountAmount, finalTotal } = computeDiscount(
 *   [{ sku: "A1", unitPrice: 100, quantity: 2 }],
 *   "gold"
 * );
 * // => { discountAmount: 20, finalTotal: 180 }
 *
 * @see {@link applyCoupon} for coupon resolution logic.
 * @since 1.2.0
 * @public
 */
function computeDiscount(cart, customerTier, coupon = null) {
  // ...
}
```

## TSDoc (TypeScript) variant

```typescript
/**
 * Compute discount for a cart.
 *
 * @param cart - List of cart items. Must be non-empty.
 * @param customerTier - Customer loyalty tier. Case-insensitive.
 * @param coupon - Optional coupon code. Pass null for no coupon.
 * @returns Discount amount and final total.
 * @throws {TypeError} If cart is empty.
 *
 * @example
 * ```ts
 * const r = computeDiscount([{ sku: "A1", unitPrice: 100, quantity: 2 }], "gold");
 * ```
 */
function computeDiscount(
  cart: CartItem[],
  customerTier: "bronze" | "silver" | "gold",
  coupon: string | null = null,
): { discountAmount: number; finalTotal: number } {
  // ...
}
```
'''


_DOC_STRATEGY = """# Documentation Generation Strategy

## Phase 1: Static Extraction
- Parse AST, collect: function signatures, parameter types, return types,
  raised exceptions, decorators, class hierarchy.
- Infer types when missing (mypy/pyright inference).

## Phase 2: Purpose Inference
- Heuristics: function name + first assignment + last return + called functions.
- LLM fallback: ask for one-sentence summary, validate against signature.

## Phase 3: Parameter Description
- Per parameter: infer from usage (read once? written? returned?).
- Look at type hints + constraint annotations.
- Generate description: "<param> is the <role>: <constraint>".

## Phase 4: Examples
- Generate 1 happy-path + 1 error example.
- Make examples doctest-runnable (Python) or runnable snippets (JS).

## Phase 5: Style Compliance
- Match existing docstring style in module (auto-detect: google/numpy/sphinx).
- Match indentation, line length, terminology.

## Phase 6: Validation
- doctest: every `>>>` block must pass.
- darglint / pydocstyle / flake8-docstrings lint.
- Verify all params in signature have `Args:` entries.
"""
