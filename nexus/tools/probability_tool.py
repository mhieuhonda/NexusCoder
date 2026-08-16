"""
Probability Tool - Phân phối xác suất: pdf, cdf, sampling.
===========================================
Hỗ trợ: normal, binomial, poisson, exponential. Ưu tiên scipy.stats,
fallback stdlib `math` + `random` cho pdf/cdf của normal/exponential.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


DISTRIBUTIONS = {"normal", "binomial", "poisson", "exponential"}
OPERATIONS = {"pdf", "cdf", "pmf", "sample", "stats"}


class ProbabilityTool(Tool):
    """Phân phối xác suất: normal/binomial/poisson/exponential — pdf, cdf, sampling."""

    category = ToolCategory.MATH
    safety = ToolSafety.SAFE

    @property
    def name(self) -> str:
        return "probability"

    @property
    def description(self) -> str:
        return "Probability distributions (normal/binomial/poisson/exponential): pdf/cdf/pmf/sample/stats."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "distribution": {
                    "type": "string",
                    "enum": sorted(DISTRIBUTIONS),
                    "default": "normal",
                },
                "operation": {
                    "type": "string",
                    "enum": sorted(OPERATIONS),
                    "default": "pdf",
                },
                "params": {
                    "type": "object",
                    "description": (
                        "normal: {mu, sigma, x}; binomial: {n, p, k}; "
                        "poisson: {lambda, k}; exponential: {lambda, x}"
                    ),
                },
            },
            "required": ["distribution", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        dist = args.get("distribution", "normal")
        if dist not in DISTRIBUTIONS:
            return f"Invalid distribution='{dist}'. Supported: {sorted(DISTRIBUTIONS)}"
        op = args.get("operation", "pdf")
        if op not in OPERATIONS:
            return f"Invalid operation='{op}'. Supported: {sorted(OPERATIONS)}"
        return None

    # ---- Pdf/Cdf thủ công (fallback không cần scipy) ---------------------

    @staticmethod
    def _normal_pdf(x: float, mu: float, sigma: float) -> float:
        if sigma <= 0:
            return 0.0
        return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

    @staticmethod
    def _normal_cdf(x: float, mu: float, sigma: float) -> float:
        if sigma <= 0:
            return 0.0
        return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))

    @staticmethod
    def _exponential_pdf(x: float, lam: float) -> float:
        if lam <= 0 or x < 0:
            return 0.0
        return lam * math.exp(-lam * x)

    @staticmethod
    def _exponential_cdf(x: float, lam: float) -> float:
        if lam <= 0 or x < 0:
            return 0.0
        return 1.0 - math.exp(-lam * x)

    @staticmethod
    def _poisson_pmf(k: int, lam: float) -> float:
        if lam <= 0 or k < 0:
            return 0.0
        # exp(-lam) * lam^k / k!
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    @staticmethod
    def _poisson_cdf(k: int, lam: float) -> float:
        if lam <= 0 or k < 0:
            return 0.0
        # Tổng PMF từ 0..k / sum PMF 0..k
        return sum(ProbabilityTool._poisson_pmf(i, lam) for i in range(k + 1))

    @staticmethod
    def _binomial_pmf(k: int, n: int, p: float) -> float:
        if not (0 <= p <= 1) or k < 0 or k > n or n <= 0:
            return 0.0
        # C(n,k) * p^k * (1-p)^(n-k)
        return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    @staticmethod
    def _binomial_cdf(k: int, n: int, p: float) -> float:
        if not (0 <= p <= 1) or k < 0:
            return 0.0
        if k >= n:
            return 1.0
        return sum(ProbabilityTool._binomial_pmf(i, n, p) for i in range(k + 1))

    # ---- Sampling (stdlib random) ---------------------------------------

    def _sample(self, dist: str, params: Dict[str, Any], n: int) -> List[float]:
        rng = random.Random(params.get("seed"))
        if dist == "normal":
            mu = float(params.get("mu", 0.0))
            sigma = float(params.get("sigma", 1.0))
            return [rng.gauss(mu, sigma) for _ in range(n)]
        if dist == "binomial":
            nn = int(params.get("n", 10))
            p = float(params.get("p", 0.5))
            return [float(sum(rng.random() < p for _ in range(nn))) for _ in range(n)]
        if dist == "poisson":
            lam = float(params.get("lambda", 1.0))
            # Knuth's algorithm / Knuth algorithm
            out: List[float] = []
            for _ in range(n):
                L = math.exp(-lam)
                k_count = 0
                p_acc = 1.0
                while True:
                    p_acc *= rng.random()
                    if p_acc <= L:
                        break
                    k_count += 1
                out.append(float(k_count))
            return out
        if dist == "exponential":
            lam = float(params.get("lambda", 1.0))
            return [rng.expovariate(lam) for _ in range(n)]
        raise ValueError(f"Unknown distribution: {dist}")

    # ---- Execute --------------------------------------------------------

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        dist = args.get("distribution", "normal")
        op = args.get("operation", "pdf")
        params: Dict[str, Any] = args.get("params", {}) or {}
        seed = params.get("seed")

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] {dist}.{op}()",
                metadata={"distribution": dist, "operation": op, "dry_run": True},
            )

        try:
            # Ưu tiên scipy.stats / prefer scipy
            use_scipy = False
            try:
                import scipy.stats as st  # type: ignore
                use_scipy = True
            except ImportError:
                pass

            if op == "sample":
                n = int(params.get("n", 1))
                if n <= 0 or n > 100000:
                    return ToolResult(success=False, error=f"n phải ∈ [1, 100000], got {n}", return_code=1)
                samples = self._sample(dist, params, n)
                return ToolResult(
                    success=True,
                    output=str(samples),
                    metadata={"distribution": dist, "n": n, "samples": samples, "backend": "stdlib-random"},
                )

            if op == "stats":
                # Trả về mean/var của phân phối (lý thuyết) / theoretical mean/var
                if dist == "normal":
                    res = {"mean": float(params.get("mu", 0.0)), "variance": float(params.get("sigma", 1.0)) ** 2}
                elif dist == "binomial":
                    n_val = int(params.get("n", 10))
                    p_val = float(params.get("p", 0.5))
                    res = {"mean": n_val * p_val, "variance": n_val * p_val * (1 - p_val)}
                elif dist == "poisson":
                    lam = float(params.get("lambda", 1.0))
                    res = {"mean": lam, "variance": lam}
                else:  # exponential
                    lam = float(params.get("lambda", 1.0))
                    res = {"mean": 1.0 / lam, "variance": 1.0 / (lam ** 2)}
                return ToolResult(success=True, output=str(res), metadata={"distribution": dist, "stats": res})

            # pdf / pmf / cdf
            if dist == "normal":
                x = float(params.get("x", 0.0))
                mu = float(params.get("mu", 0.0))
                sigma = float(params.get("sigma", 1.0))
                if op in ("pdf", "pmf"):
                    res = self._normal_pdf(x, mu, sigma) if not use_scipy else float(st.norm.pdf(x, mu, sigma))
                else:  # cdf
                    res = self._normal_cdf(x, mu, sigma) if not use_scipy else float(st.norm.cdf(x, mu, sigma))
            elif dist == "binomial":
                k = int(params.get("k", 0))
                nn = int(params.get("n", 10))
                p = float(params.get("p", 0.5))
                if op == "pmf":
                    res = self._binomial_pmf(k, nn, p) if not use_scipy else float(st.binom.pmf(k, nn, p))
                elif op == "cdf":
                    res = self._binomial_cdf(k, nn, p) if not use_scipy else float(st.binom.cdf(k, nn, p))
                else:
                    return ToolResult(success=False, error="binomial không có pdf (dùng pmf)", return_code=1)
            elif dist == "poisson":
                k = int(params.get("k", 0))
                lam = float(params.get("lambda", 1.0))
                if op == "pmf":
                    res = self._poisson_pmf(k, lam) if not use_scipy else float(st.poisson.pmf(k, lam))
                elif op == "cdf":
                    res = self._poisson_cdf(k, lam) if not use_scipy else float(st.poisson.cdf(k, lam))
                else:
                    return ToolResult(success=False, error="poisson không có pdf (dùng pmf)", return_code=1)
            else:  # exponential
                x = float(params.get("x", 0.0))
                lam = float(params.get("lambda", 1.0))
                if op == "pdf":
                    res = self._exponential_pdf(x, lam) if not use_scipy else float(st.expon.pdf(x, scale=1.0 / lam))
                elif op == "cdf":
                    res = self._exponential_cdf(x, lam) if not use_scipy else float(st.expon.cdf(x, scale=1.0 / lam))
                else:
                    return ToolResult(success=False, error="exponential không có pmf (dùng pdf)", return_code=1)

            return ToolResult(
                success=True,
                output=str(res),
                metadata={
                    "distribution": dist,
                    "operation": op,
                    "params": params,
                    "result": res,
                    "backend": "scipy.stats" if use_scipy else "stdlib-math",
                },
            )
        except (ValueError, TypeError, KeyError) as e:
            return ToolResult(success=False, error=f"Invalid params: {e}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=f"Compute failed: {e}", return_code=1)
