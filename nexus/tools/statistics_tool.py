"""
Statistics Tool - Tính toán thống kê mô tả & hồi quy.
===========================================
Dùng stdlib `statistics` + `math` (không cần numpy/scipy).

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


OPERATIONS = {
    "mean", "median", "mode", "multimode", "std", "var", "pvar", "min", "max",
    "range", "sum", "count", "quartiles", "percentile", "describe",
    "correlation", "covariance", "linear_regression", "zscore",
}


def _to_floats(data: Any) -> List[float]:
    """Ép list[Any] → list[float]. / Coerce list[Any] → list[float]."""
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"data phải là list, got {type(data).__name__}")
    return [float(x) for x in data]


class StatisticsTool(Tool):
    """Tính toán thống kê: mean/median/mode/std/var/percentiles/correlation/regression."""

    category = ToolCategory.MATH
    safety = ToolSafety.SAFE

    @property
    def name(self) -> str:
        return "statistics"

    @property
    def description(self) -> str:
        return "Thống kê mô tả (mean/median/mode/std/var/percentiles) + correlation + linear regression."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "array", "items": {"type": "number"}, "description": "Danh sách số"},
                "operation": {
                    "type": "string",
                    "enum": sorted(OPERATIONS),
                    "default": "describe",
                },
                "params": {
                    "type": "object",
                    "description": "p (percentile 0-100), data2 (cho correlation/covariance)",
                },
            },
            "required": ["data", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("data"):
            return "Missing required arg: data"
        op = args.get("operation", "describe")
        if op not in OPERATIONS:
            return f"Invalid operation='{op}'. Supported: {sorted(OPERATIONS)}"
        if op in ("correlation", "covariance") and not (args.get("params", {}) or {}).get("data2"):
            return f"Missing required param: params.data2 (cho operation='{op}')"
        if op == "percentile":
            p = (args.get("params", {}) or {}).get("p")
            if p is None:
                return "Missing required param: params.p (0-100) cho operation='percentile'"
            if not (0 <= float(p) <= 100):
                return "params.p phải nằm trong [0, 100]"
        return None

    # ---- Các phép toán / Operations ------------------------------------

    def _describe(self, data: List[float]) -> Dict[str, Any]:
        """Tính tất cả thống kê mô tả cùng lúc. / Compute all descriptive stats at once."""
        if not data:
            return {"error": "empty data"}
        sorted_data = sorted(data)
        result: Dict[str, Any] = {
            "count": len(data),
            "sum": sum(data),
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "min": sorted_data[0],
            "max": sorted_data[-1],
            "range": sorted_data[-1] - sorted_data[0],
        }
        if len(data) > 1:
            result["stdev"] = statistics.stdev(data)
            result["variance"] = statistics.variance(data)
            result["pvariance"] = statistics.pvariance(data)
        try:
            result["mode"] = statistics.mode(data)
        except statistics.StatisticsError:
            result["mode"] = None
        if len(data) >= 4:
            q = statistics.quantiles(data, n=4, method="inclusive")
            result["q1"] = q[0]
            result["q3"] = q[2]
            result["iqr"] = q[2] - q[0]
        return result

    def _percentile(self, data: List[float], p: float) -> float:
        """Tính percentile p (0-100) dùng linear interpolation (NumPy default)."""
        if not data:
            raise ValueError("empty data")
        s = sorted(data)
        if len(s) == 1:
            return float(s[0])
        rank = (p / 100.0) * (len(s) - 1)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return float(s[lo])
        # Interpolation / interpolate between two nearest values
        frac = rank - lo
        return float(s[lo] * (1 - frac) + s[hi] * frac)

    def _linear_regression(self, xs: List[float], ys: List[float]) -> Dict[str, Any]:
        """Hồi quy tuyến tính y = a*x + b. / Simple linear regression."""
        if len(xs) != len(ys) or len(xs) < 2:
            raise ValueError("Cần ít nhất 2 cặp (x,y) cùng độ dài")
        # Dùng stdlib statistics.linear_regression (Python 3.10+)
        try:
            slope, intercept = statistics.linear_regression(xs, ys)
        except AttributeError:
            # Fallback manual / manual computation
            n = len(xs)
            mean_x, mean_y = sum(xs) / n, sum(ys) / n
            num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            den = sum((x - mean_x) ** 2 for x in xs) or 1e-12
            slope = num / den
            intercept = mean_y - slope * mean_x
        # R² / R-squared
        ss_tot = sum((y - sum(ys) / len(ys)) ** 2 for y in ys) or 1e-12
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r_squared = 1 - (ss_res / ss_tot)
        return {"slope": slope, "intercept": intercept, "r_squared": r_squared}

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        op = args.get("operation", "describe")
        params: Dict[str, Any] = args.get("params", {}) or {}
        try:
            data = _to_floats(args["data"])
        except (ValueError, TypeError) as e:
            return ToolResult(success=False, error=f"Invalid data: {e}", return_code=1)

        try:
            if op == "mean":
                res: Any = statistics.mean(data)
            elif op == "median":
                res = statistics.median(data)
            elif op == "mode":
                res = statistics.mode(data)
            elif op == "multimode":
                res = statistics.multimode(data)
            elif op == "std":
                res = statistics.stdev(data) if len(data) > 1 else 0.0
            elif op == "var":
                res = statistics.variance(data) if len(data) > 1 else 0.0
            elif op == "pvar":
                res = statistics.pvariance(data)
            elif op == "min":
                res = min(data)
            elif op == "max":
                res = max(data)
            elif op == "range":
                res = max(data) - min(data)
            elif op == "sum":
                res = sum(data)
            elif op == "count":
                res = len(data)
            elif op == "quartiles":
                if len(data) < 4:
                    return ToolResult(success=False, error="Cần ≥4 samples cho quartiles", return_code=1)
                q = statistics.quantiles(data, n=4, method="inclusive")
                res = {"q1": q[0], "q2": q[1], "q3": q[2], "iqr": q[2] - q[0]}
            elif op == "percentile":
                res = self._percentile(data, float(params["p"]))
            elif op == "describe":
                res = self._describe(data)
            elif op == "correlation":
                data2 = _to_floats(params["data2"])
                if len(data) != len(data2):
                    return ToolResult(success=False, error="data và data2 phải cùng độ dài", return_code=1)
                res = statistics.correlation(data, data2)
            elif op == "covariance":
                data2 = _to_floats(params["data2"])
                if len(data) != len(data2):
                    return ToolResult(success=False, error="data và data2 phải cùng độ dài", return_code=1)
                res = statistics.covariance(data, data2)
            elif op == "linear_regression":
                data2 = _to_floats(params.get("data2", params.get("ys", [])))
                res = self._linear_regression(data, data2)
            elif op == "zscore":
                mean_val = statistics.mean(data)
                std_val = statistics.stdev(data) if len(data) > 1 else 0.0
                if std_val == 0:
                    res = [0.0 for _ in data]
                else:
                    res = [(x - mean_val) / std_val for x in data]
            else:
                return ToolResult(success=False, error=f"Unknown operation: {op}", return_code=1)

            return ToolResult(
                success=True,
                output=str(res),
                metadata={"operation": op, "count": len(data), "result": res},
            )
        except statistics.StatisticsError as e:
            return ToolResult(success=False, error=f"StatisticsError: {e}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=f"Compute failed: {e}", return_code=1)
