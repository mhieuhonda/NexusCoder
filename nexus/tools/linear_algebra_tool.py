"""
Linear Algebra Tool - Các phép toán đại số tuyến tính.
===========================================
Dùng numpy lazy import. Hỗ trợ: dot, matmul, det, inv, eigen, svd, norm,
transpose, trace, solve, rank, qr.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


OPERATIONS = {
    "dot", "matmul", "det", "inv", "eigen", "svd", "norm", "transpose",
    "trace", "solve", "rank", "qr", "lu", "identity", "ones", "zeros",
}


def _to_matrix(data: Any) -> List[List[float]]:
    """Ép list[list|list-of-rows] → list[list[float]]. / Coerce to 2D float matrix."""
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"matrix phải là list, got {type(data).__name__}")
    if not data:
        return []
    # 1D → 2D row vector / promote 1D to row vector
    if not isinstance(data[0], (list, tuple)):
        return [[float(x) for x in data]]
    return [[float(x) for x in row] for row in data]


class LinearAlgebraTool(Tool):
    """Đại số tuyến tính: dot, matmul, det, inv, eigen, svd, norm, ..."""

    category = ToolCategory.MATH
    safety = ToolSafety.SAFE

    @property
    def name(self) -> str:
        return "linear_algebra"

    @property
    def description(self) -> str:
        return "Linear algebra: dot/matmul/det/inv/eigen/svd/norm/solve/qr/lu (numpy backend)."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "matrix_a": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "matrix_b": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "operation": {
                    "type": "string",
                    "enum": sorted(OPERATIONS),
                    "default": "matmul",
                },
                "params": {"type": "object"},
            },
            "required": ["operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        op = args.get("operation", "matmul")
        if op not in OPERATIONS:
            return f"Invalid operation='{op}'. Supported: {sorted(OPERATIONS)}"
        # Phép toán 2 ngôi / binary ops require matrix_b
        if op in ("matmul", "dot", "solve") and not args.get("matrix_b"):
            return f"Missing required arg: matrix_b (cho operation='{op}')"
        # Phép 1 ngôi cần matrix_a / unary ops require matrix_a
        if op not in ("identity", "ones", "zeros") and not args.get("matrix_a"):
            return f"Missing required arg: matrix_a (cho operation='{op}')"
        if op in ("identity", "ones", "zeros"):
            params = args.get("params", {}) or {}
            if not params.get("shape"):
                return f"Missing required param: params.shape (cho operation='{op}')"
        return None

    def _to_serializable(self, obj: Any) -> Any:
        """Convert numpy object sang JSON-safe (list/number). / Convert numpy → JSON-safe."""
        try:
            import numpy as np  # type: ignore
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
        except ImportError:
            pass
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                pass
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(x) for x in obj]
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return obj

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        op = args.get("operation", "matmul")
        params: Dict[str, Any] = args.get("params", {}) or {}

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] linear_algebra op='{op}'",
                metadata={"operation": op, "dry_run": True},
            )

        try:
            import numpy as np  # type: ignore
        except ImportError:
            return ToolResult(
                success=False,
                error="numpy chưa cài. Cài đặt: pip install numpy",
                return_code=127,
            )

        try:
            # Parse matrices / parse matrices
            if op in ("identity", "ones", "zeros"):
                shape = params["shape"]
                if isinstance(shape, int):
                    shape = (shape, shape)
                if op == "identity":
                    result: Any = np.identity(int(shape[0])).tolist()
                elif op == "ones":
                    result = np.ones(tuple(int(s) for s in shape)).tolist()
                else:
                    result = np.zeros(tuple(int(s) for s in shape)).tolist()
                return ToolResult(success=True, output=str(result), metadata={"operation": op, "result": result})

            A = np.array(_to_matrix(args["matrix_a"]), dtype=float)
            B = None
            if args.get("matrix_b"):
                B = np.array(_to_matrix(args["matrix_b"]), dtype=float)

            if op in ("matmul", "dot"):
                if B is None:
                    return ToolResult(success=False, error=f"matrix_b required for {op}", return_code=1)
                result = A @ B if op == "matmul" else np.dot(A, B)
            elif op == "det":
                result = float(np.linalg.det(A))
            elif op == "inv":
                result = np.linalg.inv(A)
            elif op == "eigen":
                # Trả về values + vectors / return both eigenvalues and eigenvectors
                w, v = np.linalg.eig(A)
                result = {"eigenvalues": self._to_serializable(w), "eigenvectors": self._to_serializable(v)}
            elif op == "svd":
                U, S, Vt = np.linalg.svd(A)
                result = {
                    "U": self._to_serializable(U),
                    "singular_values": self._to_serializable(S),
                    "Vt": self._to_serializable(Vt),
                }
            elif op == "norm":
                ord_val = params.get("ord", "fro")  # type: ignore[assignment]
                result = float(np.linalg.norm(A, ord=ord_val))
            elif op == "transpose":
                result = A.T
            elif op == "trace":
                result = float(np.trace(A))
            elif op == "solve":
                # Solve Ax = B cho x / solve linear system
                if B is None:
                    return ToolResult(success=False, error="matrix_b required for solve", return_code=1)
                if B.ndim == 1:
                    result = np.linalg.solve(A, B)
                else:
                    result = np.linalg.solve(A, B)
            elif op == "rank":
                result = int(np.linalg.matrix_rank(A))
            elif op == "qr":
                Q, R = np.linalg.qr(A)
                result = {"Q": self._to_serializable(Q), "R": self._to_serializable(R)}
            elif op == "lu":
                try:
                    from scipy.linalg import lu  # type: ignore
                    P, L, U = lu(A)
                    result = {
                        "P": self._to_serializable(P),
                        "L": self._to_serializable(L),
                        "U": self._to_serializable(U),
                    }
                except ImportError:
                    return ToolResult(
                        success=False,
                        error="scipy chưa cài cho lu decomposition. Cài đặt: pip install scipy",
                        return_code=127,
                    )
            else:
                return ToolResult(success=False, error=f"Unknown operation: {op}", return_code=1)

            serializable = self._to_serializable(result)
            return ToolResult(
                success=True,
                output=str(serializable),
                metadata={
                    "operation": op,
                    "shape_a": list(A.shape),
                    "shape_b": list(B.shape) if B is not None else None,
                    "result": serializable,
                },
            )
        except np.linalg.LinAlgError as e:
            return ToolResult(success=False, error=f"LinAlgError: {e}", return_code=1)
        except ValueError as e:
            return ToolResult(success=False, error=f"ValueError: {e}", return_code=1)
        except Exception as e:
            return ToolResult(success=False, error=f"Compute failed: {e}", return_code=1)
