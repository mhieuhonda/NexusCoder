"""
Code Transpiler Tool - Transpile cơ bản giữa Python/JS/Rust.
Author: Hieu Louis (2026)

Hỗ trợ:
- python → javascript (AST-based, subset: function/return/assign/print/expr/binop)
- python → rust       (AST-based, subset: fn/let/return/print)
- javascript → python (regex-based, subset: function/var-let-const/console.log)

Đây là transpiler đơn giản, KHÔNG phải production-grade. Chỉ minh hoạ
ý tưởng AST transformation và regex substitution.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


SUPPORTED_PAIRS = {
    "python_to_javascript",
    "python_to_rust",
    "javascript_to_python",
}


# ---------------------------------------------------------------------------
# Python → JavaScript
# ---------------------------------------------------------------------------
class _PyToJSVisitor(ast.NodeVisitor):
    """Transpile Python AST subset → JavaScript."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self._indent = 0

    def _pad(self) -> str:
        return "  " * self._indent

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        params = ", ".join(a.arg for a in node.args.args)
        self.lines.append(f"{self._pad()}function {node.name}({params}) {{")
        self._indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self._indent -= 1
        self.lines.append(f"{self._pad()}}}")

    def visit_Return(self, node: ast.Return) -> None:
        val = self._expr(node.value) if node.value else ""
        self.lines.append(f"{self._pad()}return {val};")

    def visit_Assign(self, node: ast.Assign) -> None:
        target = self._expr(node.targets[0])
        val = self._expr(node.value)
        self.lines.append(f"{self._pad()}let {target} = {val};")

    def visit_Expr(self, node: ast.Expr) -> None:
        if (isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "print"):
            args = ", ".join(self._expr(a) for a in node.value.args)
            self.lines.append(f"{self._pad()}console.log({args});")
        else:
            self.lines.append(f"{self._pad()}{self._expr(node.value)};")

    def _expr(self, node: Optional[ast.AST]) -> str:
        if node is None:
            return ""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return f'"{node.value}"'
            return repr(node.value)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.BinOp):
            return f"({self._expr(node.left)} {self._op(node.op)} {self._expr(node.right)})"
        if isinstance(node, ast.Call):
            args = ", ".join(self._expr(a) for a in node.args)
            return f"{self._expr(node.func)}({args})"
        return ast.unparse(node)

    @staticmethod
    def _op(op: ast.AST) -> str:
        if isinstance(op, ast.Add): return "+"
        if isinstance(op, ast.Sub): return "-"
        if isinstance(op, ast.Mult): return "*"
        if isinstance(op, ast.Div): return "/"
        if isinstance(op, ast.Mod): return "%"
        return "?"


# ---------------------------------------------------------------------------
# Python → Rust
# ---------------------------------------------------------------------------
class _PyToRustVisitor(ast.NodeVisitor):
    """Transpile Python AST subset → Rust (i32 typed)."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self._indent = 0

    def _pad(self) -> str:
        return "  " * self._indent

    def visit_Module(self, node: ast.Module) -> None:
        self.lines.append("fn main() {")
        self._indent += 1
        for stmt in node.body:
            if not isinstance(stmt, ast.FunctionDef):
                self.visit(stmt)
        self._indent -= 1
        self.lines.append("}")
        # Top-level functions after main
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        params = ", ".join(f"{a.arg}: i32" for a in node.args.args)
        self.lines.append(f"fn {node.name}({params}) -> i32 {{")
        self._indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self._indent -= 1
        self.lines.append(f"{self._pad()}}}")

    def visit_Return(self, node: ast.Return) -> None:
        val = self._expr(node.value) if node.value else "0"
        self.lines.append(f"{self._pad()}return {val};")

    def visit_Assign(self, node: ast.Assign) -> None:
        target = self._expr(node.targets[0])
        val = self._expr(node.value)
        self.lines.append(f"{self._pad()}let {target} = {val};")

    def visit_Expr(self, node: ast.Expr) -> None:
        if (isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "print"):
            args = ", ".join(self._expr(a) for a in node.value.args)
            self.lines.append(f'{self._pad()}println!("{{:?}}", {args});')
        else:
            self.lines.append(f"{self._pad()}{self._expr(node.value)};")

    def _expr(self, node: Optional[ast.AST]) -> str:
        if node is None:
            return ""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return f'"{node.value}"'
            return repr(node.value)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.BinOp):
            return f"({self._expr(node.left)} {self._op(node.op)} {self._expr(node.right)})"
        if isinstance(node, ast.Call):
            args = ", ".join(self._expr(a) for a in node.args)
            return f"{self._expr(node.func)}({args})"
        return ast.unparse(node)

    @staticmethod
    def _op(op: ast.AST) -> str:
        if isinstance(op, ast.Add): return "+"
        if isinstance(op, ast.Sub): return "-"
        if isinstance(op, ast.Mult): return "*"
        if isinstance(op, ast.Div): return "/"
        if isinstance(op, ast.Mod): return "%"
        return "?"


# ---------------------------------------------------------------------------
# JavaScript → Python (regex)
# ---------------------------------------------------------------------------
def _js_to_python(code: str) -> str:
    """Transpile JS subset → Python bằng regex."""
    out = code
    # function name(params) {  →  def name(params):
    out = re.sub(r"function\s+(\w+)\s*\(([^)]*)\)\s*{", r"def \1(\2):", out)
    # var/let/const x = ...;  →  x = ...
    out = re.sub(r"\b(?:var|let|const)\s+(\w+)\s*=\s*", r"\1 = ", out)
    # console.log(...) → print(...)
    out = re.sub(r"console\.log\s*\(", "print(", out)
    # return ...; → return ...
    out = re.sub(r"return\s+(.*?);", r"return \1", out)
    # Loại bỏ dấu ; cuối dòng
    out = re.sub(r";\s*$", "", out, flags=re.MULTILINE)
    # Đóng block: } → pass (đơn giản)
    out = re.sub(r"^\s*}\s*$", "    pass", out, flags=re.MULTILINE)
    return out


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
class CodeTranspilerTool(Tool):
    """Transpile giữa Python/JS/Rust (basic AST + regex)."""

    category = ToolCategory.CODE
    safety = ToolSafety.MODERATE
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "code_transpiler"

    @property
    def description(self) -> str:
        return (
            "Transpile code giữa Python/JavaScript/Rust (basic, AST + regex). "
            "Hỗ trợ: python→javascript, python→rust, javascript→python."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Source code cần transpile"},
                "from_lang": {
                    "type": "string",
                    "enum": ["python", "javascript"],
                    "description": "Ngôn ngữ nguồn",
                },
                "to_lang": {
                    "type": "string",
                    "enum": ["python", "javascript", "rust"],
                    "description": "Ngôn ngữ đích",
                },
            },
            "required": ["code", "from_lang", "to_lang"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("code"):
            return "Missing required arg: code"
        fl = args.get("from_lang")
        tl = args.get("to_lang")
        if not fl:
            return "Missing required arg: from_lang"
        if not tl:
            return "Missing required arg: to_lang"
        pair = f"{fl}_to_{tl}"
        if pair not in SUPPORTED_PAIRS:
            return (
                f"Unsupported transpile pair: {pair}. "
                f"Hỗ trợ: {sorted(SUPPORTED_PAIRS)}"
            )
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        code: str = args["code"]
        fl: str = args["from_lang"]
        tl: str = args["to_lang"]
        try:
            if fl == "python" and tl == "javascript":
                tree = ast.parse(code)
                v = _PyToJSVisitor()
                v.visit(tree)
                transpiled = "\n".join(v.lines)
            elif fl == "python" and tl == "rust":
                tree = ast.parse(code)
                v = _PyToRustVisitor()
                v.visit(tree)
                transpiled = "\n".join(v.lines)
            elif fl == "javascript" and tl == "python":
                transpiled = _js_to_python(code)
            else:
                return ToolResult(success=False, error="Unsupported pair", return_code=1)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"SyntaxError: {e.msg} (line {e.lineno})",
                return_code=1,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}", return_code=1)

        return ToolResult(
            success=True,
            output=transpiled,
            metadata={
                "from_lang": fl,
                "to_lang": tl,
                "input_length": len(code),
                "output_length": len(transpiled),
            },
        )
