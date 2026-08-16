"""
Code Coverage Tool - Run tests với coverage bằng coverage.py.
Author: Hieu Louis (2026)

Lazy import `coverage` package. Trả về coverage %, uncovered lines per file.
Hỗ trợ 2 report formats: text (default, parseable) và json (structured).

DANGEROUS (executes tests), requires_confirmation.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class CodeCoverageTool(Tool):
    """Run tests với coverage.py. Trả về coverage % và uncovered lines."""

    category = ToolCategory.EXEC
    safety = ToolSafety.DANGEROUS  # executes tests
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "code_coverage"

    @property
    def description(self) -> str:
        return (
            "Run tests với coverage.py. Trả về coverage %, uncovered lines per file. "
            "Yêu cầu `coverage` package và pytest. Hỗ trợ text/json report."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Thư mục source để đo coverage"},
                "module": {
                    "type": "string",
                    "description": "Test module/path (vd 'tests/' hoặc 'test_x.py'). Mặc định 'tests/'.",
                },
                "source": {
                    "type": "string",
                    "description": "Package để đo coverage (mặc định = path)",
                },
                "report_format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "default": "text",
                    "description": "Report format",
                },
            },
            "required": ["path"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("path"):
            return "Missing required arg: path"
        fmt = args.get("report_format", "text")
        if fmt not in {"text", "json"}:
            return f"Unsupported report_format: {fmt}. Chọn: ['text', 'json']"
        return None

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path: str = args["path"]
        module: str = args.get("module", "tests/")
        source: str = args.get("source", path)
        report_format: str = args.get("report_format", "text")

        if not os.path.exists(path):
            return ToolResult(success=False, error=f"Path không tồn tại: {path}", return_code=1)

        # Lazy import coverage
        try:
            import coverage  # type: ignore
        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"coverage not installed: {e}. Cài: pip install coverage",
                return_code=127,
            )

        if context.dry_run:
            return ToolResult(
                success=True,
                output=(
                    f"[dry-run] Sẽ chạy: coverage run --source={source} -m pytest {module}, "
                    f"sau đó report (format={report_format})"
                ),
                metadata={"dry_run": True, "source": source, "module": module, "format": report_format},
            )

        # Locate pytest executable
        pytest_cmd = shutil.which("pytest")
        if not pytest_cmd:
            pytest_cmd = shutil.which("python3")
        if not pytest_cmd:
            return ToolResult(
                success=False,
                error="pytest không tìm thấy trong PATH. Cài: pip install pytest",
                return_code=127,
            )

        # Build pytest command
        full_cmd: List[str] = [pytest_cmd]
        if pytest_cmd.endswith("python3"):
            full_cmd += ["-m", "pytest"]
        full_cmd += ["-q", module]

        # Run tests under coverage
        cov = coverage.Coverage(source=[source])
        cov.start()
        try:
            try:
                cp = subprocess.run(
                    full_cmd,
                    capture_output=True,
                    text=True,
                    timeout=context.timeout,
                    cwd=context.working_dir or path,
                    env={**os.environ, **context.env},
                )
            except subprocess.TimeoutExpired:
                # finally block below sẽ chạy cov.stop()/cov.save()
                return ToolResult(
                    success=False,
                    error=f"Test timeout ({context.timeout}s)",
                    return_code=124,
                    metadata={"command": " ".join(full_cmd)},
                )
        finally:
            cov.stop()
            cov.save()

        # Build report
        try:
            if report_format == "json":
                buf = io.StringIO()
                # cov.json_report writes to a file or stdout-like stream
                cov.json_report(outfile=buf)  # type: ignore[arg-type]
                report_str = buf.getvalue()
                try:
                    report_data = json.loads(report_str)
                except json.JSONDecodeError:
                    return ToolResult(
                        success=False,
                        error="Không parse được JSON coverage report",
                        return_code=1,
                    )
                totals = report_data.get("totals", {})
                pct = float(totals.get("percent_covered", 0.0))
                files_info: List[Dict[str, Any]] = []
                for fname, finfo in report_data.get("files", {}).items():
                    summary = finfo.get("summary", {})
                    files_info.append({
                        "file": fname,
                        "covered": summary.get("covered_lines", 0),
                        "missing": summary.get("missing_lines", 0),
                        "percent": round(summary.get("percent_covered", 0.0), 2),
                        "missing_lines": finfo.get("missing_lines", []),
                    })
                output = json.dumps(
                    {"totals": totals, "files": files_info},
                    indent=2, ensure_ascii=False,
                )
            else:
                buf = io.StringIO()
                cov.report(file=buf)
                output = buf.getvalue()
                # Parse TOTAL line for summary %
                pct = 0.0
                for line in output.splitlines():
                    if line.strip().startswith("TOTAL"):
                        parts = line.split()
                        if parts:
                            try:
                                pct = float(parts[-1].rstrip("%"))
                            except ValueError:
                                pass
        except Exception as e:
            return ToolResult(success=False, error=f"Report lỗi: {e}", return_code=1)

        return ToolResult(
            success=(cp.returncode == 0),
            output=output,
            error=cp.stderr if cp.stderr else None,
            return_code=cp.returncode,
            metadata={
                "command": " ".join(full_cmd),
                "source": source,
                "module": module,
                "coverage_pct": round(pct, 2),
                "report_format": report_format,
                "test_returncode": cp.returncode,
            },
        )
