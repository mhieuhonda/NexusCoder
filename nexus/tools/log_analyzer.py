"""
Log Analyzer Tool - Phân tích log files: filter, count, top errors, exceptions.
===========================================
Pure stdlib (re + collections.Counter). Hỗ trợ log levels chuẩn
(DEBUG/INFO/WARN/ERROR/FATAL) và pattern matching linh hoạt.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


LEVELS = {"DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL", "TRACE"}
OPERATIONS = {"filter", "count", "top_errors", "extract_exceptions", "summary", "tail", "head"}


# Pattern bắt log level thường gặp / common log-line level pattern
_LEVEL_RE = re.compile(r"\b(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL|TRACE)\b", re.IGNORECASE)
# Pattern bắt exception/stacktrace / exception stacktrace pattern
_EXCEPTION_RE = re.compile(
    r"^(?:Traceback|Caused by:|^\s+at\s+|^\s*File\s+|^\s*\.\.\.|"
    r"[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Fault|Failure|Warning):)",
    re.MULTILINE,
)
# Pattern bắt message lỗi / error message capture
_ERROR_MSG_RE = re.compile(
    r"((?:[A-Za-z_][\w.]*)(?:Error|Exception|Fault|Failure|Warning))(?::\s*(.*))?",
    re.IGNORECASE,
)


class LogAnalyzerTool(Tool):
    """Phân tích log files: filter/count/top_errors/extract_exceptions/summary."""

    category = ToolCategory.MONITOR
    safety = ToolSafety.SAFE

    @property
    def name(self) -> str:
        return "log_analyzer"

    @property
    def description(self) -> str:
        return "Analyze log files: filter by level, count, top errors, extract exceptions, summary."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Đường dẫn log file"},
                "operation": {
                    "type": "string",
                    "enum": sorted(OPERATIONS),
                    "default": "summary",
                },
                "level": {
                    "type": "string",
                    "enum": sorted(LEVELS),
                    "description": "Log level để filter (cho operation='filter')",
                },
                "min_level": {
                    "type": "string",
                    "enum": sorted(LEVELS),
                    "description": "Lọc từ level này trở lên (mức nghiêm trọng tăng dần)",
                },
                "pattern": {"type": "string", "description": "Regex pattern để match line"},
                "last_n": {"type": "integer", "default": 100, "description": "Cho tail/head"},
                "top_k": {"type": "integer", "default": 10, "description": "Cho top_errors"},
            },
            "required": ["path", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("path"):
            return "Missing required arg: path"
        op = args.get("operation", "summary")
        if op not in OPERATIONS:
            return f"Invalid operation='{op}'. Supported: {sorted(OPERATIONS)}"
        if op == "filter" and not args.get("level") and not args.get("min_level") and not args.get("pattern"):
            return "filter cần ít nhất 1 trong: level, min_level, pattern"
        return None

    # ---- Helpers --------------------------------------------------------

    def _level_priority(self, level: str) -> int:
        """Trả về thứ tự nghiêm trọng của level. / Severity rank of level."""
        norm = level.upper()
        if norm in ("TRACE", "DEBUG"):
            return 0
        if norm == "INFO":
            return 1
        if norm in ("WARN", "WARNING"):
            return 2
        if norm == "ERROR":
            return 3
        if norm in ("FATAL", "CRITICAL"):
            return 4
        return -1

    def _detect_level(self, line: str) -> Optional[str]:
        m = _LEVEL_RE.search(line)
        if not m:
            return None
        return m.group(1).upper().replace("WARNING", "WARN")

    def _read_lines(self, path: str) -> List[str]:
        """Đọc file log (xử lý encoding linh hoạt). / Read log file with encoding fallback."""
        # Thử utf-8, fallback latin-1 (không bao giờ lỗi) / utf-8 first, latin-1 fallback
        for enc in ("utf-8", "latin-1"):
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.readlines()
            except UnicodeDecodeError:
                continue
        # Fallback cuối / last-resort
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()

    # ---- Operations -----------------------------------------------------

    def _op_count(self, lines: List[str]) -> Dict[str, Any]:
        counts: Counter = Counter()
        no_level = 0
        for line in lines:
            lvl = self._detect_level(line)
            if lvl:
                counts[lvl] += 1
            else:
                no_level += 1
        return {
            "counts": dict(counts),
            "no_level": no_level,
            "total_lines": len(lines),
            "total_with_level": sum(counts.values()),
        }

    def _op_filter(
        self,
        lines: List[str],
        level: Optional[str],
        min_level: Optional[str],
        pattern: Optional[str],
    ) -> Dict[str, Any]:
        regex = re.compile(pattern) if pattern else None
        threshold = self._level_priority(min_level) if min_level else None
        target = level.upper().replace("WARNING", "WARN") if level else None
        matched: List[str] = []
        for line in lines:
            lvl = self._detect_level(line)
            if target and lvl != target:
                continue
            if threshold is not None:
                p = self._level_priority(lvl) if lvl else -1
                if p < threshold:
                    continue
            if regex and not regex.search(line):
                continue
            matched.append(line.rstrip("\n"))
        return {
            "matched_count": len(matched),
            "sample_lines": matched[:50],  # Giới hạn output / limit output
            "filter": {"level": target, "min_level": min_level, "pattern": pattern},
        }

    def _op_top_errors(self, lines: List[str], top_k: int) -> Dict[str, Any]:
        """Đếm top exception types và message patterns."""
        exception_counts: Counter = Counter()
        error_line_counts: Counter = Counter()
        for line in lines:
            m = _ERROR_MSG_RE.search(line)
            if m:
                exc_type = m.group(1)
                exception_counts[exc_type] += 1
                # Normalize message: bỏ số / digits, rút gọn
                msg = (m.group(2) or "").strip()
                # Bỏ số, path, ID / strip digits, paths, IDs
                norm = re.sub(r"\d+", "N", msg)
                norm = re.sub(r"/[\w/\.]+", "/path", norm)
                norm = re.sub(r"\s+", " ", norm)[:120]
                if norm:
                    error_line_counts[f"{exc_type}: {norm}"] += 1
        return {
            "top_exception_types": exception_counts.most_common(top_k),
            "top_error_messages": error_line_counts.most_common(top_k),
        }

    def _op_extract_exceptions(self, lines: List[str]) -> Dict[str, Any]:
        """Trích các block stacktrace (từ 'Traceback' đến khi gặp dòng trống/log-level)."""
        blocks: List[Dict[str, Any]] = []
        current: List[str] = []
        for line in lines:
            if line.startswith("Traceback") or line.startswith("Caused by:"):
                if current:
                    blocks.append({"lines": current, "preview": "".join(current[:3]).strip()})
                current = [line.rstrip("\n")]
            elif current and (
                line.startswith("  ")
                or line.startswith("\t")
                or _EXCEPTION_RE.match(line)
                or not line.strip()
            ):
                current.append(line.rstrip("\n"))
                if not line.strip() and len(current) > 3:
                    # Kết thúc block khi gặp dòng trống / end on blank line
                    blocks.append({"lines": current, "preview": "".join(current[:3]).strip()})
                    current = []
            elif current:
                blocks.append({"lines": current, "preview": "".join(current[:3]).strip()})
                current = []
        if current:
            blocks.append({"lines": current, "preview": "".join(current[:3]).strip()})
        return {
            "exception_blocks": len(blocks),
            "previews": [b["preview"][:200] for b in blocks[:20]],
        }

    def _op_summary(self, lines: List[str]) -> Dict[str, Any]:
        """Tổng hợp nhanh file log."""
        counts = self._op_count(lines)
        size_bytes = sum(len(l.encode("utf-8", errors="replace")) for l in lines)
        # Phát hiện timestamp đầu & cuối / detect first/last timestamps
        ts_re = re.compile(r"(\d{4}-\d{2}-\d{2}[\dT\s:.:+\-]+\d{2}:\d{2}:\d{2})")
        first_ts = last_ts = None
        for line in lines[:200]:
            m = ts_re.search(line)
            if m:
                first_ts = m.group(1)
                break
        for line in reversed(lines[-200:]):
            m = ts_re.search(line)
            if m:
                last_ts = m.group(1)
                break
        return {
            **counts,
            "file_size_bytes": size_bytes,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
        }

    # ---- Execute --------------------------------------------------------

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        path = args["path"]
        op = args.get("operation", "summary")

        if not os.path.exists(path):
            return ToolResult(success=False, error=f"Log file không tồn tại: {path}", return_code=1)
        if os.path.isdir(path):
            return ToolResult(success=False, error=f"Path là thư mục, không phải file: {path}", return_code=1)

        try:
            lines = self._read_lines(path)
        except Exception as e:
            return ToolResult(success=False, error=f"Read log failed: {e}", return_code=1)

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] op='{op}' trên {len(lines)} lines của {path}",
                metadata={"operation": op, "n_lines": len(lines), "dry_run": True},
            )

        try:
            if op == "count":
                result: Any = self._op_count(lines)
            elif op == "filter":
                result = self._op_filter(
                    lines,
                    args.get("level"),
                    args.get("min_level"),
                    args.get("pattern"),
                )
            elif op == "top_errors":
                result = self._op_top_errors(lines, int(args.get("top_k", 10)))
            elif op == "extract_exceptions":
                result = self._op_extract_exceptions(lines)
            elif op == "summary":
                result = self._op_summary(lines)
            elif op == "tail":
                n = int(args.get("last_n", 100))
                result = {"tail_lines": [l.rstrip("\n") for l in lines[-n:]], "count": n}
            elif op == "head":
                n = int(args.get("last_n", 100))
                result = {"head_lines": [l.rstrip("\n") for l in lines[:n]], "count": n}
            else:
                return ToolResult(success=False, error=f"Unknown operation: {op}", return_code=1)

            return ToolResult(
                success=True,
                output=str(result)[:5000],  # Truncate output lớn / truncate huge outputs
                metadata={
                    "operation": op,
                    "file": path,
                    "total_lines": len(lines),
                    "result": result,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Analysis failed: {e}", return_code=1)
