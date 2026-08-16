"""Bug Reproduction Skill - Minimal Reproducible Example (MRE) framework.

Sinh khung reproduce bug: isolation steps, environment snapshot,
minimal repro script, và bisect strategy cho Git history.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


REPRO_TEMPLATE = """
# Bug Reproduction Report / Báo cáo tái lập bug

**Bug ID:** {bug_id}
**Title:** {title}
**Severity:** {severity}    (blocker / critical / major / minor / trivial)
**Reported:** {reported_at}

## 1. Environment Snapshot / Môi trường
- OS:        `uname -a`
- Runtime:   python --version  (or node -v / go version)
- Dependencies:
    pip freeze > requirements-bug.txt     # pin exact versions
- Repo state:
    git rev-parse HEAD
    git status --short
    git log -1 --format='%H %s'

## 2. Preconditions / Điều kiện tiên quyết
- ...
- ...

## 3. Steps to Reproduce / Các bước tái lập
1.
2.
3.

## 4. Expected vs. Actual / Kỳ vọng vs Thực tế
- Expected:
- Actual:

## 5. Minimal Reproducible Example (MRE) / Ví dụ tối thiểu
- Strip everything unrelated to the bug.
- Hard-code inputs (no DB / network if possible).
- Target ≤ 50 lines.

## 6. Frequency / Tần suất
- Always | Intermittent (x% of runs) | Only on CI

## 7. Logs / Traces
- Stack trace, stderr, screenshots, profiler output attached

## 8. Suspected Root Cause / Nghi ngờ nguyên nhân
- ...

## 9. Workaround / Tạm thời
- ...
"""

MRE_PYTHON = '''"""MRE: <one-line bug summary>."""
from __future__ import annotations
import sys, platform, traceback

print(f"python={sys.version} | os={platform.platform()}")

def repro() -> None:
    """Reproduce the bug deterministically."""
    # --- Arrange --- minimal setup with hard-coded inputs
    data = [1, 2, 3, None, 5]

    # --- Act --- the smallest call that triggers the bug
    try:
        result = sum(x or 0 for x in data)
    except Exception:
        traceback.print_exc()
        return

    # --- Assert --- what should happen vs what actually happens
    expected = 11
    assert result == expected, f"BUG: got {result}, expected {expected}"
    print("No bug reproduced — adjust inputs / version.")

if __name__ == "__main__":
    repro()
'''

BISECT_SCRIPT = '''#!/usr/bin/env bash
# git bisect driver — exit 0=good, 1=bad, 125=skip
# Usage: git bisect start BAD GOOD -- && git bisect run ./bisect.sh
set -euo pipefail
python -m pytest tests/test_repro.py -q || exit 1
exit 0
'''


class BugReproductionSkill(Skill):
    """Tạo khung Minimal Reproducible Example cho bug reports."""

    category = SkillCategory.CODE
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "bug", "reproduce", "repro", "mre", "minimal example",
        "minimal reproducible", "regression", "regression test",
        "bisect", "stack trace", "traceback",
    ]
    examples = [
        "Tôi gặp bug X khi chạy Y, giúp tạo repro",
        "Reproduce bug từ stack trace này",
        "Tạo minimal example cho crash",
    ]

    @property
    def name(self) -> str:
        return "bug_reproduction"

    @property
    def description(self) -> str:
        return (
            "Sinh khung Minimal Reproducible Example (MRE): isolation steps, "
            "environment snapshot, repro script, và git bisect strategy."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.22
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        bug_id = context.metadata.get("bug_id", "BUG-001")
        title = context.metadata.get("title", (context.prompt or "")[:80])
        severity = context.metadata.get("severity", "major")
        reported_at = context.metadata.get("reported_at", "2026-01-15")

        report = REPRO_TEMPLATE.format(
            bug_id=bug_id, title=title, severity=severity, reported_at=reported_at
        )

        artifacts: List[Dict[str, str]] = [
            {"name": "BUG_REPORT.md", "language": "markdown", "content": report},
            {"name": "repro.py", "language": "python", "content": MRE_PYTHON},
            {"name": "bisect.sh", "language": "bash", "content": BISECT_SCRIPT},
        ]

        return SkillResult(
            success=True,
            output=(
                f"[bug_reproduction] bug_id={bug_id} severity={severity}\n"
                f"Generated MRE framework: report + repro.py + bisect.sh"
            ),
            artifacts=artifacts,
            suggestions=[
                "Reduce the repro script until removing any line stops the bug",
                "Add `pytest -p no:randomly` if order-dependent",
                "Use `git bisect run ./bisect.sh` to localize the regression commit",
                "Attach heap profilers (tracemalloc / memray) for memory bugs",
                "If flaky: run 100× with `pytest --repeats 100` to estimate frequency",
            ],
            metadata={
                "skill": self.name,
                "bug_id": bug_id,
                "severity": severity,
                "title": title,
                "has_bisect": True,
                "version": self.version,
                "author": self.author,
            },
        )
