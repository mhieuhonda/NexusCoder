"""Release Management Skill - Semantic versioning + changelog + release notes.

Sinh chiến lược versioning (semver), changelog (Keep-a-Changelog),
và artifact release notes từ git history.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


SEMVER_RULES = """
Semantic Versioning 2.0.0 — Quick Rules / Tóm tắt
=================================================
Format: MAJOR.MINOR.PATCH  (e.g. 1.4.2)

- MAJOR : bump khi có BREAKING CHANGE (incompatible API changes)
- MINOR : bump khi thêm feature backward-compatible
- PATCH : bump khi fix bug backward-compatible

Pre-release : 1.0.0-alpha.1, 1.0.0-beta.2, 1.0.0-rc.1
Build meta  : 1.0.0+20260115

Decision tree / Cây quyết định:
  1. API breaking?            → MAJOR++  (reset MINOR=0, PATCH=0)
  2. New feature, old API OK? → MINOR++  (reset PATCH=0)
  3. Bug fix only?            → PATCH++
  4. Unstable/preview?        → append -alpha.N / -beta.N / -rc.N
"""

CHANGELOG_TEMPLATE = """# CHANGELOG

All notable changes to this project are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
-

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
-

### Security
-

## [0.3.0] - 2026-01-15

### Added
- 15 new skills across Language/Security/Data domains
- New SkillCategory enums (BLOCKCHAIN, DATABASE, NETWORK, ALGORITHM, ...)

### Changed
- Bumped base Skill interface to support `artifacts` list

### Fixed
- can_handle() now returns 0.0 on empty prompt
"""


class ReleaseManagementSkill(Skill):
    """Quản lý release: semver bump, changelog, release notes."""

    category = SkillCategory.DEVOPS
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "release", "semver", "semantic version", "changelog",
        "version bump", "release notes", "tag release", "cut release",
        "versioning",
    ]
    examples = [
        "Tạo changelog cho v0.3.0",
        "Bump version từ 1.2.3 lên release mới",
        "Generate release notes from git log",
    ]

    @property
    def name(self) -> str:
        return "release_management"

    @property
    def description(self) -> str:
        return (
            "Sinh chiến lược semantic versioning, changelog (Keep-a-Changelog), "
            "và release notes từ commit history (Conventional Commits)."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.18
        return min(1.0, score)

    @staticmethod
    def _bump(current: str, kind: str) -> str:
        """Bump semantic version / Tăng phiên bản."""
        try:
            core = current.split("+")[0].split("-")[0]
            major, minor, patch = (int(x) for x in core.split("."))
        except (ValueError, AttributeError):
            return "0.1.0"
        if kind == "major":
            return f"{major + 1}.0.0"
        if kind == "minor":
            return f"{major}.{minor + 1}.0"
        return f"{major}.{minor}.{patch + 1}"

    @staticmethod
    def _conventional_notes(commits: List[str]) -> str:
        """Group commits theo Conventional Commits / Nhóm commit."""
        buckets: Dict[str, List[str]] = {
            "Added": [], "Changed": [], "Deprecated": [],
            "Removed": [], "Fixed": [], "Security": [],
        }
        prefix_map = {
            "feat": "Added", "add": "Added",
            "change": "Changed", "refactor": "Changed",
            "deprecate": "Deprecated",
            "remove": "Removed",
            "fix": "Fixed",
            "security": "Security", "sec": "Security",
        }
        for line in commits:
            tag = line.split(":", 1)[0].lower().strip()
            bucket = prefix_map.get(tag)
            if bucket:
                buckets[bucket].append(f"- {line.split(':', 1)[-1].strip()}")
        return "\n".join(
            f"### {k}\n" + "\n".join(v) + "\n"
            for k, v in buckets.items() if v
        )

    def execute(self, context: SkillContext) -> SkillResult:
        current = context.metadata.get("current_version", "0.2.0")
        commits = context.metadata.get("commits", [])
        kind = "patch"
        if context.metadata.get("breaking"):
            kind = "major"
        elif context.metadata.get("feature"):
            kind = "minor"

        new_version = self._bump(str(current), kind)
        notes = self._conventional_notes(commits) or "_No structured commits detected._"

        release_notes = (
            f"# Release v{new_version}\n\n"
            f"**Previous:** {current}  →  **New:** {new_version}  (`{kind}` bump)\n\n"
            f"{notes}\n\n"
            f"## Upgrade / Nâng cấp\n"
            f"- Review breaking changes before upgrading\n"
            f"- Run `pip install --upgrade <pkg>=={new_version}`\n"
        )

        artifacts: List[Dict[str, str]] = [
            {"name": "SEMVER.md", "language": "markdown", "content": SEMVER_RULES},
            {"name": "CHANGELOG.md", "language": "markdown", "content": CHANGELOG_TEMPLATE},
            {"name": f"RELEASE_v{new_version}.md", "language": "markdown", "content": release_notes},
        ]

        return SkillResult(
            success=True,
            output=(
                f"[release_management] {current} → {new_version} ({kind} bump) | "
                f"commits_analyzed={len(commits)} | artifacts=3"
            ),
            artifacts=artifacts,
            suggestions=[
                f"Tag the release: git tag v{new_version} && git push --tags",
                "Run `git cliff` or `semantic-release` for fully automated notes",
                "Verify changelog dates are ISO-8601 (YYYY-MM-DD)",
                "Sign the tag (GPG / sigstore) before publishing artifacts",
            ],
            metadata={
                "skill": self.name,
                "previous_version": current,
                "new_version": new_version,
                "bump_kind": kind,
                "commits_analyzed": len(commits),
                "version": self.version,
                "author": self.author,
            },
        )
