"""Testing Skill - Sinh unit/integration/E2E tests."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class TestingSkill(Skill):
    """Sinh tests: unit, integration, E2E, property-based, mutation tests."""
    
    category = SkillCategory.CODE
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "test", "kiểm thử", "unit test", "pytest", "unittest",
        "jest", "mocha", "vitest", "integration", "e2e",
        "coverage", "mock", "fixture", "tdd", "bdd",
    ]
    
    @property
    def name(self) -> str:
        return "testing"
    
    @property
    def description(self) -> str:
        return (
            "Sinh tests toàn diện: unit tests, integration tests, E2E tests, "
            "property-based tests (Hypothesis), mutation tests, "
            "test coverage analysis, mock/fixture generation."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        test_types = {
            "unit": "Test từng function/method độc lập",
            "integration": "Test tương tác giữa modules",
            "e2e": "Test end-to-end user flows",
            "property": "Property-based testing (Hypothesis)",
            "mutation": "Mutation testing để check test quality",
            "fuzz": "Fuzz testing cho security",
            "snapshot": "Snapshot testing cho UI",
            "contract": "Contract testing cho APIs",
            "load": "Load/performance testing",
            "regression": "Regression tests cho bugs đã fix",
        }
        frameworks = {
            "python": ["pytest", "unittest", "hypothesis", "pytest-asyncio"],
            "javascript": ["jest", "vitest", "mocha", "playwright"],
            "go": ["testing", "testify", "gomock"],
            "rust": ["cargo test", "proptest", "mockall"],
            "java": ["JUnit 5", "TestNG", "Mockito"],
        }
        return SkillResult(
            success=True,
            output=f"[Testing] Can generate {len(test_types)} test types.",
            metadata={
                "skill": self.name,
                "test_types": test_types,
                "frameworks": frameworks,
                "coverage_target": ">= 80%",
            },
            suggestions=[
                "Aim for high coverage on critical paths",
                "Use TDD for new features",
                "Mock external dependencies",
            ],
        )
