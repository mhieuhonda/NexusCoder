"""CI/CD Pipeline Skill - Sinh pipeline YAML cho GitHub Actions / GitLab CI / Jenkins.

Cung cấp template pipeline CI/CD hoàn chỉnh: build, test, scan, publish,
deploy với chiến lược branch (trunk-based / GitFlow) và environment promotion.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


# Template: GitHub Actions / GitLab CI / Jenkins
GITHUB_ACTIONS_TEMPLATE = """# .github/workflows/ci.yml  (GitHub Actions)
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

permissions:
  contents: read
  packages: write

jobs:
  build-test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0          # cần cho cache key & changelog

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install ruff pytest pytest-cov safety bandit

      - name: Lint (ruff)
        run: ruff check .

      - name: Type-check (mypy)
        run: mypy --strict nexus

      - name: Test + coverage
        run: pytest --cov=nexus --cov-report=xml --cov-report=term-missing

      - name: SAST (bandit) + dependency scan (safety)
        run: |
          bandit -r nexus -q
          safety check --short

      - name: Build artifacts
        run: python -m build

      - name: Upload coverage
        if: github.event_name == 'push'
        uses: codecov/codecov-action@v4

  publish:
    needs: build-test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: "pip" }
      - run: pip install build twine
      - run: python -m build
      - run: twine upload dist/*
        env:
          TWINE_API_TOKEN: ${{ secrets.PYPI_TOKEN }}
"""

GITLAB_CI_TEMPLATE = """# .gitlab-ci.yml  (GitLab CI)
stages: [lint, test, build, deploy]

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  PYTHON_IMAGE: "python:3.12-slim"

cache:
  key: "$CI_COMMIT_REF_SLUG"
  paths: [.cache/pip, .venv/]

lint:
  stage: lint
  image: $PYTHON_IMAGE
  script:
    - pip install ruff mypy
    - ruff check .
    - mypy --strict nexus

test:
  stage: test
  image: $PYTHON_IMAGE
  script:
    - pip install -r requirements.txt pytest pytest-cov
    - pytest --cov=nexus --cov-report=xml
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
  coverage: '/TOTAL.*\\s+(\\d+\\%)$/'

build:
  stage: build
  image: $PYTHON_IMAGE
  script: python -m build
  artifacts:
    paths: [dist/]
  rules:
    - if: $CI_COMMIT_TAG

deploy:prod:
  stage: deploy
  image: $PYTHON_IMAGE
  environment: production
  script:
    - pip install twine
    - twine upload dist/*
  rules:
    - if: $CI_COMMIT_TAG =~ /^v\\d+\\.\\d+\\.\\d+$/
  when: manual
"""


class CICDPipelineSkill(Skill):
    """Sinh CI/CD pipeline template cho GitHub Actions / GitLab CI / Jenkins."""

    category = SkillCategory.DEVOPS
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "ci/cd", "ci cd", "cicd", "pipeline", "jenkins",
        "github actions", "gitlab ci", "gitlab-ci", "continuous integration",
        "continuous deployment", "workflow", "ci build",
    ]
    examples = [
        "Tạo CI/CD pipeline cho Python project dùng GitHub Actions",
        "Setup GitLab CI với test + build + deploy",
        "Configure Jenkins pipeline cho microservice",
    ]

    @property
    def name(self) -> str:
        return "ci_cd_pipeline"

    @property
    def description(self) -> str:
        return (
            "Sinh CI/CD pipeline templates (GitHub Actions / GitLab CI / Jenkins) "
            "với lint, test, SAST, build, publish và environment-gated deploy."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.22
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        # Chọn engine dựa trên prompt / Chọn engine theo keyword
        prompt_lower = (context.prompt or "").lower()
        if "gitlab" in prompt_lower:
            engine, template, filename = "gitlab_ci", GITLAB_CI_TEMPLATE, ".gitlab-ci.yml"
        elif "jenkins" in prompt_lower:
            engine, template, filename = "jenkins", (
                "# Jenkinsfile (Declarative)\n"
                "pipeline {\n"
                "  agent any\n"
                "  options { timeout(time: 30, unit: 'MINUTES') }\n"
                "  stages {\n"
                "    stage('Lint') { steps { sh 'ruff check .' } }\n"
                "    stage('Test')  { steps { sh 'pytest --cov=nexus' } }\n"
                "    stage('Build') { steps { sh 'python -m build' } }\n"
                "    stage('Deploy') {\n"
                "      when { branch 'main' }\n"
                "      steps { sh 'twine upload dist/*' }\n"
                "    }\n"
                "  }\n"
                "}\n"
            ), "Jenkinsfile"
        else:
            engine, template, filename = "github_actions", GITHUB_ACTIONS_TEMPLATE, ".github/workflows/ci.yml"

        stages = ["lint", "test", "sast", "build", "publish", "deploy"]
        artifacts: List[Dict[str, str]] = [
            {"name": filename, "language": "yaml", "content": template},
            {
                "name": "BRANCHING.md",
                "language": "markdown",
                "content": (
                    "# Branch Strategy / Chiến lược nhánh\n\n"
                    "- `main`     : luôn deployable (trunk-based)\n"
                    "- `develop`  : integration branch (GitFlow optional)\n"
                    "- `feat/*`   : short-lived feature branches\n"
                    "- Tag `vMAJOR.MINOR.PATCH`  → trigger release\n"
                ),
            },
        ]

        return SkillResult(
            success=True,
            output=(
                f"[ci_cd_pipeline] engine={engine} | stages={','.join(stages)}\n"
                f"Generated {filename} ({len(template)} bytes) with branch strategy guide."
            ),
            artifacts=artifacts,
            suggestions=[
                "Add matrix build for multiple Python versions if cross-version support is required",
                "Enable required status checks + branch protection on main",
                "Configure environment secrets per stage (staging → production)",
                "Add a dependabot/renovate workflow to keep actions pinned",
            ],
            metadata={
                "skill": self.name,
                "engine": engine,
                "stages": stages,
                "filename": filename,
                "version": self.version,
                "author": self.author,
            },
        )
