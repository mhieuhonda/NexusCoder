"""DevOps Skill - Sinh Dockerfile, k8s manifests, Terraform, CI/CD pipelines.

Tạo artifacts DevOps từ mô tả tự nhiên: container images, Kubernetes
deployments, infrastructure-as-code, và CI/CD pipeline templates.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class DevOpsSkill(Skill):
    """Sinh DevOps artifacts: Docker, Kubernetes, Terraform, CI/CD."""

    category = SkillCategory.DEVOPS
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "docker", "dockerfile", "container", "kubernetes", "k8s",
        "terraform", "ansible", "ci/cd", "cicd", "jenkins",
        "github actions", "gitlab ci", "helm", "deploy",
        "pod", "deployment", "service", "ingress", "manifest",
    ]
    examples = [
        "Tạo Dockerfile cho FastAPI app",
        "Generate k8s deployment manifest for Redis",
        "Write Terraform to provision an EC2 instance",
        "Setup GitHub Actions CI/CD pipeline for Python project",
    ]

    @property
    def name(self) -> str:
        return "devops"

    @property
    def description(self) -> str:
        return (
            "Sinh DevOps artifacts: Dockerfile (multi-stage), Kubernetes "
            "manifests (Deployment/Service/Ingress), Terraform modules, "
            "Helm charts, và CI/CD pipelines (GitHub Actions / GitLab CI)."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.25
        if any(tag in prompt_lower for tag in ("yaml", "yml", ".tf", "manifest")):
            score += 0.2
        return min(1.0, score)

    def _detect_target(self, prompt: str) -> str:
        p = prompt.lower()
        if "dockerfile" in p or "docker" in p and "image" in p:
            return "dockerfile"
        if "kubernetes" in p or "k8s" in p or "manifest" in p or "helm" in p:
            return "k8s"
        if "terraform" in p or ".tf" in p or "infrastructure" in p:
            return "terraform"
        if "ci/cd" in p or "cicd" in p or "github actions" in p or "gitlab" in p or "jenkins" in p:
            return "cicd"
        if "ansible" in p:
            return "ansible"
        return "dockerfile"

    def execute(self, context: SkillContext) -> SkillResult:
        target = self._detect_target(context.prompt)
        artifact = self._build_artifact(target, context)

        return SkillResult(
            success=True,
            output=f"[DevOps/{target}] Artifact ready for: {context.prompt[:180]}",
            artifacts=[artifact],
            metadata={
                "skill": self.name,
                "target": target,
                "language": context.language or "yaml",
                "tools": ["docker", "kubectl", "terraform", "helm", "act"],
            },
            suggestions=[
                "Pin base image digests for reproducible builds (e.g. python:3.12-slim@sha256:...)",
                "Scan images for CVEs (trivy, grype) before pushing",
                "Use multi-stage builds to shrink final image size",
                "Apply least-privilege RBAC and network policies in k8s",
                "Store secrets in a vault (Vault, AWS SM, Sealed Secrets)",
                "Enable image signature verification (cosign) in CI",
            ],
        )

    def _build_artifact(self, target: str, context: SkillContext) -> Dict[str, str]:
        if target == "k8s":
            return {
                "path": "k8s/deployment.yaml",
                "content": _K8S_MANIFEST,
            }
        if target == "terraform":
            return {
                "path": "infra/main.tf",
                "content": _TERRAFORM_SNIPPET,
            }
        if target == "cicd":
            return {
                "path": ".github/workflows/ci.yml",
                "content": _GITHUB_ACTIONS,
            }
        if target == "ansible":
            return {
                "path": "ansible/playbook.yml",
                "content": _ANSIBLE_PLAYBOOK,
            }
        return {"path": "Dockerfile", "content": _DOCKERFILE}


_DOCKERFILE = """# Multi-stage Dockerfile — Python service
FROM python:3.12-slim AS builder
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.12-slim AS runtime
RUN useradd -m -u 10001 appuser
WORKDIR /app
COPY --from=builder /root/.local /home/appuser/.local
COPY . .
USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app:main", "--host", "0.0.0.0", "--port", "8000"]
"""

_K8S_MANIFEST = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-api
  labels: {app: nexus-api}
spec:
  replicas: 3
  selector:
    matchLabels: {app: nexus-api}
  template:
    metadata:
      labels: {app: nexus-api}
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        fsGroup: 10001
      containers:
        - name: api
          image: ghcr.io/nexus/api:0.3.0
          ports: [{containerPort: 8000}]
          resources:
            requests: {cpu: "250m", memory: "256Mi"}
            limits:   {cpu: "1000m", memory: "1Gi"}
          livenessProbe:
            httpGet: {path: /health, port: 8000}
            initialDelaySeconds: 10
          readinessProbe:
            httpGet: {path: /ready, port: 8000}
---
apiVersion: v1
kind: Service
metadata:
  name: nexus-api
spec:
  selector: {app: nexus-api}
  ports: [{port: 80, targetPort: 8000}]
  type: ClusterIP
"""

_TERRAFORM_SNIPPET = """# Infrastructure as Code — AWS EC2
terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

variable "region"  { default = "ap-southeast-1" }
variable "instance_type" { default = "t3.small" }

resource "aws_instance" "nexus" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  vpc_security_group_ids = [aws_security_group.nexus.id]
  tags                   = { Name = "nexus-coder" }
}

resource "aws_security_group" "nexus" {
  name = "nexus-sg"
  ingress {
    from_port = 443
    to_port   = 443
    protocol  = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
"""

_GITHUB_ACTIONS = """name: CI
on:
  push: { branches: [main] }
  pull_request: { branches: [main] }
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check .
      - run: mypy src
      - run: pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v4
"""

_ANSIBLE_PLAYBOOK = """---
- name: Provision Nexus Coder host
  hosts: webservers
  become: true
  vars:
    app_version: "0.3.0"
  tasks:
    - name: Install system deps
      apt:
        name: [python3, python3-pip, nginx]
        update_cache: true
    - name: Create app user
      user: { name: nexus, shell: /sbin/nologin, system: true }
    - name: Deploy app
      copy:
        src: ../dist/
        dest: /opt/nexus/
        owner: nexus
    - name: Ensure nginx running
      service: { name: nginx, state: started, enabled: true }
"""
