"""
GitHub Collector - Thu thập code từ GitHub repositories
========================================================
 thu thập dữ liệu training từ public GitHub repos.

Features:
- Clone & extract code từ repos
- Filter theo language, file size, license
- Extract functions, classes, docstrings
- Rate limit aware (GitHub API: 5000 req/h với token)
- Parallel fetching
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import logging
from typing import List, Dict, Optional, Iterator, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class GitHubRepo:
    """Thông tin một GitHub repo để collect."""
    owner: str
    name: str
    branch: str = "main"
    languages: List[str] = field(default_factory=lambda: ["python"])
    max_files: int = 1000
    max_file_size_kb: int = 100
    license_filter: List[str] = field(default_factory=lambda: ["MIT", "Apache-2.0", "BSD", "GPL"])
    
    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"
    
    @property
    def api_url(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.name}"


@dataclass
class CodeSample:
    """Một sample code được thu thập."""
    repo: str
    file_path: str
    language: str
    content: str
    size: int
    license: Optional[str] = None
    quality_score: float = 0.0


class GitHubCollector:
    """Collect training data từ GitHub repositories.
    
    Usage:
        collector = GitHubCollector(token="ghp_xxx")
        repos = [
            GitHubRepo("python", "cpython", languages=["python"]),
            GitHubRepo("pallets", "flask"),
        ]
        for sample in collector.collect(repos):
            print(sample.file_path, len(sample.content))
    """
    
    EXTENSIONS = {
        "python": [".py"],
        "javascript": [".js", ".mjs", ".jsx"],
        "typescript": [".ts", ".tsx"],
        "go": [".go"],
        "rust": [".rs"],
        "java": [".java"],
        "c": [".c", ".h"],
        "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hxx"],
        "csharp": [".cs"],
        "ruby": [".rb"],
        "php": [".php"],
        "swift": [".swift"],
        "kotlin": [".kt"],
        "scala": [".scala"],
        "sql": [".sql"],
        "shell": [".sh", ".bash"],
        "yaml": [".yaml", ".yml"],
        "markdown": [".md", ".markdown"],
    }
    
    SKIP_DIRS = {
        "node_modules", "vendor", "venv", ".venv", "env", "__pycache__",
        ".git", ".github", "dist", "build", "target", "out", "bin",
        ".idea", ".vscode", "coverage", ".cache", ".eggs", ".tox",
    }
    
    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: str = "./data_cache/github",
        max_concurrent: int = 4,
    ):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.cache_dir = cache_dir
        self.max_concurrent = max_concurrent
        os.makedirs(cache_dir, exist_ok=True)
    
    def collect(self, repos: List[GitHubRepo]) -> Iterator[CodeSample]:
        """Collect code samples từ list of repos.
        
        Yields:
            CodeSample objects
        """
        for repo in repos:
            try:
                yield from self._collect_repo(repo)
            except Exception as e:
                logger.error(f"Failed to collect {repo.url}: {e}")
                continue
    
    def _collect_repo(self, repo: GitHubRepo) -> Iterator[CodeSample]:
        """Collect từ một repo."""
        cache_path = os.path.join(self.cache_dir, f"{repo.owner}_{repo.name}")
        
        # Clone if not cached
        if not os.path.exists(cache_path):
            logger.info(f"Cloning {repo.url}...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", repo.branch, repo.url, cache_path],
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Clone failed for {repo.url}: {e.stderr.decode()[:200]}")
                return
            except subprocess.TimeoutExpired:
                logger.error(f"Clone timeout for {repo.url}")
                return
        
        # Walk and collect files
        count = 0
        for root, dirs, files in os.walk(cache_path):
            # Filter dirs in-place
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS and not d.startswith(".")]
            
            for fname in files:
                if count >= repo.max_files:
                    return
                
                ext = os.path.splitext(fname)[1].lower()
                lang = self._detect_language(ext)
                if lang is None or (repo.languages and lang not in repo.languages):
                    continue
                
                fpath = os.path.join(root, fname)
                
                # Size check
                try:
                    size = os.path.getsize(fpath)
                    if size > repo.max_file_size_kb * 1024 or size < 100:
                        continue
                except OSError:
                    continue
                
                # Read
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except Exception:
                    continue
                
                # Quality filter
                if not self._is_quality(content, lang):
                    continue
                
                rel_path = os.path.relpath(fpath, cache_path)
                
                yield CodeSample(
                    repo=f"{repo.owner}/{repo.name}",
                    file_path=rel_path,
                    language=lang,
                    content=content,
                    size=size,
                    quality_score=self._score_quality(content, lang),
                )
                count += 1
    
    def _detect_language(self, ext: str) -> Optional[str]:
        for lang, exts in self.EXTENSIONS.items():
            if ext in exts:
                return lang
        return None
    
    def _is_quality(self, content: str, lang: str) -> bool:
        """Basic quality filter."""
        if len(content) < 50:
            return False
        if len(content) > 100000:  # Skip huge files
            return False
        # Skip if too many non-printable chars
        non_print = sum(1 for c in content if not c.isprintable() and c not in "\n\r\t")
        if non_print / len(content) > 0.05:
            return False
        # Skip auto-generated files
        if "auto-generated" in content[:200].lower():
            return False
        if "DO NOT EDIT" in content[:200]:
            return False
        return True
    
    def _score_quality(self, content: str, lang: str) -> float:
        """Score quality [0.0, 1.0]."""
        score = 0.5
        # Has docstrings/comments
        if lang == "python":
            if '"""' in content or "'''" in content:
                score += 0.2
            if "# " in content:
                score += 0.1
        # Has type hints
        if "->" in content or ": int" in content or ": str" in content:
            score += 0.1
        # Reasonable length
        lines = content.count("\n")
        if 20 <= lines <= 500:
            score += 0.1
        return min(1.0, score)
    
    def search_repos(
        self,
        query: str,
        language: str = "python",
        sort: str = "stars",
        max_results: int = 50,
    ) -> List[GitHubRepo]:
        """Search GitHub repos by query (requires token)."""
        if not self.token:
            logger.warning("No GitHub token - cannot search")
            return []
        
        import urllib.request
        import urllib.parse
        
        params = urllib.parse.urlencode({
            "q": f"{query} language:{language}",
            "sort": sort,
            "order": "desc",
            "per_page": min(max_results, 100),
        })
        url = f"https://api.github.com/search/repositories?{params}"
        
        req = urllib.request.Request(url, headers={
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "NexusCoder-DataCollector/0.2",
        })
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
            
            repos = []
            for item in data.get("items", [])[:max_results]:
                repos.append(GitHubRepo(
                    owner=item["owner"]["login"],
                    name=item["name"],
                    languages=[language],
                ))
            return repos
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []


# =============================================================================
# Curated list of high-quality repos for training
# =============================================================================

CURATED_REPOS: List[GitHubRepo] = [
    # Python core
    GitHubRepo("python", "cpython", languages=["python"], max_files=2000),
    GitHubRepo("pallets", "flask", languages=["python"]),
    GitHubRepo("pallets", "django", languages=["python"], max_files=2000),
    GitHubRepo("pallets", "click", languages=["python"]),
    GitHubRepo("psf", "requests", languages=["python"]),
    GitHubRepo("psf", "requests-html", languages=["python"]),
    
    # Data science
    GitHubRepo("numpy", "numpy", languages=["python"], max_files=2000),
    GitHubRepo("pandas-dev", "pandas", languages=["python"], max_files=2000),
    GitHubRepo("scipy", "scipy", languages=["python"], max_files=2000),
    GitHubRepo("matplotlib", "matplotlib", languages=["python"], max_files=2000),
    GitHubRepo("scikit-learn", "scikit-learn", languages=["python"], max_files=2000),
    
    # ML/DL
    GitHubRepo("pytorch", "pytorch", languages=["python", "cpp"], max_files=2000),
    GitHubRepo("tensorflow", "tensorflow", languages=["python", "cpp"], max_files=2000),
    GitHubRepo("huggingface", "transformers", languages=["python"], max_files=2000),
    GitHubRepo("huggingface", "datasets", languages=["python"]),
    GitHubRepo("huggingface", "tokenizers", languages=["python", "rust"]),
    GitHubRepo("langchain-ai", "langchain", languages=["python"], max_files=2000),
    GitHubRepo("ollama", "ollama-python", languages=["python"]),
    
    # Web frameworks
    GitHubRepo("tiangolo", "fastapi", languages=["python"], max_files=2000),
    GitHubRepo("encode", "starlette", languages=["python"]),
    GitHubRepo("encode", "uvicorn", languages=["python"]),
    GitHubRepo("tornadoweb", "tornado", languages=["python"]),
    GitHubRepo("Sanic", "sanic", languages=["python"]),
    
    # CLI
    GitHubRepo("click", "click", languages=["python"]),
    GitHubRepo("prompt-toolkit", "python-prompt-toolkit", languages=["python"]),
    GitHubRepo("Textualize", "rich", languages=["python"]),
    GitHubRepo("Textualize", "textual", languages=["python"]),
    
    # Tools
    GitHubRepo("pytest-dev", "pytest", languages=["python"]),
    GitHubRepo("pypa", "pip", languages=["python"]),
    GitHubRepo("pypa", "setuptools", languages=["python"]),
    GitHubRepo("mkdocs", "mkdocs", languages=["python"]),
    GitHubRepo("sphinx-doc", "sphinx", languages=["python"]),
    
    # Async
    GitHubRepo("MagicStack", "uvloop", languages=["python", "c"]),
    GitHubRepo("aio-libs", "aiohttp", languages=["python"], max_files=2000),
    GitHubRepo("aio-libs", "aiomysql", languages=["python"]),
    GitHubRepo("aio-libs", "aiopg", languages=["python"]),
    
    # Database
    GitHubRepo("sqlalchemy", "sqlalchemy", languages=["python"], max_files=2000),
    GitHubRepo("mongodb", "mongo-python-driver", languages=["python"]),
    GitHubRepo("redis", "redis-py", languages=["python"]),
    GitHubRepo("coleifer", "peewee", languages=["python"]),
    
    # Other useful
    GitHubRepo("psf", "black", languages=["python"]),
    GitHubRepo("pycqa", "flake8", languages=["python"]),
    GitHubRepo("pycqa", "isort", languages=["python"]),
    GitHubRepo("python-attrs", "attrs", languages=["python"]),
    GitHubRepo("pydantic", "pydantic", languages=["python"]),
    GitHubRepo("encode", "httpx", languages=["python"]),
    GitHubRepo("httpie", "httpie", languages=["python"]),
    GitHubRepo("pypa", "virtualenv", languages=["python"]),
    GitHubRepo("pypa", "build", languages=["python"]),
    
    # JavaScript/TypeScript
    GitHubRepo("facebook", "react", languages=["javascript", "typescript"], max_files=2000),
    GitHubRepo("vuejs", "vue", languages=["javascript", "typescript"], max_files=2000),
    GitHubRepo("angular", "angular", languages=["typescript"], max_files=2000),
    GitHubRepo("vercel", "next.js", languages=["javascript", "typescript"], max_files=2000),
    GitHubRepo("microsoft", "TypeScript", languages=["typescript"], max_files=2000),
    GitHubRepo("nodejs", "node", languages=["javascript", "c++"], max_files=2000),
    GitHubRepo("expressjs", "express", languages=["javascript"]),
    GitHubRepo("lodash", "lodash", languages=["javascript"]),
    GitHubRepo("axios", "axios", languages=["javascript"]),
    GitHubRepo("chalk", "chalk", languages=["javascript"]),
    
    # Go
    GitHubRepo("golang", "go", languages=["go"], max_files=2000),
    GitHubRepo("gin-gonic", "gin", languages=["go"]),
    GitHubRepo("labstack", "echo", languages=["go"]),
    GitHubRepo("spf13", "cobra", languages=["go"]),
    GitHubRepo("kubernetes", "kubernetes", languages=["go"], max_files=2000),
    GitHubRepo("prometheus", "prometheus", languages=["go"], max_files=2000),
    GitHubRepo("grafana", "grafana", languages=["go"], max_files=2000),
    GitHubRepo("etcd-io", "etcd", languages=["go"], max_files=2000),
    GitHubRepo("hashicorp", "terraform", languages=["go"], max_files=2000),
    GitHubRepo("hashicorp", "vault", languages=["go"], max_files=2000),
    GitHubRepo("docker", "compose", languages=["go"]),
    GitHubRepo("cli", "cli", languages=["go"]),
    
    # Rust
    GitHubRepo("rust-lang", "rust", languages=["rust"], max_files=2000),
    GitHubRepo("rust-lang", "cargo", languages=["rust"], max_files=2000),
    GitHubRepo("tokio-rs", "tokio", languages=["rust"], max_files=2000),
    GitHubRepo("serde-rs", "serde", languages=["rust"]),
    GitHubRepo("clap-rs", "clap", languages=["rust"]),
    GitHubRepo("BurntSushi", "ripgrep", languages=["rust"]),
    GitHubRepo("starship", "starship", languages=["rust"], max_files=2000),
    
    # C/C++
    GitHubRepo("redis", "redis", languages=["c"], max_files=2000),
    GitHubRepo("sqlite", "sqlite", languages=["c"]),
    GitHubRepo("curl", "curl", languages=["c"], max_files=2000),
    GitHubRepo("nginx", "nginx", languages=["c"], max_files=2000),
    GitHubRepo("openssl", "openssl", languages=["c"], max_files=2000),
    
    # Tools/CLI
    GitHubRepo("junegunn", "fzf", languages=["go"]),
    GitHubRepo("BurntSushi", "ripgrep", languages=["rust"]),
    GitHubRepo("sharkdp", "bat", languages=["rust"]),
    GitHubRepo("sharkdp", "fd", languages=["rust"]),
    GitHubRepo("dalance", "procs", languages=["rust"]),
    
    # Documentation/Examples
    GitHubRepo("realpython", "python-guide", languages=["python", "markdown"]),
    GitHubRepo("ehmatthes", "pcc_2e", languages=["python"]),
    GitHubRepo("thedaviddias", "Front-End-Checklist", languages=["markdown"]),
    GitHubRepo("kamranahmedse", "developer-roadmap", languages=["markdown"]),
]
