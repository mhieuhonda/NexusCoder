"""
StarCoder2-data Collector for Nexus Coder v0.3
===============================================
Pulls from BigCode's StarCoder2 training data (github-code, commits, jupyter).

Components:
  - github_code:  raw code files from GitHub (subset of The-Stack v2)
  - github_commits: commit diffs — great for code-editing / instruction tasks
  - github_jupyter: notebook cells with markdown + code interleaved

Each component has different schema; this collector unifies them into the
Nexus format: {source, lang, content, metadata}.

Reference:
  BigCode. "StarCoder 2 and The Stack v2: Building the Next Generation of
  Transparent Code Models."
  https://huggingface.co/datasets/bigcode/starcoder2data

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional


COMPONENT_DATASETS = {
    "github_code": "bigcode/starcoder2data",
    "github_commits": "bigcode/starcoder2data",
    "github_jupyter": "bigcode/starcoder2data",
}

SUPPORTED_LANGS = [
    "python", "javascript", "typescript", "java",
    "go", "rust", "c", "cpp",
]


class StarCoder2Collector:
    """Collect from StarCoder2 training data."""

    def __init__(
        self,
        cache_dir: str = "./data_cache/starcoder2",
        components: Optional[List[str]] = None,
        max_samples_per_component: int = 20000,
        languages: Optional[List[str]] = None,
        streaming: bool = True,
    ):
        self.cache_dir = cache_dir
        self.components = components or list(COMPONENT_DATASETS.keys())
        self.max_samples_per_component = max_samples_per_component
        self.languages = languages or SUPPORTED_LANGS
        self.streaming = streaming
        os.makedirs(cache_dir, exist_ok=True)

    def _iter_github_code(self) -> Iterator[Dict]:
        """Iterate github_code component."""
        try:
            from datasets import load_dataset
        except ImportError:
            return
        for lang in self.languages:
            count = 0
            try:
                ds = load_dataset(
                    "bigcode/starcoder2data",
                    split="train",
                    streaming=self.streaming,
                    data_dir=f"data/{lang}",
                )
            except Exception:
                continue
            for example in ds:
                if count >= self.max_samples_per_component // len(self.languages):
                    break
                content = example.get("content", "")
                if not content or len(content) < 50:
                    continue
                yield {
                    "source": "starcoder2_github_code",
                    "lang": lang,
                    "content": content,
                    "metadata": {
                        "repo": example.get("repository", ""),
                        "path": example.get("path", ""),
                        "size": example.get("size", 0),
                        "license": example.get("license", ""),
                    },
                }
                count += 1

    def _iter_github_commits(self) -> Iterator[Dict]:
        """Iterate github_commits component (commit diffs)."""
        try:
            from datasets import load_dataset
        except ImportError:
            return
        count = 0
        try:
            ds = load_dataset(
                "bigcode/starcoder2data",
                split="train",
                streaming=self.streaming,
                name="commits",
            )
        except Exception:
            return
        for example in ds:
            if count >= self.max_samples_per_component:
                break
            diff = example.get("diff", "") or example.get("content", "")
            if not diff or len(diff) < 50:
                continue
            yield {
                "source": "starcoder2_commits",
                "lang": example.get("language", "unknown"),
                "content": diff,
                "metadata": {
                    "commit": example.get("commit", ""),
                    "repo": example.get("repository", ""),
                    "author": example.get("author", ""),
                },
            }
            count += 1

    def _iter_github_jupyter(self) -> Iterator[Dict]:
        """Iterate github_jupyter component (notebook cells)."""
        try:
            from datasets import load_dataset
        except ImportError:
            return
        count = 0
        try:
            ds = load_dataset(
                "bigcode/starcoder2data",
                split="train",
                streaming=self.streaming,
                name="jupyter",
            )
        except Exception:
            return
        for example in ds:
            if count >= self.max_samples_per_component:
                break
            content = example.get("content", "")
            if not content or len(content) < 50:
                continue
            yield {
                "source": "starcoder2_jupyter",
                "lang": "python",
                "content": content,
                "metadata": {
                    "repo": example.get("repository", ""),
                    "notebook_path": example.get("path", ""),
                    "cell_type": example.get("cell_type", ""),
                },
            }
            count += 1

    def __iter__(self) -> Iterator[Dict]:
        """Stream samples from all enabled components."""
        for component in self.components:
            if component == "github_code":
                yield from self._iter_github_code()
            elif component == "github_commits":
                yield from self._iter_github_commits()
            elif component == "github_jupyter":
                yield from self._iter_github_jupyter()

    def collect(self, output_dir: Optional[str] = None) -> str:
        """Collect all samples and write to JSONL. Returns the output file path."""
        output_dir = output_dir or self.cache_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "starcoder2.jsonl")
        total = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in self:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1
        print(f"[StarCoder2Collector] Collected {total} samples → {output_path}")
        return output_path


__all__ = ["StarCoder2Collector", "COMPONENT_DATASETS", "SUPPORTED_LANGS"]
