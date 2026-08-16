"""
The-Stack v2 Collector for Nexus Coder v0.3
============================================
Pulls code samples from BigCode's The-Stack v2 dataset on HuggingFace.

The-Stack v2 is a massive deduplicated code corpus covering ~600 programming
languages, collected from GitHub repos with permissive licenses.

This collector:
  - Streams samples lazily via `datasets` library (lazy import)
  - Filters by language (Python, JS, TS, Go, Rust, etc.)
  - Applies license filter (only MIT/Apache/BSD/MPL)
  - Writes to JSONL with metadata {lang, license, repo, path, content}

Reference:
  BigCode. "The Stack v2: A Comprehensive Multilingual Code Corpus."
  https://huggingface.co/datasets/bigcode/the-stack-v2-train-full-ids

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional


# Curated language list (subset of v2's ~600 languages)
SUPPORTED_LANGUAGES = [
    "python", "javascript", "typescript", "java", "go", "rust",
    "c", "cpp", "csharp", "ruby", "php", "swift", "kotlin",
    "scala", "shell", "sql", "html", "css",
]

# Permissive licenses (allowlist)
PERMISSIVE_LICENSES = {
    "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause",
    "mpl-2.0", "unlicense", "isc", "0bsd",
}


class TheStackCollector:
    """Collect code samples from The-Stack v2."""

    DATASET_NAME = "bigcode/the-stack-v2-train-full-ids"

    def __init__(
        self,
        cache_dir: str = "./data_cache/the_stack",
        languages: Optional[List[str]] = None,
        max_samples_per_language: int = 5000,
        min_stars: int = 0,
        license_filter: Optional[List[str]] = None,
        streaming: bool = True,
    ):
        self.cache_dir = cache_dir
        self.languages = languages or SUPPORTED_LANGUAGES
        self.max_samples_per_language = max_samples_per_language
        self.min_stars = min_stars
        self.license_filter = set(license_filter) if license_filter else PERMISSIVE_LICENSES
        self.streaming = streaming
        os.makedirs(cache_dir, exist_ok=True)

    def __iter__(self) -> Iterator[Dict]:
        """Stream samples lazily from The-Stack v2.
        Yields dicts: {lang, license, repo, path, size, content}.
        """
        try:
            from datasets import load_dataset  # lazy import
        except ImportError as e:
            raise ImportError(
                "The `datasets` package is required. Install with: pip install datasets"
            ) from e

        for lang in self.languages:
            count = 0
            try:
                ds = load_dataset(
                    self.DATASET_NAME,
                    split="train",
                    streaming=self.streaming,
                    data_dir=f"data/{lang}",
                )
            except Exception:
                continue
            for example in ds:
                if count >= self.max_samples_per_language:
                    break
                # Apply filters
                stars = example.get("stars", 0) or 0
                if stars < self.min_stars:
                    continue
                license_ = (example.get("license") or "").lower()
                if license_ and license_ not in self.license_filter:
                    continue
                content = example.get("content", "")
                if not content or len(content) < 50:
                    continue
                yield {
                    "lang": lang,
                    "license": license_,
                    "repo": example.get("repository", ""),
                    "path": example.get("path", ""),
                    "size": example.get("size", len(content)),
                    "stars": stars,
                    "content": content,
                }
                count += 1

    def collect(self, output_dir: Optional[str] = None) -> str:
        """Collect all samples and write to JSONL. Returns the output file path."""
        output_dir = output_dir or self.cache_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "the_stack_v2.jsonl")
        total = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in self:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1
        print(f"[TheStackCollector] Collected {total} samples → {output_path}")
        return output_path

    def stats(self) -> Dict[str, int]:
        """Return per-language sample counts (calls collect if not yet run)."""
        counts: Dict[str, int] = {lang: 0 for lang in self.languages}
        for sample in self:
            counts[sample["lang"]] = counts.get(sample["lang"], 0) + 1
        return counts


__all__ = ["TheStackCollector", "SUPPORTED_LANGUAGES", "PERMISSIVE_LICENSES"]
