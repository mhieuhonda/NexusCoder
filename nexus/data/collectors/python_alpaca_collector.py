"""
Python-Alpaca Collector for Nexus Coder v0.3
=============================================
Aggregates multiple high-quality Python instruction-tuning datasets.

Sources (all on HuggingFace):
  - sahil2801/codealpaca                 ~20K samples
  - HuggingFaceH4/CodeAlpaca_20K         ~20K
  - nickroany/Evol-Instruct-Code         ~15K
  - TheBloke/CodeAlpaca-13B              ~5K
  - codeparrot/codeparrot-clean          ~50K (filterable)
  - nampdn-ai/tiny-codes                 ~50K (filterable)

Output: unified JSONL with Nexus format {system, user, assistant}.
Converts Alpaca-style {instruction, input, output} → unified via
nexus.integrations.llamafactory.alpaca_to_nexus.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional

# We import the converter for type hints only — actual import at runtime
# to keep the module importable when llamafactory deps are missing.
try:
    from ...integrations.llamafactory import convert_to_nexus
    _HAS_CONVERTER = True
except Exception:
    _HAS_CONVERTER = False


DEFAULT_SOURCES = [
    {"name": "sahil2801/codealpaca", "max_samples": 20000},
    {"name": "HuggingFaceH4/CodeAlpaca_20K", "max_samples": 20000},
    {"name": "nickroany/Evol-Instruct-Code", "max_samples": 15000},
    {"name": "TheBloke/CodeAlpaca-13B", "max_samples": 5000},
    {"name": "codeparrot/codeparrot-clean", "max_samples": 50000, "is_completion": True},
    {"name": "nampdn-ai/tiny-codes", "max_samples": 50000},
]


class PythonAlpacaCollector:
    """Aggregate Python instruction datasets."""

    def __init__(
        self,
        cache_dir: str = "./data_cache/python_alpaca",
        sources: Optional[List[Dict]] = None,
    ):
        self.cache_dir = cache_dir
        self.sources = sources or DEFAULT_SOURCES
        os.makedirs(cache_dir, exist_ok=True)

    def _iter_source(self, source: Dict) -> Iterator[Dict]:
        name = source["name"]
        max_samples = source.get("max_samples", 10000)
        is_completion = source.get("is_completion", False)
        try:
            from datasets import load_dataset
        except ImportError:
            return
        try:
            ds = load_dataset(name, split="train", streaming=True)
        except Exception:
            return
        count = 0
        for example in ds:
            if count >= max_samples:
                break
            # Normalize to Nexus format
            try:
                if _HAS_CONVERTER:
                    turns = convert_to_nexus(example)
                else:
                    # Inline fallback for Alpaca format
                    turns = [{
                        "system": example.get("system_prompt", ""),
                        "user": example.get("instruction", ""),
                        "assistant": example.get("output", ""),
                    }]
                for turn in turns:
                    if not turn.get("user") or not turn.get("assistant"):
                        continue
                    yield {
                        "source": name,
                        "system": turn.get("system", ""),
                        "user": turn["user"],
                        "assistant": turn["assistant"],
                    }
                    count += 1
                    if count >= max_samples:
                        break
            except Exception:
                continue

    def __iter__(self) -> Iterator[Dict]:
        for source in self.sources:
            yield from self._iter_source(source)

    def collect(self, output_dir: Optional[str] = None) -> str:
        """Collect and write JSONL. Returns output path."""
        output_dir = output_dir or self.cache_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "python_alpaca.jsonl")
        total = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in self:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1
        print(f"[PythonAlpacaCollector] Collected {total} samples → {output_path}")
        return output_path


__all__ = ["PythonAlpacaCollector", "DEFAULT_SOURCES"]
