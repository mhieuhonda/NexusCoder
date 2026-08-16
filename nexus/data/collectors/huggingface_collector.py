"""
HuggingFace Collector - Thu thập datasets từ HuggingFace Hub
=============================================================
Pull datasets từ HuggingFace Hub cho training Nexus Coder.

Recommended datasets for code/text training:
- codeparrot/codeparrot-clean: Clean Python code
- GitHub CODE: Code from GitHub
- the-stack: Massive code dataset (3TB)
- oscar: Multilingual web text
- wikipedia: Wikipedia dumps
- openwebtext: Web text
- c4: Colossal Clean Crawled Corpus
- bookcorpus: Books
- arxiv: Scientific papers
- pubmed: Biomedical papers
"""
from __future__ import annotations

import os
import json
import logging
from typing import List, Dict, Optional, Iterator, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HFDataset:
    """Thông tin một HuggingFace dataset."""
    name: str  # e.g. "codeparrot/codeparrot-clean"
    subset: Optional[str] = None
    split: str = "train"
    streaming: bool = True  # Use streaming for large datasets
    max_samples: int = 10000
    field_mapping: Dict[str, str] = field(default_factory=lambda: {"text": "text"})
    description: str = ""
    language: Optional[str] = None  # programming language for code datasets
    size_gb: Optional[float] = None


# =============================================================================
# Curated list of high-quality datasets for Nexus Coder training
# =============================================================================

CURATED_DATASETS: List[HFDataset] = [
    # === Code datasets ===
    HFDataset(
        name="codeparrot/codeparrot-clean",
        max_samples=50000,
        language="python",
        description="Clean Python code from GitHub (preprocessed)",
        size_gb=15,
    ),
    HFDataset(
        name="codeparrot/github-code",
        max_samples=30000,
        language="multiple",
        description="Code from GitHub across multiple languages",
        size_gb=115,
    ),
    HFDataset(
        name="bigcode/the-stack-dedup",
        max_samples=20000,
        language="multiple",
        description="Deduplicated code from The Stack v2 (3TB)",
        size_gb=3000,
    ),
    HFDataset(
        name="bigcode/the-stack-v2-train-full-ids",
        max_samples=10000,
        language="multiple",
        description="The Stack v2 full training set",
        size_gb=3000,
    ),
    HFDataset(
        name="nampdn-ai/tiny-codes",
        max_samples=30000,
        language="multiple",
        description="Small high-quality code samples with instructions",
        size_gb=2,
    ),
    HFDataset(
        name="HuggingFaceH4/CodeAlpaca_20K",
        max_samples=20000,
        language="python",
        description="Code instruction dataset",
        size_gb=0.1,
    ),
    
    # === General text (Vietnamese + English) ===
    HFDataset(
        name="wikimedia/wikipedia",
        subset="20231101.vi",
        max_samples=20000,
        description="Vietnamese Wikipedia",
        size_gb=2,
    ),
    HFDataset(
        name="wikimedia/wikipedia",
        subset="20231101.en",
        max_samples=20000,
        description="English Wikipedia",
        size_gb=20,
    ),
    HFDataset(
        name="allenai/c4",
        subset="multilingual",
        split="train",
        max_samples=10000,
        description="Colossal Clean Crawled Corpus (multilingual)",
        size_gb=25000,
    ),
    HFDataset(
        name="oscar-corpus/OSCAR-2301",
        subset="vi",
        max_samples=10000,
        description="OSCAR Vietnamese web text",
        size_gb=10,
    ),
    
    # === Conversational / Instruction ===
    HFDataset(
        name="HuggingFaceH4/ultrachat_200k",
        max_samples=20000,
        description="High-quality multi-turn chat data",
        size_gb=8,
    ),
    HFDataset(
        name="Open-Orca/OpenOrca",
        max_samples=15000,
        description="GPT-4 augmented FLAN instructions",
        size_gb=50,
    ),
    HFDataset(
        name="teknium/OpenHermes-2.5",
        max_samples=20000,
        description="1M instruction samples",
        size_gb=5,
    ),
    HFDataset(
        name="databricks/databricks-dolly-15k",
        max_samples=15000,
        description="Human-generated instruction data",
        size_gb=0.2,
    ),
    HFDataset(
        name="allenai/RLVR-Chat",
        max_samples=10000,
        description="Reinforcement Learning from Verifiable Rewards chat data",
        size_gb=2,
    ),
    
    # === Math/Reasoning ===
    HFDataset(
        name="meta-math/MetaMathQA",
        max_samples=20000,
        description="Math Q&A with step-by-step solutions",
        size_gb=1,
    ),
    HFDataset(
        name="gsm8k",
        max_samples=8000,
        description="Grade School Math 8K",
        size_gb=0.01,
    ),
    HFDataset(
        name="lighteval/MATH",
        max_samples=10000,
        description="Competition math problems",
        size_gb=0.05,
    ),
    
    # === Scientific ===
    HFDataset(
        name="allenai/sciq",
        max_samples=13000,
        description="Science exam questions",
        size_gb=0.05,
    ),
    HFDataset(
        name="allenai/openbookqa",
        max_samples=5000,
        description="Open-book science Q&A",
        size_gb=0.02,
    ),
    
    # === Vietnamese specific ===
    HFDataset(
        name="vietgpt/news_corpus",
        max_samples=10000,
        description="Vietnamese news corpus",
        size_gb=2,
    ),
    HFDataset(
        name="PhoATC",
        max_samples=5000,
        description="Vietnamese text classification",
        size_gb=0.1,
    ),
]


class HuggingFaceCollector:
    """Collect training data từ HuggingFace Hub.
    
    Usage:
        collector = HuggingFaceCollector(cache_dir="./data_cache/hf")
        for sample in collector.collect(CURATED_DATASETS[:3]):
            print(sample["text"][:100])
    """
    
    def __init__(
        self,
        cache_dir: str = "./data_cache/hf",
        token: Optional[str] = None,
    ):
        self.cache_dir = cache_dir
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        os.makedirs(cache_dir, exist_ok=True)
    
    def collect(self, datasets: List[HFDataset]) -> Iterator[Dict[str, Any]]:
        """Collect samples từ list of HF datasets.
        
        Yields:
            Dict with keys: text, source, language, metadata
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets lib not installed. Run: pip install datasets")
            return
        
        for ds in datasets:
            try:
                yield from self._collect_dataset(ds, load_dataset)
            except Exception as e:
                logger.error(f"Failed to collect {ds.name}: {e}")
                continue
    
    def _collect_dataset(
        self,
        ds: HFDataset,
        load_fn,
    ) -> Iterator[Dict[str, Any]]:
        """Collect từ một dataset."""
        logger.info(f"Loading {ds.name} ({ds.subset or 'default'})...")
        
        try:
            if ds.streaming:
                dataset = load_fn(
                    ds.name,
                    name=ds.subset,
                    split=ds.split,
                    streaming=True,
                    token=self.token,
                )
            else:
                dataset = load_fn(
                    ds.name,
                    name=ds.subset,
                    split=ds.split,
                    token=self.token,
                    cache_dir=self.cache_dir,
                )
        except Exception as e:
            logger.error(f"Failed to load {ds.name}: {e}")
            return
        
        count = 0
        text_field = ds.field_mapping.get("text", "text")
        
        for item in dataset:
            if count >= ds.max_samples:
                break
            
            # Extract text using field mapping
            text = item.get(text_field) or item.get("text") or item.get("content") or ""
            
            if not text or not isinstance(text, str):
                # Try concatenating fields
                text = " ".join(str(v) for v in item.values() if isinstance(v, str))
            
            if not text or len(text) < 50:
                continue
            
            yield {
                "text": text,
                "source": ds.name,
                "language": ds.language or "text",
                "metadata": {
                    "dataset": ds.name,
                    "subset": ds.subset,
                    "split": ds.split,
                    "original_size": len(text),
                },
            }
            count += 1
        
        logger.info(f"Collected {count} samples from {ds.name}")
    
    def list_available(self) -> List[HFDataset]:
        """Return curated list of datasets."""
        return CURATED_DATASETS
    
    def estimate_total_size(self, datasets: List[HFDataset]) -> float:
        """Estimate total size in GB."""
        return sum(ds.size_gb or 0 for ds in datasets)
