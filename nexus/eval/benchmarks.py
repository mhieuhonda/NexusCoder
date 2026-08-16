"""Benchmark Suite - Đánh giá model trên multiple benchmarks."""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class BenchmarkType(str, Enum):
    MMLU = "mmlu"           # General knowledge
    HUMANEVAL = "humaneval"  # Code generation
    GSM8K = "gsm8k"          # Math reasoning
    BBH = "bbh"              # Big-bench hard
    truthful_qa = "truthful_qa"
    MT_BENCH = "mt_bench"    # Multi-turn chat
    VI_BENCH = "vi_bench"    # Vietnamese specific


@dataclass
class Benchmark:
    """Một benchmark evaluation."""
    name: str
    type: BenchmarkType
    description: str
    num_examples: int
    languages: List[str] = field(default_factory=lambda: ["en"])
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    estimated_time_minutes: int = 30


class BenchmarkSuite:
    """Run model on multiple benchmarks.
    
    Usage:
        suite = BenchmarkSuite()
        suite.add(Benchmark(name="humaneval", ...))
        results = suite.run(model, tokenizer)
    """
    
    SUPPORTED_BENCHMARKS = [
        Benchmark(
            name="humaneval",
            type=BenchmarkType.HUMANEVAL,
            description="HumanEval - Code generation (164 problems)",
            num_examples=164,
            languages=["en"],
            metrics=["pass@1", "pass@10"],
            estimated_time_minutes=60,
        ),
        Benchmark(
            name="mbpp",
            type=BenchmarkType.HUMANEVAL,
            description="MBPP - Mostly Basic Python Problems (974 problems)",
            num_examples=974,
            languages=["en"],
            metrics=["pass@1"],
            estimated_time_minutes=90,
        ),
        Benchmark(
            name="gsm8k",
            type=BenchmarkType.GSM8K,
            description="Grade School Math 8K",
            num_examples=1319,
            languages=["en"],
            metrics=["accuracy"],
            estimated_time_minutes=45,
        ),
        Benchmark(
            name="mmlu",
            type=BenchmarkType.MMLU,
            description="Massive Multitask Language Understanding",
            num_examples=14042,
            languages=["en"],
            metrics=["accuracy"],
            estimated_time_minutes=120,
        ),
        Benchmark(
            name="bbh",
            type=BenchmarkType.BBH,
            description="BIG-Bench Hard (23 tasks)",
            num_examples=6511,
            languages=["en"],
            metrics=["accuracy"],
            estimated_time_minutes=180,
        ),
        Benchmark(
            name="truthful_qa",
            type=BenchmarkType.truthful_qa,
            description="TruthfulQA - Measure truthfulness",
            num_examples=817,
            languages=["en"],
            metrics=["truthful", "informative"],
            estimated_time_minutes=20,
        ),
        Benchmark(
            name="mt_bench",
            type=BenchmarkType.MT_BENCH,
            description="Multi-turn benchmark for chat assistants",
            num_examples=80,
            languages=["en"],
            metrics=["gpt4_score", "judge_score"],
            estimated_time_minutes=30,
        ),
        Benchmark(
            name="vi_bench",
            type=BenchmarkType.VI_BENCH,
            description="Vietnamese language understanding",
            num_examples=500,
            languages=["vi"],
            metrics=["accuracy", "fluency"],
            estimated_time_minutes=15,
        ),
    ]
    
    def __init__(self):
        self._benchmarks: Dict[str, Benchmark] = {
            b.name: b for b in self.SUPPORTED_BENCHMARKS
        }
        self._results: Dict[str, Dict] = {}
    
    def add(self, benchmark: Benchmark) -> None:
        self._benchmarks[benchmark.name] = benchmark
    
    def list_available(self) -> List[Benchmark]:
        return list(self._benchmarks.values())
    
    def run(
        self,
        model,
        tokenizer,
        benchmarks: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks on model.
        
        Args:
            model: NexusCoderForCausalLM
            tokenizer: NexusTokenizer
            benchmarks: List of benchmark names (None = all)
            sample_size: Limit examples per benchmark (for quick eval)
        """
        to_run = benchmarks or list(self._benchmarks.keys())
        results = {}
        
        for name in to_run:
            if name not in self._benchmarks:
                results[name] = {"error": f"Unknown benchmark: {name}"}
                continue
            
            bench = self._benchmarks[name]
            results[name] = {
                "status": "not_implemented",
                "benchmark": bench.name,
                "description": bench.description,
                "num_examples": bench.num_examples,
                "sample_size": sample_size,
                "note": "Evaluation requires downloading dataset. Run scripts/evaluate.py with --download flag.",
            }
        
        self._results = results
        return results
    
    def summary(self) -> str:
        """Generate summary report."""
        if not self._results:
            return "No results yet. Run benchmarks first."
        
        lines = ["Benchmark Results Summary", "=" * 50]
        for name, result in self._results.items():
            if "error" in result:
                lines.append(f"  {name}: ERROR - {result['error']}")
            elif "scores" in result:
                lines.append(f"  {name}: {result['scores']}")
            else:
                lines.append(f"  {name}: {result.get('status', 'unknown')}")
        return "\n".join(lines)
