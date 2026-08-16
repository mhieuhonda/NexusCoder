"""
omp-gym-inspired benchmark hooks for Nexus Coder v0.3
=====================================================
Ported & simplified from dylantirandaz/omp-gym (MIT).

omp-gym provides OpenMP performance benchmarks as a gym environment.
We adapt the IDEA (sample real OpenMP programs of varying complexity,
have the model predict an optimization) into a benchmark hook that
Nexus Coder's evaluation pipeline can consume.

This is an EVALUATION-only adapter — it does not train anything.

Original attribution:
    omp-gym: An OpenMP optimization gym environment.
    Author: Dylan Tirandaz
    License: MIT
    Source: https://github.com/dylantirandaz/omp-gym
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class OMPTask:
    """A single OpenMP optimization task."""
    task_id: str
    source_code: str                          # original C/C++ with OpenMP pragmas
    language: str = "c"                       # c | cpp
    target_metric: str = "speedup"           # speedup | cache_misses | energy
    ground_truth: Optional[str] = None       # optimized code (if known)
    description: Optional[str] = None
    difficulty: str = "medium"                # easy | medium | hard
    parallel_pattern: str = "for"            # for | sections | task | simd


# Curated sample tasks (synthetic, illustrative)
SAMPLE_TASKS: List[OMPTask] = [
    OMPTask(
        task_id="omp_pi_001",
        source_code="""
#include <omp.h>
double compute_pi(long n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; i++) {
        double x = (i + 0.5) / n;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum / n;
}
""",
        description="Compute pi via numerical integration. Already uses reduction.",
        difficulty="easy",
        parallel_pattern="for",
        target_metric="speedup",
    ),
    OMPTask(
        task_id="omp_matmul_002",
        source_code="""
void matmul(double *A, double *B, double *C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++) {
                s += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = s;
        }
    }
}
""",
        description="Naive matrix multiply. Optimize with cache blocking, SIMD, scheduling.",
        difficulty="hard",
        parallel_pattern="for",
        target_metric="speedup",
    ),
    OMPTask(
        task_id="omp_task_003",
        source_code="""
long fib(int n) {
    if (n < 2) return n;
    long a, b;
    #pragma omp task shared(a)
    a = fib(n - 1);
    #pragma omp task shared(b)
    b = fib(n - 2);
    #pragma omp taskwait
    return a + b;
}
""",
        description="Recursive Fibonacci with OpenMP tasks. Optimize cutoff.",
        difficulty="medium",
        parallel_pattern="task",
        target_metric="speedup",
    ),
]


def load_omp_benchmarks() -> List[OMPTask]:
    """Load all available OMP benchmark tasks.
    Returns a static list for now; future versions may pull from the
    upstream omp-gym dataset (or scrape C/C++ programs from GitHub).
    """
    return list(SAMPLE_TASKS)


def evaluate_prediction(
    task: OMPTask,
    predicted_code: str,
    speedup_factor: Optional[float] = None,
    cache_miss_reduction: Optional[float] = None,
) -> Dict[str, float]:
    """Score a predicted optimization against the original.

    Returns a dict of metrics. Higher = better. 0.0 = no improvement
    (or regression).
    """
    score: Dict[str, float] = {"valid": 1.0 if predicted_code.strip() else 0.0}
    if speedup_factor is not None:
        # log-scale reward: 2x speedup → 1.0, 1x → 0.0, 0.5x → -1.0
        import math
        score["speedup_reward"] = math.log2(max(0.01, speedup_factor))
    if cache_miss_reduction is not None:
        score["cache_reward"] = float(cache_miss_reduction)
    # Heuristic: did the model actually add new pragmas?
    if "#pragma" in predicted_code and predicted_code != task.source_code:
        score["modified"] = 1.0
    else:
        score["modified"] = 0.0
    score["total"] = sum(v for k, v in score.items() if k != "valid") / max(1, len(score) - 1)
    return score


__all__ = ["OMPTask", "SAMPLE_TASKS", "load_omp_benchmarks", "evaluate_prediction"]
