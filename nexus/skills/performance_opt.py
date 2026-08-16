"""Performance Optimization Skill - Tối ưu hiệu năng."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class PerformanceOptimizationSkill(Skill):
    """Tối ưu hiệu năng: profiling, bottleneck identification, optimization."""
    
    category = SkillCategory.DEVOPS
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "optimize", "tối ưu", "performance", "hiệu năng",
        "slow", "chậm", "latency", "throughput",
        "profile", "benchmark", "bottleneck", "speedup",
    ]
    
    @property
    def name(self) -> str:
        return "performance_optimization"
    
    @property
    def description(self) -> str:
        return (
            "Tối ưu hiệu năng: profiling, bottleneck identification, "
            "algorithmic optimization, memory optimization, "
            "concurrency/parallelism, caching, I/O optimization."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        optimizations = {
            "algorithmic": [
                "Replace O(n²) with O(n log n)",
                "Memoization / caching intermediate results",
                "Lazy evaluation",
                "Early termination",
            ],
            "memory": [
                "Generator vs list (reduce memory)",
                "Slots in classes (__slots__)",
                "Avoid unnecessary object creation",
                "Use array.array for numeric data",
                "Memory pool / object pool",
            ],
            "concurrency": [
                "asyncio for I/O-bound",
                "multiprocessing for CPU-bound",
                "Threading for I/O + GIL-friendly code",
                "Concurrent.futures.ThreadPoolExecutor",
                "Ray / Dask for distributed",
            ],
            "caching": [
                "functools.lru_cache",
                "Redis / Memcached",
                "HTTP caching headers",
                "CDN for static assets",
            ],
            "io": [
                "Batch I/O operations",
                "Streaming instead of loading all",
                "Connection pooling",
                "Async I/O (aiofiles, aiohttp)",
            ],
            "python_specific": [
                "Use builtins (map, filter, comprehensions)",
                "NumPy vectorization",
                "Cython / Numba for hot loops",
                "PyPy for compatible code",
            ],
        }
        return SkillResult(
            success=True,
            output=f"[PerformanceOpt] {len(optimizations)} optimization categories.",
            metadata={
                "skill": self.name,
                "optimizations": optimizations,
                "profilers": ["cProfile", "line_profiler", "memory_profiler", "py-spy"],
                "benchmarkers": ["pytest-benchmark", "timeit"],
            },
            suggestions=[
                "Profile before optimizing (don't guess)",
                "Optimize hot paths first (80/20 rule)",
                "Benchmark before/after each change",
            ],
        )
