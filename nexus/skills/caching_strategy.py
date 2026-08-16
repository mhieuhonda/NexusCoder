"""Caching Strategy Skill - Cache design analysis + Redis / Memcached config.

Phân tích chiến lược cache: write-through, write-back, cache-aside, refresh-ahead;
TTL & eviction policy; stampede protection (lock + jitter); Redis config mẫu.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


CACHE_STRATEGIES = """
Cache Strategies Comparison / So sánh chiến lược cache
=======================================================

| Strategy        | Write Path        | Pros                        | Cons                          |
|-----------------|-------------------|-----------------------------|-------------------------------|
| Cache-Aside     | App updates both  | Simple, resilient           | Stale on failure             |
| Write-Through   | Cache then DB    | Strong consistency          | Higher write latency         |
| Write-Back      | Cache only (async)| Fast writes                | Data loss risk on crash      |
| Refresh-Ahead   | Pre-emptive      | Hides latency for hot keys  | Extra infra & complexity     |

Eviction Policies: LRU (Redis default), LFU, FIFO, TTL-based
Invalidation: explicit `DEL`, key-bucket versioning, pub/sub bust, tag-based (Redis 7.4)

Stampede Protection:
  - Single-flight lock (SET NX EX 30) before recompute
  - Early refresh (TTL * 0.8) with random jitter
  - Bloom filter for negative caching (anti cache-penetration)
"""

REDIS_CONFIG = """# redis.conf — Production tuning / Cấu hình production
bind 0.0.0.0
protected-mode yes
port 6379
tcp-keepalive 300
timeout 0

# Memory & eviction / Bộ nhớ & loại bỏ
maxmemory 4gb
maxmemory-policy allkeys-lru        # evict least-recently-used
lfu-log-factor 10

# Persistence / Độ bền dữ liệu
appendonly yes
appendfsync everysec               # balance durability vs throughput
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
save 900 1
save 300 10

# Replication / Sao chép
replica-read-only yes
repl-backlog-size 64mb

# Security / Bảo mật
requirepass ${REDIS_PASSWORD}
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command KEYS ""
"""

STAMPEDE_GUARD = '''"""Cache-aside with stampede protection / Cache-aside chống dồn dập."""
import json, time, random, uuid
import redis

r = redis.Redis(host="redis", port=6379, decode_responses=True)
LOCK_TTL = 30   # seconds
BASE_TTL = 3600

def cached(key: str, loader, ttl: int = BASE_TTL):
    val = r.get(key)
    if val is not None:
        return json.loads(val)

    # Single-flight: acquire lock to recompute / Chỉ 1 worker recompute
    lock = f"{key}:lock"
    token = str(uuid.uuid4())
    if not r.set(lock, token, nx=True, ex=LOCK_TTL):
        time.sleep(0.05 + random.random() * 0.1)  # jitter
        return cached(key, loader, ttl)            # retry read

    try:
        val = loader()
        # Early-refresh: keep value warm but mark as stale soon / Giữ ấm giá trị
        effective_ttl = int(ttl * 0.8) + random.randint(0, 60)
        r.setex(key, effective_ttl, json.dumps(val))
        return val
    finally:
        # Lua CAS to safely release our lock (avoid removing others')
        r.eval(
            "if redis.call('get',KEYS[1])==ARGV[1] then return redis.call('del',KEYS[1]) end",
            1, lock, token,
        )
'''


class CachingStrategySkill(Skill):
    """Phân tích & sinh caching strategy + Redis/Memcached config."""

    category = SkillCategory.SYSTEM
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "cache", "caching", "redis", "memcached", "cache invalidation",
        "cache stampede", "cache-aside", "write-through", "eviction",
        "ttl", "lru", "lfu", "cdn",
    ]
    examples = [
        "Thiết kế caching cho API endpoint",
        "Setup Redis cluster with stampede protection",
        "Choose cache eviction policy for session store",
    ]

    @property
    def name(self) -> str:
        return "caching_strategy"

    @property
    def description(self) -> str:
        return (
            "Phân tích cache strategy (cache-aside / write-through / write-back) "
            "+ sinh Redis config với stampede protection và eviction tuning."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.15
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        prompt_lower = (context.prompt or "").lower()
        # Pick default strategy / Chọn strategy mặc định
        if "write-through" in prompt_lower:
            strategy = "write_through"
        elif "write-back" in prompt_lower or "writeback" in prompt_lower:
            strategy = "write_back"
        elif "refresh-ahead" in prompt_lower or "refresh_ahead" in prompt_lower:
            strategy = "refresh_ahead"
        else:
            strategy = "cache_aside"

        backend = "memcached" if "memcached" in prompt_lower else "redis"

        artifacts: List[Dict[str, str]] = [
            {"name": "CACHE_STRATEGIES.md", "language": "markdown", "content": CACHE_STRATEGIES},
            {"name": "redis.conf", "language": "ini", "content": REDIS_CONFIG},
            {"name": "stampede_guard.py", "language": "python", "content": STAMPEDE_GUARD},
        ]

        return SkillResult(
            success=True,
            output=(
                f"[caching_strategy] strategy={strategy} | backend={backend}\n"
                f"Generated strategy doc + {backend} config + stampede guard."
            ),
            artifacts=artifacts,
            suggestions=[
                "Benchmark with realistic read/write ratio (e.g. 95/5 read-heavy)",
                "Add monitoring: hit ratio, latency p99, eviction rate, memory usage",
                "Use Redis Sentinel / Cluster for HA in production",
                "Negative cache empty results to prevent cache-penetration",
                "Tag-based invalidation (Redis 7.4+) for multi-key busts",
            ],
            metadata={
                "skill": self.name,
                "strategy": strategy,
                "backend": backend,
                "eviction_policy": "allkeys-lru",
                "version": self.version,
                "author": self.author,
            },
        )
