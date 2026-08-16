"""System Design Skill - Sinh design doc template cho distributed systems.

Bao gồm components, data flow, capacity estimation, trade-offs,
scalability analysis, và SLO definitions.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class SystemDesignSkill(Skill):
    """Sinh system design doc: components, data flow, trade-offs, capacity."""

    category = SkillCategory.SYSTEM
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "design", "architecture", "architect", "scalable",
        "microservices", "monolith", "load balancer", "load balancer",
        "sharding", "partitioning", "cache", "cdn", "queue",
        "pubsub", "event sourcing", "cqrs", "rate limiter",
        "system design", "capacity planning",
    ]
    examples = [
        "Design a URL shortener at scale",
        "Design a Twitter-like timeline system",
        "Architecture for a real-time chat backend",
    ]

    @property
    def name(self) -> str:
        return "system_design"

    @property
    def description(self) -> str:
        return (
            "Sinh system design doc template: requirements, capacity estimation, "
            "high-level architecture, components, data flow, detailed design, "
            "trade-offs (CAP, consistency), scalability, và SLO/SLA definitions."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.13
        if any(k in prompt_lower for k in ("design ", "architect", "scalable", "at scale")):
            score += 0.2
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output=f"[SystemDesign] Design doc template ready.",
            artifacts=[{"path": "design/design_doc.md", "content": _DESIGN_DOC}],
            metadata={
                "skill": self.name,
                "sections": [
                    "1. Requirements (functional + non-functional)",
                    "2. Capacity estimation (QPS, storage, bandwidth)",
                    "3. High-level architecture (diagram)",
                    "4. Component design",
                    "5. Data model & schema",
                    "6. Data flow (sync/async)",
                    "7. Trade-offs (CAP, consistency vs availability)",
                    "8. Scalability (horizontal, sharding, caching)",
                    "9. Reliability (redundancy, failover, idempotency)",
                    "10. SLO/SLA + monitoring",
                    "11. Security & compliance",
                ],
                "patterns": {
                    "load_balancing": ["L4 (NLB)", "L7 (ALB / NGINX / Envoy)", "DNS geo"],
                    "caching": ["Redis (hot keys)", "CDN (static)", "application-local (LRU)"],
                    "queueing": ["Kafka (log)", "RabbitMQ (AMQP)", "SQS (managed)"],
                    "db_scaling": ["read replicas", "sharding (hash/range)", "CQRS + ES"],
                    "idempotency": ["client request id + dedup table", "idempotency-key header"],
                },
                "capacity_rules_of_thumb": {
                    "qps_per_core": "~500-2000 (API CRUD)",
                    "read_write_ratio": "100:1 (read-heavy), 1:1 (transactional)",
                    "cache_hit_rate_target": "> 95%",
                    "storage_overhead": "~1.5x raw (indexes + replication)",
                },
                "slo_template": {
                    "availability": "99.9% (monthly)",
                    "latency_p99": "< 500ms for read, < 2s for write",
                    "error_budget": "0.1% per month",
                },
            },
            suggestions=[
                "Start with capacity estimation — it drives every component choice",
                "Pick the simplest architecture that meets SLO — don't over-engineer",
                "Document trade-offs explicitly (CAP, latency vs consistency)",
                "Define idempotency keys for all mutating endpoints",
                "Design for failure: every dependency has a circuit breaker + fallback",
            ],
        )


_DESIGN_DOC = '''# System Design Document

## 1. Requirements

### Functional
- <F1: user can ...>
- <F2: system must ...>

### Non-functional
- Availability: 99.9% (3 nines)
- Latency: p99 < 500ms (read), < 2s (write)
- Throughput: 10k QPS read, 1k QPS write
- Consistency: eventual (read-replica lag < 1s)
- Durability: 99.999999999% (11 nines)

## 2. Capacity Estimation

| Metric            | Value             | Calculation                       |
|-------------------|-------------------|-----------------------------------|
| DAU               | 1,000,000         | given                             |
| Read QPS (peak)   | ~11,600           | DAU * 50 reads / 3600s * 3x burst |
| Write QPS (peak)  | ~1,200            | DAU * 5 writes / 3600s * 3x burst |
| Storage / year    | ~12 TB            | 1M * 5 writes * 365 * 0.6 KB      |
| Bandwidth (out)   | ~50 Mbps          | 11.6k * 0.5 KB                    |
| Cache size        | ~5 GB hot         | top 1% keys * 0.5 KB              |

## 3. High-Level Architecture

```
            ┌─────────┐    ┌──────────────┐
  clients ─▶│  CDN    │───▶│  Load Balancer│
            └─────────┘    └──────┬───────┘
                                   │
                  ┌────────────────┼────────────────┐
                  ▼                ▼                 ▼
            ┌──────────┐    ┌──────────┐      ┌──────────┐
            │ API (x N)│    │ API (x N)│      │ API (x N)│   (stateless, autoscaled)
            └────┬─────┘    └────┬─────┘      └────┬─────┘
                 │               │                 │
            ┌────┴───────────────┴─────────────────┴────┐
            │              Service Mesh (mTLS)           │
            └────┬───────────────┬─────────────────┬────┘
                 ▼               ▼                 ▼
           ┌──────────┐   ┌──────────┐      ┌──────────┐
           │ Postgres │   │  Redis   │      │  Kafka   │
           │ (primary │   │ (cache + │      │ (event   │
           │  + 3 RR) │   │  pubsub) │      │  log)    │
           └──────────┘   └──────────┘      └────┬─────┘
                                                  ▼
                                          ┌──────────────┐
                                          │  Workers     │ (async consumers)
                                          └──────────────┘
```

## 4. Component Design

| Component        | Tech                  | Reason                            |
|------------------|-----------------------|-----------------------------------|
| API              | Python (FastAPI)     | async, typed, OpenAPI             |
| Primary DB       | PostgreSQL 16         | ACID, mature, logical replication |
| Cache            | Redis 7 (cluster)     | sub-ms, pub/sub, streams          |
| Queue            | Kafka 3               | replay, partitioning, exactly-once|
| Search           | OpenSearch            | full-text, faceted                |
| Object storage   | S3                    | 11 nines durability, lifecycle    |
| CDN              | CloudFront            | edge POPs, signed URLs            |

## 5. Data Model

- `users(id PK, email UK, created_at)` — partition by hash(id), 64 shards
- `objects(id PK, owner_id FK, ...)` — partition by hash(owner_id)
- Indexes: `(owner_id, created_at DESC)` for timeline lookups

## 6. Data Flow

### Write (synchronous)
client -> LB -> API -> Postgres (primary) -> binlog -> Debezium -> Kafka -> workers
                                └-> invalidate Redis key

### Read (cached)
client -> CDN (static) | LB -> API -> Redis (hit) ? return : Postgres (RR) -> warm Redis

## 7. Trade-offs

| Decision               | Trade-off                                  |
|------------------------|--------------------------------------------|
| Eventual consistency   | Stronger availability + lower read latency |
| Read replicas          | Lag up to ~1s; tolerate via read-your-write|
| Kafka over RabbitMQ    | Replay + partition scalability; complexity |
| Hash sharding          | Even distribution; resharding is expensive  |

## 8. Scalability

- Horizontal scale API behind ALB (target CPU 60%)
- DB: 1 primary + N read replicas; partition large tables by hash (64 shards)
- Cache: Redis Cluster, write-through for hot keys, TTL 5 min for cold
- Queue: 32 partitions, consumer group per service

## 9. Reliability

- Multi-AZ deployment; warm standby in secondary region
- Circuit breakers on every outbound call (pybreaker / resilience4j)
- Idempotency: `Idempotency-Key` header + dedup table (24h TTL)
- Backups: nightly WAL archive to S3, monthly restore drill

## 10. SLO / SLA

| SLO                | Target       | Error budget (30d) |
|--------------------|--------------|-------------------|
| Availability       | 99.9%        | 43m 12s           |
| Read latency p99   | < 500 ms     | —                 |
| Write latency p99  | < 2 s        | —                 |

## 11. Security & Compliance

- TLS 1.3 everywhere; mTLS in service mesh
- OAuth2 + OIDC for users; short-lived JWT (15 min) + refresh tokens
- PII encrypted at rest (AES-256, KMS-managed keys)
- Audit log to immutable storage (S3 Object Lock)
'''
