"""Microservices Skill - Sinh microservices architecture template.

Bao gồm: service boundaries (DDD bounded contexts), communication patterns
(sync REST/gRPC, async events), data ownership, service mesh, observability,
và deployment topology.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class MicroservicesSkill(Skill):
    """Sinh microservices architecture template (boundaries + comms)."""

    category = SkillCategory.SYSTEM
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "microservice", "microservices", "service mesh", "grpc",
        "service boundary", "bounded context", "event driven",
        "event sourcing", "cqrs", "saga", "service decomposition",
        "kiến trúc microservices",
    ]
    examples = [
        "Design microservices for an e-commerce platform",
        "How to split a monolith into microservices (strangler fig)",
        "Service mesh setup with Istio for mTLS + traffic shifting",
    ]

    @property
    def name(self) -> str:
        return "microservices"

    @property
    def description(self) -> str:
        return (
            "Sinh microservices architecture: service boundaries (DDD), "
            "communication patterns (REST/gRPC/events), data ownership, "
            "service mesh, observability, deployment."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.13
        if "microservice" in prompt_lower:
            score += 0.3
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output="[Microservices] Architecture template ready (boundaries + comms + ops).",
            artifacts=[
                {"path": "microservices/architecture.md", "content": _ARCHITECTURE_DOC},
                {"path": "microservices/communication_patterns.md", "content": _COMMUNICATION_PATTERNS},
            ],
            metadata={
                "skill": self.name,
                "service_decomposition": {
                    "strategies": [
                        "DDD bounded contexts — decompose by business capability",
                        "Strangler Fig — gradually extract from monolith",
                        "Domain-driven subdomain analysis (core / supporting / generic)",
                        "Conway's Law — team topology drives service boundaries",
                    ],
                    "guidelines": [
                        "Service owns its data (no shared DB)",
                        "Single responsibility per service",
                        "Service should be replaceable in 2 weeks of work",
                        "Bounded context = team ownership boundary",
                        "Avoid chatty services (network is slow)",
                    ],
                },
                "communication_patterns": {
                    "sync_rest": "HTTP/JSON — simple, ubiquitous, debugging friendly",
                    "sync_grpc": "HTTP/2 + Protobuf — fast, typed, streaming, strict contracts",
                    "async_events": "Kafka/RabbitMQ — decoupled, scalable, eventual consistency",
                    "async_cqrs": "Command + separate read model — optimize reads independently",
                    "saga_choreography": "Each service emits + listens to events (no orchestrator)",
                    "saga_orchestration": "Central orchestrator manages saga state (Temporal / Camunda)",
                },
                "data_patterns": {
                    "db_per_service": "Each service owns its schema (no shared tables)",
                    "event_sourcing": "Append-only event log is source of truth; projections for reads",
                    "cqrs": "Separate write model (commands) from read models (projections)",
                    "saga": "Distributed transaction = sequence of local tx + compensating actions",
                    "outbox_pattern": "Write to business table + outbox table in same tx; relay publishes",
                    "idempotency": "Every write op has idempotency key (dedup table or event log)",
                },
                "service_mesh": {
                    "purpose": "mTLS, traffic shifting, retries, circuit breaking without code changes",
                    "options": ["Istio", "Linkerd", "Consul Connect", "AWS App Mesh"],
                    "features": ["mTLS automatic", "canary / blue-green", "retry + timeout",
                                 "circuit breaker", "distributed tracing"],
                    "sidecar_overhead": "+50-200ms p99 latency, +100-300MB RAM per pod",
                },
                "observability": {
                    "logs": "Structured JSON, correlation ID across services",
                    "metrics": "RED (Rate, Errors, Duration) per service",
                    "traces": "OpenTelemetry, Jaeger/Tempo — distributed tracing",
                    "alerts": "SLO-based (error budget burn), per-service dashboards",
                },
                "deployment": {
                    "containers": "Docker image per service (multi-stage build, distroless)",
                    "orchestration": "Kubernetes (Deployments, Services, Ingress, HPA)",
                    "ci_cd": "Per-service pipeline; independent deploy cadence",
                    "infrastructure": "Terraform — GitOps via ArgoCD / Flux",
                    "database_migration": "Per-service; expand/contract pattern (no breaking)",
                },
                "trade_offs": {
                    "monolith_pros": "Simple deployment, fast local calls, easy debugging",
                    "monolith_cons": "Scaling whole app, team conflicts, slow build",
                    "microservices_pros": "Independent scaling, deploy cadence, team autonomy",
                    "microservices_cons": "Network complexity, distributed tx pain, ops overhead",
                },
                "when_not_to_use_microservices": [
                    "Team < 10 engineers — monolith is faster",
                    "Single domain (no clear bounded contexts)",
                    "Tight latency requirements (sub-ms)",
                    "No platform engineering team (SRE)",
                    "Immature CI/CD (no automated deploy pipeline)",
                ],
            },
            suggestions=[
                "Specify the business domain (e-commerce, fintech, social, etc.)",
                "Indicate team size + org structure (Conway's Law)",
                "Mention if extracting from existing monolith (strangler fig)",
                "Ask for specific pattern (CQRS, event sourcing, saga) deep-dive",
            ],
        )


_ARCHITECTURE_DOC = '''# Microservices Architecture — E-commerce Example

Author: Hieu Louis (2026)

## 1. Service Boundaries (DDD Bounded Contexts)

| Service         | Bounded Context   | Owns                                | Tech            |
|-----------------|-------------------|-------------------------------------|-----------------|
| user-svc        | Identity          | users, auth, profiles              | Go + Postgres   |
| catalog-svc     | Product Catalog   | products, categories, inventory    | Java + Postgres |
| cart-svc        | Shopping Cart      | cart items, session carts          | Node + Redis    |
| order-svc       | Order              | orders, order items, fulfillment   | Python + Postgres |
| payment-svc     | Payment            | transactions, refunds, gateway     | Go + Postgres   |
| shipping-svc    | Logistics          | shipments, tracking, carriers      | Rust + Postgres |
| notification-svc| Notifications      | email, SMS, push templates         | Node + Mongo    |
| search-svc      | Search             | product search index, autocomplete | Go + OpenSearch|

## 2. Topology

```
                          ┌──────────────────┐
                          │   API Gateway    │   (Kong / Apollo Router / NGINX)
                          │  (auth, rate)    │
                          └────────┬─────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        ┌──────────┐         ┌──────────┐         ┌──────────┐
        │ user-svc │         │catalog-svc│        │ cart-svc │
        └────┬─────┘         └────┬─────┘         └────┬─────┘
             │                    │                    │
             │                    │                    │
             └──────────┬─────────┴──────────┬─────────┘
                        ▼                    ▼
                  ┌──────────┐         ┌──────────┐
                  │ order-svc│         │payment-svc│
                  └────┬─────┘         └────┬─────┘
                       │                    │
                       └────────┬───────────┘
                                ▼
                         ┌──────────────┐
                         │ Kafka (events)│
                         └──────┬───────┘
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                  ▼
        ┌──────────┐    ┌──────────┐       ┌─────────────────┐
        │shipping- │    │notif-svc │       │ search-svc      │
        │  svc     │    │          │       │ (reads events)  │
        └──────────┘    └──────────┘       └─────────────────┘
```

## 3. Communication Patterns

### Synchronous (request/response)
- **user-svc**: REST + JWT (external clients via gateway)
- **catalog-svc -> cart-svc**: gRPC (typed contract, fast)
- **order-svc -> payment-svc**: gRPC (synchronous authorization needed)

### Asynchronous (events)
- **order-svc** publishes `OrderCreated` event to Kafka
- **payment-svc** consumes `OrderCreated`, publishes `PaymentCompleted` / `PaymentFailed`
- **shipping-svc** consumes `PaymentCompleted`, publishes `ShipmentDispatched`
- **notification-svc** consumes all events, sends user notifications
- **search-svc** consumes `ProductCreated`/`ProductUpdated` to rebuild index

### Saga (orchestrated)
- Orchestrator: `order-svc` (uses Temporal workflow)
- Steps:
  1. Create order (pending)
  2. Reserve inventory (catalog-svc) — compensating: release reservation
  3. Charge payment (payment-svc) — compensating: refund
  4. Create shipment (shipping-svc) — compensating: cancel shipment
  5. Mark order complete

## 4. Data Ownership

| Data               | Owner           | Read by (via event projection)  |
|--------------------|-----------------|--------------------------------|
| users              | user-svc        | order-svc, notification-svc    |
| products           | catalog-svc     | cart-svc, order-svc, search-svc|
| cart_items         | cart-svc        | order-svc (snapshot at checkout) |
| orders             | order-svc       | notification-svc, shipping-svc  |
| payments           | payment-svc     | order-svc (via event)            |
| shipments          | shipping-svc    | order-svc (via event)            |

**Rule**: No service reads another service's database directly. All cross-service
reads go via: (a) synchronous API call, or (b) local projection built from events.

## 5. Service Mesh

- **Istio** with automatic mTLS
- **Traffic shifting**: 5% canary -> 25% -> 100% over 1 hour
- **Circuit breaker**: 5 consecutive 5xx -> open for 30s
- **Retry**: 3 attempts, exponential backoff (50ms, 200ms, 800ms)
- **Timeout**: 2s per call (gRPC deadline), 30s total chain

## 6. Observability

- **Logs**: structlog JSON with `trace_id` propagated via OpenTelemetry
- **Metrics**: Prometheus scraping `/metrics` per service (RED: Rate, Errors, Duration)
- **Traces**: OpenTelemetry SDK -> Jaeger; sampling 10% in prod, 100% in staging
- **Alerts**: SLO-based — error budget burn > 2% in 1h pages on-call

## 7. Deployment

- **Container**: Multi-stage Dockerfile, distroless base, ~30MB image
- **Orchestration**: Kubernetes with HPA (CPU 60%, custom QPS metric)
- **CI/CD**: GitHub Actions -> build image -> push to ECR -> ArgoCD deploys
- **Database migration**: expand/contract (add column, deploy, backfill, switch, drop)
- **Rollback**: ArgoCD one-click; DB rollback manual (contract changes)

## 8. Resilience Patterns

- **Circuit breaker** on all outbound calls (gRPC + HTTP)
- **Bulkhead**: separate thread pools per downstream (prevent cascade)
- **Timeout**: every call has explicit deadline
- **Retry with jitter**: prevent thundering herd
- **Idempotency key**: all POST/PATCH mutations
- **Outbox pattern**: atomic write to business table + event outbox

## 9. Anti-patterns to Avoid

- **Distributed monolith**: services deployed together, tight coupling, shared DB
- **Chatty services**: 10+ round trips for one user request -> batch or co-locate
- **Shared library bloat**: every service pulls in 100MB shared "common" lib
- **CRUD over events**: events mirror DB rows instead of business intent
- **Synchronous chains**: A -> B -> C -> D (4 sync calls) — convert to async events
- **Premature decomposition**: split before domain is understood
'''


_COMMUNICATION_PATTERNS = '''# Microservices Communication Patterns

## 1. Synchronous

### REST (HTTP/JSON)
- **When**: External clients, public API, simple CRUD
- **Pros**: ubiquitous, debuggable, content negotiation
- **Cons**: no streaming, verbose, latency overhead
- **Tools**: FastAPI (Py), Express (JS), Gin (Go), Spring Boot (Java)

### gRPC (HTTP/2 + Protobuf)
- **When**: Internal service-to-service, low-latency, streaming
- **Pros**: typed contracts, 7-10x faster than REST, bidirectional streaming
- **Cons**: harder to debug (binary), browser needs grpc-web proxy
- **Tools**: grpc (multi-lang), buf (schema management), connect-go

### GraphQL Federation
- **When**: Multiple services compose one GraphQL API for clients
- **Pros**: clients query exactly what they need, gateway composes
- **Cons**: complex to operate, N+1 risk without DataLoader
- **Tools**: Apollo Federation, Router

## 2. Asynchronous

### Pub/Sub Events
- **When**: Decoupled services, eventual consistency OK, fan-out
- **Pros**: producer doesn't wait for consumer, scalability
- **Cons**: harder to debug, ordering guarantees tricky
- **Tools**: Kafka (log), RabbitMQ (AMQP), NATS JetStream, SNS+SQS

### Event Sourcing
- **When**: Audit trail required, temporal queries, complex state transitions
- **Pros**: replayable, debuggable, natural CQRS fit
- **Cons**: event schema evolution is hard, eventually consistent reads
- **Tools**: EventStoreDB, Kafka (as event log), Axon Framework

### CQRS (Command Query Responsibility Segregation)
- **When**: Read/write workloads differ significantly (100:1 reads)
- **Pros**: optimize read model independently (denormalized, indexed)
- **Cons**: eventual consistency between write and read models
- **Tools**: custom projections, Kafka Streams, Materialize

## 3. Distributed Transactions

### Saga (Choreography)
- Each service emits event on local commit
- Next service listens, performs local tx, emits next event
- Compensating actions on failure
- **Pros**: no central orchestrator, naturally decoupled
- **Cons**: hard to follow flow, debugging is painful

### Saga (Orchestration)
- Central orchestrator (Temporal / Camunda / AWS Step Functions)
- Sends commands to services, awaits responses, tracks state
- Compensating actions on failure
- **Pros**: visible workflow, easier debugging, retries built-in
- **Cons**: orchestrator becomes coupling point

## 4. Reliable Delivery Patterns

### Outbox Pattern
```
BEGIN TX
  INSERT INTO orders (...) VALUES (...);
  INSERT INTO outbox (event_type, payload) VALUES ('OrderCreated', ...);
COMMIT TX
-- Relay process publishes outbox rows to Kafka, marks as published
```
- Guarantees: business write + event publish atomic
- Trade-off: eventual consistency (relay runs periodically)

### Idempotent Consumer
- Consumer stores processed message IDs (dedup table)
- Skip if already processed
- Critical for at-least-once delivery systems (Kafka)

### Inbox Pattern
- Consumer writes received event to `inbox` table first
- Process inbox in transaction with business logic
- Guarantees: no double-processing of duplicate events

## 5. Service Mesh (cross-cutting)

| Feature            | Implementation                       |
|--------------------|--------------------------------------|
| mTLS               | Sidecar (Envoy) rotates certs       |
| Retry + timeout    | Sidecar retries on 5xx (configurable)|
| Circuit breaker    | Sidecar trips after N failures       |
| Traffic shifting   | VirtualService + DestinationRule    |
| Distributed tracing| Sidecar injects trace headers       |
| Load balancing     | L7 (round-robin, least-conn, random)|

## 6. Failure Modes & Mitigations

| Failure                  | Detection                  | Mitigation                  |
|--------------------------|----------------------------|-----------------------------|
| Downstream service down   | Health check fails         | Circuit breaker + fallback  |
| Slow downstream          | p99 latency > threshold    | Timeout + bulkhead          |
| Network partition        | Failed health checks       | Bulkhead + retry to replica|
| Cascade failure          | Cascading 5xx              | Bulkhead + shed load (429)  |
| Thundering herd on recovery | Cache stampede          | Jittered retry + warm-up    |
| Duplicate event          | Dedup table miss           | Idempotent consumer         |
'''
