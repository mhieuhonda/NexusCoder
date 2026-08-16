"""Logging Analytics Skill - Structured logging config + log analysis queries.

Sinh structured JSON logging (structlog / loguru), ELK / Loki /
Datadog / Splunk ingestion, và example KQL/Lucene queries.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class LoggingAnalyticsSkill(Skill):
    """Sinh structured logging config + log analysis query examples."""

    category = SkillCategory.DEVOPS
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "log", "logs", "logging", "elasticsearch", "logstash",
        "kibana", "datadog", "splunk", "loki", "grafana loki",
        "structured logging", "log analysis", "correlation id",
        "trace", "tracing", "jaeger", "zipkin", "otel",
    ]
    examples = [
        "Setup structured JSON logging in Python",
        "Write KQL query to find slow API requests",
        "Configure Datadog log pipeline for FastAPI",
    ]

    @property
    def name(self) -> str:
        return "logging_analytics"

    @property
    def description(self) -> str:
        return (
            "Sinh structured logging config (structlog / loguru / Python logging), "
            "ELK / Loki / Datadog / Splunk ingestion pipelines, correlation IDs, "
            "sampling, PII redaction, và example analysis queries (KQL / Lucene / Splunk SPL)."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.15
        return min(1.0, score)

    def _pick_platform(self, prompt: str) -> str:
        p = prompt.lower()
        if "datadog" in p:
            return "datadog"
        if "splunk" in p:
            return "splunk"
        if "loki" in p:
            return "loki"
        if "kibana" in p or "elasticsearch" in p or "kusto" in p or "kql" in p:
            return "elk"
        return "elk"

    def execute(self, context: SkillContext) -> SkillResult:
        platform = self._pick_platform(context.prompt)
        query_artifact = self._query_artifact(platform)

        return SkillResult(
            success=True,
            output=f"[LoggingAnalytics/{platform}] Logging config + queries ready.",
            artifacts=[
                {"path": "logging/structlog_setup.py", "content": _STRUCTLOG_SETUP},
                query_artifact,
            ],
            metadata={
                "skill": self.name,
                "platform": platform,
                "schema": {
                    "timestamp": "ISO 8601 UTC with timezone",
                    "level": "DEBUG|INFO|WARNING|ERROR|CRITICAL",
                    "service": "logical service name",
                    "env": "dev|staging|prod",
                    "trace_id": "W3C traceparent (32 hex)",
                    "span_id": "16 hex",
                    "request_id": "UUID v4",
                    "user_id": "hashed if PII",
                    "method": "GET|POST|...",
                    "path": "URL path (no query string)",
                    "status": "HTTP status int",
                    "latency_ms": "float",
                    "error": "exception class name (no stack in fields)",
                },
                "principles": [
                    "Logs are JSON, one event per line (NDJSON)",
                    "Never log secrets / PII — use a redaction filter",
                    "Always emit trace_id + span_id for cross-correlation",
                    "Sample DEBUG at < 1% in prod; never sample ERROR",
                    "Log to stdout only — let the platform handle rotation",
                ],
                "platforms": {
                    "elk": "Elasticsearch + Logstash + Kibana (KQL / Lucene)",
                    "loki": "Grafana Loki + Promtail (LogQL)",
                    "datadog": "Datadog Logs (DD pipeline + facets)",
                    "splunk": "Splunk (SPL)",
                },
            },
            suggestions=[
                "Add a middleware that injects request_id + trace_id into every log",
                "Use structlog processors: add_log_level, TimeStamper(UTC), JSONRenderer",
                "Build a redaction processor that masks emails / tokens before rendering",
                "Define log-based alerts (ERROR rate, latency outliers) in Loki/Kibana",
            ],
        )

    def _query_artifact(self, platform: str) -> Dict[str, str]:
        if platform == "datadog":
            return {"path": "logging/datadog_queries.md", "content": _DATADOG_QUERIES}
        if platform == "splunk":
            return {"path": "logging/splunk_queries.md", "content": _SPLUNK_QUERIES}
        if platform == "loki":
            return {"path": "logging/loki_queries.md", "content": _LOKI_QUERIES}
        return {"path": "logging/kql_queries.md", "content": _KQL_QUERIES}


_STRUCTLOG_SETUP = '''"""Structured logging setup — structlog + JSON + correlation IDs."""
import logging
import sys
import uuid
from typing import Any

import structlog
from structlog.types import EventDict, Processor

REDACT_KEYS = {"password", "token", "api_key", "authorization", "ssn", "email"}


def add_request_context(_: Any, __: Any, event_dict: EventDict) -> EventDict:
    """Inject request_id + trace_id from contextvar (set by middleware)."""
    import contextvars
    ctx = contextvars.ContextVar("request_context", default={}).get()
    for k, v in ctx.items():
        event_dict.setdefault(k, v)
    return event_dict


def redact_sensitive(_: Any, __: Any, event_dict: EventDict) -> EventDict:
    """Mask values of sensitive keys (recursive)."""
    for key in list(event_dict):
        if key.lower() in REDACT_KEYS:
            event_dict[key] = "***REDACTED***"
        elif isinstance(event_dict[key], dict):
            redact_sensitive(_, __, event_dict[key])
    return event_dict


def configure_logging(env: str = "prod", level: str = "INFO") -> None:
    """Call once at app startup."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_request_context,
        redact_sensitive,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# FastAPI middleware (pseudo) — sets contextvar per request
#   request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
#   trace_id   = extract_traceparent(request.headers.get("traceparent"))
#   structlog.contextvars.bind_contextvars(
#       request_id=request_id, trace_id=trace_id,
#       method=request.method, path=request.url.path,
#   )
# Usage:
#   import structlog; log = structlog.get_logger()
#   log.info("order.created", order_id=123, amount=99.90)
'''

_KQL_QUERIES = '''# KQL / Lucene — Kibana Discover queries

# 1. Top 10 slowest requests in the last hour
logs-*
| where timestamp > ago(1h)
| where level == "INFO" and status >= 200 and status < 300
| top 10 by latency_ms desc
| project timestamp, service, method, path, latency_ms, request_id

# 2. Error rate per service (last 24h)
logs-*
| where timestamp > ago(24h) and level == "ERROR"
| summarize error_count = count() by service
| join kind=inner (
    logs-* | where timestamp > ago(24h)
    | summarize total = count() by service
  ) on service
| extend error_rate = error_count * 1.0 / total
| where error_rate > 0.01
| sort by error_rate desc

# 3. Trace a single request_id across services
logs-* | where request_id == "abc-123" | sort by timestamp asc

# 4. Exceptions grouped by error type
logs-*
| where level == "ERROR" and isnotempty(error)
| summarize count() by error, service | top 20 by count_

# 5. P99 latency per route (last 7d, bucketed 5m)
logs-*
| where timestamp > ago(7d)
| summarize p99 = percentile(latency_ms, 99) by bin(timestamp, 5m), path
| render timechart

# Lucene (Discover bar) equivalents:
#   level:ERROR AND service:nexus-api AND @timestamp:[now-1h/h TO *]
#   request_id:"abc-123"
#   error:* AND NOT error:"TimeoutError"
'''

_LOKI_QUERIES = '''# LogQL — Grafana Loki / Promtail

# 1. Top slowest requests (last 1h, JSON parsing)
{service="nexus-api"} | json
  | latency_ms > 1000
  | line_format "{{.method}} {{.path}} took {{.latency_ms}}ms (req={{.request_id}})"

# 2. Error rate per service (alert-style metric query)
sum by (service) (
  count_over_time({level="ERROR"}[5m])
)
/
sum by (service) (
  count_over_time({service="nexus-api"}[5m])
) > 0.01

# 3. Trace a request_id across services
{service=~"nexus-.*"} | json | request_id="abc-123"

# 4. Top error types (last 1h)
topk(10,
  sum by (error) (
    count_over_time({level="ERROR"}[1h])
  )
)

# 5. Latency histogram via bucketed latency_ms
sum by (le) (
  count_over_time({service="nexus-api"} | json | latency_ms > 0 [5m])
)
'''

_DATADOG_QUERIES = '''# Datadog Logs — search syntax + facets

# 1. Slow requests last hour (facet on latency_ms)
env:prod service:nexus-api status:info -status:5* @latency_ms:>1000

# 2. Error rate by route (split by path facet)
env:prod service:nexus-api status:error *
# Then in UI: Group by -> @path, Measure -> count

# 3. Trace a request_id across services
@request_id:abc-123

# 4. Top exceptions by type
env:prod service:nexus-api status:error @error:*
# Measure -> count, Facet -> @error, Top -> 10

# 5. Set up a log-based metric (UI -> Generate Metric)
#    Name: log.errors.nexus_api
#    Query: env:prod service:nexus-api status:error
#    Then alert: logs.errors.nexus_api{env:prod} > threshold
'''

_SPLUNK_QUERIES = '''# Splunk SPL — search + analytics

# 1. Top 10 slowest requests in the last hour
index=nexus_logs level=INFO status=2*
| head 10000
| sort -latency_ms
| head 10
| table _time service method path latency_ms request_id

# 2. Error rate per service
index=nexus_logs earliest=-24h
| stats count(eval(level="ERROR")) as errors, count as total by service
| eval error_rate = errors / total
| where error_rate > 0.01
| sort -error_rate

# 3. Trace a single request_id
index=nexus_logs request_id="abc-123"
| sort _time

# 4. Exceptions grouped by error type
index=nexus_logs level=ERROR
| stats count by error, service
| sort -count
| head 20

# 5. P99 latency per route (bucketed 5m)
index=nexus_logs earliest=-7d
| bucket _time span=5m
| stats p99(latency_ms) as p99 by _time, path
| timechart span=5m max(p99) by path
'''
