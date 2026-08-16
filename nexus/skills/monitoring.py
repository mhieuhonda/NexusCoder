"""Monitoring Skill - Prometheus alerting rules + Grafana dashboard JSON.

Sinh Prometheus rules (recording + alerting), Alertmanager config,
Grafana dashboard JSON, và SLO/SLI definitions.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class MonitoringSkill(Skill):
    """Sinh Prometheus alerts + Grafana dashboard cho observability."""

    category = SkillCategory.DEVOPS
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "monitor", "monitoring", "prometheus", "grafana", "alert",
        "alertmanager", "observability", "slo", "sli",
        "metrics", "telemetry", "opentelemetry", "otel", "victoria metrics",
    ]
    examples = [
        "Create Prometheus alerting rules for high CPU",
        "Build a Grafana dashboard for API latency",
        "Define SLOs with error budgets",
    ]

    @property
    def name(self) -> str:
        return "monitoring"

    @property
    def description(self) -> str:
        return (
            "Sinh observability stack: Prometheus recording + alerting rules, "
            "Alertmanager routing, Grafana dashboard JSON, OpenTelemetry "
            "instrumentation, và SLO/SLI với error budgets."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.16
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output=f"[Monitoring] Prometheus rules + Grafana dashboard ready.",
            artifacts=[
                {"path": "monitoring/prometheus/rules.yml", "content": _PROM_RULES},
                {"path": "monitoring/alertmanager.yml", "content": _ALERTMANAGER},
                {"path": "monitoring/grafana/dashboard.json", "content": _GRAFANA_DASHBOARD},
            ],
            metadata={
                "skill": self.name,
                "stack": ["Prometheus", "Alertmanager", "Grafana", "OpenTelemetry", "Loki/Tempo"],
                "metric_types": {
                    "counter": "monotonic increase (e.g. http_requests_total)",
                    "gauge": "current value (e.g. queue_depth)",
                    "histogram": "distribution (latency buckets)",
                    "summary": "quantiles (alternative to histogram)",
                },
                "golden_signals": ["latency", "traffic", "errors", "saturation"],
                "slo_template": {
                    "availability": "99.9% (success_ratio > 0.999 over 30d)",
                    "latency_p99": "0.99 quantile < 500ms over 30d",
                    "error_budget": "0.1% per 30d = ~43m downtime",
                },
                "labeling_convention": {
                    "service": "logical service name",
                    "env": "dev|staging|prod",
                    "region": "deployment region",
                    "version": "image tag",
                },
            },
            suggestions=[
                "Use histograms (not summaries) for latency — aggregatable across instances",
                "Cardinality < 10 per label — never put user_id in labels",
                "Page on symptoms (user-visible), not on causes (CPU high)",
                "Burn rate alerts: 2% budget in 1h OR 5% in 6h -> page",
                "Always include a runbook link in alert annotations",
            ],
        )


_PROM_RULES = '''# Prometheus rules: recording + alerting (SLO-based burn rate)
groups:
  - name: nexus_recording
    interval: 30s
    rules:
      - record: nexus:http_request_rate
        expr: sum by (service, route) (rate(http_requests_total[5m]))
      - record: nexus:http_error_ratio
        expr: |
          sum by (service) (rate(http_requests_total{status=~"5.."}[5m]))
          /
          sum by (service) (rate(http_requests_total[5m]))
      - record: nexus:http_latency_p99
        expr: histogram_quantile(0.99,
                sum by (service, le) (rate(http_request_duration_seconds_bucket[5m])))

  - name: nexus_alerts
    rules:
      - alert: HighErrorRate
        expr: nexus:http_error_ratio{service="nexus-api"} > 0.01
        for: 5m
        labels: { severity: page, team: platform }
        annotations:
          summary: "nexus-api error ratio > 1% ({{ $value | humanizePercentage }})"
          runbook: "https://runbooks.nexus.io/high-error-rate"

      - alert: HighLatencyP99
        expr: nexus:http_latency_p99{service="nexus-api"} > 0.5
        for: 10m
        labels: { severity: ticket, team: platform }
        annotations:
          summary: "nexus-api p99 latency > 500ms ({{ $value | humanizeDuration }})"

      # Multi-window multi-burn-rate SLO alert (Google SRE pattern)
      - alert: SLOBurnRateFast
        expr: |
          (nexus:http_error_ratio{service="nexus-api"} > 14.4 * 0.001
           and rate(nexus:http_error_ratio[5m]) > 14.4 * 0.001)
        for: 2m
        labels: { severity: page, team: platform, slo: "availability-99.9" }
        annotations:
          summary: "SLO burn rate fast window exceeded (2% budget in 1h)"

      - alert: PodCrashLooping
        expr: increase(kube_pod_container_status_restarts_total[1h]) > 5
        for: 10m
        labels: { severity: page }
        annotations:
          summary: "Pod {{ $labels.pod }} restarted {{ $value }} times in 1h"

      - alert: DiskWillFillIn4h
        expr: predict_linear(node_filesystem_avail_bytes[1h], 4*3600) < 0
        for: 10m
        labels: { severity: ticket }
        annotations:
          summary: "Disk {{ $labels.device }} on {{ $labels.instance }} full in < 4h"
'''

_ALERTMANAGER = '''# Alertmanager routing — severity-based
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/XXX'

route:
  receiver: default
  group_by: ['alertname', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - matchers: ['severity="page"']
      receiver: oncall-pagerduty
      group_wait: 0s
      repeat_interval: 1h
    - matchers: ['severity="ticket"']
      receiver: slack-tickets

receivers:
  - name: default
    slack_configs:
      - channel: '#alerts-platform'
        send_resolved: true
        title: '{{ .CommonLabels.alertname }}'
        text: '{{ .CommonAnnotations.summary }}'

  - name: oncall-pagerduty
    pagerduty_configs:
      - routing_key: 'PAGERDUTY_KEY'
        severity: critical
        send_resolved: true

  - name: slack-tickets
    slack_configs:
      - channel: '#data-tickets'
        send_resolved: true

inhibit_rules:
  - source_matchers: ['alertname="PodCrashLooping"']
    target_matchers: ['alertname="HighErrorRate"']
    equal: ['service']
'''

_GRAFANA_DASHBOARD = '''{
  "title": "Nexus API — Overview",
  "uid": "nexus-api-overview",
  "schemaVersion": 39,
  "timezone": "browser",
  "refresh": "30s",
  "tags": ["nexus", "api", "overview"],
  "time": {"from": "now-6h", "to": "now"},
  "templating": {
    "list": [
      {"name": "service", "type": "query", "datasource": "Prometheus",
       "query": "label_values(http_requests_total, service)"}
    ]
  },
  "panels": [
    {
      "id": 1, "type": "stat", "title": "Requests / sec",
      "gridPos": {"h": 6, "w": 6, "x": 0, "y": 0},
      "targets": [{"expr": "sum(rate(http_requests_total{service=\\"$service\\"}[5m]))",
                   "legendFormat": "rps", "refId": "A"}]
    },
    {
      "id": 2, "type": "stat", "title": "Error rate",
      "gridPos": {"h": 6, "w": 6, "x": 6, "y": 0},
      "fieldConfig": {"defaults": {"thresholds": {"mode": "absolute",
        "steps": [{"color": "green", "value": null},
                  {"color": "yellow", "value": 0.005},
                  {"color": "red", "value": 0.01}]}}},
      "targets": [{"expr": "sum(rate(http_requests_total{service=\\"$service\\",status=~\\"5..\\"}[5m])) / sum(rate(http_requests_total{service=\\"$service\\"}[5m]))",
                   "legendFormat": "err%", "refId": "A"}]
    },
    {
      "id": 3, "type": "timeseries", "title": "Latency p50/p90/p99",
      "gridPos": {"h": 10, "w": 12, "x": 0, "y": 6},
      "targets": [
        {"expr": "histogram_quantile(0.50, sum by (le)(rate(http_request_duration_seconds_bucket{service=\\"$service\\"}[5m])))", "legendFormat": "p50", "refId": "A"},
        {"expr": "histogram_quantile(0.90, sum by (le)(rate(http_request_duration_seconds_bucket{service=\\"$service\\"}[5m])))", "legendFormat": "p90", "refId": "B"},
        {"expr": "histogram_quantile(0.99, sum by (le)(rate(http_request_duration_seconds_bucket{service=\\"$service\\"}[5m])))", "legendFormat": "p99", "refId": "C"}
      ],
      "fieldConfig": {"defaults": {"unit": "s"}}
    },
    {
      "id": 4, "type": "timeseries", "title": "SLO error budget burn",
      "gridPos": {"h": 10, "w": 12, "x": 12, "y": 6},
      "targets": [{"expr": "1 - (sum(rate(http_requests_total{service=\\"$service\\"}[1h])) - sum(rate(http_requests_total{service=\\"$service\\",status=\\"500\\"}[1h]))) / sum(rate(http_requests_total{service=\\"$service\\"}[1h]))", "legendFormat": "burn_rate", "refId": "A"}]
    }
  ]
}
'''
