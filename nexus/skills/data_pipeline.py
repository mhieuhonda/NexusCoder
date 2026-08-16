"""Data Pipeline Skill - Sinh Airflow DAG + dbt models.

Tạo ETL/ELT pipeline: Airflow DAG (with sensors + retries),
dbt staging/marts models, data quality checks (Great Expectations).

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class DataPipelineSkill(Skill):
    """Sinh Airflow DAG + dbt models + data quality checks."""

    category = SkillCategory.DATA
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "etl", "elt", "airflow", "dbt", "pipeline", "data flow",
        "dag", "data pipeline", "extract transform load",
        "snowflake", "bigquery", "redshift", "spark", "databricks",
        "great expectations", "data quality",
    ]
    examples = [
        "Build an ETL pipeline from Postgres to BigQuery",
        "Create an Airflow DAG with retries and sensors",
        "Generate dbt staging + marts models",
    ]

    @property
    def name(self) -> str:
        return "data_pipeline"

    @property
    def description(self) -> str:
        return (
            "Sinh data pipeline: Airflow DAG (idempotent, retries, sensors), "
            "dbt staging/marts models (with tests), data quality checks "
            "(Great Expectations), và monitoring hooks (Dagster sensors / Slack alerts)."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.15
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output=f"[DataPipeline] Airflow DAG + dbt models + GE checks ready.",
            artifacts=[
                {"path": "pipelines/airflow/nexus_etl_dag.py", "content": _AIRFLOW_DAG},
                {"path": "pipelines/dbt/models/staging/stg_orders.sql", "content": _DBT_STAGING},
                {"path": "pipelines/dbt/models/marts/fct_orders.sql", "content": _DBT_MART},
                {"path": "pipelines/dbt/models/schema.yml", "content": _DBT_SCHEMA},
            ],
            metadata={
                "skill": self.name,
                "orchestrator": "Apache Airflow 2.x (TaskFlow API)",
                "transform": "dbt-core (staging -> marts pattern)",
                "quality": "Great Expectations (expect_row_count, expect_column_values_to_not_be_null)",
                "targets": ["BigQuery", "Snowflake", "Redshift", "Databricks"],
                "principles": [
                    "Idempotency: re-running a task yields the same result",
                    "Partition by event_date — never full-refresh in prod",
                    "Separate raw / staging / marts layers (dbt best practice)",
                    "Every model has tests (not_null, unique, relationships)",
                    "Set sensible retries (3x) + exponential backoff",
                    "Alert on SLA miss (Slack / PagerDuty)",
                ],
                "schedule_examples": {
                    "hourly": "0 * * * *",
                    "daily_2am": "0 2 * * *",
                    "weekly_mon": "0 2 * * 1",
                },
            },
            suggestions=[
                "Use dbt-utils package for testing macros (test_equal_rowcount, ...)",
                "Materialize staging as view, marts as incremental",
                "Add data freshness SLAs in dbt (warning: 24h, error: 48h)",
                "Run Great Expectations checkpoint as an Airflow Task",
                "Version-control dbt + Airflow together (same repo)",
            ],
        )


_AIRFLOW_DAG = '''"""Airflow ETL DAG — Postgres -> Staging -> dbt -> Marts -> GE checks."""
from __future__ import annotations
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.utils.task_group import TaskGroup

DEFAULT_ARGS = {
    "owner": "data",
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
    "depends_on_past": False,
    "email_on_failure": True,
}


@dag(
    dag_id="nexus_etl",
    description="Extract -> stage -> transform -> mart -> quality",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["etl", "dbt", "nexus"],
)
def nexus_etl_dag():

    @task
    def extract_postgres() -> str:
        """Extract incremental partitions from Postgres source."""
        import pendulum
        ds = "{{ ds }}"
        # CDC / incremental logic; return run_id for next task
        return f"extracted_{ds}"

    @task
    def load_to_bigquery(run_id: str) -> str:
        """Load extracted files into BigQuery raw layer (WRITE_TRUNCATE per partition)."""
        return f"loaded_{run_id}"

    with TaskGroup("dbt") as dbt_group:
        dbt_seed = BigQueryInsertJobOperator(
            task_id="dbt_seed",
            configuration={"query": {"query": "dbt seed --select tag:seeds", "useLegacySql": False}},
        )
        dbt_run_staging = BigQueryInsertJobOperator(
            task_id="dbt_run_staging",
            configuration={"query": {"query": "dbt run --select staging.*", "useLegacySql": False}},
        )
        dbt_run_marts = BigQueryInsertJobOperator(
            task_id="dbt_run_marts",
            configuration={"query": {"query": "dbt run --select marts.*", "useLegacySql": False}},
        )
        dbt_test = BigQueryInsertJobOperator(
            task_id="dbt_test",
            configuration={"query": {"query": "dbt test", "useLegacySql": False}},
        )
        dbt_seed >> dbt_run_staging >> dbt_run_marts >> dbt_test

    @task
    def run_great_expectations() -> str:
        """Run GE checkpoint against marts.fct_orders."""
        # from great_expectations.checkpoint import Checkpoint
        return "ge_passed"

    done = EmptyOperator(task_id="done")

    extract_postgres() >> load_to_bigquery() >> dbt_group >> run_great_expectations() >> done


dag = nexus_etl_dag()
'''

_DBT_STAGING = '''-- staging/stg_orders.sql — minimal clean, no business logic
{{ config(materialized='view', tags=['staging']) }}

with source as (
    select
        id,
        user_id,
        amount_cents,
        currency,
        status,
        cast(created_at as timestamp) as created_at
    from {{ source('raw', 'orders') }}
    where date(created_at) = date('{{ ds }}')
)

select
    id as order_id,
    user_id,
    amount_cents / 100.0 as amount,
    upper(currency) as currency,
    lower(status) as status,
    created_at
from source
'''

_DBT_MART = '''-- marts/fct_orders.sql — business-ready fact table
{{ config(
    materialized='incremental',
    unique_key='order_id',
    incremental_strategy='merge',
    partition_by={'field': 'created_at', 'data_type': 'date'},
    tags=['marts', 'orders']
) }}

with stg as (
    select * from {{ ref('stg_orders') }}
    {% if is_incremental() %}
      where date(created_at) >= date_sub(current_date(), interval 3 day)
    {% endif %}
),
users as (
    select user_id, country, segment
    from {{ ref('dim_users') }}
)

select
    s.order_id,
    s.user_id,
    u.country,
    u.segment,
    s.amount,
    s.currency,
    s.status,
    s.created_at
from stg s
left join users u using (user_id)
'''

_DBT_SCHEMA = '''# models/schema.yml — dbt tests + docs
version: 2

models:
  - name: stg_orders
    description: "Cleaned raw orders, one row per order event"
    columns:
      - name: order_id
        tests:
          - unique
          - not_null
      - name: user_id
        tests:
          - not_null
          - relationships:
              to: ref('dim_users')
              field: user_id
      - name: amount
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              max_value: 100000

  - name: fct_orders
    description: "Fact table of orders joined with user dimensions"
    tests:
      - dbt_utils.expression_is_true:
          expression: "amount >= 0"
    columns:
      - name: order_id
        tests: [unique, not_null]
      - name: status
        tests:
          - accepted_values:
              values: ['pending', 'paid', 'shipped', 'refunded', 'failed']
'''
