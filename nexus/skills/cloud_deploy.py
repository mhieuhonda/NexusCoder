"""Cloud Deploy Skill - Sinh cloud deployment templates.

Hỗ trợ AWS (EC2/S3/Lambda), GCP (Cloud Run / Compute), Azure
(Container Apps / Functions). Tạo IaC + deploy command.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class CloudDeploySkill(Skill):
    """Sinh IaC + deploy command cho AWS / GCP / Azure."""

    category = SkillCategory.CLOUD
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "deploy", "deployment", "aws", "gcp", "azure",
        "ec2", "s3", "lambda", "cloud run", "cloudrun",
        "cloud functions", "ecs", "eks", "fargate",
        "compute engine", "container apps", "app service",
        "deploy command", "iac", "pulumi", "cdk",
    ]
    examples = [
        "Deploy FastAPI to AWS Lambda",
        "Deploy container to GCP Cloud Run",
        "Provision S3 bucket + CloudFront for static site",
    ]

    @property
    def name(self) -> str:
        return "cloud_deploy"

    @property
    def description(self) -> str:
        return (
            "Sinh cloud deployment templates cho AWS / GCP / Azure: "
            "Lambda (SAM), Cloud Run, ECS/Fargate, Container Apps, "
            "S3+CloudFront static hosting, với IaC (Terraform / CDK / Pulumi) "
            "và deploy commands (aws / gcloud / az)."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.15
        return min(1.0, score)

    def _detect_target(self, prompt: str) -> str:
        p = prompt.lower()
        if "lambda" in p:
            return "aws_lambda"
        if "cloud run" in p or "cloudrun" in p:
            return "gcp_cloudrun"
        if "ecs" in p or "fargate" in p:
            return "aws_ecs"
        if "container apps" in p:
            return "azure_containerapps"
        if "app service" in p:
            return "azure_appservice"
        if "compute engine" in p or "gce" in p:
            return "gcp_gce"
        if "ec2" in p:
            return "aws_ec2"
        if "s3" in p and ("static" in p or "cloudfront" in p or "website" in p):
            return "aws_s3_static"
        return "gcp_cloudrun"  # sane default for container

    def execute(self, context: SkillContext) -> SkillResult:
        target = self._detect_target(context.prompt)
        artifact, deploy_cmd = self._build(target)

        return SkillResult(
            success=True,
            output=f"[CloudDeploy/{target}] IaC + deploy command ready.",
            artifacts=[artifact],
            metadata={
                "skill": self.name,
                "target": target,
                "deploy_command": deploy_cmd,
                "providers": {
                    "aws": ["aws-cli", "sam", "terraform", "cdk"],
                    "gcp": ["gcloud", "terraform", "pulumi"],
                    "azure": ["az", "terraform", "bicep"],
                },
                "checklist": [
                    "Pin runtime versions (Python 3.12, Node 20, ...)",
                    "Set least-privilege IAM role",
                    "Enable VPC flow logs / CloudTrail / Audit Logs",
                    "Configure autoscaling + health checks",
                    "Set up alarms on error rate / latency / cost",
                    "Store secrets in Secrets Manager / Secret Manager",
                ],
            },
            suggestions=[
                "Run `terraform plan` in CI before `apply`",
                "Use blue/green or canary for production deploys",
                "Tag resources with owner / cost-center / env",
                "Enable WAF + rate-limiting on public endpoints",
            ],
        )

    def _build(self, target: str) -> tuple[Dict[str, str], str]:
        if target == "aws_lambda":
            return ({"path": "deploy/aws_lambda/template.yaml", "content": _AWS_LAMBDA_SAM},
                    "sam build && sam deploy --guided")
        if target == "aws_ecs":
            return ({"path": "deploy/aws_ecs/main.tf", "content": _AWS_ECS_TF},
                    "terraform apply -auto-approve")
        if target == "aws_ec2":
            return ({"path": "deploy/aws_ec2/main.tf", "content": _AWS_EC2_TF},
                    "terraform apply -auto-approve")
        if target == "aws_s3_static":
            return ({"path": "deploy/aws_s3_static/main.tf", "content": _AWS_S3_STATIC_TF},
                    "aws s3 sync ./dist s3://$BUCKET --delete")
        if target == "azure_containerapps":
            return ({"path": "deploy/azure_containerapps/main.bicep", "content": _AZURE_CONTAINERAPPS},
                    "az deployment group create -g rg-prod -f main.bicep")
        if target == "azure_appservice":
            return ({"path": "deploy/azure_appservice/main.bicep", "content": _AZURE_APPSERVICE},
                    "az webapp up --runtime PYTHON:3.12 --sku B1")
        if target == "gcp_gce":
            return ({"path": "deploy/gcp_gce/main.tf", "content": _GCP_GCE_TF},
                    "terraform apply -auto-approve")
        return ({"path": "deploy/gcp_cloudrun/main.tf", "content": _GCP_CLOUDRUN_TF},
                "gcloud run deploy nexus-api --source . --region asia-southeast1")


_AWS_LAMBDA_SAM = '''# AWS SAM template — Lambda + API Gateway
AWSTemplateFormatVersion: "2010-09-09"
Transform: AWS::Serverless-2016-10-31
Globals:
  Function:
    Runtime: python3.12
    MemorySize: 512
    Timeout: 30
    Tracing: Active
Resources:
  NexusApi:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: ../src
      Handler: app.handler
      Policies:
        - AWSLambdaBasicExecutionRole
        - DynamoDBCrud: { TableName: !Ref NexusTable }
      Environment:
        Variables: { LOG_LEVEL: INFO }
      Events:
        Api:
          Type: Api
          Properties:
            Path: /{proxy+}
            Method: ANY
  NexusTable:
    Type: AWS::DynamoDB::Table
    Properties:
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - { AttributeName: pk, AttributeType: S }
      KeySchema:
        - { AttributeName: pk, KeyType: HASH }
Outputs:
  ApiUrl: { Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod" }
'''

_AWS_ECS_TF = '''# AWS ECS Fargate + ALB
terraform {
  required_version = ">= 1.7"
  required_providers { aws = { source = "hashicorp/aws", version = "~> 5.0" } }
}
resource "aws_ecs_cluster" "nexus" { name = "nexus-cluster" }
resource "aws_ecs_task_definition" "nexus" {
  family                   = "nexus-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "512"
  memory                   = "1024"
  container_definitions     = jsonencode([{
    name  = "api"
    image = "ghcr.io/nexus/api:0.3.0"
    portMappings = [{ containerPort = 8000 }]
    logConfiguration = { logDriver = "awslogs",
      options = { "awslogs-group" = "/ecs/nexus", "awslogs-region" = "ap-southeast-1" } }
  }])
}
resource "aws_ecs_service" "nexus" {
  name            = "nexus-api"
  cluster         = aws_ecs_cluster.nexus.id
  task_definition = aws_ecs_task_definition.nexus.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  network_configuration {
    subnets         = module.vpc.private_subnets
    security_groups = [aws_security_group.nexus.id]
  }
}
'''

_AWS_EC2_TF = '''# AWS EC2 with EIP + user_data
terraform {
  required_version = ">= 1.7"
  required_providers { aws = { source = "hashicorp/aws", version = "~> 5.0" } }
}
resource "aws_instance" "nexus" {
  ami                    = "ami-0abc1234def56789"
  instance_type          = "t3.small"
  vpc_security_group_ids = [aws_security_group.nexus.id]
  iam_instance_profile   = aws_iam_instance_profile.nexus.name
  user_data              = file("deploy/aws_ec2/userdata.sh")
  tags                   = { Name = "nexus-api", Env = "prod" }
}
resource "aws_eip" "nexus" {
  instance = aws_instance.nexus.id
  domain   = "vpc"
}
'''

_AWS_S3_STATIC_TF = '''# S3 + CloudFront static website
terraform {
  required_version = ">= 1.7"
  required_providers { aws = { source = "hashicorp/aws", version = "~> 5.0" } }
}
resource "aws_s3_bucket" "static" { bucket = "nexus-static-prod" }
resource "aws_s3_bucket_website_configuration" "static" {
  bucket = aws_s3_bucket.static.id
  index_document { suffix = "index.html" }
  error_document { key = "404.html" }
}
resource "aws_cloudfront_distribution" "cdn" {
  origin {
    domain_name = aws_s3_bucket_website_configuration.static.website_endpoint
    origin_id    = "s3-nexus"
    custom_origin_config { origin_protocol_policy = "http-only" }
  }
  enabled             = true
  is_ipv6_enabled     = true
  default_cache_behavior {
    target_origin_id       = "s3-nexus"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    forwarded_values { query_string = false; cookies { forward = "none" } }
    min_ttl = 0; default_ttl = 3600; max_ttl = 86400
  }
  restrictions { geo_restriction { restriction_type = "none" } }
  viewer_certificate { cloudfront_default_certificate = true }
}
'''

_GCP_CLOUDRUN_TF = '''# GCP Cloud Run service
terraform {
  required_version = ">= 1.7"
  required_providers { google = { source = "hashicorp/google", version = "~> 5.0" } }
}
resource "google_cloud_run_service" "nexus" {
  name     = "nexus-api"
  location = "asia-southeast1"
  template {
    spec {
      container_concurrency = 80
      timeout_seconds        = 300
      containers {
        image = "gcr.io/PROJECT/nexus-api:0.3.0"
        env { name = "LOG_LEVEL"; value = "INFO" }
        resources {
          limits = { cpu = "1000m", memory = "1Gi" }
        }
      }
    }
  }
  traffic { percent = 100; latest_revision = true }
  autogenerate_revision_name = true
}
resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.nexus.name
  location = google_cloud_run_service.nexus.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
'''

_GCP_GCE_TF = '''# GCP Compute Engine with startup script
terraform {
  required_version = ">= 1.7"
  required_providers { google = { source = "hashicorp/google", version = "~> 5.0" } }
}
resource "google_compute_instance" "nexus" {
  name         = "nexus-api"
  machine_type = "e2-small"
  zone         = "asia-southeast1-a"
  boot_disk {
    initialize_params { image = "debian-cloud/debian-12" }
  }
  network_interface {
    network = "default"
    access_config {}  # ephemeral IP
  }
  metadata = { startup-script = file("deploy/gcp_gce/startup.sh") }
  tags     = ["nexus", "http-server"]
  service_account {
    scopes = ["cloud-platform"]
  }
}
'''

_AZURE_CONTAINERAPPS = '''// Azure Container Apps (Bicep)
param location string = 'southeastasia'
param imageName string = 'ghcr.io/nexus/api:0.3.0'

resource managedEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'nexus-env'
  location: location
}

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'nexus-api'
  location: location
  properties: {
    managedEnvironmentId: managedEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        traffic: [{ weight: 100, latestRevision: true }]
        allowInsecure: false
      }
      secrets: [
        { name: 'api-key', value: '@Microsoft.KeyVault(VaultName=nexus-kv;SecretName=ApiKey)' }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: imageName
          env: [
            { name: 'LOG_LEVEL', value: 'INFO' }
          ]
          resources: { cpu: json('1.0'), memory: '1.0Gi' }
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 10 }
    }
  }
}
'''

_AZURE_APPSERVICE = '''// Azure App Service (Linux, Python) — Bicep
param location string = 'southeastasia'
param sku string = 'B1'

resource plan 'Microsoft.Web/serverfarms@2023-12-01' = {
  name: 'nexus-plan'
  location: location
  sku: { name: sku, tier: 'Basic' }
  properties: { reserved: true } // Linux
}

resource app 'Microsoft.Web/sites@2023-12-01' = {
  name: 'nexus-api'
  location: location
  properties: {
    serverFarmId: plan.id
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.12'
      appCommandLine: 'gunicorn -w 4 -b 0.0.0.0:8000 app:main'
      alwaysOn: true
    }
  }
  identity: { type: 'SystemAssigned' }
}
'''
