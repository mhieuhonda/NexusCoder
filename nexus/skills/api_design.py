"""API Design Skill - Sinh OpenAPI 3.0 spec + REST API templates.

Cung cấp template cho RESTful API design: resource naming, status codes,
pagination, versioning, error format, idempotency, và OpenAPI 3.0 spec.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class APIDesignSkill(Skill):
    """Sinh REST API design + OpenAPI 3.0 spec template."""

    category = SkillCategory.SYSTEM
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "api design", "rest api", "restful", "openapi",
        "swagger", "api spec", "api documentation",
        "endpoint design", "thiết kế api", "resource naming",
    ]
    examples = [
        "Design a REST API for a blog platform",
        "Generate OpenAPI 3.0 spec for users and posts endpoints",
        "REST API versioning strategy for breaking changes",
    ]

    @property
    def name(self) -> str:
        return "api_design"

    @property
    def description(self) -> str:
        return (
            "Sinh REST API design + OpenAPI 3.0 spec: resource naming, "
            "status codes, pagination, versioning, errors, idempotency."
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
            output="[APIDesign] OpenAPI 3.0 spec template + REST guidelines ready.",
            artifacts=[
                {"path": "api/openapi.yaml", "content": _OPENAPI_SPEC},
                {"path": "api/guidelines.md", "content": _REST_GUIDELINES},
            ],
            metadata={
                "skill": self.name,
                "rest_principles": {
                    "resource_naming": "Plural nouns, lowercase, hyphenated: /users, /order-items",
                    "http_methods": {
                        "GET": "Read (idempotent, cacheable)",
                        "POST": "Create (non-idempotent — use Idempotency-Key for safety)",
                        "PUT": "Full replace (idempotent)",
                        "PATCH": "Partial update (idempotent if operation-based, not if value-based)",
                        "DELETE": "Remove (idempotent)",
                    },
                    "status_codes": {
                        "2xx_success": ["200 OK", "201 Created", "202 Accepted", "204 No Content"],
                        "3xx_redirect": ["301 Moved Permanently", "304 Not Modified"],
                        "4xx_client_error": ["400 Bad Request", "401 Unauthorized", "403 Forbidden",
                                             "404 Not Found", "409 Conflict", "422 Unprocessable Entity",
                                             "429 Too Many Requests"],
                        "5xx_server_error": ["500 Internal Server Error", "502 Bad Gateway",
                                             "503 Service Unavailable", "504 Gateway Timeout"],
                    },
                    "versioning": {
                        "uri_versioning": "/v1/users (simple, visible, cacheable)",
                        "header_versioning": "Accept: application/vnd.api+json;version=1",
                        "query_param": "?api-version=1 (rare)",
                        "media_type": "application/vnd.example.v1+json (most RESTful)",
                    },
                },
                "pagination": {
                    "offset_limit": "?page=2&limit=20 — simple, slow on large offsets",
                    "cursor": "?cursor=abc123 — stable, fast for infinite scroll",
                    "keyset": "?after_id=1234 — fastest for ordered data",
                    "links": "Use Link header (RFC 5988) or response body _links",
                },
                "errors": {
                    "format": "RFC 9457 (formerly RFC 7807) Problem Details for HTTP APIs",
                    "fields": ["type", "title", "status", "detail", "instance", "errors[]"],
                    "example": "application/problem+json",
                },
                "idempotency": {
                    "header": "Idempotency-Key: <uuid>",
                    "ttl": "24-48 hours",
                    "scope": "per-client (use API key + key as dedup index)",
                    "applies_to": "POST, PATCH (write operations); not needed for GET/PUT/DELETE",
                },
                "security": {
                    "auth": ["OAuth 2.0 (PKCE for SPAs)", "API key (server-to-server)",
                             "JWT (short-lived, refresh tokens)"],
                    "transport": "TLS 1.3 mandatory; HSTS header",
                    "rate_limit": "X-RateLimit-Remaining / -Limit / -Reset headers",
                },
                "tooling": {
                    "spec": "OpenAPI 3.1 (latest) — Swagger 2.0 deprecated",
                    "validators": ["openapi-generator", "oas-validator", "redocly-cli"],
                    "doc_ui": ["Swagger UI", "Redoc", "Elements (Stoplight)"],
                    "mock": ["Prism", "WireMock", "MSW"],
                    "lint": ["Spectral (Stoplight)", "vacuum (daveshanley)"],
                },
            },
            suggestions=[
                "Specify resource names (e.g. users, orders, posts)",
                "Indicate auth method (OAuth2 / API key / JWT)",
                "Mention if pagination cursor or offset preferred",
                "Ask for SDK generation (openapi-generator for many languages)",
            ],
        )


_OPENAPI_SPEC = '''openapi: 3.1.0
info:
  title: Example Blog API
  version: 1.0.0
  description: |
    RESTful API for a blog platform with users, posts, and comments.
    Author: Hieu Louis (2026)
  contact:
    name: API Support
    email: api@example.com
  license:
    name: MIT
    url: https://opensource.org/license/mit

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: https://staging-api.example.com/v1
    description: Staging

security:
  - bearerAuth: []

tags:
  - name: users
    description: User account management
  - name: posts
    description: Blog post CRUD
  - name: comments
    description: Comments on posts

paths:
  /users:
    get:
      tags: [users]
      summary: List users
      operationId: listUsers
      parameters:
        - $ref: '#/components/parameters/PageParam'
        - $ref: '#/components/parameters/LimitParam'
        - name: sort
          in: query
          schema: { type: string, enum: [created_at, -created_at, name] }
      responses:
        '200':
          description: A page of users
          headers:
            X-Total-Count:
              schema: { type: integer }
          content:
            application/json:
              schema:
                type: object
                required: [data, meta]
                properties:
                  data:
                    type: array
                    items: { $ref: '#/components/schemas/User' }
                  meta:
                    $ref: '#/components/schemas/PageMeta'
    post:
      tags: [users]
      summary: Create user
      operationId: createUser
      parameters:
        - name: Idempotency-Key
          in: header
          required: true
          schema: { type: string, format: uuid }
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/UserCreate' }
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema: { $ref: '#/components/schemas/User' }
        '409':
          $ref: '#/components/responses/Conflict'
        '422':
          $ref: '#/components/responses/Unprocessable'

  /users/{userId}:
    parameters:
      - $ref: '#/components/parameters/UserIdParam'
    get:
      tags: [users]
      summary: Get user by id
      operationId: getUser
      responses:
        '200':
          description: A user
          content:
            application/json:
              schema: { $ref: '#/components/schemas/User' }
        '404':
          $ref: '#/components/responses/NotFound'
    patch:
      tags: [users]
      summary: Update user
      operationId: updateUser
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/UserUpdate' }
      responses:
        '200':
          description: Updated user
          content:
            application/json:
              schema: { $ref: '#/components/schemas/User' }
        '404':
          $ref: '#/components/responses/NotFound'
    delete:
      tags: [users]
      summary: Delete user
      operationId: deleteUser
      responses:
        '204': { description: Deleted }

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  parameters:
    PageParam:
      name: page
      in: query
      schema: { type: integer, minimum: 1, default: 1 }
    LimitParam:
      name: limit
      in: query
      schema: { type: integer, minimum: 1, maximum: 100, default: 20 }
    UserIdParam:
      name: userId
      in: path
      required: true
      schema: { type: string, format: uuid }

  schemas:
    User:
      type: object
      required: [id, email, created_at]
      properties:
        id: { type: string, format: uuid }
        email: { type: string, format: email }
        name: { type: string, minLength: 1, maxLength: 100 }
        created_at: { type: string, format: date-time }
        updated_at: { type: string, format: date-time }
    UserCreate:
      type: object
      required: [email]
      properties:
        email: { type: string, format: email }
        name: { type: string, minLength: 1, maxLength: 100 }
    UserUpdate:
      type: object
      properties:
        name: { type: string, minLength: 1, maxLength: 100 }
    PageMeta:
      type: object
      required: [page, limit, total]
      properties:
        page: { type: integer }
        limit: { type: integer }
        total: { type: integer }
        has_next: { type: boolean }
    Error:
      type: object
      required: [type, title, status]
      properties:
        type: { type: string, format: uri }
        title: { type: string }
        status: { type: integer }
        detail: { type: string }
        instance: { type: string }
        errors:
          type: array
          items:
            type: object
            properties:
              field: { type: string }
              message: { type: string }

  responses:
    NotFound:
      description: Resource not found
      content:
        application/problem+json:
          schema: { $ref: '#/components/schemas/Error' }
    Conflict:
      description: Conflict with current state
      content:
        application/problem+json:
          schema: { $ref: '#/components/schemas/Error' }
    Unprocessable:
      description: Validation failed
      content:
        application/problem+json:
          schema: { $ref: '#/components/schemas/Error' }
'''


_REST_GUIDELINES = """# REST API Design Guidelines

## 1. Resource Naming
- Use **plural nouns**: `/users`, `/orders`, `/order-items` (not `/orderItem`).
- Use **hyphens** for multi-word: `/order-items` (not `/order_items` or `/orderitems`).
- Nest for sub-resources: `/users/{userId}/posts`.
- Never use verbs in path: `/users/{id}/posts` not `/users/{id}/getPosts`.

## 2. HTTP Methods (CRUD mapping)
| Action  | Method | Path             | Success Codes       | Idempotent |
|---------|--------|------------------|---------------------|------------|
| List    | GET    | /users           | 200                 | Yes        |
| Create  | POST   | /users           | 201 + Location hdr  | No*        |
| Read    | GET    | /users/{id}      | 200 / 404           | Yes        |
| Replace | PUT    | /users/{id}      | 200                 | Yes        |
| Update  | PATCH  | /users/{id}      | 200                 | No*        |
| Delete  | DELETE | /users/{id}      | 204 / 404           | Yes        |

*Use `Idempotency-Key` header for safe retries on POST/PATCH.

## 3. Status Codes (most common)
- **200** OK — generic success
- **201** Created — POST success (include `Location: /users/{id}`)
- **204** No Content — successful but empty body (DELETE)
- **400** Bad Request — malformed syntax
- **401** Unauthorized — auth missing/invalid
- **403** Forbidden — authenticated but no permission
- **404** Not Found — resource does not exist
- **409** Conflict — duplicate / state violation
- **422** Unprocessable Entity — semantic validation failure
- **429** Too Many Requests — rate limited
- **500** Internal Server Error — bug
- **503** Service Unavailable — maintenance / overload

## 4. Pagination
- **offset/limit**: simple but slow on large datasets (O(offset))
- **cursor**: opaque token, stable, fast (preferred for public APIs)
- **keyset**: `?after_id=1234` — fastest for ordered data
- Always return pagination metadata: `page`, `limit`, `total`, `has_next`

## 5. Versioning
- **URI versioning** (most common): `/v1/users` — simple, visible, cacheable
- **Media type versioning** (most RESTful): `Accept: application/vnd.example.v1+json`
- **Header versioning**: `Api-Version: 1` — invisible, harder to test
- Avoid query param versioning (breaks caching)

## 6. Error Format (RFC 9457)
```
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/problem+json

{
  "type": "https://example.com/errors/validation",
  "title": "Validation failed",
  "status": 422,
  "detail": "Email already in use",
  "instance": "/users",
  "errors": [
    { "field": "email", "message": "must be unique" }
  ]
}
```

## 7. Idempotency
- Header: `Idempotency-Key: <uuid>`
- Store: `(api_key, idempotency_key) -> response` with 24-48h TTL
- Same key + same body -> return cached response
- Same key + different body -> 422 Conflict (likely client bug)

## 8. Security
- TLS 1.3 mandatory; HSTS header
- Auth: OAuth 2.0 PKCE for SPAs, API key for server-to-server, JWT short-lived
- Rate limit: `X-RateLimit-Limit`, `-Remaining`, `-Reset` headers
- Never expose internal errors / stack traces in production
- Audit log all mutations

## 9. Documentation
- Generate from OpenAPI 3.1 spec (single source of truth)
- Swagger UI / Redoc for interactive docs
- Provide examples in multiple languages (curl, Python, JS)
- Changelog with breaking vs non-breaking tags
"""
