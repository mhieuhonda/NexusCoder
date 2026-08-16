"""GraphQL Skill - Sinh GraphQL schema, queries, mutations, và resolvers.

Cung cấp template cho: type design, query/mutation/subscription,
pagination (Relay-style), error handling, federation, và N+1 mitigation.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class GraphQLSkill(Skill):
    """Sinh GraphQL schema + queries + resolvers."""

    category = SkillCategory.SYSTEM
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "graphql", "schema", "query", "mutation",
        "subscription", "resolver", "apollo", "federation",
        "thiết kế graphql", "graphql schema",
    ]
    examples = [
        "Design a GraphQL schema for a blog",
        "GraphQL query with pagination and filtering",
        "Apollo Federation subgraph for users service",
    ]

    @property
    def name(self) -> str:
        return "graphql"

    @property
    def description(self) -> str:
        return (
            "Sinh GraphQL schema + queries + mutations + resolvers. "
            "Hỗ trợ Relay pagination, federation, DataLoader (N+1 fix)."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.17
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            output="[GraphQL] Schema + queries + resolvers template ready.",
            artifacts=[
                {"path": "graphql/schema.graphql", "content": _GRAPHQL_SCHEMA},
                {"path": "graphql/example_queries.graphql", "content": _EXAMPLE_QUERIES},
                {"path": "graphql/resolvers.py", "content": _RESOLVERS},
            ],
            metadata={
                "skill": self.name,
                "schema_design": {
                    "types": "Object types use PascalCase, fields use camelCase",
                    "ids": "Global ID (base64-encoded typename:id) for Relay",
                    "enums": "SCREAMING_SNAKE_CASE",
                    "input_types": "Suffixed with Input (e.g. CreateUserInput)",
                    "interfaces": "For shared contracts (Node, Entity)",
                    "unions": "For polymorphic results (SearchResult = User | Post)",
                    "scalars": "Custom scalars for DateTime, UUID, JSON, URL",
                },
                "operations": {
                    "query": "Read-only — parallel-safe, idempotent",
                    "mutation": "Write — sequential (root mutation fields run in series)",
                    "subscription": "Long-lived, server-push over WebSocket / SSE",
                },
                "pagination": {
                    "offset_limit": "Simple but discouraged (no cursor stability)",
                    "relay_cursor": "Industry standard — Connections + Edges + PageInfo",
                    "edges": "Wrapper per item, allows per-edge cursor + metadata",
                    "page_info": "hasNextPage, hasPreviousPage, startCursor, endCursor",
                },
                "error_handling": {
                    "partial_success": "GraphQL returns 200 with `errors` array alongside `data`",
                    "error_format": "{ message, locations, path, extensions: { code, ...} }",
                    "error_codes": ["UNAUTHENTICATED", "FORBIDDEN", "NOT_FOUND",
                                    "BAD_USER_INPUT", "INTERNAL_SERVER_ERROR"],
                    "suggestion": "Use `extensions.code` for machine-readable errors",
                },
                "performance": {
                    "n_plus_one": "Top issue — DataLoader batches + caches per-request",
                    "query_complexity": "Limit depth + complexity to prevent abuse",
                    "persisted_queries": "APQ — send hash instead of full query",
                    "introspection": "Disable in production (security)",
                    "depth_limit": "Reject queries with depth > 7",
                },
                "federation": {
                    "purpose": "Compose multiple subgraphs into one supergraph",
                    "directives": ["@key", "@extends", "@external", "@requires", "@provides"],
                    "gateway": "Apollo Gateway / Router composes schemas",
                    "entities": "Resolve via `__resolveEntity` reference resolver",
                },
                "tooling": {
                    "schema_lint": ["graphql-eslint", "graphql-schema-linter"],
                    "codegen": ["graphql-code-generator (TS/JS)", "ariadne-codegen (Python)"],
                    "servers": ["Apollo Server (JS)", "Strawberry (Python)",
                                "graphql-go", "async-graphql (Rust)"],
                    "testing": ["easygraphql-tester", "apollo-server-testing"],
                },
            },
            suggestions=[
                "Specify the domain (users, posts, products, etc.)",
                "Indicate if Relay-style pagination is required",
                "Mention if federation / subgraphs are needed",
                "Ask for codegen target language (TS / Python / Go / Rust)",
            ],
        )


_GRAPHQL_SCHEMA = '''# GraphQL Schema - Blog Platform Example
# Author: Hieu Louis (2026)

# ---- Custom Scalars -------------------------------------------------------
scalar DateTime
scalar UUID
scalar URL

# ---- Enums ----------------------------------------------------------------
enum PostStatus {
  DRAFT
  PUBLISHED
  ARCHIVED
}

enum SortOrder {
  ASC
  DESC
}

# ---- Interfaces (shared contracts) ---------------------------------------
interface Node {
  id: ID!
}

# ---- Types ----------------------------------------------------------------
type User implements Node {
  id: ID!
  email: String!
  name: String!
  avatarUrl: URL
  posts(first: Int = 10, after: String): PostConnection!
  createdAt: DateTime!
}

type Post implements Node {
  id: ID!
  title: String!
  body: String!
  status: PostStatus!
  author: User!
  comments(first: Int = 10, after: String): CommentConnection!
  tags: [String!]!
  publishedAt: DateTime
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Comment implements Node {
  id: ID!
  body: String!
  author: User!
  post: Post!
  createdAt: DateTime!
}

# ---- Relay-style Connections (pagination) --------------------------------
type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type PostEdge {
  cursor: String!
  node: Post!
}

type CommentConnection {
  edges: [CommentEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type CommentEdge {
  cursor: String!
  node: Comment!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

# ---- Inputs ---------------------------------------------------------------
input CreateUserInput {
  email: String!
  name: String!
  password: String!
}

input UpdateUserInput {
  name: String
  avatarUrl: URL
}

input CreatePostInput {
  title: String!
  body: String!
  tags: [String!] = []
  status: PostStatus = DRAFT
}

input PostFilter {
  status: PostStatus
  tags: [String!]
  authorId: ID
}

input PostOrder {
  field: PostOrderField!
  direction: SortOrder!
}

enum PostOrderField {
  CREATED_AT
  PUBLISHED_AT
  TITLE
}

# ---- Errors ---------------------------------------------------------------
type FieldError {
  field: String!
  message: String!
}

type UserError {
  message: String!
  code: String!
  field: String
}

# ---- Query Root -----------------------------------------------------------
type Query {
  # Node interface lookup (Relay pattern)
  node(id: ID!): Node

  # Resource queries
  user(id: ID!): User
  users(first: Int = 10, after: String): UserConnection!
  me: User

  post(id: ID!): Post
  posts(
    first: Int = 10
    after: String
    filter: PostFilter
    orderBy: [PostOrder!]
  ): PostConnection!

  search(query: String!, types: [SearchType!] = [USER, POST]): [SearchResult!]!
}

enum SearchType { USER POST COMMENT }
union SearchResult = User | Post | Comment

type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}
type UserEdge { cursor: String!, node: User! }

# ---- Mutation Root --------------------------------------------------------
type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(id: ID!, input: UpdateUserInput!): UpdateUserPayload!
  deleteUser(id: ID!): DeletePayload!

  createPost(input: CreatePostInput!): CreatePostPayload!
  publishPost(id: ID!): PublishPostPayload!
  deletePost(id: ID!): DeletePayload!

  addComment(postId: ID!, body: String!): AddCommentPayload!
}

type CreateUserPayload {
  user: User
  errors: [UserError!]!
}

type UpdateUserPayload {
  user: User
  errors: [UserError!]!
}

type DeletePayload {
  deletedId: ID
  errors: [UserError!]!
}

type CreatePostPayload {
  post: Post
  errors: [UserError!]!
}

type PublishPostPayload {
  post: Post
  errors: [UserError!]!
}

type AddCommentPayload {
  comment: Comment
  errors: [UserError!]!
}

# ---- Subscription Root (real-time) ---------------------------------------
type Subscription {
  postPublished: Post!
  commentAdded(postId: ID!): Comment!
  userUpdated(id: ID!): User!
}

# ---- Schema ---------------------------------------------------------------
schema {
  query: Query
  mutation: Mutation
  subscription: Subscription
}
'''


_EXAMPLE_QUERIES = '''# GraphQL Example Queries
# Author: Hieu Louis (2026)

# 1. Simple query - get user by id
query GetUser($userId: ID!) {
  user(id: $userId) {
    id
    email
    name
    avatarUrl
    createdAt
  }
}

# 2. Nested query with pagination (Relay-style)
query GetUserWithPosts($userId: ID!, $first: Int, $after: String) {
  user(id: $userId) {
    id
    name
    posts(first: $first, after: $after) {
      edges {
        cursor
        node {
          id
          title
          status
          publishedAt
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
      totalCount
    }
  }
}

# 3. Mutation with input + error handling
mutation CreateUser($input: CreateUserInput!) {
  createUser(input: $input) {
    user {
      id
      email
      name
    }
    errors {
      message
      code
      field
    }
  }
}

# 4. Mutation with optimistic ID
mutation PublishPost($id: ID!) {
  publishPost(id: $id) {
    post {
      id
      status
      publishedAt
    }
    errors {
      message
      code
    }
  }
}

# 5. Subscription (real-time updates)
subscription OnPostPublished {
  postPublished {
    id
    title
    author {
      name
    }
  }
}

# 6. Union type - polymorphic search
query Search($q: String!) {
  search(query: $q) {
    __typename
    ... on User {
      id
      name
      email
    }
    ... on Post {
      id
      title
      status
    }
    ... on Comment {
      id
      body
    }
  }
}

# 7. Node interface lookup (Relay global ID)
query GetNode($id: ID!) {
  node(id: $id) {
    __typename
    ... on User { id name }
    ... on Post { id title status }
    ... on Comment { id body }
  }
}

# 8. Filtered + sorted list
query FilteredPosts($filter: PostFilter, $orderBy: [PostOrder!]) {
  posts(filter: $filter, orderBy: $orderBy, first: 20) {
    edges {
      node {
        id
        title
        tags
      }
    }
    totalCount
  }
}
'''


_RESOLVERS = '''"""GraphQL resolvers (Python / Strawberry or Ariadne style).

Uses DataLoader to mitigate N+1 queries for nested author/comments.

Author: Hieu Louis (2026)
"""
from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass


# DataLoader stub — replace with actual db access
class DataLoader:
    """Batched loader: collects ids within a single tick, then fetches all at once."""

    def __init__(self, batch_load_fn):
        self._batch_load_fn = batch_load_fn
        self._queue: List = []
        self._cache: dict = {}

    async def load(self, key):
        if key in self._cache:
            return self._cache[key]
        # In production: schedule a microtask to flush queue
        self._queue.append(key)
        # ... flush + cache (omitted for brevity)
        return self._cache.setdefault(key, None)


@dataclass
class User:
    id: str
    email: str
    name: str
    avatar_url: Optional[str] = None


@dataclass
class Post:
    id: str
    title: str
    body: str
    status: str
    author_id: str
    published_at: Optional[str] = None


# ---- Per-request context (DataLoader instances) ---------------------------
def build_context(db_session) -> dict:
    """Create per-request context with fresh DataLoaders (avoid cross-request leakage)."""
    return {
        "db": db_session,
        "user_loader": DataLoader(lambda ids: _batch_load_users(db_session, ids)),
        "post_loader": DataLoader(lambda ids: _batch_load_posts(db_session, ids)),
        "comment_loader": DataLoader(lambda ids: _batch_load_comments(db_session, ids)),
    }


async def _batch_load_users(db, ids: List[str]) -> List[Optional[User]]:
    rows = await db.fetch_all(
        "SELECT * FROM users WHERE id = ANY($1)", ids
    )
    by_id = {r["id"]: User(**r) for r in rows}
    return [by_id.get(i) for i in ids]


async def _batch_load_posts(db, ids: List[str]):
    rows = await db.fetch_all("SELECT * FROM posts WHERE id = ANY($1)", ids)
    by_id = {r["id"]: Post(**r) for r in rows}
    return [by_id.get(i) for i in ids]


async def _batch_load_comments(db, ids: List[str]):
    # stub
    return [None] * len(ids)


# ---- Resolvers -------------------------------------------------------------
async def resolve_user(user, info) -> User:
    """Field resolver: user (for Post.author)."""
    loader = info.context["user_loader"]
    return await loader.load(user.author_id)


async def resolve_posts(user, info, first: int = 10, after: Optional[str] = None):
    """User.posts connection — uses cursor pagination."""
    db = info.context["db"]
    # Decode cursor -> offset (in production: use opaque cursor encoding)
    offset = int(after, 36) + 1 if after else 0
    rows = await db.fetch_all(
        "SELECT * FROM posts WHERE author_id = $1 ORDER BY created_at "
        "LIMIT $2 OFFSET $3",
        user.id, first + 1, offset,
    )
    has_next = len(rows) > first
    edges = rows[:first]
    return {
        "edges": [
            {"cursor": _encode_cursor(offset + i), "node": Post(**r)}
            for i, r in enumerate(edges)
        ],
        "pageInfo": {
            "hasNextPage": has_next,
            "hasPreviousPage": offset > 0,
            "startCursor": _encode_cursor(offset) if edges else None,
            "endCursor": _encode_cursor(offset + len(edges) - 1) if edges else None,
        },
        "totalCount": await db.fetch_val(
            "SELECT count(*) FROM posts WHERE author_id = $1", user.id
        ),
    }


async def resolve_node(_, info, id: str):
    """Relay Node interface resolver."""
    # Decode global id: base64(typename:local_id)
    import base64
    decoded = base64.b64decode(id).decode()
    typename, local_id = decoded.split(":", 1)
    if typename == "User":
        return await info.context["user_loader"].load(local_id)
    if typename == "Post":
        return await info.context["post_loader"].load(local_id)
    return None


def _encode_cursor(offset: int) -> str:
    """Encode offset as base36 cursor."""
    import base64
    return base64.b64encode(str(offset).encode()).decode()
'''
