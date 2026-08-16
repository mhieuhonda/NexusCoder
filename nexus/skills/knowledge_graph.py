"""Knowledge Graph Skill - Neo4j Cypher schema + queries + RDF mapping.

Sinh đồ thị tri thức trong Neo4j: schema constraints, indexed labels/relationships,
ingestion queries (MERGE), traversal queries (1-hop / multi-hop / shortest path),
page-rank style analytics, và mapping sang RDF (n10s / neosemantics).

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


NEO4J_SCHEMA = """// Neo4j schema for a knowledge graph / Schema đồ thị tri thức
// ============================================================
// Constraints & indexes (run once)
CREATE CONSTRAINT person_id IF NOT EXISTS
  FOR (n:Person) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT org_id IF NOT EXISTS
  FOR (n:Organization) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT concept_id IF NOT EXISTS
  FOR (n:Concept) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT article_id IF NOT EXISTS
  FOR (n:Article) REQUIRE n.doi IS UNIQUE;

CREATE INDEX person_name IF NOT EXISTS FOR (n:Person) ON (n.name);
CREATE INDEX org_name   IF NOT EXISTS FOR (n:Organization) ON (n.name);
CREATE INDEX article_year IF NOT EXISTS FOR (n:Article) ON (n.year);

// Full-text index for fuzzy search / Index full-text để tìm mờ
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
  FOR (n:Person|Organization|Concept) ON EACH [n.name, n.description];

// Labels: Person, Organization, Concept, Article, Location, Event
// Relationships:
//   (Person)-[:WORKS_AT]->(Organization)
//   (Person)-[:AUTHORED]->(Article)
//   (Article)-[:MENTIONS]->(Concept)
//   (Person)-[:KNOWS {since: date}]->(Person)
//   (Organization)-[:LOCATED_IN]->(Location)
//   (Concept)-[:SUBCLASS_OF]->(Concept)
"""

INGEST_QUERIES = """// Ingest with MERGE (idempotent) / Nhập liệu bằng MERGE
// ---- People & organizations ----
UNWIND $people AS p
MERGE (person:Person {id: p.id})
  SET person.name = p.name, person.bio = p.bio, person.updated_at = datetime()
MERGE (org:Organization {id: p.org_id})
  SET org.name = p.org_name
MERGE (person)-[:WORKS_AT]->(org);

// ---- Articles & concepts ----
UNWIND $articles AS a
MERGE (art:Article {doi: a.doi})
  SET art.title = a.title, art.year = a.year, art.abstract = a.abstract
WITH art, a
UNWIND a.author_ids AS aid
  MATCH (au:Person {id: aid})
  MERGE (au)-[:AUTHORED]->(art)
WITH art, a
UNWIND a.concepts AS c
  MERGE (con:Concept {id: c.id}) SET con.name = c.name
  MERGE (art)-[:MENTIONS]->(con);

// ---- Concept taxonomy (subclass-of) ----
UNWIND $edges AS e
MATCH (c1:Concept {id: e.from}), (c2:Concept {id: e.to})
MERGE (c1)-[:SUBCLASS_OF]->(c2);
"""

TRAVERSAL_QUERIES = """// Common traversal & analytics queries / Truy vấn phổ biến

// 1. Co-authors (1-hop) / Đồng tác giả
MATCH (p:Person {id: $person_id})-[:AUTHORED]->(:Article)<-[:AUTHORED]-(co)
RETURN co.name AS coauthor, count(*) AS joint_papers
ORDER BY joint_papers DESC LIMIT 10;

// 2. Shortest path between two people / Đường đi ngắn nhất
MATCH path = shortestPath(
  (p1:Person {id: $from})-[:KNOWS|AUTHORED*..6]-(p2:Person {id: $to})
)
RETURN [n IN nodes(path) | coalesce(n.name, n.title)] AS hops;

// 3. Top influential concepts (degree centrality) / Khái niệm quan trọng
MATCH (c:Concept)<-[:MENTIONS]-(:Article)
RETURN c.name AS concept, count(*) AS mentions
ORDER BY mentions DESC LIMIT 20;

// 4. PageRank via GDS (Graph Data Science library)
CALL gds.pageRank.stream('conceptGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS concept, score
ORDER BY score DESC LIMIT 25;

// 5. Community detection (Louvain) / Phát hiện cộng đồng
CALL gds.louvain.write('entityGraph', { writeProperty: 'community' })
YIELD communityCount, modularity;

// 6. Find experts on a topic (with hop limit) / Tìm chuyên gia
MATCH (c:Concept {name: $topic})<-[:MENTIONS]-(a:Article)<-[:AUTHORED]-(p:Person)
WITH p, count(a) AS papers, collect(a.year) AS years
RETURN p.name AS expert, papers, years
ORDER BY papers DESC LIMIT 10;

// 7. Org collaboration network / Mạng hợp tác tổ chức
MATCH (o1:Organization)<-[:WORKS_AT]-(p1)-[:AUTHORED]->(a)<-[:AUTHORED]-(p2)-[:WORKS_AT]->(o2)
WHERE id(o1) < id(o2)
RETURN o1.name, o2.name, count(DISTINCT a) AS joint_papers
ORDER BY joint_papers DESC LIMIT 10;
"""

RDF_MAPPING = """
RDF Export / Mapping (neosemantics / n10s)
==========================================
1. Enable RDF in Neo4j:
   CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS
     FOR (r:Resource) REQUIRE r.uri IS UNIQUE;
   CALL n10s.graphconfig.init({handleVocabUris: "MAP"});
   CALL n10s.nsprefixes.add("schema", "https://schema.org/");
   CALL n10s.nsprefixes.add("ex", "https://example.org/kg/");

2. Export as Turtle:
   :Person_123 a schema:Person ;
       schema:name "Hieu Louis" ;
       schema:worksFor :Org_42 .
   :Org_42 a schema:Organization ;
       schema:name "ACME" .
   :Article_doi a schema:ScholarlyArticle ;
       schema:author :Person_123 ;
       schema:about :Concept_ML .

3. SPARQL federated query (on exported RDF):
   SELECT ?expert ?paper WHERE {
     ?paper schema:about/schema:name "Machine Learning" ;
            schema:author ?expert .
   }
"""


class KnowledgeGraphSkill(Skill):
    """Sinh Neo4j Cypher schema, queries, và RDF mapping cho knowledge graph."""

    category = SkillCategory.DATA
    priority = SkillPriority.LOW
    keywords: List[str] = [
        "knowledge graph", "neo4j", "cypher", "graph database",
        "entity relation", "entity-relationship", "rdf", "owl",
        "sparql", "n10s", "neosemantics", "kg",
    ]
    examples = [
        "Tạo knowledge graph schema trên Neo4j",
        "Sinh Cypher queries cho co-author network",
        "Map Neo4j entities to RDF / OWL",
    ]

    @property
    def name(self) -> str:
        return "knowledge_graph"

    @property
    def description(self) -> str:
        return (
            "Sinh Neo4j Cypher schema (constraints + indexes), ingestion (MERGE), "
            "traversal + analytics queries (PageRank, Louvain), và RDF mapping "
            "via neosemantics."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.16
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        artifacts: List[Dict[str, str]] = [
            {"name": "schema.cypher", "language": "cypher", "content": NEO4J_SCHEMA},
            {"name": "ingest.cypher", "language": "cypher", "content": INGEST_QUERIES},
            {"name": "queries.cypher", "language": "cypher", "content": TRAVERSAL_QUERIES},
            {"name": "RDF_MAPPING.md", "language": "markdown", "content": RDF_MAPPING},
        ]

        return SkillResult(
            success=True,
            output=(
                "[knowledge_graph] Generated Neo4j Cypher: schema (constraints+indexes), "
                "ingestion (MERGE idempotent), 7 traversal/analytics queries "
                "(PageRank, Louvain, shortest path) + RDF export guide."
            ),
            artifacts=artifacts,
            suggestions=[
                "Install APOC + Graph Data Science (GDS) plugin for PageRank/Louvain",
                "Use EXPLAIN / PROFILE to verify query plans use indexes",
                "Batch ingest with `:auto` + periodic.commit for >10k nodes",
                "Add schema validation (SHACL) when exporting to RDF",
                "Consider Stardog / GraphDB if SPARQL reasoning (OWL) is required",
            ],
            metadata={
                "skill": self.name,
                "labels": ["Person", "Organization", "Concept", "Article", "Location", "Event"],
                "relationships": ["WORKS_AT", "AUTHORED", "MENTIONS", "KNOWS", "SUBCLASS_OF"],
                "algorithms": ["PageRank", "Louvain", "shortestPath", "degree centrality"],
                "version": self.version,
                "author": self.author,
            },
        )
