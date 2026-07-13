"""Neo4j 概念 Schema 的只读查询仓储。"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from neo4j import GraphDatabase

from config.neo4j_config import settings as neo4j_settings
from src.utils.embedding_client import EmbeddingClient

from .schema_models import SchemaConcept, SchemaRelation


ALL_CONCEPTS_CYPHER = """
MATCH (node:EntityConcept)
OPTIONAL MATCH (node)-[:BELONGS_TO_CATEGORY]->(category:ConceptCategory)
RETURN node.schema AS schema,
       node.zhName AS zh_name,
       coalesce(category.name, node.category, '') AS category,
       node.description AS description,
       node.examples AS examples,
       node.embedding AS embedding
ORDER BY node.schema
"""

VECTOR_SEARCH_CYPHER = """
CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
YIELD node, score
WHERE score >= $min_score
OPTIONAL MATCH (node)-[:BELONGS_TO_CATEGORY]->(category:ConceptCategory)
RETURN node.schema AS schema,
       node.zhName AS zh_name,
       coalesce(category.name, node.category, '') AS category,
       node.description AS description,
       node.examples AS examples,
       score AS vector_score
ORDER BY vector_score DESC
"""

NEIGHBORHOOD_CYPHER = """
MATCH (source:EntityConcept)-[relation:SCHEMA_RELATION]->(target:EntityConcept)
WHERE source.schema IN $seed_schemas OR target.schema IN $seed_schemas
RETURN source.schema AS source_schema,
       source.zhName AS source_zh_name,
       source.category AS source_category,
       source.description AS source_description,
       source.examples AS source_examples,
       target.schema AS target_schema,
       target.zhName AS target_zh_name,
       target.category AS target_category,
       target.description AS target_description,
       target.examples AS target_examples,
       relation.relationEn AS relation_en,
       relation.relationZh AS relation_zh
LIMIT $limit
"""

INDUCED_RELATIONS_CYPHER = """
MATCH (source:EntityConcept)-[relation:SCHEMA_RELATION]->(target:EntityConcept)
WHERE source.schema IN $schemas AND target.schema IN $schemas
RETURN source.schema AS source_schema,
       relation.relationEn AS relation_en,
       relation.relationZh AS relation_zh,
       target.schema AS target_schema
LIMIT $limit
"""


QueryRunner = Callable[[str, Mapping[str, Any]], Iterable[Mapping[str, Any]]]


class Neo4jSchemaRepository:
    """封装 Schema 概念、向量召回和一跳关系查询，并提供索引失败回退。"""

    def __init__(
        self,
        *,
        query_runner: QueryRunner | None = None,
        embedding_client: EmbeddingClient | None = None,
        vector_index_name: str = "entity_concept_embedding_idx",
    ) -> None:
        """创建只读仓储；测试可注入 query_runner 避免连接真实数据库。"""

        self._query_runner = query_runner
        self.embedding_client = embedding_client or EmbeddingClient()
        self.vector_index_name = vector_index_name
        self._concept_cache: tuple[SchemaConcept, ...] | None = None

    def _run(self, query: str, parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
        """统一执行参数化 Cypher，并把 Neo4j Record 转换为普通字典。"""

        if self._query_runner is not None:
            return [dict(row) for row in self._query_runner(query, parameters)]
        config = neo4j_settings.schema_db
        driver = GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
            connection_timeout=10.0,
            connection_acquisition_timeout=10.0,
        )
        try:
            driver.verify_connectivity()
            with driver.session(database=config.database) as session:
                return [dict(record) for record in session.run(query, **dict(parameters))]
        finally:
            driver.close()

    def all_concepts(self, *, refresh: bool = False) -> tuple[SchemaConcept, ...]:
        """读取并缓存全部概念节点，用于词法召回和向量索引降级。"""

        if self._concept_cache is None or refresh:
            rows = self._run(ALL_CONCEPTS_CYPHER, {})
            self._concept_cache = tuple(
                SchemaConcept.from_record(row) for row in rows if row.get("schema")
            )
        return self._concept_cache

    def vector_search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        min_score: float,
    ) -> tuple[tuple[SchemaConcept, ...], bool]:
        """优先查询 Neo4j 向量索引，失败时对全部概念执行内存余弦检索。"""

        if not query_embedding:
            return (), True
        parameters = {
            "index_name": self.vector_index_name,
            "top_k": int(top_k),
            "query_embedding": list(query_embedding),
            "min_score": float(min_score),
        }
        try:
            rows = self._run(VECTOR_SEARCH_CYPHER, parameters)
            concepts = tuple(
                SchemaConcept.from_record(row, vector_score=float(row.get("vector_score") or 0.0))
                for row in rows
                if row.get("schema")
            )
            return concepts, False
        except Exception:
            scored = []
            for concept in self.all_concepts():
                score = self.embedding_client.cosine_similarity(query_embedding, concept.embedding)
                if score >= min_score:
                    scored.append(
                        SchemaConcept(
                            **{
                                **concept.__dict__,
                                "vector_score": score,
                                "selection_reasons": ("全量余弦回退",),
                            }
                        )
                    )
            scored.sort(key=lambda item: (-item.vector_score, item.schema))
            return tuple(scored[:top_k]), True

    def neighborhood(
        self,
        seed_schemas: Sequence[str],
        *,
        limit: int = 200,
    ) -> tuple[tuple[SchemaConcept, ...], tuple[SchemaRelation, ...]]:
        """读取种子概念的一跳入边和出边，并返回邻居概念与有向关系。"""

        seeds = sorted({str(item) for item in seed_schemas if item})
        if not seeds:
            return (), ()
        rows = self._run(NEIGHBORHOOD_CYPHER, {"seed_schemas": seeds, "limit": int(limit)})
        concepts: dict[str, SchemaConcept] = {}
        relations: dict[tuple[str, str, str], SchemaRelation] = {}
        for row in rows:
            for prefix in ("source", "target"):
                schema = str(row.get(f"{prefix}_schema") or "")
                if schema:
                    concepts[schema] = SchemaConcept.from_record(
                        {
                            "schema": schema,
                            "zh_name": row.get(f"{prefix}_zh_name"),
                            "category": row.get(f"{prefix}_category"),
                            "description": row.get(f"{prefix}_description"),
                            "examples": row.get(f"{prefix}_examples"),
                        }
                    )
            relation = SchemaRelation(
                str(row.get("source_schema") or ""),
                str(row.get("relation_en") or ""),
                str(row.get("relation_zh") or ""),
                str(row.get("target_schema") or ""),
            )
            if all(relation.key):
                relations[relation.key] = relation
        return tuple(concepts.values()), tuple(relations.values())

    def induced_relations(
        self,
        schemas: Sequence[str],
        *,
        limit: int = 500,
    ) -> tuple[SchemaRelation, ...]:
        """读取指定概念集合内部的全部合法有向关系。"""

        names = sorted({str(item) for item in schemas if item})
        if len(names) < 2:
            return ()
        rows = self._run(INDUCED_RELATIONS_CYPHER, {"schemas": names, "limit": int(limit)})
        relations = {
            SchemaRelation(
                str(row.get("source_schema") or ""),
                str(row.get("relation_en") or ""),
                str(row.get("relation_zh") or ""),
                str(row.get("target_schema") or ""),
            )
            for row in rows
        }
        return tuple(sorted(relations, key=lambda item: item.key))


__all__ = [
    "ALL_CONCEPTS_CYPHER",
    "INDUCED_RELATIONS_CYPHER",
    "NEIGHBORHOOD_CYPHER",
    "VECTOR_SEARCH_CYPHER",
    "Neo4jSchemaRepository",
]
