"""把 postSchema 新增二级概念连入 Neo4j，并统一附加可识别标签和审计属性。"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from config.model_config import settings as model_settings
from config.neo4j_config import Neo4jSchemaDatabaseConfig, settings as neo4j_settings
from src.utils.embedding_client import EmbeddingClient

from .models import NewConceptProposal


LOGGER = logging.getLogger(__name__)


READ_EXISTING_CYPHER = """
UNWIND $schemas AS schema
OPTIONAL MATCH (concept:EntityConcept {schema: schema})
OPTIONAL MATCH (concept)-[:BELONGS_TO_CATEGORY]->(category:ConceptCategory)
RETURN schema,
       labels(concept) AS labels,
       concept.zhName AS zhName,
       concept.category AS category,
       collect(category.name) AS linked_categories
ORDER BY schema
"""

READ_CATEGORIES_CYPHER = """
UNWIND $categories AS category_name
OPTIONAL MATCH (category:ConceptCategory {name: category_name})
RETURN category_name, category IS NOT NULL AS exists
ORDER BY category_name
"""

CREATE_CONCEPTS_CYPHER = """
UNWIND $rows AS row
MATCH (category:ConceptCategory {name: row.category})
CREATE (concept:EntityConcept:PostSchemaAdded {
    schema: row.schema,
    zhName: row.zhName,
    category: row.category,
    description: row.description,
    examples: row.examples,
    embedding: row.embedding,
    embeddingText: row.embedding_text,
    embeddingModel: $embedding_model,
    embeddedAt: datetime(),
    postSchemaAdded: true,
    postSchemaVersion: 'v1',
    postSchemaSourceRawTypes: row.source_raw_types,
    postSchemaModel: $llm_model,
    postSchemaCreatedAt: datetime()
})
CREATE (concept)-[:BELONGS_TO_CATEGORY]->(category)
SET category.postSchemaNeedsDescriptionRefresh = true
RETURN count(concept) AS created_count
"""

VERIFY_CONCEPTS_CYPHER = """
UNWIND $schemas AS schema
MATCH (concept:EntityConcept:PostSchemaAdded {schema: schema})
MATCH (concept)-[:BELONGS_TO_CATEGORY]->(category:ConceptCategory)
RETURN concept.schema AS schema,
       concept.category AS category,
       category.name AS linked_category,
       concept.postSchemaAdded AS marked,
       size(coalesce(concept.embedding, [])) AS embedding_dimensions
ORDER BY schema
"""


def build_embedding_text(concept: NewConceptProposal) -> str:
    """按既有 Schema 向量格式拼接新概念文本。"""

    return "\n".join(
        (
            f"类别：{concept.category}",
            f"描述：{concept.description}",
            f"示例：{'、'.join(concept.examples)}",
            f"中文名称：{concept.zh_name}",
        )
    )


class Neo4jPostSchemaWriter:
    """以先校验、后向量化、再单事务创建的方式写入新增二级概念。"""

    def __init__(
        self,
        config: Neo4jSchemaDatabaseConfig | None = None,
        embedding_client: EmbeddingClient | None = None,
        driver: Any | None = None,
    ) -> None:
        """保存 Neo4j、Embedding 配置和可选测试驱动。"""

        self.config = config or neo4j_settings.schema_db
        self.embedding_client = embedding_client or EmbeddingClient()
        self._driver = driver
        self._owns_driver = driver is None

    @staticmethod
    def _deduplicate(concepts: Sequence[NewConceptProposal]) -> list[NewConceptProposal]:
        """按 schema 去重，并拒绝同名但定义冲突的新概念。"""

        unique: dict[str, NewConceptProposal] = {}
        for concept in concepts:
            previous = unique.get(concept.schema)
            if previous is not None and previous != concept:
                raise ValueError(f"新增概念定义冲突：{concept.schema}")
            unique[concept.schema] = concept
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _validate_existing(rows: Sequence[dict[str, Any]], expected: dict[str, NewConceptProposal]) -> set[str]:
        """允许完全一致的 postSchema 节点幂等复用，并拒绝覆盖原有 Schema。"""

        reusable: set[str] = set()
        for row in rows:
            schema = str(row.get("schema") or "")
            labels = set(row.get("labels") or ())
            if not labels:
                continue
            concept = expected[schema]
            linked = {str(item) for item in row.get("linked_categories") or () if item}
            if (
                "PostSchemaAdded" in labels
                and str(row.get("zhName") or "") == concept.zh_name
                and str(row.get("category") or "") == concept.category
                and linked == {concept.category}
            ):
                reusable.add(schema)
                continue
            raise RuntimeError(f"Neo4j 已存在不可覆盖的 EntityConcept.schema：{schema}")
        return reusable

    def write(
        self,
        concepts: Sequence[NewConceptProposal],
        *,
        llm_model: str,
    ) -> dict[str, Any]:
        """写入并验证新增概念，返回创建数、复用数和向量维度。"""

        unique = self._deduplicate(concepts)
        if not unique:
            return {
                "created_count": 0,
                "reused_count": 0,
                "ensured_count": 0,
                "embedding_dimensions": 0,
            }
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
            )
        try:
            self._driver.verify_connectivity()
            expected = {item.schema: item for item in unique}
            with self._driver.session(database=self.config.database) as session:
                category_rows = [
                    dict(row)
                    for row in session.run(
                        READ_CATEGORIES_CYPHER,
                        categories=sorted({item.category for item in unique}),
                    )
                ]
                missing_categories = [
                    str(row.get("category_name") or "")
                    for row in category_rows
                    if not row.get("exists")
                ]
                if missing_categories:
                    raise RuntimeError(f"Neo4j 缺少一级 ConceptCategory：{missing_categories}")

                existing_rows = [
                    dict(row)
                    for row in session.run(READ_EXISTING_CYPHER, schemas=sorted(expected))
                ]
                reusable = self._validate_existing(existing_rows, expected)
                pending = [item for item in unique if item.schema not in reusable]
                texts = [build_embedding_text(item) for item in pending]
                vectors = self.embedding_client.embed_batch(texts) if texts else []
                if len(vectors) != len(pending):
                    raise RuntimeError(
                        f"新增概念向量数量不一致：预期 {len(pending)}，实际 {len(vectors)}"
                    )
                dimensions = {len(vector) for vector in vectors}
                if len(dimensions) > 1 or (vectors and not next(iter(dimensions))):
                    raise RuntimeError(f"新增概念向量维度无效：{dimensions}")
                rows = [
                    {
                        **concept.to_dict(),
                        "embedding": vector,
                        "embedding_text": text,
                    }
                    for concept, vector, text in zip(pending, vectors, texts)
                ]
                created_count = 0
                if rows:
                    record = session.run(
                        CREATE_CONCEPTS_CYPHER,
                        rows=rows,
                        embedding_model=model_settings.embedding.model,
                        llm_model=llm_model,
                    ).single()
                    created_count = int(record["created_count"] if record else 0)
                    if created_count != len(rows):
                        raise RuntimeError(
                            f"Neo4j 新概念创建数不一致：预期 {len(rows)}，实际 {created_count}"
                        )

                verification = [
                    dict(row)
                    for row in session.run(VERIFY_CONCEPTS_CYPHER, schemas=sorted(expected))
                ]
                if len(verification) != len(expected):
                    raise RuntimeError(
                        f"Neo4j 新概念验证数不一致：预期 {len(expected)}，实际 {len(verification)}"
                    )
                for row in verification:
                    if (
                        row.get("category") != row.get("linked_category")
                        or not row.get("marked")
                        or int(row.get("embedding_dimensions") or 0) <= 0
                    ):
                        raise RuntimeError(f"Neo4j 新概念验证失败：{row.get('schema')}")
                embedding_dimensions = (
                    int(verification[0].get("embedding_dimensions") or 0) if verification else 0
                )
                LOGGER.info(
                    "postSchema 新概念写入完成：创建=%s，幂等复用=%s，向量维度=%s",
                    created_count,
                    len(reusable),
                    embedding_dimensions,
                )
                return {
                    "created_count": created_count,
                    "reused_count": len(reusable),
                    "ensured_count": len(expected),
                    "embedding_dimensions": embedding_dimensions,
                }
        finally:
            if self._owns_driver and self._driver is not None:
                self._driver.close()
                self._driver = None


__all__ = [
    "CREATE_CONCEPTS_CYPHER",
    "Neo4jPostSchemaWriter",
    "VERIFY_CONCEPTS_CYPHER",
    "build_embedding_text",
]
