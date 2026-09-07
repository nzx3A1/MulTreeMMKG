"""从 Neo4j 或 JSON 配置动态加载概念 Schema。"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol

from config.neo4j_config import Neo4jSchemaDatabaseConfig, settings as neo4j_settings
from src.utils.json_io import read_json

from .models import SchemaConcept, SchemaRelation, SchemaSnapshot


logger = logging.getLogger(__name__)


READ_CONCEPTS_CYPHER = """
MATCH (concept:EntityConcept)
RETURN concept.schema AS schema,
       concept.zhName AS zhName,
       concept.category AS category,
       concept.description AS description,
       concept.examples AS examples,
       concept.embedding AS embedding,
       concept.embeddingText AS embeddingText
ORDER BY concept.schema
"""

READ_RELATIONS_CYPHER = """
MATCH (source:EntityConcept)-[relation:SCHEMA_RELATION]->(target:EntityConcept)
RETURN source.schema AS source,
       relation.relationEn AS relationEn,
       relation.relationZh AS relationZh,
       target.schema AS target
ORDER BY source.schema, target.schema, relation.relationEn
"""


class SchemaRepository(Protocol):
    """可注入的 Schema 数据源接口。"""

    def load(self) -> SchemaSnapshot:
        """加载一次完整快照。"""


class Neo4jSchemaRepository:
    """从用户指定的概念图谱数据库读取 65/297 定义。"""

    def __init__(
        self,
        config: Neo4jSchemaDatabaseConfig | None = None,
        driver: Any = None,
    ) -> None:
        self.config = config or neo4j_settings.schema_db
        self._driver = driver
        self._owns_driver = driver is None

    def load(self) -> SchemaSnapshot:
        """连接 Neo4j，并一次性加载全部实体概念和有向关系定义。"""

        started_at = time.perf_counter()
        logger.info(
            "[Schema 1/2] 正在连接 Neo4j：%s，数据库=%s",
            self.config.uri,
            self.config.database,
        )
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
            )
        try:
            self._driver.verify_connectivity()
            with self._driver.session(database=self.config.database) as session:
                concepts = [SchemaConcept.from_mapping(dict(row)) for row in session.run(READ_CONCEPTS_CYPHER)]
                relations = [SchemaRelation.from_mapping(dict(row)) for row in session.run(READ_RELATIONS_CYPHER)]
        finally:
            if self._owns_driver and self._driver is not None:
                self._driver.close()
                self._driver = None

        if not concepts:
            raise RuntimeError(f"Neo4j 数据库 {self.config.database!r} 中没有 EntityConcept")
        logger.info(
            "[Schema 2/2] Neo4j 加载完成：概念=%s，关系=%s，耗时 %.2f 秒",
            len(concepts),
            len(relations),
            time.perf_counter() - started_at,
        )
        return SchemaSnapshot(
            concepts=concepts,
            relations=relations,
            source=f"neo4j://{self.config.database}",
        )


class JsonSchemaRepository:
    """从可版本化 JSON 快照读取 Schema，便于离线运行和单元测试。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> SchemaSnapshot:
        """从 JSON 文件读取完整 Schema 快照并记录加载规模。"""

        started_at = time.perf_counter()
        logger.info("[Schema 1/2] 正在读取离线 Schema：%s", self.path)
        payload = read_json(self.path)
        if not isinstance(payload, dict):
            raise TypeError("Schema JSON 顶层必须是对象")
        concepts = [SchemaConcept.from_mapping(item) for item in payload.get("concepts", [])]
        relations = [SchemaRelation.from_mapping(item) for item in payload.get("relations", [])]
        if not concepts:
            raise ValueError(f"Schema JSON 中没有 concepts：{self.path}")
        logger.info(
            "[Schema 2/2] 离线 Schema 加载完成：概念=%s，关系=%s，耗时 %.2f 秒",
            len(concepts),
            len(relations),
            time.perf_counter() - started_at,
        )
        return SchemaSnapshot(concepts, relations, source=str(self.path.resolve()))


class InMemorySchemaRepository:
    """测试或上层编排直接注入快照时使用。"""

    def __init__(self, snapshot: SchemaSnapshot) -> None:
        self.snapshot = snapshot

    def load(self) -> SchemaSnapshot:
        """返回测试或上层调用方预先构造的内存 Schema。"""

        return self.snapshot


__all__ = [
    "InMemorySchemaRepository",
    "JsonSchemaRepository",
    "Neo4jSchemaRepository",
    "READ_CONCEPTS_CYPHER",
    "READ_RELATIONS_CYPHER",
    "SchemaRepository",
]
