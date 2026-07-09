"""Neo4j 数据库连接配置。

配置从环境变量或 .env 读取，默认值仅用于本地开发。写入器可分别使用 schema
数据库和 document 数据库存放概念 schema 与论文抽取结果。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .model_config import PROJECT_ROOT, _load_env_file


_load_env_file(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Neo4jSchemaDatabaseConfig:
    """单个 Neo4j 数据库连接配置。"""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "123456"
    database: str = "petrommkg-schema"

@dataclass(frozen=True)
class Neo4jDocumentDatabaseConfig:
    """单个 Neo4j 数据库连接配置。"""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "123456"
    database: str = "petrommkg-document"

@dataclass(frozen=True)
class Neo4jSettings:
    """Neo4j 写入相关配置。"""

    schema_db: Neo4jSchemaDatabaseConfig
    document_db: Neo4jDocumentDatabaseConfig
    batch_size: int = 500
    write_timeout_secs: float = 60.0
    create_constraints: bool = True


def load_neo4j_settings() -> Neo4jSettings:
    """从环境变量加载 Neo4j 配置。"""

    schema_db = Neo4jSchemaDatabaseConfig(
        uri=os.getenv("NEO4J_SCHEMA_URI", os.getenv("NEO4J_URI", Neo4jSchemaDatabaseConfig.uri)),
        username=os.getenv("NEO4J_SCHEMA_USER", os.getenv("NEO4J_USER", Neo4jSchemaDatabaseConfig.username)),
        password=os.getenv("NEO4J_SCHEMA_PASSWORD", os.getenv("NEO4J_PASSWORD", Neo4jSchemaDatabaseConfig.password)),
        database=os.getenv("NEO4J_SCHEMA_DATABASE", "petrommkg-schema"),
    )
    document_db = Neo4jDocumentDatabaseConfig(
        uri=os.getenv("NEO4J_DOCUMENT_URI", os.getenv("NEO4J_URI", Neo4jDocumentDatabaseConfig.uri)),
        username=os.getenv("NEO4J_DOCUMENT_USER", os.getenv("NEO4J_USER", Neo4jDocumentDatabaseConfig.username)),
        password=os.getenv("NEO4J_DOCUMENT_PASSWORD", os.getenv("NEO4J_PASSWORD", Neo4jDocumentDatabaseConfig.password)),
        database=os.getenv("NEO4J_DOCUMENT_DATABASE", "petrommkg-document"),
    )
    return Neo4jSettings(
        schema_db=schema_db,
        document_db=document_db,
        batch_size=int(os.getenv("NEO4J_BATCH_SIZE", "500")),
        write_timeout_secs=float(os.getenv("NEO4J_WRITE_TIMEOUT_SECS", "60")),
        create_constraints=os.getenv("NEO4J_CREATE_CONSTRAINTS", "true").lower() in {"1", "true", "yes"},
    )


settings = load_neo4j_settings()
