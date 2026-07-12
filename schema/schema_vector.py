"""为 Neo4j 中的 EntityConcept 节点生成并写入向量嵌入。

脚本读取 ``config`` 中的 Neo4j 与 Embedding 配置，将节点的 category、
description、examples、zhName 和 schema 属性拼成结构化文本后调用 SiliconFlow。
默认仅处理没有 embedding 的节点，可用 ``--force`` 强制重新生成全部向量。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import requests
from neo4j import GraphDatabase


# 直接运行本文件时，将项目根目录加入模块搜索路径，确保可以导入 config。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_config import EmbeddingConfig, settings as model_settings  # noqa: E402
from config.neo4j_config import settings as neo4j_settings  # noqa: E402


LOGGER = logging.getLogger("schema_vector")

READ_NODES_CYPHER = """
MATCH (node:EntityConcept)
WHERE $force OR node.embedding IS NULL
RETURN node.schema AS schema,
       node.zhName AS zhName,
       node.category AS category,
       node.description AS description,
       node.examples AS examples
ORDER BY node.schema
"""

WRITE_EMBEDDINGS_CYPHER = """
UNWIND $rows AS row
MATCH (node:EntityConcept {schema: row.schema})
SET node.embedding = row.embedding,
    node.embeddingModel = $model,
    node.embeddingText = row.embedding_text,
    node.embeddedAt = datetime()
RETURN count(node) AS updated_count
"""

VERIFY_CYPHER = """
MATCH (node:EntityConcept)
RETURN count(node) AS total_count,
       count(node.embedding) AS embedded_count,
       count(CASE WHEN node.embedding IS NOT NULL
                       AND size(node.embedding) = $dimensions THEN 1 END) AS valid_count
"""


def batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    """按指定大小切分序列，避免单次 API 请求和 Neo4j 事务过大。"""

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_embedding_text(node: dict[str, Any]) -> str:
    """按固定顺序拼接实体概念属性，使相同节点始终得到一致的输入文本。"""

    examples = node.get("examples") or []
    if not isinstance(examples, list):
        examples = [examples]
    return "\n".join(
        (
            f"类别：{node.get('category') or ''}",
            f"描述：{node.get('description') or ''}",
            f"示例：{'、'.join(str(item) for item in examples)}",
            f"中文名称：{node.get('zhName') or ''}"
        )
    )


def request_embeddings(
    session: requests.Session,
    texts: list[str],
    config: EmbeddingConfig,
    retry_times: int,
) -> list[list[float]]:
    """批量调用 SiliconFlow Embeddings API，并对暂时性请求错误进行重试。"""

    if not config.api_key:
        raise ValueError("Embedding API Key 为空，请在 config 或 EMBEDDING_API_KEY 中配置")

    for attempt in range(1, retry_times + 1):
        try:
            response = session.post(
                config.base_url,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                json={"input": texts, "model": config.model, "encoding_format": "float"},
                timeout=config.timeout_secs,
            )
            response.raise_for_status()
            rows = sorted(response.json().get("data", []), key=lambda row: row.get("index", 0))
            vectors = [row.get("embedding") for row in rows]
            if len(vectors) != len(texts) or any(not isinstance(vector, list) for vector in vectors):
                raise ValueError("Embedding API 返回的向量数量或格式不正确")
            return vectors
        except (requests.RequestException, ValueError) as exc:
            if attempt == retry_times:
                raise RuntimeError(f"Embedding API 连续 {retry_times} 次调用失败") from exc
            wait_seconds = 2 ** (attempt - 1)
            LOGGER.warning("Embedding API 调用失败，%s 秒后重试：%s", wait_seconds, exc)
            time.sleep(wait_seconds)

    raise RuntimeError("未能获取向量")


def vectorize_entity_concepts(force: bool = False, batch_size: int | None = None) -> tuple[int, int]:
    """读取 EntityConcept 节点、生成嵌入并回写，返回更新数和向量维度。"""

    database = neo4j_settings.schema_db
    embedding_config = model_settings.embedding
    actual_batch_size = batch_size or embedding_config.batch_size
    if actual_batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    driver = GraphDatabase.driver(database.uri, auth=(database.username, database.password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database.database) as neo4j_session:
            nodes = [dict(record) for record in neo4j_session.run(READ_NODES_CYPHER, force=force)]
            LOGGER.info("待处理 EntityConcept 节点数：%s", len(nodes))
            if not nodes:
                return 0, 0

            updated_count = 0
            vector_dimensions = 0
            with requests.Session() as http_session:
                for batch_number, node_batch in enumerate(batched(nodes, actual_batch_size), start=1):
                    texts = [build_embedding_text(node) for node in node_batch]
                    vectors = request_embeddings(
                        http_session,
                        texts,
                        embedding_config,
                        retry_times=3,
                    )
                    dimensions = {len(vector) for vector in vectors}
                    if len(dimensions) != 1:
                        raise ValueError(f"同一批次返回了不同维度的向量：{dimensions}")
                    vector_dimensions = dimensions.pop()
                    if embedding_config.dimensions and vector_dimensions != embedding_config.dimensions:
                        raise ValueError(
                            f"向量维度不符：配置为 {embedding_config.dimensions}，实际为 {vector_dimensions}"
                        )

                    rows = [
                        {
                            "schema": node["schema"],
                            "embedding": vector,
                            "embedding_text": text,
                        }
                        for node, vector, text in zip(node_batch, vectors, texts)
                    ]
                    result = neo4j_session.run(
                        WRITE_EMBEDDINGS_CYPHER,
                        rows=rows,
                        model=embedding_config.model,
                    ).single()
                    batch_updated = int(result["updated_count"] if result else 0)
                    updated_count += batch_updated
                    LOGGER.info("第 %s 批完成，写入 %s 个节点", batch_number, batch_updated)

            verification = neo4j_session.run(
                VERIFY_CYPHER,
                dimensions=vector_dimensions,
            ).single()
            LOGGER.info(
                "验证结果：总节点 %s，已有向量 %s，本次维度有效 %s",
                verification["total_count"],
                verification["embedded_count"],
                verification["valid_count"],
            )
            return updated_count, vector_dimensions
    finally:
        driver.close()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="为 EntityConcept 节点添加向量嵌入属性")
    parser.add_argument("--force", default=True, action="store_true", help="重新生成所有节点的向量")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖配置中的 API 批大小")
    return parser.parse_args()


def main() -> int:
    """执行自动化向量写入流程，并以进程退出码报告结果。"""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        updated_count, dimensions = vectorize_entity_concepts(args.force, args.batch_size)
    except Exception:
        LOGGER.exception("EntityConcept 向量写入失败")
        return 1
    LOGGER.info("处理完成：更新 %s 个节点，向量维度 %s", updated_count, dimensions or "无")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
