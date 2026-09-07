"""实体与 Schema Concept 的向量召回。"""
from __future__ import annotations

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.embedding_client import EmbeddingClient
from src.utils.json_io import read_json, write_json

from .models import SchemaConcept, SchemaSnapshot, SemanticCandidate


logger = logging.getLogger(__name__)


def _stable_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(value)


def build_schema_embedding_text(concept: SchemaConcept) -> str:
    """按固定字段顺序构造概念向量文本，字段与需求一一对应。"""

    return "\n".join(
        (
            f"schema: {concept.schema}",
            f"zhName: {concept.zh_name}",
            f"category: {concept.category}",
            f"description: {concept.description}",
            f"examples: {'、'.join(concept.examples)}",
        )
    )


def build_entity_embedding_text(entity: Mapping[str, Any]) -> str:
    """按固定字段顺序构造开放实体向量文本，不丢弃属性和溯源。"""

    return "\n".join(
        (
            f"name: {entity.get('name') or entity.get('official_name') or ''}",
            f"raw_type: {entity.get('raw_type', entity.get('type')) or ''}",
            f"type_zh: {entity.get('type_zh') or ''}",
            f"attributes: {_stable_json(entity.get('attributes'))}",
            f"provenance: {_stable_json(entity.get('provenance'))}",
        )
    )


class VectorSemanticRetriever:
    """编码全部动态概念，并为待映射实体召回余弦 Top-K。"""

    def __init__(
        self,
        snapshot: SchemaSnapshot,
        embedding_client: EmbeddingClient | None = None,
        top_k: int = 5,
        batch_size: int = 32,
        schema_embedding_cache_path: str | Path | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.snapshot = snapshot
        self.embedding_client = embedding_client or EmbeddingClient()
        self.top_k = top_k
        self.batch_size = batch_size
        self.schema_embedding_cache_path = (
            Path(schema_embedding_cache_path) if schema_embedding_cache_path is not None else None
        )
        self._concept_vectors: list[list[float]] | None = None

    def _encode_in_batches(self, texts: Sequence[str]) -> list[list[float]]:
        """优先调用客户端批量接口，并兼容仅提供 encode 的测试客户端。"""

        embed_batch = getattr(self.embedding_client, "embed_batch", None)
        if callable(embed_batch):
            vectors = embed_batch(texts, batch_size=self.batch_size)
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Embedding 服务返回数量不符：期望 {len(texts)}，实际 {len(vectors)}"
                )
            return vectors
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self.embedding_client.encode(batch)
            if len(encoded) != len(batch):
                raise RuntimeError(
                    f"Embedding 服务返回数量不符：期望 {len(batch)}，实际 {len(encoded)}"
                )
            vectors.extend(encoded)
        return vectors

    def _schema_embedding_metadata(self, texts: Sequence[str]) -> dict[str, Any]:
        """构造 Schema 向量缓存的版本元数据。"""

        config = getattr(self.embedding_client, "config", None)
        model = str(getattr(config, "model", ""))
        dimensions = getattr(config, "dimensions", None)
        digest_source = json.dumps(list(texts), ensure_ascii=False, separators=(",", ":"))
        return {
            "version": 1,
            "schema_hash": hashlib.sha256(digest_source.encode("utf-8")).hexdigest(),
            "embedding_model": model,
            "embedding_dimensions": dimensions,
        }

    def _load_schema_embedding_cache(
        self,
        metadata: Mapping[str, Any],
        concept_count: int,
    ) -> list[list[float]] | None:
        """读取并校验 Schema 向量缓存，失效时返回 None。"""

        if self.schema_embedding_cache_path is None:
            return None
        try:
            payload = read_json(self.schema_embedding_cache_path)
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        for key, value in metadata.items():
            if payload.get(key) != value:
                return None
        vectors = payload.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != concept_count:
            return None
        if any(not isinstance(vector, list) or not vector for vector in vectors):
            return None
        return vectors

    def _save_schema_embedding_cache(
        self,
        metadata: Mapping[str, Any],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """持久化当前 Schema 向量，供下一次 Stage 05 直接复用。"""

        if self.schema_embedding_cache_path is None:
            return
        try:
            write_json(
                self.schema_embedding_cache_path,
                {**dict(metadata), "vectors": [list(vector) for vector in vectors]},
            )
        except OSError as exc:
            logger.warning("写入 Schema embedding 缓存失败：%s", exc)

    def prepare(self) -> None:
        """加载有效的 Schema embedding 缓存，否则批量编码并持久化概念向量。"""

        if self._concept_vectors is not None:
            return
        texts = [build_schema_embedding_text(concept) for concept in self.snapshot.concepts]
        metadata = self._schema_embedding_metadata(texts)
        cached_vectors = self._load_schema_embedding_cache(metadata, len(texts))
        if cached_vectors is not None:
            self._concept_vectors = cached_vectors
            logger.info(
                "[Schema Embedding] 命中持久化缓存：%s，共 %s 个概念",
                self.schema_embedding_cache_path,
                len(cached_vectors),
            )
            return
        logger.info(
            "[Schema Embedding] 缓存未命中，开始编码 %s 个 Schema 概念",
            len(texts),
        )
        started_at = time.perf_counter()
        self._concept_vectors = self._encode_in_batches(texts)
        if not self._concept_vectors or any(not vector for vector in self._concept_vectors):
            raise RuntimeError("Schema Concept embedding 为空")
        self._save_schema_embedding_cache(metadata, self._concept_vectors)
        logger.info(
            "[Schema Embedding] 编码完成并写入缓存：%s，耗时 %.2f 秒",
            self.schema_embedding_cache_path,
            time.perf_counter() - started_at,
        )

    def retrieve_many(
        self,
        entities: Sequence[Mapping[str, Any]],
        categories: Sequence[str | None] | None = None,
    ) -> list[list[SemanticCandidate]]:
        """批量召回；有可靠一级分类时仅比较该类别内的 Schema 概念。"""

        if not entities:
            return []
        if categories is not None and len(categories) != len(entities):
            raise ValueError("实体数量与 Schema 一级类别数量不一致")
        self.prepare()
        routed_count = sum(category is not None for category in categories or ())
        logger.info(
            "[实体向量] 开始编码 %s 个去重实体；类别内召回=%s，全 Schema 召回=%s",
            len(entities),
            routed_count,
            len(entities) - routed_count,
        )
        started_at = time.perf_counter()
        entity_vectors = self._encode_in_batches(
            [build_entity_embedding_text(entity) for entity in entities]
        )
        assert self._concept_vectors is not None
        results: list[list[SemanticCandidate]] = []
        for index, entity_vector in enumerate(entity_vectors):
            requested_category = categories[index] if categories is not None else None
            ranked = [
                SemanticCandidate(
                    concept=concept,
                    similarity=self.embedding_client.cosine_similarity(entity_vector, concept_vector),
                )
                for concept, concept_vector in zip(self.snapshot.concepts, self._concept_vectors)
                if requested_category is None or concept.category == requested_category
            ]
            if not ranked and requested_category is not None:
                ranked = [
                    SemanticCandidate(
                        concept=concept,
                        similarity=self.embedding_client.cosine_similarity(
                            entity_vector, concept_vector
                        ),
                    )
                    for concept, concept_vector in zip(
                        self.snapshot.concepts, self._concept_vectors
                    )
                ]
            ranked.sort(key=lambda item: (-item.similarity, item.concept.schema))
            results.append(ranked[: min(self.top_k, len(ranked))])
        logger.info(
            "[实体向量] Top-%s 召回完成，耗时 %.2f 秒",
            self.top_k,
            time.perf_counter() - started_at,
        )
        return results


__all__ = [
    "VectorSemanticRetriever",
    "build_entity_embedding_text",
    "build_schema_embedding_text",
]
