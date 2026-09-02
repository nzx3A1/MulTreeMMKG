"""实体与 Schema Concept 的向量召回。"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from src.utils.embedding_client import EmbeddingClient

from .models import SchemaConcept, SchemaSnapshot, SemanticCandidate


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
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.snapshot = snapshot
        self.embedding_client = embedding_client or EmbeddingClient()
        self.top_k = top_k
        self.batch_size = batch_size
        self._concept_vectors: list[list[float]] | None = None

    def _encode_in_batches(self, texts: Sequence[str]) -> list[list[float]]:
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

    def prepare(self) -> None:
        """一次性按规范文本编码 65 个概念，而非依赖业务逻辑硬编码。"""

        if self._concept_vectors is not None:
            return
        texts = [build_schema_embedding_text(concept) for concept in self.snapshot.concepts]
        self._concept_vectors = self._encode_in_batches(texts)
        if not self._concept_vectors or any(not vector for vector in self._concept_vectors):
            raise RuntimeError("Schema Concept embedding 为空")

    def retrieve_many(
        self,
        entities: Sequence[Mapping[str, Any]],
    ) -> list[list[SemanticCandidate]]:
        """批量召回，每个实体返回按相似度降序排列的 Top-K。"""

        if not entities:
            return []
        self.prepare()
        entity_vectors = self._encode_in_batches(
            [build_entity_embedding_text(entity) for entity in entities]
        )
        assert self._concept_vectors is not None
        results: list[list[SemanticCandidate]] = []
        for entity_vector in entity_vectors:
            ranked = [
                SemanticCandidate(
                    concept=concept,
                    similarity=self.embedding_client.cosine_similarity(entity_vector, concept_vector),
                )
                for concept, concept_vector in zip(self.snapshot.concepts, self._concept_vectors)
            ]
            ranked.sort(key=lambda item: (-item.similarity, item.concept.schema))
            results.append(ranked[: min(self.top_k, len(ranked))])
        return results


__all__ = [
    "VectorSemanticRetriever",
    "build_entity_embedding_text",
    "build_schema_embedding_text",
]
