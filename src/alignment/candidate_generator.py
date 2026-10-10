"""为同一文档中的实体 mention 生成高召回候选对。

候选生成先利用名称和 ``schema_type_zh`` 分块，再补充邻接结构与可选向量相似度，
避免把全量两两比较直接交给 LLM。
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Mapping, Sequence

from src.utils.embedding_client import EmbeddingClient

from .models import CandidatePair, EntityMention, SourceKey
from .normalizer import name_similarity
from .scope_rules import AlignmentRules


def _jaccard(left: set[str], right: set[str]) -> float:
    """计算两个集合的 Jaccard 相似度，无证据时返回中性值。"""

    if not left or not right:
        return 0.5
    return len(left & right) / len(left | right)


def _section_parts(chunk_id: str) -> tuple[str, ...]:
    """从 Chunk ID 中提取章节层级片段。"""

    section = str(chunk_id or "").split(":", 1)[0]
    return tuple(part for part in section.split(".") if part)


class EntityCandidateGenerator:
    """按文档、作用域、类型、名称和图结构召回候选实体对。"""

    def __init__(
        self,
        rules: AlignmentRules,
        min_name_similarity: float = 0.58,
        min_embedding_similarity: float = 0.84,
    ) -> None:
        """保存候选召回阈值和作用域规则。"""

        self.rules = rules
        self.min_name_similarity = min_name_similarity
        self.min_embedding_similarity = min_embedding_similarity

    @staticmethod
    def _context_score(left: EntityMention, right: EntityMention) -> float:
        """根据同 Chunk、章节层级和交叉文本提及计算上下文相似度。"""

        if left.chunk_id == right.chunk_id:
            return 1.0
        left_parts = _section_parts(left.chunk_id)
        right_parts = _section_parts(right.chunk_id)
        common = 0
        for left_part, right_part in zip(left_parts, right_parts):
            if left_part != right_part:
                break
            common += 1
        section_score = min(0.75, 0.25 + 0.2 * common) if common else 0.25
        left_in_right = bool(left.normalized_name and left.normalized_name in right.context_text)
        right_in_left = bool(right.normalized_name and right.normalized_name in left.context_text)
        if left_in_right and right_in_left:
            return max(section_score, 0.9)
        if left_in_right or right_in_left:
            return max(section_score, 0.75)
        return section_score

    @staticmethod
    def _neighbor_score(left: EntityMention, right: EntityMention) -> float:
        """比较两个实体原始邻居的名称和类型签名。"""

        left_values = {
            f"{item.neighbor_name}|{item.neighbor_type_zh}"
            for item in left.neighbors
            if item.neighbor_name or item.neighbor_type_zh
        }
        right_values = {
            f"{item.neighbor_name}|{item.neighbor_type_zh}"
            for item in right.neighbors
            if item.neighbor_name or item.neighbor_type_zh
        }
        return _jaccard(left_values, right_values)

    @staticmethod
    def _embedding_score(
        left: SourceKey,
        right: SourceKey,
        embeddings: Mapping[SourceKey, Sequence[float]] | None,
    ) -> tuple[float, bool]:
        """读取可选实体向量并计算余弦相似度。"""

        if not embeddings or left not in embeddings or right not in embeddings:
            return 0.5, False
        return EmbeddingClient.cosine_similarity(embeddings[left], embeddings[right]), True

    def generate(
        self,
        mentions: Sequence[EntityMention],
        embeddings: Mapping[SourceKey, Sequence[float]] | None = None,
    ) -> list[CandidatePair]:
        """在同一文档内生成可解释且稳定排序的候选实体对。"""

        by_document: dict[str, list[EntityMention]] = defaultdict(list)
        for mention in mentions:
            by_document[mention.document_id].append(mention)

        candidates: list[CandidatePair] = []
        for document_id in sorted(by_document):
            document_mentions = sorted(by_document[document_id], key=lambda item: item.key)
            for left, right in combinations(document_mentions, 2):
                if left.scope != right.scope:
                    continue
                if left.scope == "chunk_local" and left.chunk_id != right.chunk_id:
                    continue

                left_names = sorted(left.all_normalized_names)
                right_names = sorted(right.all_normalized_names)
                name_score = name_similarity(left_names, right_names)
                exact_name = float(
                    bool(left.normalized_name)
                    and left.normalized_name == right.normalized_name
                )
                alias_name = float(bool(set(left_names) & set(right_names)))
                type_relation = self.rules.type_compatibility(
                    left.schema_type_zh,
                    right.schema_type_zh,
                    left.schema_type,
                    right.schema_type,
                    left.raw_type,
                    right.raw_type,
                )
                type_score = {
                    "exact": 1.0,
                    "compatible": 0.75,
                    "unknown": 0.35,
                    "incompatible": 0.0,
                }[type_relation]
                neighbor_score = self._neighbor_score(left, right)
                context_score = self._context_score(left, right)
                if type_relation in {"exact", "compatible"}:
                    embedding_score, embedding_available = self._embedding_score(
                        left.key, right.key, embeddings
                    )
                else:
                    embedding_score, embedding_available = 0.5, False

                should_recall = bool(exact_name or alias_name)
                should_recall = should_recall or (
                    type_relation in {"exact", "compatible"}
                    and name_score >= self.min_name_similarity
                )
                should_recall = should_recall or (
                    type_relation == "unknown" and name_score >= 0.82
                )
                should_recall = should_recall or (
                    type_relation in {"exact", "compatible"}
                    and neighbor_score >= 0.5
                    and name_score >= 0.45
                )
                should_recall = should_recall or (
                    embedding_available
                    and embedding_score >= self.min_embedding_similarity
                    and type_relation in {"exact", "compatible"}
                )
                if not should_recall:
                    continue

                reasons: list[str] = []
                if exact_name:
                    reasons.append("normalized_name_exact")
                elif alias_name:
                    reasons.append("alias_intersection")
                if type_relation == "exact":
                    reasons.append("schema_type_zh_exact")
                elif type_relation == "compatible":
                    reasons.append("schema_type_zh_compatible")
                elif type_relation == "incompatible":
                    reasons.append("schema_type_zh_incompatible")
                if neighbor_score > 0.5:
                    reasons.append("shared_neighbors")
                if embedding_available and embedding_score >= self.min_embedding_similarity:
                    reasons.append("embedding_recall")

                candidates.append(
                    CandidatePair(
                        left=left.key,
                        right=right.key,
                        signals={
                            "name": name_score,
                            "exact_name": exact_name,
                            "alias_name": alias_name,
                            "type": type_score,
                            "context": context_score,
                            "neighbor": neighbor_score,
                            "embedding": embedding_score,
                            "embedding_available": float(embedding_available),
                        },
                        reasons=tuple(reasons),
                    )
                )
        return sorted(candidates, key=lambda item: item.pair_id)

    def generate_graph_refinement(
        self,
        mentions: Sequence[EntityMention],
        cluster_by_source: Mapping[SourceKey, str],
        existing_pair_ids: set[str],
    ) -> list[CandidatePair]:
        """利用首轮融合后的共同全局邻居补召回结构相似实体。"""

        neighbor_clusters: dict[SourceKey, set[str]] = {}
        for mention in mentions:
            neighbor_clusters[mention.key] = {
                cluster_by_source[item.neighbor_key]
                for item in mention.neighbors
                if item.neighbor_key in cluster_by_source
                and cluster_by_source[item.neighbor_key] != cluster_by_source.get(mention.key)
            }

        result: list[CandidatePair] = []
        by_document: dict[str, list[EntityMention]] = defaultdict(list)
        for mention in mentions:
            by_document[mention.document_id].append(mention)
        for document_mentions in by_document.values():
            for left, right in combinations(sorted(document_mentions, key=lambda item: item.key), 2):
                if left.scope != right.scope:
                    continue
                if left.scope == "chunk_local" and left.chunk_id != right.chunk_id:
                    continue
                if cluster_by_source.get(left.key) == cluster_by_source.get(right.key):
                    continue
                left_neighbor_clusters = neighbor_clusters.get(left.key, set())
                right_neighbor_clusters = neighbor_clusters.get(right.key, set())
                if not left_neighbor_clusters or not right_neighbor_clusters:
                    continue
                names_score = name_similarity(
                    sorted(left.all_normalized_names), sorted(right.all_normalized_names)
                )
                graph_score = _jaccard(
                    left_neighbor_clusters,
                    right_neighbor_clusters,
                )
                type_relation = self.rules.type_compatibility(
                    left.schema_type_zh,
                    right.schema_type_zh,
                    left.schema_type,
                    right.schema_type,
                    left.raw_type,
                    right.raw_type,
                )
                if (
                    graph_score < 0.5
                    or names_score < 0.45
                    or type_relation not in {"exact", "compatible"}
                ):
                    continue
                candidate = CandidatePair(
                    left=left.key,
                    right=right.key,
                    signals={
                        "name": names_score,
                        "exact_name": float(left.normalized_name == right.normalized_name),
                        "alias_name": float(
                            bool(left.all_normalized_names & right.all_normalized_names)
                        ),
                        "type": 1.0 if type_relation == "exact" else 0.75,
                        "context": self._context_score(left, right),
                        "neighbor": graph_score,
                        "embedding": 0.5,
                        "embedding_available": 0.0,
                    },
                    reasons=("resolved_neighbor_overlap",),
                )
                if candidate.pair_id not in existing_pair_ids:
                    result.append(candidate)
        return sorted(result, key=lambda item: item.pair_id)


__all__ = ["EntityCandidateGenerator"]
