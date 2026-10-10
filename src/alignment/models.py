"""Stage 06 使用的轻量数据模型。

模型只保存对齐期间需要的索引、来源和判定信息；最终 JSON 仍使用普通字典，
避免把内部实现类型泄露给下游阶段。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, order=True)
class SourceKey:
    """唯一标识一个文档、Chunk 中的原始实体 mention。"""

    document_id: str
    chunk_id: str
    entity_id: str

    @property
    def token(self) -> str:
        """返回可稳定排序和持久化的来源键字符串。"""

        return f"{self.document_id}|{self.chunk_id}|{self.entity_id}"

    def to_dict(self) -> dict[str, str]:
        """将来源键转换为 JSON 友好的字典。"""

        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "original_entity_id": self.entity_id,
        }


@dataclass(frozen=True)
class NeighborSignature:
    """记录原始实体的一条邻接关系摘要。"""

    relation: str
    neighbor_key: SourceKey
    neighbor_name: str
    neighbor_type_zh: str


@dataclass
class EntityMention:
    """一个带完整来源上下文的原始实体 mention。"""

    key: SourceKey
    graph_index: int
    entity_index: int
    document_id: str
    chunk_id: str
    modality: str
    scope: str
    name: str
    normalized_name: str
    aliases: tuple[str, ...]
    normalized_aliases: tuple[str, ...]
    schema_type: str
    schema_type_zh: str
    raw_type: str
    attributes: dict[str, Any]
    provenance: Any
    context_text: str
    schema_alignment_status: str
    original: dict[str, Any]
    neighbors: list[NeighborSignature] = field(default_factory=list)

    @property
    def all_normalized_names(self) -> set[str]:
        """返回主名称和别名组成的非空规范化名称集合。"""

        return {
            value
            for value in (self.normalized_name, *self.normalized_aliases)
            if value
        }


@dataclass(frozen=True)
class CandidatePair:
    """保存一个待判定实体对及其可审计特征。"""

    left: SourceKey
    right: SourceKey
    signals: dict[str, float]
    reasons: tuple[str, ...] = ()

    @property
    def pair_id(self) -> str:
        """生成与实体输入顺序无关的稳定候选对标识。"""

        left, right = sorted((self.left.token, self.right.token))
        return f"{left}||{right}"


@dataclass(frozen=True)
class PairDecision:
    """表示一对实体的 SAME、DIFFERENT 或 UNCERTAIN 判定。"""

    left: SourceKey
    right: SourceKey
    decision: str
    confidence: float
    method: str
    reason: str
    score: float
    signals: dict[str, float] = field(default_factory=dict)

    @property
    def pair_id(self) -> str:
        """返回与候选对象相同的稳定实体对标识。"""

        left, right = sorted((self.left.token, self.right.token))
        return f"{left}||{right}"


__all__ = [
    "CandidatePair",
    "EntityMention",
    "NeighborSignature",
    "PairDecision",
    "SourceKey",
]
