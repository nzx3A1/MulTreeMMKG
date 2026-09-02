"""Schema 对齐使用的轻量数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SchemaConcept:
    """概念图谱中的一个 ``EntityConcept``。"""

    schema: str
    zh_name: str = ""
    category: str = ""
    description: str = ""
    examples: tuple[str, ...] = ()
    embedding: tuple[float, ...] = ()
    embedding_text: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SchemaConcept":
        examples = value.get("examples") or ()
        if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
            examples = (examples,)
        embedding = value.get("embedding") or ()
        return cls(
            schema=str(value.get("schema") or "").strip(),
            zh_name=str(value.get("zhName") or value.get("zh_name") or "").strip(),
            category=str(value.get("category") or "").strip(),
            description=str(value.get("description") or "").strip(),
            examples=tuple(str(item) for item in examples if item is not None),
            embedding=tuple(float(item) for item in embedding),
            embedding_text=str(value.get("embeddingText") or value.get("embedding_text") or ""),
        )


@dataclass(frozen=True)
class SchemaRelation:
    """两个概念类型之间允许的一条 ``SCHEMA_RELATION``。"""

    source_schema: str
    relation_en: str
    relation_zh: str
    target_schema: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SchemaRelation":
        return cls(
            source_schema=str(value.get("source") or value.get("source_schema") or "").strip(),
            relation_en=str(value.get("relationEn") or value.get("relation_en") or "").strip(),
            relation_zh=str(value.get("relationZh") or value.get("relation_zh") or "").strip(),
            target_schema=str(value.get("target") or value.get("target_schema") or "").strip(),
        )


@dataclass
class SchemaSnapshot:
    """一次运行所使用的不可变语义快照及查询索引。"""

    concepts: list[SchemaConcept]
    relations: list[SchemaRelation]
    source: str = ""
    concepts_by_schema: dict[str, SchemaConcept] = field(init=False)
    relations_by_endpoints: dict[tuple[str, str], list[SchemaRelation]] = field(init=False)

    def __post_init__(self) -> None:
        self.concepts_by_schema = {}
        for concept in self.concepts:
            if not concept.schema:
                raise ValueError("Schema Concept 缺少 schema")
            if concept.schema in self.concepts_by_schema:
                raise ValueError(f"Schema Concept 重复：{concept.schema}")
            self.concepts_by_schema[concept.schema] = concept

        self.relations_by_endpoints = {}
        seen: set[tuple[str, str, str]] = set()
        for relation in self.relations:
            key = (relation.source_schema, relation.relation_en, relation.target_schema)
            if not all(key):
                raise ValueError(f"Schema Relation 字段不完整：{relation}")
            if key in seen:
                continue
            seen.add(key)
            self.relations_by_endpoints.setdefault(
                (relation.source_schema, relation.target_schema), []
            ).append(relation)

        for candidates in self.relations_by_endpoints.values():
            candidates.sort(key=lambda item: (item.relation_en, item.relation_zh))

    def relation_candidates(self, source_schema: str, target_schema: str) -> list[SchemaRelation]:
        """严格按有向端点类型查询关系候选，不自行反转方向。"""

        return list(self.relations_by_endpoints.get((source_schema, target_schema), ()))


@dataclass(frozen=True)
class SemanticCandidate:
    """向量召回的一个候选概念。"""

    concept: SchemaConcept
    similarity: float


@dataclass(frozen=True)
class SelectionDecision:
    """LLM 在候选集合（含 NONE）中的选择。"""

    selected: str | None
    confidence: float
    reason: str


def clamp_confidence(value: Any) -> float:
    """把外部模型给出的置信度安全归一到 ``[0, 1]``。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
