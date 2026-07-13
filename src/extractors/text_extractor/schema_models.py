"""文本 Schema 选择阶段使用的数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SchemaConcept:
    """描述一个概念节点及其在当前选择任务中的各项相关性分数。"""

    schema: str
    zh_name: str = ""
    category: str = ""
    description: str = ""
    examples: tuple[str, ...] = ()
    embedding: tuple[float, ...] = field(default=(), repr=False, compare=False)
    vector_score: float = 0.0
    lexical_score: float = 0.0
    context_score: float = 0.0
    schema_key_score: float = 0.0
    document_score: float = 0.0
    final_score: float = 0.0
    selection_reasons: tuple[str, ...] = ()

    @classmethod
    def from_record(cls, record: Mapping[str, Any], **scores: Any) -> "SchemaConcept":
        """把 Neo4j 查询记录转换为稳定、不可变的概念对象。"""

        examples = record.get("examples") or ()
        embedding = record.get("embedding") or ()
        return cls(
            schema=str(record.get("schema") or ""),
            zh_name=str(record.get("zh_name") or record.get("zhName") or ""),
            category=str(record.get("category") or ""),
            description=str(record.get("description") or ""),
            examples=tuple(str(item) for item in examples),
            embedding=tuple(float(item) for item in embedding),
            **scores,
        )

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """输出可序列化字典，默认省略体积较大的向量字段。"""

        result = {
            "schema": self.schema,
            "zhName": self.zh_name,
            "category": self.category,
            "description": self.description,
            "examples": list(self.examples),
            "vector_score": self.vector_score,
            "lexical_score": self.lexical_score,
            "context_score": self.context_score,
            "schema_key_score": self.schema_key_score,
            "document_score": self.document_score,
            "final_score": self.final_score,
            "selection_reasons": list(self.selection_reasons),
        }
        if include_embedding:
            result["embedding"] = list(self.embedding)
        return result


@dataclass(frozen=True)
class SchemaRelation:
    """描述 Schema 图中一条有方向的合法概念关系。"""

    source_schema: str
    relation_en: str
    relation_zh: str
    target_schema: str
    edge_score: float = 0.0

    @property
    def key(self) -> tuple[str, str, str]:
        """返回可用于去重和合法性判断的关系三元组键。"""

        return self.source_schema, self.relation_en, self.target_schema

    def to_dict(self) -> dict[str, Any]:
        """输出与设计文档字段命名一致的可序列化字典。"""

        return {
            "source_schema": self.source_schema,
            "relationEn": self.relation_en,
            "relationZh": self.relation_zh,
            "target_schema": self.target_schema,
            "edge_score": self.edge_score,
        }


@dataclass(frozen=True)
class RelevantSchema:
    """表示文档级候选池或单个 Chunk 的局部 Schema 子图。"""

    concepts: tuple[SchemaConcept, ...] = ()
    relations: tuple[SchemaRelation, ...] = ()
    core_categories: tuple[str, ...] = ()
    query_terms: tuple[str, ...] = ()
    selection_confidence: float = 0.0
    fallback_used: bool = False
    selector_version: str = "hybrid_graph_v1"

    @property
    def concept_map(self) -> dict[str, SchemaConcept]:
        """按英文 Schema 名建立只读式查找映射。"""

        return {concept.schema: concept for concept in self.concepts}

    def to_dict(self) -> dict[str, Any]:
        """输出可直接写入日志或 Chunk Graph 元数据的选择结果。"""

        return {
            "selector_version": self.selector_version,
            "concepts": [concept.to_dict() for concept in self.concepts],
            "relations": [relation.to_dict() for relation in self.relations],
            "core_categories": list(self.core_categories),
            "query_terms": list(self.query_terms),
            "selection_confidence": self.selection_confidence,
            "fallback_used": self.fallback_used,
        }


@dataclass(frozen=True)
class DocumentContext:
    """保存整篇论文的有序文本 Chunk、章节索引和全局主题画像。"""

    document_id: str
    chunks: tuple[Mapping[str, Any], ...]
    chunk_indexes: Mapping[str, int]
    section_chunks: Mapping[str, tuple[str, ...]]
    section_titles: tuple[str, ...]
    document_schema_keys: tuple[str, ...]
    section_schema_keys: Mapping[str, tuple[str, ...]]
    domain_terms: tuple[str, ...]
    representative_chunk_ids: tuple[str, ...]
    topic_profile: str

    def local_context(self, chunk_id: str, *, neighbor_chars: int = 180) -> dict[str, str]:
        """返回当前 Chunk 的章节信息及同章节前后相邻文本片段。"""

        index = self.chunk_indexes[chunk_id]
        current = self.chunks[index]
        section_id = str(current.get("section_id") or "")
        same_section_ids = self.section_chunks.get(section_id, ())
        position = same_section_ids.index(chunk_id)
        previous_text = ""
        next_text = ""
        if position > 0:
            previous = self.chunks[self.chunk_indexes[same_section_ids[position - 1]]]
            previous_text = str(previous.get("text") or "")[-neighbor_chars:]
        if position + 1 < len(same_section_ids):
            following = self.chunks[self.chunk_indexes[same_section_ids[position + 1]]]
            next_text = str(following.get("text") or "")[:neighbor_chars]
        return {
            "section_title": str(current.get("section_title") or ""),
            "section_summary": str(current.get("section_summary") or ""),
            "schema_keys": "、".join(self.section_schema_keys.get(section_id, ())),
            "previous_tail": previous_text,
            "next_head": next_text,
            "document_topic_profile": self.topic_profile,
        }


@dataclass(frozen=True)
class DocumentSchemaContext:
    """聚合整篇论文上下文、文档级 Schema 池和逐 Chunk 局部选择结果。"""

    document: DocumentContext
    document_schema_pool: RelevantSchema
    chunk_schemas: Mapping[str, RelevantSchema]
