"""Immutable data models for the shared concept-schema layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SchemaConcept:
    """A concept node and its relevance scores for a selection task."""

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
        """Convert a Neo4j record into a stable concept object."""

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
        """Return a serializable representation."""

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
    """A directed relationship that is valid in the concept schema."""

    source_schema: str
    relation_en: str
    relation_zh: str
    target_schema: str
    edge_score: float = 0.0

    @property
    def key(self) -> tuple[str, str, str]:
        return self.source_schema, self.relation_en, self.target_schema

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_schema": self.source_schema,
            "relationEn": self.relation_en,
            "relationZh": self.relation_zh,
            "target_schema": self.target_schema,
            "edge_score": self.edge_score,
        }


@dataclass(frozen=True)
class RelevantSchema:
    """A local concept-schema subgraph selected for one extraction task."""

    concepts: tuple[SchemaConcept, ...] = ()
    relations: tuple[SchemaRelation, ...] = ()
    core_categories: tuple[str, ...] = ()
    query_terms: tuple[str, ...] = ()
    selection_confidence: float = 0.0
    fallback_used: bool = False
    selector_version: str = "hybrid_graph_v1"

    @property
    def concept_map(self) -> dict[str, SchemaConcept]:
        return {concept.schema: concept for concept in self.concepts}

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_version": self.selector_version,
            "concepts": [concept.to_dict() for concept in self.concepts],
            "relations": [relation.to_dict() for relation in self.relations],
            "core_categories": list(self.core_categories),
            "query_terms": list(self.query_terms),
            "selection_confidence": self.selection_confidence,
            "fallback_used": self.fallback_used,
        }


__all__ = ["RelevantSchema", "SchemaConcept", "SchemaRelation"]
