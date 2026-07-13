"""文本模态 Schema 选择与后续抽取器的公开入口。"""
from __future__ import annotations

from typing import Any

from .document_context import build_document_context, extract_domain_terms
from .schema_models import (
    DocumentContext,
    DocumentSchemaContext,
    RelevantSchema,
    SchemaConcept,
    SchemaRelation,
)
from .schema_repository import Neo4jSchemaRepository
from .schema_selector import SchemaSelector, SchemaSelectorConfig, build_chunk_query_text
from .pipeline import extract_text_chunks_to_file
from .text_extractor import extract_from_text, extract_text_chunk_graph

__all__ = [
    "DocumentContext",
    "DocumentSchemaContext",
    "Neo4jSchemaRepository",
    "RelevantSchema",
    "SchemaConcept",
    "SchemaRelation",
    "SchemaSelector",
    "SchemaSelectorConfig",
    "build_chunk_query_text",
    "build_document_context",
    "extract_domain_terms",
    "extract_from_text",
    "extract_text_chunks_to_file",
    "extract_text_chunk_graph",
]
