"""数据模型统一导出入口。

外部模块优先从 model 包导入类型，避免绑定到具体文件路径，后续重构模型文件
也不会影响调用方。
"""
from .alignment import AlignedEntity, AlignedRelation, AlignmentCluster, AlignmentResult
from .base import ID, MMKGBaseModel, Provenance, SourceModality, Timestamp
from .chunk import Chunk, ChunkBase, ChunkList, FormulaChunk, ImageChunk, TableChunk, TextChunk
from .document import Document, DocumentMetadata, DocumentSection
from .entity import Entity
from .event import Event, EventType
from .extraction import (
    ExtractionResult,
    FormulaExtractionResult,
    ImageExtractionResult,
    TableExtractionResult,
    TextExtractionResult,
)
from .graph import GraphEdge, GraphMetadata, GraphNode, KnowledgeGraph
from .relation import Relation
from .skeleton import (
    DocumentSkeleton,
    SkeletonEdge,
    SkeletonEdgeType,
    SkeletonNode,
    SkeletonNodeType,
)

__all__ = [
    "ID",
    "MMKGBaseModel",
    "Provenance",
    "SourceModality",
    "Timestamp",
    "DocumentMetadata",
    "DocumentSection",
    "Document",
    "ChunkBase",
    "TextChunk",
    "TableChunk",
    "ImageChunk",
    "FormulaChunk",
    "Chunk",
    "ChunkList",
    "Entity",
    "Relation",
    "Event",
    "EventType",
    "ExtractionResult",
    "TextExtractionResult",
    "TableExtractionResult",
    "ImageExtractionResult",
    "FormulaExtractionResult",
    "AlignedEntity",
    "AlignmentCluster",
    "AlignedRelation",
    "AlignmentResult",
    "SkeletonNodeType",
    "SkeletonEdgeType",
    "SkeletonNode",
    "SkeletonEdge",
    "DocumentSkeleton",
    "GraphNode",
    "GraphEdge",
    "GraphMetadata",
    "KnowledgeGraph",
]
