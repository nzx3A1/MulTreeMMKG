"""分模态 Chunk 模型。

Chunk 是摘要、抽取、对齐和图谱融合的最小处理单元。不同模态的内容结构不同，
因此这里按文本、表格、图片和公式分别建模，而不是使用一个泛化的单一 Chunk。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypeAlias, Union

from pydantic import Field

from .base import ID, MMKGBaseModel, Provenance, SourceModality


class ChunkBase(MMKGBaseModel):
    """所有 Chunk 类型共享的基础字段。"""

    id: ID
    document_id: ID
    section_id: Optional[ID] = None
    content_id: Optional[ID] = None
    order: int = Field(default=0, ge=0)
    summary: Optional[str] = None
    provenance: Optional[Provenance] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextChunk(ChunkBase):
    """正文文本 Chunk。"""

    modality: SourceModality = SourceModality.TEXT
    text: str
    token_count: Optional[int] = Field(default=None, ge=0)


class TableChunk(ChunkBase):
    """表格 Chunk，保留 markdown 表格和可选结构化单元格。"""

    modality: SourceModality = SourceModality.TABLE
    markdown: str
    caption: Optional[str] = None
    cells: List[List[str]] = Field(default_factory=list)


class ImageChunk(ChunkBase):
    """图片 Chunk，保存图片资源引用和图注。"""

    modality: SourceModality = SourceModality.IMAGE
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    caption: Optional[str] = None
    ocr_text: Optional[str] = None


class FormulaChunk(ChunkBase):
    """公式 Chunk，保存 LaTeX 公式及上下文说明。"""

    modality: SourceModality = SourceModality.FORMULA
    latex: str
    caption: Optional[str] = None
    context: Optional[str] = None


Chunk: TypeAlias = Union[TextChunk, TableChunk, ImageChunk, FormulaChunk]


class ChunkList(MMKGBaseModel):
    """某篇论文的分模态 Chunk 集合。"""

    document_id: ID
    chunks: List[Chunk] = Field(default_factory=list)
