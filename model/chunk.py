"""分模态 Chunk 模型。

Chunk 是摘要、抽取、对齐和图谱融合的最小处理单元。不同模态的内容结构不同，
因此这里按文本、表格、图片和公式分别建模，而不是使用一个泛化的单一 Chunk。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypeAlias, Union

from pydantic import Field

from .base import ID, MMKGBaseModel, SourceModality


class ChunkBase(MMKGBaseModel):
    """所有 Chunk 类型共享的基础字段。"""

    id: ID
    """Chunk 唯一标识符。"""

    order: int = Field(default=0, ge=0)
    """在文档中的顺序编号，从0开始。"""


class TextChunk(ChunkBase):
    """正文文本 Chunk。"""

    modality: SourceModality = SourceModality.TEXT
    """内容模态类型，固定为 TEXT。"""

    text: str
    """文本内容。"""



class TableChunk(ChunkBase):
    """表格 Chunk，保留 markdown 表格和可选结构化单元格。"""

    modality: SourceModality = SourceModality.TABLE
    """内容模态类型，固定为 TABLE。"""

    markdown: str
    """表格的 Markdown 格式内容。"""

    caption: Optional[str] = None
    """表格标题（可选）。"""

    references: Optional[List[str]] = None
    """表格的引用列表，每个引用是一个字符串（可选）。"""


class ImageChunk(ChunkBase):
    """图片 Chunk，保存图片资源引用和图注。"""

    modality: SourceModality = SourceModality.IMAGE
    """内容模态类型，固定为 IMAGE。"""

    image_path: List[str] = Field(default_factory=list)
    """图片的本地文件路径列表。"""

    caption: Optional[str] = None
    """图片标题或图注（可选）。"""

    references: Optional[List[str]] = None
    """图片的引用列表，每个引用是一个字符串（可选）。"""


class FormulaChunk(ChunkBase):
    """公式 Chunk，保存 LaTeX 公式及上下文说明。"""

    modality: SourceModality = SourceModality.FORMULA
    """内容模态类型，固定为 FORMULA。"""

    latex: str
    """公式的 LaTeX 表示。"""

    caption: Optional[str] = None
    """公式标题或编号（可选）。"""



Chunk: TypeAlias = Union[TextChunk, TableChunk, ImageChunk, FormulaChunk]


class ChunkList(MMKGBaseModel):
    """某篇论文的分模态 Chunk 集合。"""

    document_id: ID
    """所属文档的唯一标识符。"""

    chunks: List[Chunk] = Field(default_factory=list)
    """Chunk 列表，包含文档中的所有分模态内容块。"""
