"""文档结构模型。

这些模型承载 MinerU 解析和 Markdown 章节切分后的论文层级结构：
Paper -> Section -> Block。后续骨架图、Chunk 构建和抽取模块都基于该结构。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import ID, MMKGBaseModel
from .content import ContentBlock


class BlockType(str, Enum):
    """文档块类型。"""

    TEXT = "text"           # 文本块
    TABLE = "table"         # 表格块
    IMAGE = "image"         # 图片块
    FORMULA = "formula"     # 公式块
    CAPTION = "caption"     # 说明文字块


class PaperMeta(MMKGBaseModel):
    """论文元信息。"""

    title: str = ""                    # 论文标题
    authors: List[str] = Field(default_factory=list)  # 作者列表
    organizations: List[str] = Field(default_factory=list)  # 机构列表
    year: Optional[int] = None         # 发表年份
    doi: Optional[str] = None          # DOI 编号
    keywords: List[str] = Field(default_factory=list)  # 关键词列表
    abstract: Optional[str] = None     # 摘要
    source_path: Optional[str] = None  # 原始 PDF 路径
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


class Block(MMKGBaseModel):
    """章节中的原始内容块引用或内联内容。"""

    id: ID                            # 块唯一标识
    type: BlockType                   # 块类型
    order: int = Field(default=0, ge=0)  # 在章节内的顺序号
    content: Optional[str] = None     # 内联内容（短文本）
    content_id: Optional[ID] = None   # 引用的 ContentBlock ID
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


class Section(MMKGBaseModel):
    """论文章节或子章节节点。"""

    id: ID                            # 章节唯一标识
    title: str                        # 章节标题
    level: int = Field(default=1, ge=1)  # 层级（1 为一级章节）
    order: int = Field(default=0, ge=0)  # 在同级中的顺序号
    parent_id: Optional[ID] = None    # 父章节 ID
    number: Optional[str] = None      # 章节编号（如 "1.2"）
    text: str = ""                    # 章节正文合并文本
    blocks: List[Block] = Field(default_factory=list)  # 内容块列表
    children: List["Section"] = Field(default_factory=list)  # 子章节列表
    summary: Optional[str] = None     # 章节摘要
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


class SectionTree(MMKGBaseModel):
    """论文的章节树。"""

    document_id: ID                   # 所属文档 ID
    sections: List[Section] = Field(default_factory=list)  # 根章节列表


class Paper(MMKGBaseModel):
    """归一化后的论文对象。"""

    id: ID                            # 论文唯一标识
    meta: PaperMeta = Field(default_factory=PaperMeta)  # 元信息
    sections: List[Section] = Field(default_factory=list)  # 章节树
    contents: List[ContentBlock] = Field(default_factory=list)  # 内容块列表
    raw_markdown: Optional[str] = None  # 原始 Markdown 文本
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


Section.update_forward_refs()
