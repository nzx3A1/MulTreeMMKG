"""第二阶段的文档目录树数据模型。

本模块只定义内存中的文档、章节和内容块层级，不负责读取 JSON 或构建节点；
JSON 适配和对象构建逻辑位于 :mod:`src.skeleton.document_tree_builder`。
"""
from __future__ import annotations

from enum import Enum
from typing import Iterator, List, Optional

from pydantic import Field, PrivateAttr

from .base import ID, MMKGBaseModel
from .chunk import Chunk

# 定义文档骨架节点类型和关系类型，用于图谱写入时定义节点标签和边类型。
class SkeletonNodeType(str, Enum):
    """文档骨架节点类型。"""

    PAPER = "Paper"           # 论文文档节点
    SECTION = "Section"       # 一级章节节点
    SUBSECTION = "SubSection" # 二级或三级章节节点
    CHUNK = "Chunk"           # 最小内容块节点
    FIGURE = "Figure"         # 图片节点
    TABLE = "Table"           # 表格节点
    FORMULA = "Formula"       # 公式节点


class SkeletonEdgeType(str, Enum):
    """文档骨架关系类型。"""

    HAS_SECTION = "HAS_SECTION"   # 文档包含章节
    HAS_CHUNK = "HAS_CHUNK"       # 章节包含 Chunk
    NEXT = "NEXT"                 # 顺序关系
    SOURCE_FROM = "SOURCE_FROM"   # 实体来源 Chunk


class DocumentMetadata(MMKGBaseModel):
    """保存第一阶段抽取的、与目录树相关的文档元数据。"""

    abstract: Optional[str] = None  # 文档摘要
    authors: List[str] = Field(default_factory=list)  # 作者列表
    keywords: List[str] = Field(default_factory=list)  # 关键词列表
    publish_organization: Optional[str] = None  # 作者单位或出版机构
    references: Optional[str] = None  # 原始参考文献文本


class DocumentSection(MMKGBaseModel):
    """文档中的一个章节节点，可递归容纳子章节和各模态 Chunk。"""

    id: ID  # 在所属文档内唯一的章节标识
    title: str  # 章节标题
    level: int = Field(ge=1)  # 标题层级，数值越大层级越深
    order: int = Field(ge=0)  # 在同级章节中的顺序
    parent_id: Optional[ID] = None  # 父章节标识，一级章节为 None
    children: List["DocumentSection"] = Field(default_factory=list)  # 子章节对象
    chunks: List[Chunk] = Field(default_factory=list)  # 直接归属本章节的内容块

    def iter_sections(self) -> Iterator["DocumentSection"]:
        """按深度优先顺序遍历当前章节及其全部子章节。"""

        yield self
        for child in self.children:
            yield from child.iter_sections()

    def iter_chunks(self) -> Iterator[Chunk]:
        """按目录顺序遍历当前章节及其子章节中的全部 Chunk。"""

        yield from self.chunks
        for child in self.children:
            yield from child.iter_chunks()


class Document(MMKGBaseModel):
    """完整的文档目录树，并提供供后续抽取阶段使用的内存索引。"""

    id: ID  # 文档唯一标识
    title: str  # 文档标题
    sections: List[DocumentSection] = Field(default_factory=list)  # 一级章节列表
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)  # 文档元数据
    source_file: Optional[str] = None  # 第一阶段 JSON 中记录的原始 Markdown 路径

    _section_index: dict[ID, DocumentSection] = PrivateAttr(default_factory=dict)
    _chunk_index: dict[ID, Chunk] = PrivateAttr(default_factory=dict)
    _chunk_section_index: dict[ID, DocumentSection] = PrivateAttr(default_factory=dict)

    def rebuild_indexes(self) -> None:
        """重建章节和 Chunk 索引，同时校验整棵目录树中的 ID 唯一性。"""

        self._section_index.clear()
        self._chunk_index.clear()
        self._chunk_section_index.clear()
        for section in self.iter_sections():
            if section.id in self._section_index:
                raise ValueError(f"文档 {self.id} 存在重复章节 ID: {section.id}")
            self._section_index[section.id] = section
            for chunk in section.chunks:
                if chunk.id in self._chunk_index:
                    raise ValueError(f"文档 {self.id} 存在重复 Chunk ID: {chunk.id}")
                self._chunk_index[chunk.id] = chunk
                self._chunk_section_index[chunk.id] = section

    def iter_sections(self) -> Iterator[DocumentSection]:
        """按深度优先顺序遍历文档中的全部章节。"""

        for section in self.sections:
            yield from section.iter_sections()

    def iter_chunks(self) -> Iterator[Chunk]:
        """按目录顺序遍历文档中的全部内容块。"""

        for section in self.sections:
            yield from section.iter_chunks()

    def get_section(self, section_id: ID) -> Optional[DocumentSection]:
        """按 ID 获取已加载到内存中的章节对象。"""

        return self._section_index.get(section_id)

    def get_chunk(self, chunk_id: ID) -> Optional[Chunk]:
        """按 ID 获取已加载到内存中的内容块对象。"""

        return self._chunk_index.get(chunk_id)

    def get_chunk_section(self, chunk_id: ID) -> Optional[DocumentSection]:
        """按 Chunk ID 获取其直接归属的章节对象，用于保留抽取结果的章节依赖。"""

        return self._chunk_section_index.get(chunk_id)


DocumentSection.update_forward_refs()
