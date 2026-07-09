"""文档骨架图模型。

骨架图只描述论文结构节点和结构关系，不混入领域实体抽取结果。这样后续阶段
可以把抽取出的知识通过 SOURCE_FROM 等边回连到章节和 Chunk。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import Field

from .base import ID, MMKGBaseModel


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
    SOURCE_FROM = "SOURCE_FROM"   # 来源关系


class SkeletonNode(MMKGBaseModel):
    """文档骨架节点。"""

    id: ID                             # 节点唯一标识
    type: SkeletonNodeType | str       # 节点类型
    title: str = ""                    # 节点标题/名称
    properties: Dict[str, Any] = Field(default_factory=dict)  # 属性键值对


class SkeletonEdge(MMKGBaseModel):
    """文档骨架关系。"""

    id: ID                             # 边唯一标识
    type: SkeletonEdgeType | str       # 关系类型
    source_id: ID                      # 源节点 ID
    target_id: ID                      # 目标节点 ID
    properties: Dict[str, Any] = Field(default_factory=dict)  # 属性键值对


class DocumentSkeleton(MMKGBaseModel):
    """某篇论文的完整文档骨架图。"""

    document_id: ID                   # 所属文档 ID
    nodes: List[SkeletonNode] = Field(default_factory=list)  # 骨架节点列表
    edges: List[SkeletonEdge] = Field(default_factory=list)  # 骨架边列表
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据