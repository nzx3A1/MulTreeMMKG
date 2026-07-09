"""统一知识图谱模型。

图谱阶段统一使用 nodes/edges/metadata 三段式结构，既可承载文档骨架，也可承载
实体、关系、事件等增强知识。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import ID, MMKGBaseModel, Provenance


class GraphNode(MMKGBaseModel):
    """图谱节点。"""

    id: ID                             # 节点唯一标识
    labels: List[str] = Field(default_factory=list)  # Neo4j 标签列表
    type: str                          # 节点类型
    name: Optional[str] = None         # 节点名称
    properties: Dict[str, Any] = Field(default_factory=dict)  # 属性键值对
    provenance: List[Provenance] = Field(default_factory=list)  # 来源证据列表


class GraphEdge(MMKGBaseModel):
    """图谱关系边。"""

    id: ID                             # 边唯一标识
    type: str                          # 关系类型
    source_id: ID                      # 源节点 ID
    target_id: ID                      # 目标节点 ID
    properties: Dict[str, Any] = Field(default_factory=dict)  # 属性键值对
    provenance: List[Provenance] = Field(default_factory=list)  # 来源证据列表


class GraphMetadata(MMKGBaseModel):
    """图谱产物元信息。"""

    document_id: Optional[ID] = None   # 所属文档 ID
    stage: Optional[str] = None        # 生成阶段（如 "stage_03"）
    schema_version: Optional[str] = None  # Schema 版本
    created_at: datetime = Field(default_factory=datetime.utcnow)  # 创建时间
    extra: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


class KnowledgeGraph(MMKGBaseModel):
    """完整知识图谱。"""

    nodes: List[GraphNode] = Field(default_factory=list)  # 节点列表
    edges: List[GraphEdge] = Field(default_factory=list)  # 边列表
    metadata: GraphMetadata = Field(default_factory=GraphMetadata)  # 元信息

    def node_ids(self) -> set[ID]:
        """返回当前图谱中所有节点 ID，供校验器检查边引用完整性。"""

        return {node.id for node in self.nodes}