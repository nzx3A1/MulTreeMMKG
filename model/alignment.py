"""跨模态对齐结果模型。

对齐阶段会把不同模态或不同章节中指向同一真实对象的实体合并到 cluster，
并记录关系归一化后的结果。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import ID, MMKGBaseModel


class AlignedEntity(MMKGBaseModel):
    """单个被对齐实体的引用。"""

    entity_id: ID                      # 原始实体 ID
    name: str                          # 实体名称
    type: str                          # 实体类型
    source_modality: Optional[str] = None  # 来源模态
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)  # 对齐置信度


class AlignmentCluster(MMKGBaseModel):
    """实体对齐后的聚类。"""

    id: ID                             # 聚类唯一标识
    canonical_name: str                # 规范化名称（代表该聚类）
    type: str                          # 聚类的实体类型
    members: List[AlignedEntity] = Field(default_factory=list)  # 成员实体列表
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)  # 聚类置信度
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


class AlignedRelation(MMKGBaseModel):
    """关系对齐后的规范化边。"""

    id: ID                             # 关系唯一标识
    type: str                          # 关系类型
    source_cluster_id: ID              # 源聚类 ID
    target_cluster_id: ID              # 目标聚类 ID
    source_relation_ids: List[ID] = Field(default_factory=list)  # 原始关系 ID 列表
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)  # 关系置信度
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


class AlignmentResult(MMKGBaseModel):
    """阶段十输出的对齐和去重结果。"""

    document_id: Optional[ID] = None   # 所属文档 ID
    clusters: List[AlignmentCluster] = Field(default_factory=list)  # 实体聚类列表
    relations: List[AlignedRelation] = Field(default_factory=list)  # 对齐关系列表
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据