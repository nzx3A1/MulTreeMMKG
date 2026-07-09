"""关系模型。

Relation 保存实体之间的有向语义边，并携带证据、置信度和属性信息，供对齐、
关系发现和图谱融合模块复用。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import ID, MMKGBaseModel, Provenance


class Relation(MMKGBaseModel):
    """知识图谱关系。"""

    id: ID                             # 关系唯一标识
    type: str                          # 关系类型（来自 Schema 白名单）
    official_name: Optional[str] = None  # 关系官方名称或标准名称
    type_zh: Optional[str] = None       # 关系类型中文名称
    source_id: ID                      # 源实体 ID
    target_id: ID                      # 目标实体 ID
    label: Optional[str] = None        # 关系中文标签
    attributes: Dict[str, Any] = Field(default_factory=dict)  # 属性键值对
    provenance: List[Provenance] = Field(default_factory=list)  # 来源证据列表
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)  # 抽取置信度
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据
