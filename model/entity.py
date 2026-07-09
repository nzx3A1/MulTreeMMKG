"""实体模型。

Entity 表示从文本、表格、图片或公式中抽取出的领域对象，字段设计与后续
schema/entity_schema.json 校验保持松耦合，允许 schema 阶段继续扩展类型清单。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import ID, MMKGBaseModel, Provenance


class Entity(MMKGBaseModel):
    """知识图谱实体。"""

    id: ID                             # 实体唯一标识
    name: str                          # 实体名称
    official_name: Optional[str] = None  # 实体官方名称或标准名称
    type: str                          # 实体类型（来自 Schema 白名单）
    type_zh: Optional[str] = None       # 实体类型中文名称
    aliases: List[str] = Field(default_factory=list)  # 别名列表
    description: Optional[str] = None  # 实体描述
    attributes: Dict[str, Any] = Field(default_factory=dict)  # 属性键值对
    provenance: List[Provenance] = Field(default_factory=list)  # 来源证据列表
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)  # 抽取置信度
    normalized_id: Optional[ID] = None # 对齐后的规范化 ID
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据
