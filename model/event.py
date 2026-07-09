"""事件模型。

Event 表示成藏、运移、沉积、实验等具有时间或过程含义的知识单元。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import ID, MMKGBaseModel, Provenance


class EventType(str, Enum):
    """常见事件类别，schema 阶段可继续细化。"""

    GEOLOGICAL_PROCESS = "geological_process"  # 地质过程（沉积、成岩等）
    EXPERIMENT = "experiment"                  # 实验事件
    OBSERVATION = "observation"                # 观测事件
    CHARGING = "charging"                      # 充注事件
    MIGRATION = "migration"                    # 运移事件
    ACCUMULATION = "accumulation"              # 聚集事件
    OTHER = "other"                            # 其他事件


class Event(MMKGBaseModel):
    """知识图谱事件。"""

    id: ID                             # 事件唯一标识
    type: EventType | str              # 事件类型
    name: str                          # 事件名称
    participants: List[ID] = Field(default_factory=list)  # 参与实体 ID 列表
    time: Optional[str] = None         # 时间信息
    location: Optional[str] = None     # 位置信息
    attributes: Dict[str, Any] = Field(default_factory=dict)  # 属性键值对
    provenance: List[Provenance] = Field(default_factory=list)  # 来源证据列表
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)  # 抽取置信度
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据