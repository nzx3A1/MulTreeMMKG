"""流水线运行元信息模型。

PipelineStage 和 RunMetadata 用于记录每个阶段的状态、输入输出和错误信息，
为后续实现断点续跑与阶段产物追踪提供数据结构。
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import MMKGBaseModel


class StageStatus(str, Enum):
    """流水线阶段状态。"""

    PENDING = "pending"   # 待执行
    RUNNING = "running"   # 执行中
    SUCCESS = "success"   # 成功
    FAILED = "failed"     # 失败
    SKIPPED = "skipped"   # 跳过


class StageOutput(MMKGBaseModel):
    """单个阶段的输出文件描述。"""

    path: str                          # 输出文件路径
    exists: bool = False               # 文件是否已存在
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


class PipelineStage(MMKGBaseModel):
    """流水线阶段定义与运行状态。"""

    number: int = Field(ge=1)          # 阶段编号（从 1 开始）
    name: str                          # 阶段名称
    status: StageStatus = StageStatus.PENDING  # 当前状态
    input_paths: List[str] = Field(default_factory=list)  # 输入文件路径列表
    outputs: List[StageOutput] = Field(default_factory=list)  # 输出文件描述列表
    started_at: Optional[datetime] = None  # 开始时间
    finished_at: Optional[datetime] = None  # 结束时间
    error: Optional[str] = None        # 错误信息（失败时）


class RunMetadata(MMKGBaseModel):
    """一次流水线运行的整体元信息。"""

    run_id: str                        # 运行唯一标识
    input_path: str                    # 输入文件路径
    stages: List[PipelineStage] = Field(default_factory=list)  # 阶段列表
    started_at: datetime = Field(default_factory=datetime.utcnow)  # 开始时间
    finished_at: Optional[datetime] = None  # 结束时间
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据