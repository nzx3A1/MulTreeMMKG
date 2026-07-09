"""抽取结果模型。

四路抽取器共享同一结果结构：实体、关系、事件和原始响应。不同模态的结果类
主要用于标识来源阶段和输出文件类型。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import ID, MMKGBaseModel, SourceModality
from .entity import Entity
from .event import Event
from .relation import Relation


class ExtractionResult(MMKGBaseModel):
    """单次抽取的通用结果。"""

    document_id: ID                    # 所属文档 ID
    chunk_id: Optional[ID] = None      # 来源 Chunk ID
    modality: SourceModality           # 抽取模态类型
    entities: List[Entity] = Field(default_factory=list)  # 抽取出的实体列表
    relations: List[Relation] = Field(default_factory=list)  # 抽取出的关系列表
    events: List[Event] = Field(default_factory=list)  # 抽取出的事件列表
    raw_response: Optional[Any] = None  # LLM/VLM 原始响应
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


class TextExtractionResult(ExtractionResult):
    """文本抽取结果。"""

    modality: SourceModality = SourceModality.TEXT  # 固定为文本模态


class TableExtractionResult(ExtractionResult):
    """表格抽取结果。"""

    modality: SourceModality = SourceModality.TABLE  # 固定为表格模态


class ImageExtractionResult(ExtractionResult):
    """图片抽取结果。"""

    modality: SourceModality = SourceModality.IMAGE  # 固定为图像模态


class FormulaExtractionResult(ExtractionResult):
    """公式抽取结果。"""

    modality: SourceModality = SourceModality.FORMULA  # 固定为公式模态