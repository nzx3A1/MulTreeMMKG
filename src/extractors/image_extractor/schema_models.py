"""图片知识抽取架构共享的数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class ImageExtractorKind(str, Enum):
    """六类图片抽取器的稳定标识，供路由、注册和结果溯源共同使用。"""

    MAP_SPATIAL = "map_spatial"
    STRATIGRAPHIC_PROFILE = "stratigraphic_profile"
    ROCK_MICRO = "rock_micro"
    QUANTITATIVE_CHART = "quantitative_chart"
    GEOLOGICAL_PROCESS = "geological_process"
    COMPOSITE_PANEL = "composite_panel"


@dataclass(frozen=True)
class ImageExtractionTask:
    """单张图片的规范化抽取任务；一个 ImageChunk 可展开为多个任务。"""

    document_id: str
    chunk_id: str
    image_id: str
    image_index: int
    image_path: str
    caption: str = ""
    references: tuple[str, ...] = ()
    classification_code: str = ""
    classification_type: str = ""
    section_id: str = ""
    section_title: str = ""
    raw_chunk: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class ImageExtractionContext:
    """向专用抽取器传递共享依赖；当前骨架只保存依赖，不发起模型请求。"""

    llm_client: Any | None = None
    vlm_client: Any | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


def as_string_tuple(value: Any) -> tuple[str, ...]:
    """中文说明：把字符串或序列统一转换为去空值的字符串元组。"""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),)

