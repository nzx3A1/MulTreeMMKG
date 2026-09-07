"""识别抽取阶段已经确定、无需再次执行 Schema 映射的结构类型。"""
from __future__ import annotations

import re
from typing import Any


TABLE_PREDEFINED_TYPES = frozenset(
    {
        "table",
        "caption",
        "table_header",
        "table_row",
        "table_column",
        "table_cell",
        "parameter",
        "unit",
        "data_series",
    }
)

CHART_PREDEFINED_TYPES = frozenset(
    {
        "quantitative_chart",
        "chart_panel",
        "chart_axis",
        "data_series",
        "chart_trend",
        "chart_anomaly",
        "chart_correlation",
        "chart_threshold",
        "temporal_change",
    }
)

IMAGE_PREDEFINED_TYPES = CHART_PREDEFINED_TYPES | frozenset(
    {
        # 地层图的版面/轨道辅助节点。
        "stratigraphic_profile",
        "two_dimensional_stratigraphic_log",
        "three_dimensional_stratigraphic_model",
        "diagram_track",
        "track_interval",
        "curve_response_interval",
        # 地图抽取器用于表达图面结构的辅助节点。
        "map_theme",
        "geological_attribute",
        "attribute_value",
        "facies_transition",
        "spatial_distribution",
        "high_value_zone",
        "low_value_zone",
        "section_line",
        "boundary",
        "stratigraphic_erosion_line",
        "relative_direction",
        "spatial_trend",
    }
)


def normalize_predefined_type(value: Any) -> str:
    """将 snake_case、连字符和 CamelCase 类型统一为小写下划线形式。"""

    text = str(value or "").strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()


def is_predefined_non_schema_type(raw_type: Any, source_modality: Any) -> bool:
    """判断类型是否由表格/图像抽取器确定，因而不需要概念 Schema 映射。"""

    entity_type = normalize_predefined_type(raw_type)
    modality = str(source_modality or "").strip().lower()
    if modality == "table":
        return entity_type in TABLE_PREDEFINED_TYPES
    if modality == "image":
        # ``chart_*`` 同时覆盖定量图表抽取器以后新增的结构图元类型。
        return entity_type in IMAGE_PREDEFINED_TYPES or entity_type.startswith("chart_")
    return False


__all__ = [
    "CHART_PREDEFINED_TYPES",
    "IMAGE_PREDEFINED_TYPES",
    "TABLE_PREDEFINED_TYPES",
    "is_predefined_non_schema_type",
    "normalize_predefined_type",
]
