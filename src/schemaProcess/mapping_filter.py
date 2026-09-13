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


# 图像/表格抽取器的结构类型在抽取阶段已确定，不应再映射到领域 Schema。
# 该表只提供稳定的中文展示名称，避免 ``schema_type_zh`` 留空。
PREDEFINED_TYPE_ZH = {
    "table": "表格",
    "caption": "题注",
    "table_header": "表头",
    "table_row": "表格行",
    "table_column": "表格列",
    "table_cell": "表格单元格",
    "parameter": "参数",
    "unit": "单位",
    "data_series": "数据序列",
    "quantitative_chart": "定量图表",
    "chart_panel": "图表面板",
    "chart_axis": "图表坐标轴",
    "chart_trend": "图表趋势",
    "chart_anomaly": "图表异常",
    "chart_correlation": "图表相关性",
    "chart_threshold": "图表阈值",
    "temporal_change": "时间变化",
    "stratigraphic_profile": "地层剖面图",
    "two_dimensional_stratigraphic_log": "二维地层柱状图",
    "three_dimensional_stratigraphic_model": "三维地层模型",
    "diagram_track": "图件轨道",
    "track_interval": "轨道区间",
    "curve_response_interval": "曲线响应区间",
    "map_theme": "地图主题",
    "geological_attribute": "地质属性",
    "attribute_value": "属性值",
    "facies_transition": "相带转换",
    "spatial_distribution": "空间分布",
    "high_value_zone": "高值区",
    "low_value_zone": "低值区",
    "section_line": "剖面线",
    "boundary": "边界",
    "stratigraphic_erosion_line": "地层剥蚀线",
    "relative_direction": "相对方向",
    "spatial_trend": "空间趋势",
}


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


def resolve_schema_type_zh(
    raw_type: Any,
    type_zh: Any = None,
    schema_type_zh: Any = None,
) -> str:
    """解析实体展示中文类型，优先保留已有值并保证返回非空字符串。"""

    for value in (schema_type_zh, type_zh):
        text = str(value or "").strip()
        if text:
            return text

    normalized_type = normalize_predefined_type(raw_type)
    if normalized_type in PREDEFINED_TYPE_ZH:
        return PREDEFINED_TYPE_ZH[normalized_type]

    raw_text = str(raw_type or "").strip()
    return f"原始类型（{raw_text}）" if raw_text else "未命名类型"


def resolve_schema_type(type_value: Any, schema_type: Any = None) -> str | None:
    """解析实体 Schema 类型；缺失时使用抽取器输出的 ``type``。"""

    for value in (schema_type, type_value):
        text = str(value or "").strip()
        if text:
            return text
    return None


__all__ = [
    "CHART_PREDEFINED_TYPES",
    "IMAGE_PREDEFINED_TYPES",
    "PREDEFINED_TYPE_ZH",
    "TABLE_PREDEFINED_TYPES",
    "is_predefined_non_schema_type",
    "normalize_predefined_type",
    "resolve_schema_type",
    "resolve_schema_type_zh",
]
