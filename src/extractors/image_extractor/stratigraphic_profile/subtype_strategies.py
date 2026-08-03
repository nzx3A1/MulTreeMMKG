"""地层图片子类型的路由元数据。"""
from __future__ import annotations

from dataclasses import dataclass

from .subclassifier import StratigraphicProfileSubtype


@dataclass(frozen=True)
class StratigraphicSubtypeExtractionStrategy:
    """保存子类型对应的算法标识，不在外层定义任何具体抽取步骤。"""

    subtype: StratigraphicProfileSubtype
    algorithm_name: str
    algorithm_description: str


# 中文说明：这些值只用于路由记录和结果审计，具体算法由各子目录自己实现。
_STRATEGIES = {
    StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID: StratigraphicSubtypeExtractionStrategy(
        subtype=StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID,
        algorithm_name="ppstructurev3_geometry_semantic_vlm_depth_alignment_graph_assembly",
        algorithm_description="PP-StructureV3 像素几何 + VLM 地质语义 + 确定性深度对齐算法",
    ),
    StratigraphicProfileSubtype.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL: StratigraphicSubtypeExtractionStrategy(
        subtype=StratigraphicProfileSubtype.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL,
        algorithm_name="surface_topology_spatial_extraction",
        algorithm_description="三维表面拓扑 + 同层级上下层序 + 上下文证据三元组的专用抽取算法",
    ),
    StratigraphicProfileSubtype.TWO_DIMENSIONAL_STRATIGRAPHIC_LOG: StratigraphicSubtypeExtractionStrategy(
        subtype=StratigraphicProfileSubtype.TWO_DIMENSIONAL_STRATIGRAPHIC_LOG,
        algorithm_name="axis_layer_correlation_extraction",
        algorithm_description="二维地层—测井子目录专用抽取算法",
    ),
}


def get_stratigraphic_subtype_strategy(
    subtype: StratigraphicProfileSubtype,
) -> StratigraphicSubtypeExtractionStrategy:
    """中文说明：返回子类型的路由元数据，未知类型直接报错。"""

    return _STRATEGIES[subtype]
