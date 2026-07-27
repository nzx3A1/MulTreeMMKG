"""三类地层图片的差异化视觉抽取策略。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..schema_models import ImageExtractionTask
from .prompt import build_stratigraphic_visual_prompt
from .subclassifier import StratigraphicProfileSubtype, StratigraphicSubtypeClassification


@dataclass(frozen=True)
class StratigraphicSubtypeExtractionStrategy:
    """定义子类型对应的算法名、处理重点和专用 Prompt。"""

    subtype: StratigraphicProfileSubtype
    algorithm_name: str
    algorithm_description: str
    prompt_suffix: str

    def build_prompt(
        self,
        task: ImageExtractionTask,
        classification: StratigraphicSubtypeClassification,
    ) -> str:
        """中文说明：在共享地质证据约束后追加该子类型独立的处理步骤。"""

        base_prompt = build_stratigraphic_visual_prompt(task)
        return (
            f"{base_prompt}\n\n"
            "【大类内部子分类结果】\n"
            f"子类型：{classification.subtype.value}（{classification.subtype_name}）\n"
            f"分类依据：{classification.evidence}\n"
            f"抽取算法：{self.algorithm_name}\n"
            f"{self.prompt_suffix.strip()}"
        )

    def normalize_visual_result(self, visual: Mapping[str, Any]) -> dict[str, Any]:
        """中文说明：保留模型原始字段，同时写入实际采用的子类型算法标识供后续审计。"""

        normalized = dict(visual)
        normalized["stratigraphic_subtype"] = self.subtype.value
        normalized["subtype_extraction_algorithm"] = self.algorithm_name
        return normalized


TABLE_EMBEDDED_HYBRID_STRATEGY = StratigraphicSubtypeExtractionStrategy(
    subtype=StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID,
    algorithm_name="coordinate_track_specialized_depth_alignment_graph_assembly",
    algorithm_description="坐标系重建 + 轨道拆分 + 专用解析 + 深度/层序对齐 + 确定性知识图谱装配",
    prompt_suffix="""
专用步骤：
1. 先定位表头、纵横分隔线、合并单元格、深度列、地层列、岩性列、曲线轨和嵌入图片面板。
2. 以深度或层段边界为公共行锚点，将同一高度的地层、岩性、曲线响应、储层标记和井名对齐。
3. 嵌入照片、成像测井或小剖面必须作为独立面板读取，再通过可见行列位置关联到对应深度或层段。
4. 只能输出网格或轨道中明确对应的关系；跨列错位、裁切缺失和无法辨认的表头写入 uncertainties。
5. 按 table_embedded_hybrid.v1 专用结构输出像素坐标、标尺锚点、轨道和图元；不得直接生成跨轨道关系。
""",
)


THREE_DIMENSIONAL_MODEL_STRATEGY = StratigraphicSubtypeExtractionStrategy(
    subtype=StratigraphicProfileSubtype.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL,
    algorithm_name="surface_topology_spatial_extraction",
    algorithm_description="三维表面识别 + 块体拓扑 + 断裂/井轨迹空间关系抽取",
    prompt_suffix="""
专用步骤：
1. 区分顶面、侧壁、剖切面和透视方向，读取可见方位箭头、比例尺及前后遮挡关系。
2. 识别地层表面、三维体、断裂带、相带、井轨迹和流体箭头；同一对象跨两个可见面时合并为一个实体。
3. cuts_through、offsets、located_in、flows_to 等空间关系必须由相交、错断、包围或箭头等明确图形证据支持。
4. 透视投影产生的远近关系不得当作真实地质层序；不可见背面和体积不得补猜。
5. 在 JSON 顶层补充 spatial_model：view_direction、visible_surfaces、volumes、topology_evidence。
""",
)


TWO_DIMENSIONAL_LOG_STRATEGY = StratigraphicSubtypeExtractionStrategy(
    subtype=StratigraphicProfileSubtype.TWO_DIMENSIONAL_STRATIGRAPHIC_LOG,
    algorithm_name="axis_layer_correlation_extraction",
    algorithm_description="二维坐标轴识别 + 层界追踪 + 井/曲线/地震同层对比",
    prompt_suffix="""
专用步骤：
1. 先判断横轴是距离、井位或道号，纵轴是深度、海拔、时间或层序；记录单位和增大方向。
2. 沿二维平面追踪层界、断层、井轨迹和地震反射轴，建立上下、切穿、错断和横向延伸关系。
3. 连井图按井分别读取曲线轨，再用可见连接线或同名层段建立 correlates_with，不得仅凭相近深度强行对比。
4. 单井曲线保留轨道名称、刻度范围和高低响应区间；二维地震只记录明确标注的解释层位和轨迹。
5. 在 JSON 顶层补充 plane_analysis：horizontal_axis、vertical_axis、layer_boundaries、correlation_lines。
""",
)


_STRATEGIES = {
    strategy.subtype: strategy
    for strategy in (
        TABLE_EMBEDDED_HYBRID_STRATEGY,
        THREE_DIMENSIONAL_MODEL_STRATEGY,
        TWO_DIMENSIONAL_LOG_STRATEGY,
    )
}


def get_stratigraphic_subtype_strategy(
    subtype: StratigraphicProfileSubtype,
) -> StratigraphicSubtypeExtractionStrategy:
    """中文说明：按稳定子类型返回唯一抽取策略，未知枚举会立即暴露配置错误。"""

    return _STRATEGIES[subtype]
