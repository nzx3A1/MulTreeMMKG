from __future__ import annotations

from typing import Dict, List

from .GeoOntologyRules import GeoOntologyRules


class GeoQuestionPlanner:
    """Generate type-specific visual questions for GeoVLM-ECA."""

    def __init__(self, ontology: GeoOntologyRules | None = None):
        self.ontology = ontology or GeoOntologyRules()

    def plan(self, image_type: str, caption: str = "", references: str = "", context: str = "") -> List[Dict[str, str]]:
        image_type = image_type or "Other Geological Image"
        key = image_type.lower()

        if "curve" in key or "chart" in key:
            questions = self._curve_questions()
        elif "map" in key or "geological" in key:
            questions = self._map_questions()
        elif "core" in key or "thin" in key or "section" in key:
            questions = self._core_questions()
        elif "schematic" in key or "diagram" in key:
            questions = self._schematic_questions()
        else:
            questions = self._generic_questions()

        if caption or references or context:
            questions.append({
                "id": f"q{len(questions) + 1}",
                "focus": "text_visual_alignment",
                "question": "图题、references 或上下文中提到的地层、岩性、构造、沉积相、过程或参数，哪些能在图中找到明确视觉证据？",
            })

        return questions[:7]

    @staticmethod
    def _with_ids(items: List[tuple[str, str]]) -> List[Dict[str, str]]:
        return [
            {"id": f"q{i}", "focus": focus, "question": question}
            for i, (focus, question) in enumerate(items, start=1)
        ]

    def _curve_questions(self) -> List[Dict[str, str]]:
        return self._with_ids([
            ("axes", "图中有哪些坐标轴、变量名和单位？"),
            ("series", "图中有哪些曲线、图例系列或测井响应？"),
            ("trend", "主要曲线整体呈上升、下降、波动还是分段变化？"),
            ("anomaly", "是否存在峰值、低值、拐点、异常段或关键深度段？"),
            ("relation", "曲线与参数、地层或储层特征之间能形成哪些证据明确的关系？"),
        ])

    def _map_questions(self) -> List[Dict[str, str]]:
        return self._with_ids([
            ("geological_objects", "图中出现哪些地层、沉积相、构造单元、井位或区域名称？"),
            ("spatial_relation", "不同地质对象之间存在包含、邻接、分布、展布方向或上下游关系吗？"),
            ("legend", "图例、颜色、符号或标注分别对应哪些地质含义？"),
            ("trend", "图中是否体现砂体、湖岸线、物源、水系或相带的迁移、变薄、萎缩、推进等趋势？"),
            ("relation", "哪些关系可以由图像证据直接支撑，例如 contains、has_facies、located_in 或 adjacent_to？"),
        ])

    def _core_questions(self) -> List[Dict[str, str]]:
        return self._with_ids([
            ("lithology", "图中可见哪些岩性、矿物、孔隙、裂缝、纹理或沉积构造？"),
            ("reservoir", "这些现象说明了哪些储层特征、孔渗条件或成岩作用？"),
            ("scale", "是否有尺度、放大倍数、样品层位或井深信息？"),
            ("relation", "可见岩性或构造现象能支持哪些 has_lithology、has_component、indicates 或 formed_by 关系？"),
        ])

    def _schematic_questions(self) -> List[Dict[str, str]]:
        return self._with_ids([
            ("objects", "图中有哪些地层、岩性、构造、沉积相、油气或储层要素？"),
            ("process", "箭头、流程或空间组合表达了哪些迁移、沉积、成藏或演化关系？"),
            ("order", "哪些过程或层位先发生、位于下部或作为后续过程的基础？"),
            ("control", "哪些因素控制了油气聚集、储层发育、沉积相展布或构造演化？"),
            ("relation", "哪些候选关系可以由图像证据和文字证据共同支持？"),
        ])

    def _generic_questions(self) -> List[Dict[str, str]]:
        return self._with_ids([
            ("objects", "图中可识别哪些关键石油地质对象、参数、符号或标注？"),
            ("structure", "这些对象之间存在什么空间、组成、趋势、因果或支撑关系？"),
            ("evidence", "每个关键实体和关系分别对应哪些视觉证据片段？"),
            ("relation", "哪些关系能映射到石油地质领域本体模板？"),
        ])

