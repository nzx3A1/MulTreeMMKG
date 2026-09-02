"""地图与平面空间图片的证据优先视觉抽取 Prompt。"""
from __future__ import annotations

import json

from ..schema_models import ImageExtractionTask


_FOCUS_BY_CODE = {
    "A01": "区域位置、盆地/构造分区、边界、井位与油气田",
    "A02": "岩相、沉积相、古地理单元、相带边界及横向过渡",
    "A07": "地层或层位、参数名称与单位、等值线、数值范围、高低值区及空间趋势",
    "A18": "井/油气田/有利区、预测边界、评价属性及空间展布",
}


def build_map_spatial_prompt(task: ImageExtractionTask) -> str:
    """构造覆盖四类地图的统一 JSON 协议，禁止无视觉证据的拓扑推断。"""

    focus = _FOCUS_BY_CODE.get(task.classification_code, "地图中的地质对象、属性与空间拓扑")
    return f"""
你是石油地质地图判读与多模态知识图谱专家。请直接分析随消息提供的平面地图，不要只复述图题。

图片信息：
- 图片 ID：{task.image_id}
- 图片类别：{task.classification_code or "未提供"} {task.classification_type or ""}
- 图题：{task.caption or "无"}
- 所属章节：{task.section_title or "无"}
- 正文参考：{json.dumps(list(task.references), ensure_ascii=False)}
- 本类重点：{focus}

一级目标（图中存在时必须抽取，直接进入知识图谱）：
1. 地图主题；地层/层位；地理位置；井；油气田；岩相；沉积相；古地理单元；构造单元。
2. LOCATED_IN、CONTAINS、OVERLAPS、ADJACENT_TO 四类平面空间拓扑。
3. 地质属性及其可见属性值、区间和单位；A07 必须优先读取图题、图例和等值线中的参数名与数值。

二级目标（图中存在时推荐抽取）：
相带过渡、空间展布、高值区、低值区、剖面线、普通边界、地层剥蚀线、相对方向、空间趋势。

判读约束：
1. 先读图题、图例、比例尺、指北针、文字标注和符号，再匹配面、线、点。井的圆圈符号与地名的方框符号必须区分；井名不得误作断层或地名。
2. 图例项只定义类别，不等于一个现实地质对象。只有图内实际出现的带名区域或能由图题明确命名的区域才建立对象。
3. 空白区、被裁切区和图框外区域不是地质单元。接触边界明确才可写 ADJACENT_TO；面积相交才可写 OVERLAPS；完全包容才可写 LOCATED_IN/CONTAINS。
4. LOCATED_IN 方向为“小对象 -> 容器”，CONTAINS 方向为“容器 -> 小对象”。不要同时输出同一事实的正反两条边。
5. 颜色相近不能单独证明同类；必须结合图例纹理、边界和文字。无法区分“岩相/沉积相”时降低 confidence 并写入 uncertainties，不能臆造。
6. 地层/层位保留图中原字符，如马五段、奥陶系；按可见层级填写 formation/member/submember/bed/reservoir_interval/unknown。
7. 属性数值必须在图中、图题或正文参考中明确出现；minimum/maximum 不能靠色彩主观估算。正文证据的 evidence_scope=context，图片证据为 visual。
8. 相对方向必须以指北针、经纬网或明确方位文字为依据；仅凭页面上/下/左/右时 direction 写 page_left_of/page_right_of/page_above/page_below，不得擅自改成东南西北。
9. 每个关系端点和 subject_ids 都只能引用本 JSON 已定义的 id。每项给出简短 evidence 和 0~1 confidence；不清楚的文字不要猜。
10. 只输出一个 JSON 对象，不要 Markdown。不要为填满字段而创造图中不存在的对象，空类返回 []。

实体 type 取值优先使用：
geographic_location, basin, study_area, block, well, oil_gas_field, lithofacies,
sedimentary_facies, paleogeographic_unit, structural_unit, depression, sag, uplift, slope, structural_belt。

secondary_features.type 仅使用：
facies_transition, spatial_distribution, high_value_zone, low_value_zone, section_line,
boundary, stratigraphic_erosion_line, relative_direction, spatial_trend。

JSON 格式：
{{
  "schema_version": "map_spatial.v1",
  "map": {{
    "id": "map", "title": "地图题名", "theme": "地图主题", "map_type": "regional_geology|facies|paleogeography|parameter_distribution|prospect|other",
    "scale": "可见比例尺", "north_direction": "可见指北信息", "coordinate_system": "", "evidence": "", "confidence": 0.0
  }},
  "stratigraphic_units": [
    {{"id": "strat_1", "name": "地层/层位原名", "rank": "formation|member|submember|bed|reservoir_interval|unknown", "evidence": "", "evidence_scope": "visual|caption|context", "confidence": 0.0}}
  ],
  "entities": [
    {{
      "id": "obj_1", "name": "图中原名", "type": "上述实体类型之一",
      "geometry": {{"kind": "point|line|polygon|unknown", "position": "图中位置", "bbox": []}},
      "attributes": {{}}, "evidence": "", "evidence_scope": "visual|caption|context", "confidence": 0.0
    }}
  ],
  "geological_attributes": [
    {{
      "id": "attr_1", "name": "厚度/储层厚度/孔隙度等", "subject_ids": ["map或对象id"],
      "value": null, "minimum": null, "maximum": null, "unit": "m/%等", "qualifier": "", "method": "等值线/分区/标注",
      "evidence": "", "evidence_scope": "visual|caption|context", "confidence": 0.0
    }}
  ],
  "secondary_features": [
    {{
      "id": "feature_1", "name": "特征名", "type": "上述二级类型之一", "subject_ids": ["对象或属性id"],
      "source_id": "相对方向或相带过渡的起点id/空", "target_id": "终点id/空", "direction": "north_of/east_of/page_left_of/increasing_toward_north等",
      "geometry": {{"kind": "point|line|polygon|unknown", "position": "", "bbox": []}},
      "evidence": "", "evidence_scope": "visual|caption|context", "confidence": 0.0
    }}
  ],
  "spatial_relations": [
    {{"source_id": "id", "type": "LOCATED_IN|CONTAINS|OVERLAPS|ADJACENT_TO", "target_id": "id", "explicit": true, "evidence": "直接可见的符号/边界依据", "confidence": 0.0}}
  ],
  "uncertainties": ["无法可靠确认的文字、边界或数值"]
}}
""".strip()


__all__ = ["build_map_spatial_prompt"]
