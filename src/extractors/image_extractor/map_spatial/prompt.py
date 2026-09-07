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
1. 地图主题；地层/层位；地理位置；井；油气田；岩性；岩相；沉积相；古地理单元；构造单元。每个实体必须给出明确的 type 和 type_zh。
2. 分开抽取地质语义关系与纯空间关系。语义关系使用 DISTRIBUTED_IN、DEVELOPED_IN、PART_OF、HAS_LITHOLOGY、HAS_FACIES；空间关系使用 WITHIN、CONTAINS、COVERS、OVERLAPS、INTERSECTS、CROSSES、TOUCHES、ADJACENT_TO、NEAR 和八方位关系。
3. 为图中真实出现的点、线、面重建几何边界，坐标统一为左上角原点、右下角 [1000,1000] 的 normalized_1000 坐标。
4. 地质属性及其可见属性值、区间和单位；A07 必须优先读取图题、图例和等值线中的参数名与数值。

二级目标（图中存在时推荐抽取）：
相带过渡、空间展布、高值区、低值区、剖面线、普通边界、地层剥蚀线、相对方向、空间趋势。

判读约束：
1. 先读图题、图例、比例尺、指北针、文字标注和符号，再匹配面、线、点。井的圆圈符号与地名的方框符号必须区分；井名不得误作断层或地名。
2. 图例项只定义类别，不等于一个现实地质对象。只有图内实际出现的带名区域或能由图题明确命名的区域才建立对象。
3. 空白区、被裁切区和图框外区域不是地质单元。接触边界明确才可写 ADJACENT_TO/TOUCHES；面积相交才可写 OVERLAPS；线穿过面才写 CROSSES；完全包容才可写 WITHIN/CONTAINS。
4. 岩性和岩相是地质概念，不是地点。禁止输出“岩性/岩相 LOCATED_IN 地点”；应按证据输出“岩性 DISTRIBUTED_IN 区域”“沉积相 DEVELOPED_IN 区域”或“区域 HAS_LITHOLOGY 岩性”。构造单元与盆地的层级使用 PART_OF。井、油气田、行政地点等有位置的对象才可使用 LOCATED_IN；几何包含优先使用 WITHIN。
5. 颜色相近不能单独证明同类；必须结合图例纹理、边界和文字。无法区分“岩相/沉积相”时降低 confidence 并写入 uncertainties，不能臆造。
6. 地层/层位保留图中原字符，如马五段、奥陶系；按可见层级填写 formation/member/submember/bed/reservoir_interval/unknown。
7. 属性数值必须在图中、图题或正文参考中明确出现；minimum/maximum 不能靠色彩主观估算。正文证据的 evidence_scope=context，图片证据为 visual。
8. 相对方向必须以指北针、经纬网或明确方位文字为依据；在 georeference 中记录指北旋转角。没有指北依据时只能写 PAGE_LEFT_OF/PAGE_RIGHT_OF/PAGE_ABOVE/PAGE_BELOW，不得擅自改成东南西北。
9. 点使用 [x,y]；线和面使用 [[x1,y1],...]，面沿可见边界给出至少 3 个顶点，不能只用文字 position 代替坐标。文字标签框不是地质区域边界。
10. 比例尺清楚时记录比例尺线段端点、实际距离和单位；至少 3 个经纬度/坐标控制点清楚时才填写 control_points，否则保持空数组，禁止猜坐标。
11. 每个关系端点和 subject_ids 都只能引用本 JSON 已定义的 id。每项给出简短 evidence 和 0~1 confidence；不清楚的文字不要猜。
12. 关系必须稀疏。相邻地点只保留形成连续邻接链所需的直接边，例如已有 A—B、B—C 时不要再输出 A—C；嵌套区域只输出最近一级归属，例如井→区块、区块→盆地，不再输出井→盆地。
13. 每个实体最多主动输出一条最有地质意义的方位关系；空间关系总数原则上不超过可定位实体数的两倍。
14. 只输出一个 JSON 对象，不要 Markdown。不要为填满字段而创造图中不存在的对象，空类返回 []。

实体 type 取值优先使用：
geographic_location, province, city, county, town, basin, study_area, block, well,
oil_gas_field, prospect_area, lithology, lithofacies, sedimentary_facies, subfacies,
microfacies, paleogeographic_unit, structural_unit, depression, sag, uplift, slope,
structural_belt, fault, fault_zone。

secondary_features.type 仅使用：
facies_transition, spatial_distribution, high_value_zone, low_value_zone, section_line,
boundary, stratigraphic_erosion_line, relative_direction, spatial_trend。

JSON 格式：
{{
  "schema_version": "map_spatial.v2",
  "map": {{
    "id": "map", "title": "地图题名", "theme": "地图主题", "map_type": "regional_geology|facies|paleogeography|parameter_distribution|prospect|other",
    "scale": "可见比例尺", "north_direction": "可见指北信息", "coordinate_system": "", "evidence": "", "confidence": 0.0
  }},
  "georeference": {{
    "coordinate_space": "normalized_1000",
    "crs": "图中明确标注的坐标参考系或空字符串",
    "world_coordinate_unit": "degree|m|km|",
    "north_rotation_degrees_clockwise_from_page_up": null,
    "scale_bar": {{"start": [], "end": [], "distance": null, "unit": ""}},
    "control_points": [{{"normalized_point": [0.0,0.0], "world_point": [0.0,0.0], "label": "可见刻度文字"}}]
  }},
  "stratigraphic_units": [
    {{"id": "strat_1", "name": "地层/层位原名", "rank": "formation|member|submember|bed|reservoir_interval|unknown", "evidence": "", "evidence_scope": "visual|caption|context", "confidence": 0.0}}
  ],
  "entities": [
    {{
      "id": "obj_1", "name": "图中原名", "type": "上述实体类型之一", "type_zh": "类型中文名",
      "geometry": {{"kind": "point|line|polygon|unknown", "coordinate_space": "normalized_1000", "coordinates": [], "position": "图中位置", "confidence": 0.0}},
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
      "geometry": {{"kind": "point|line|polygon|unknown", "coordinate_space": "normalized_1000", "coordinates": [], "position": "", "confidence": 0.0}},
      "evidence": "", "evidence_scope": "visual|caption|context", "confidence": 0.0
    }}
  ],
  "semantic_relations": [
    {{"source_id": "id", "type": "DISTRIBUTED_IN|DEVELOPED_IN|PART_OF|HAS_LITHOLOGY|HAS_FACIES", "target_id": "id", "explicit": true, "evidence": "地质语义依据", "confidence": 0.0}}
  ],
  "spatial_relations": [
    {{"source_id": "id", "type": "LOCATED_IN|WITHIN|CONTAINS|COVERS|OVERLAPS|INTERSECTS|CROSSES|TOUCHES|ADJACENT_TO|NEAR|NORTH_OF|SOUTH_OF|EAST_OF|WEST_OF|NORTHEAST_OF|NORTHWEST_OF|SOUTHEAST_OF|SOUTHWEST_OF|PAGE_LEFT_OF|PAGE_RIGHT_OF|PAGE_ABOVE|PAGE_BELOW", "target_id": "id", "explicit": true, "evidence": "直接可见的符号、边界或方向依据", "confidence": 0.0}}
  ],
  "uncertainties": ["无法可靠确认的文字、边界或数值"]
}}
""".strip()


__all__ = ["build_map_spatial_prompt"]
