"""地层、剖面、地震与测井图片的结构化抽取 Prompt。"""
from __future__ import annotations

import json

from ..schema_models import ImageExtractionTask


def build_stratigraphic_visual_prompt(task: ImageExtractionTask) -> str:
    """中文说明：要求视觉模型按统一 schema 穷举图内层段、井、曲线、区带和证据关系。"""

    return f"""
你是石油地质多模态知识图谱专家。请逐项读取这张地层—剖面—测井类图片，并返回可直接建图的 JSON。

来源图片：{task.image_path}
图片分类：{task.classification_code} {task.classification_type}
图题：{task.caption}
正文参考（只可用于消歧，正文补充关系的 explicit 必须为 false）：
{json.dumps(list(task.references), ensure_ascii=False)}

必须完成：
1. 先判断 diagram_type：stratigraphic_column、geological_section、seismic_profile、single_well_log、multiwell_correlation、borehole_image 或 composite_profile。
2. 原样抄写全部可辨认的地层、组、段、亚段、小层和标志层；按从下到上填写 order_bottom_to_top。大套地层与子层同时出现时必须用 parent_unit_id 建层级。
3. 读取图例中的全部可辨认岩性。仅有编号而没有岩性名称时，名称写“图例岩性1”等并在 uncertainties 说明，不得凭花纹猜岩性。
4. 读取全部井名、深度范围、井轨迹、断裂、构造区、沉积相、储层、油层、页岩层、顶板、底板、研究层和取心段。
5. 读取全部测井/地震道名称、单位、刻度和明确可见的高低响应；不能从曲线形态编造精确数值。
6. 连井剖面必须建立井—层段穿过关系和同层跨井对比；构造剖面必须建立断裂切层、井所在构造区和相带邻接；地震剖面必须区分原轨迹与调整后轨迹。
7. 每个对象必须有 id、evidence、image_region、confidence；每条关系端点必须引用已定义 id，并带 explicit、evidence、confidence。
8. 看不清的字或数值写入 uncertainties，不得猜测。只输出 JSON，不要 Markdown。

关系类型优先使用：
- 层级/组成：part_of, contains, has_lithology, contains_interval
- 垂向/横向：directly_overlies, directly_underlies, above, below, adjacent_to, lateral_transition_to, correlates_with
- 井与曲线：intersects, has_log_curve, measured_in, tracks_along, deviates_from, adjusted_from
- 构造/储层：located_in, cuts_through, offsets, contains_reservoir, contains_oil_layer, bounded_by, acts_as_source_rock, acts_as_reservoir, acts_as_seal
- 指标比较：higher_response_than, lower_response_than, characterizes

JSON 格式：
{{
  "diagram_type": "",
  "diagram_summary": "",
  "coordinate_system": {{"horizontal_meaning": "", "vertical_meaning": "", "depth_unit": "", "normal_stratigraphic_order": true, "evidence": ""}},
  "lithologies": [
    {{"id": "lith_1", "name": "", "visual_pattern": "", "evidence": "", "image_region": "", "confidence": 0.0}}
  ],
  "stratigraphic_units": [
    {{
      "id": "unit_1", "name": "图中原始标签", "rank": "组/段/亚段/小层/标志层", "parent_unit_id": "",
      "order_bottom_to_top": 0, "top_depth_m": null, "bottom_depth_m": null, "geometry": "",
      "lithologies": [{{"id": "lith_1", "role": "主要/次要/夹层/不确定", "evidence": "", "confidence": 0.0}}],
      "evidence": "", "image_region": "", "confidence": 0.0
    }}
  ],
  "wells": [
    {{"id": "well_1", "name": "", "top_depth_m": null, "bottom_depth_m": null, "attributes": {{}}, "evidence": "", "image_region": "", "confidence": 0.0}}
  ],
  "log_curves": [
    {{"id": "curve_1", "name": "", "well_id": "", "unit": "", "scale": "", "response": "", "evidence": "", "image_region": "", "confidence": 0.0}}
  ],
  "zones": [
    {{"id": "zone_1", "name": "", "type": "reservoir/oil_layer/core_interval/shale/facies/structure/marker/top_seal/bottom_seal/study_interval", "top_depth_m": null, "bottom_depth_m": null, "attributes": {{}}, "evidence": "", "image_region": "", "confidence": 0.0}}
  ],
  "entities": [
    {{"id": "obj_1", "name": "", "type": "fault/structure/trajectory/fluid/mineral/other", "attributes": {{}}, "evidence": "", "image_region": "", "confidence": 0.0}}
  ],
  "relations": [
    {{"source": "id", "type": "关系类型", "target": "id", "dimension": "stratigraphic/spatial/structural/logging/correlation/process", "explicit": true, "basis": "", "evidence": "", "confidence": 0.0}}
  ],
  "uncertainties": [""]
}}
""".strip()


def build_stratigraphic_relation_audit_prompt(task: ImageExtractionTask, visual_result: dict) -> str:
    """中文说明：让文本模型只审查端点、方向和证据，不凭常识扩写视觉结果。"""

    return f"""
你是地层剖面与测井知识图谱关系审查专家。请审查以下视觉抽取 JSON。
来源图片：{task.image_path}
图题：{task.caption}
视觉结果：
{json.dumps(visual_result, ensure_ascii=False, indent=2)}

约束：
1. 不得创建视觉结果中不存在的实体 id。
2. 检查地层上下关系方向、井—层段穿过关系、曲线—井归属、断裂切层和跨井对比是否正确。
3. added_relations 只补图题、正文或图内明确支持而视觉结果遗漏的关系；正文证据 explicit=false。
4. reject_relations 用 source/type/target 指出必须删除的错误边。
5. 只输出 JSON。

JSON 格式：
{{
  "accepted": true,
  "added_relations": [{{"source": "id", "type": "", "target": "id", "dimension": "", "explicit": false, "basis": "", "evidence": "", "confidence": 0.0}}],
  "reject_relations": [{{"source": "id", "type": "", "target": "id", "reason": ""}}],
  "warnings": []
}}
""".strip()
