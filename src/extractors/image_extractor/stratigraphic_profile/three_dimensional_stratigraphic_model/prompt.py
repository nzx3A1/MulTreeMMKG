"""三维地层建模图的视觉—上下文联合结构化抽取 Prompt。"""
from __future__ import annotations

import json

from ...schema_models import ImageExtractionTask
from ..subclassifier import StratigraphicSubtypeClassification


def build_three_dimensional_stratigraphic_model_prompt(
    task: ImageExtractionTask,
    classification: StratigraphicSubtypeClassification,
) -> str:
    """中文说明：要求 VLM 优先读取真实垂向层序，并把图内事实与正文补充关系严格分栏。"""

    return f"""
你是三维地层建模图知识抽取专家。请联合读取图片、图题和正文参考，输出严格 JSON。

来源图片：{task.image_path}
图片分类：{task.classification_code} {task.classification_type}
子类型：{classification.subtype.value}（{classification.subtype_name}）
子分类依据：{classification.evidence}
抽取算法：surface_topology_spatial_extraction
图题：{task.caption}
正文参考：
{json.dumps(list(task.references), ensure_ascii=False, indent=2)}

核心任务（按优先级执行）：
1. 重点读取地层剖面从下到上的真实叠置关系。必须沿可见侧壁从最底部到最顶部逐层扫描，不能跳过薄层或黄色等窄色带。每个可辨认地层都填写 order_bottom_to_top，最下层为 0；只有处于同一 column_id 且具有同一 parent_unit_id 的地层才能直接比较上下。
2. 区分顶面、侧壁、剖切面、前后遮挡和透视方向。透视画面中的远近关系不等于地层上下关系，不得据此编造层序。
3. 识别断层/裂缝带、井轨迹、构造或沉积环境分区、流体/离子箭头、岩性和其他有标签对象。同一对象跨多个可见面时只建一个实体。
4. relations 只写图片中由标签、相交、包围、错断或箭头明确支持的关系；上下关系若仅来自层序排列，不要在 relations 重复填写，程序会按 order_bottom_to_top 确定性生成。
5. context_relations 只写图题或正文参考明确支持且与图中对象可锚定的三元组。端点必须引用本 JSON 已定义 id，explicit 必须为 false，并逐字摘录简短 context_evidence；不得凭地质常识扩写。
6. 图片模糊、地层层级不一致、被遮挡或上下顺序无法确认时写入 uncertainties，不猜测。
7. 每个对象必须包含 evidence、image_region、confidence 和 source_kind。source_kind 只能是 visual 或 visual_context；visual_context 对象还必须有 context_evidence。
8. 岩性必须直接归属到具体 stratigraphic_unit，lithology_ids 是可包含多个 ID 的数组；不得创建 visual_layer_segment 或其他中间节点来连接地层与岩性。
9. 只输出 JSON，不要 Markdown，不要解释。

地层文字与上下文关系的额外质量约束：
- 地层名称必须原样保留图中的地质年代符号、上下标和组名，例如 Є、O、下标数字；若图片字符与正文中的同一名称相互印证，可用正文纠正 OCR，并把 source_kind 标为 visual_context、写入 context_evidence。
- 逐层复核每个可见地层标签；看不清时记录原始可见片段和 uncertainty，不能用标准地层表补齐图中没有的层。
- transports 的宾语只能是油、气、油气、流体或离子，不能是烃源岩、断裂、地层模型；断裂输导油气时必须单独定义“油气”对象。
- generates 的主语可以是烃源岩，宾语必须是油气；controls 的主语通常是断裂或构造，宾语是储层、油气藏或富集区。
- dolomitizes 的宾语应优先使用正文明确指出的灰岩/泥晶灰岩等岩性实体，不能因为空间相邻而随意改成整个地层段。
- 不得把 model 根实体作为 acts_as_source_rock、transports、generates、dolomitizes 的宾语。若缺少语义准确的端点，应先在 objects、zones、fluids 或 lithologies 中定义带 visual_context 证据的实体。

关系类型优先使用：
- 层序与组成：part_of, has_lithology, directly_overlies, directly_underlies
- 空间与构造：contains, located_in, cuts_through, offsets, intersects, adjacent_to
- 地质作用与流体：flows_to, transports, generates, supplies_hydrocarbons_to, controls, dolomitizes, acts_as_source_rock, acts_as_reservoir, acts_as_seal

JSON 格式：
{{
  "schema_version": "three_dimensional_stratigraphic_model.v1",
  "model": {{
    "id": "model_1",
    "name": "图题或三维模型名称",
    "view_direction": "",
    "visible_surfaces": ["top", "front", "side", "cutaway"],
    "topology_evidence": "",
    "evidence": "",
    "image_region": "整图",
    "confidence": 0.0
  }},
  "lithologies": [
    {{"id": "lith_1", "name": "", "visual_pattern": "", "source_kind": "visual", "evidence": "", "image_region": "", "confidence": 0.0}}
  ],
  "stratigraphic_units": [
    {{
      "id": "unit_1", "name": "图中原始标签", "rank": "系/统/组/段/层", "parent_unit_id": "",
      "column_id": "main_section", "order_bottom_to_top": 0, "sequence_role": "layer",
      "lithology_ids": ["一个或多个岩性 ID"], "geometry": "", "source_kind": "visual",
      "evidence": "", "image_region": "", "confidence": 0.0
    }}
  ],
  "wells": [
    {{"id": "well_1", "name": "", "trajectory": "", "source_kind": "visual", "evidence": "", "image_region": "", "confidence": 0.0}}
  ],
  "faults": [
    {{"id": "fault_1", "name": "", "fault_style": "", "source_kind": "visual", "evidence": "", "image_region": "", "confidence": 0.0}}
  ],
  "zones": [
    {{"id": "zone_1", "name": "", "zone_type": "structure/facies/environment/reservoir", "source_kind": "visual", "evidence": "", "image_region": "", "confidence": 0.0}}
  ],
  "fluids": [
    {{"id": "fluid_1", "name": "", "direction": "", "source_kind": "visual", "evidence": "", "image_region": "", "confidence": 0.0}}
  ],
  "objects": [
    {{"id": "obj_1", "name": "", "entity_type": "geological_object", "source_kind": "visual", "evidence": "", "image_region": "", "confidence": 0.0}}
  ],
  "relations": [
    {{
      "source": "已定义 id", "type": "cuts_through", "target": "已定义 id",
      "explicit": true, "basis": "visible_intersection", "evidence_scope": "visual",
      "evidence": "", "visual_anchor_evidence": "", "confidence": 0.0
    }}
  ],
  "context_relations": [
    {{
      "source": "已定义 id", "type": "acts_as_source_rock", "target": "已定义 id",
      "explicit": false, "basis": "reference_context", "evidence_scope": "context",
      "context_evidence": "正文中的短证据", "visual_anchor_evidence": "端点在图中的可见锚点", "confidence": 0.0
    }}
  ],
  "uncertainties": [""]
}}
""".strip()
