"""二维平面地层—测井图的专用视觉抽取 Prompt。"""
from __future__ import annotations

import json

from ...schema_models import ImageExtractionTask
from ..subclassifier import StratigraphicSubtypeClassification


def build_two_dimensional_stratigraphic_log_prompt(
    task: ImageExtractionTask,
    classification: StratigraphicSubtypeClassification,
) -> str:
    """中文说明：要求 VLM 抄录二维图元、层序和相对位置，并把可确定关系限定在已定义实体之间。"""

    return f"""
你是石油地质二维剖面、连井测井和地震剖面知识抽取专家。请直接读取图片并只输出 JSON。

来源图片：{task.image_path}
图题：{task.caption}
正文参考（只可消歧；正文独有信息不得伪装成图中显式证据）：
{json.dumps(list(task.references), ensure_ascii=False)}
已确认子类型：{classification.subtype.value}（{classification.subtype_name}）
子分类依据：{classification.evidence}
抽取算法：axis_layer_correlation_extraction

核心目标：
1. 完整抽取图中可辨认的地层、组、段、亚段、小层、层位、页岩段、储层、顶板、底板、井、井轨迹、断层、构造区、沉积相、岩性和测井曲线。
2. 重点建立地层从上到下的序列，以及井、构造区、相带等对象从左到右的相对位置。
3. 抽取图中直接画出或标注的井—层段钻遇、轨迹沿层、断层切层、跨井对比、相带横向过渡和边界关系。
4. 看不清的文字、深度和数值写入 uncertainties；不得凭颜色、花纹或常识猜测实体名称。

坐标与顺序约束：
- position.bbox 使用归一化整数坐标 [x0,y0,x1,y1]，整图左上角为 [0,0]，右下角为 [1000,1000]；无法可靠定位时用空数组。
- position.x_order 和 position.y_order 从 0 开始；只在图中顺序明确时填写整数，否则为 null。
- stratigraphic_sequences.ordered_unit_ids_top_to_bottom 必须严格按图中从上到下排列；同一父层的子层单独建序列。
- stratigraphic_sequences 中只放实际参与层序的节点；程序只连接列表中的相邻节点，不会连接跨层节点。
- spatial_groups.ordered_member_ids_left_to_right 只作版式审计，不生成 left_of/right_of/adjacent_to 三元组。
- 每个实体 id 在本图内唯一；parent_id、序列、横向组和关系端点只能引用 entities 中已有 id。
- 同一地层在多口井分别出现时：若图中表达的是同一跨井层位，可只建一个区域实体；若各井深度段需要分别保留，则建立井内区间实体并用 correlates_with 连接。
- 除图片/Chunk 根节点外，每个实体必须至少出现在 parent_id、stratigraphic_sequences 或 relations 之一；完全孤立的实体不要输出。

关系类型只使用：
part_of, contains, directly_overlies, directly_underlies,
intersects, cuts_through, offsets,
correlates_with, lateral_transition_to, tracks_along, adjusted_from,
located_in, bounded_by, has_lithology, has_log_curve,
contains_reservoir, contains_shale, characterizes, related_to。

explicit 规则：
- 图中有连线、穿越、边界、同层对比线或明确文字标注时 explicit=true。
- 只由正文参考补充或由视觉顺序推导时 explicit=false，并在 basis 中写清依据。
- 纯层序位置不要输出 above、below、left_of、right_of 或 adjacent_to。
- 不必在 relations 中重复上下层序；程序只为 ordered_unit_ids_top_to_bottom 中相邻的两个节点生成 directly_overlies 和 directly_underlies。

JSON 格式：
{{
  "schema_version": "two_dimensional_stratigraphic_log.v1",
  "diagram_type": "geological_section|stratigraphic_column|seismic_profile|multiwell_correlation|single_well_log|other_2d",
  "diagram_name": "",
  "coordinate_system": {{
    "horizontal_meaning": "",
    "horizontal_unit": "",
    "vertical_meaning": "",
    "vertical_unit": "",
    "vertical_increases": "downward|upward|unknown",
    "evidence": ""
  }},
  "entities": [
    {{
      "id": "",
      "name": "图中原始名称",
      "type": "stratigraphic_unit|horizon|well|well_trajectory|fault|structure_zone|sedimentary_facies|reservoir_interval|shale_interval|seal|source_rock|log_curve|lithology|marker|other",
      "parent_id": "",
      "position": {{"bbox": [0,0,0,0], "x_order": null, "y_order": null, "top_value": null, "bottom_value": null, "unit": ""}},
      "attributes": {{}},
      "evidence": "图中可见文字、线条或区域",
      "confidence": 0.0
    }}
  ],
  "stratigraphic_sequences": [
    {{
      "id": "",
      "context_id": "regional 或 entities 中的井/父层 id",
      "ordered_unit_ids_top_to_bottom": [""],
      "evidence": "图中层界或柱状顺序",
      "confidence": 0.0
    }}
  ],
  "spatial_groups": [
    {{
      "id": "",
      "kind": "wells|structure_zones|facies_zones|other",
      "ordered_member_ids_left_to_right": [""],
      "evidence": "图中横向排列",
      "confidence": 0.0
    }}
  ],
  "relations": [
    {{
      "source_id": "",
      "type": "",
      "target_id": "",
      "dimension": "stratigraphic|spatial|structural|logging|correlation|semantic",
      "explicit": true,
      "basis": "",
      "evidence": "",
      "confidence": 0.0
    }}
  ],
  "uncertainties": [""]
}}
""".strip()
