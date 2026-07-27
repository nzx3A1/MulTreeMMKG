"""地质过程与成因模式图的视觉抽取及关系审查 Prompt。"""
from __future__ import annotations

import json

from ..schema_models import ImageExtractionTask


def _diagram_focus(task: ImageExtractionTask) -> tuple[str, str]:
    """中文说明：按图片分类注入专用观察清单，避免提示词被某一张样例图硬编码。"""

    if task.classification_code == "A08":
        return (
            "油气藏、成藏与富集模式图",
            """
- 按图中从左到右的顺序识别不同构造分段，重点区分走滑压隆、走滑平移、走滑拉分等样式；名称必须以图中文字为准。
- 把井名、断裂、裂缝、角砾岩/断溶体、断裂破碎带分别建模。SB1、SB1-8 这类井号只可作为 well，除非原图明确写出“××断裂”，不得据井号创造同名断裂。
- 建立烃源岩、储集空间、盖层、输导通道、聚集部位五类油气系统角色；角色可来自图题或正文，但必须在 evidence 中说明来源。
- 抽取“供烃—成储—运移—聚集—富集控制”过程链，尤其保留沿断裂的垂向连通、断裂切层及不同分段的储集样式差异。
""".strip(),
        )
    return (
        "沉积、成岩与孔隙演化模式图",
        """
- 识别沉积环境、流体、古地貌、成岩作用和孔隙演化阶段。
- 逐一读取过程箭头的起点、路径和终点，抽取沉积、蒸发浓缩、回流渗透、白云石化等图中实际出现的过程。
- 对同一层段的横向岩性变化、局部透镜体和夹层分别记录，不能用局部填充代表整个层段。
""".strip(),
    )


def build_visual_extraction_prompt(task: ImageExtractionTask) -> str:
    """中文说明：生成证据优先的分类专用视觉 Prompt，同时识别层段、岩性、时空关系与过程。"""

    diagram_type, specialist_checklist = _diagram_focus(task)
    references = "\n".join(task.references)
    return f"""
你是石油地质多模态知识图谱专家。请分析一张“{diagram_type}”。
任务不是普通图片描述，而是重建可验证的地质时空结构、构造分段和过程图。

图题：
{task.caption}

正文参考（仅用于消歧或补充机理；不能覆盖图片中相反的证据，正文关系的 explicit 必须为 false）：
{references}

本类型专项观察：
{specialist_checklist}

必须完成：
1. 读取图例中的全部岩石类型及其颜色/填充样式；断裂、裂缝和井符号不是岩性，必须另建实体。
2. 读取图内全部地层/层段标签，按从下到上给出 order_bottom_to_top，最下部为 0。若标签只表示一个大套地层，不得臆造其亚层。
3. 对每个层段逐一匹配可见岩性。薄层、夹层、局部透镜体或断裂带充填不能写成整个层段的主要岩性；横向变化写入 lateral_zones。
4. 识别剖面方向、构造位置、不同构造分段、垂向叠置、横向邻接、断裂切层和断裂连通关系。
5. 逐一识别井名、断裂/裂缝、断溶体或储集体以及所有箭头的起点、路径、终点和方向。
6. 抽取事件的相对时间和因果链。图中直接可见或文字明示的关系 explicit=true；仅由正文或地质规律解释的关系 explicit=false。
7. 每个关系端点只能引用本 JSON 已定义的 id。看不清的字符不得猜测，写入 uncertainties。
8. 只输出 JSON，不要 Markdown；每个判断都必须带具体 evidence 和 0~1 confidence。
9. 控制输出规模：evidence 每项不超过 40 个汉字；不输出互为反向的重复关系；spatial_relations 最多 24 条、temporal_relations 最多 10 条、causal_relations 最多 12 条。
10. 第一轮岩性只做可靠的粗匹配，无法确认时留空；后续会用第二次视觉调用专项复核，不能为了填满字段而猜测。

关系类型优先使用：
- 空间/构造：above, below, directly_overlies, directly_underlies, within, contains, adjacent_to,
  west_of, east_of, lateral_transition_to, located_in, cuts_through, offsets, vertically_connects, connected_to
- 流体/过程：flows_from, flows_to, flows_through, migrates_along, accumulates_in
- 时间：before, after, contemporaneous_with, older_than, younger_than
- 因果/控制：causes, enables, controls, transforms_into, dolomitizes, concentrates, evaporates_from

JSON 格式：
{{
  "diagram_summary": "",
  "coordinate_system": {{
    "section_line": "剖面线或空",
    "horizontal_direction": "",
    "vertical_meaning": "",
    "normal_stratigraphic_order": true,
    "evidence": ""
  }},
  "legend_lithologies": [
    {{"id": "lith_1", "name": "岩性名", "visual_pattern": "颜色/纹理", "evidence": "", "confidence": 0.0}}
  ],
  "stratigraphic_units": [
    {{
      "id": "unit_1", "name": "图中原始层段标签", "rank": "组/段/亚段/层", "order_bottom_to_top": 0,
      "geometry": "图中位置、厚度和横向变化",
      "lithologies": [
        {{
          "lithology_id": "lith_1", "name": "岩性名", "role": "主要/次要/夹层/局部/不确定",
          "lateral_variation": "总体横向变化概述",
          "lateral_zones": [{{"zone": "左/中/右或构造分段id", "role": "主要/次要/夹层/局部", "evidence": ""}}],
          "evidence": "颜色/纹理和层段内位置", "confidence": 0.0
        }}
      ],
      "evidence": "层段标签及边界", "confidence": 0.0
    }}
  ],
  "entities": [
    {{
      "id": "obj_1", "name": "图中地质对象或井名",
      "type": "fluid/location/structure/fault/fracture/reservoir_body/source_rock/well/environment/facies/process_material",
      "attributes": {{}}, "evidence": "", "confidence": 0.0
    }}
  ],
  "structural_segments": [
    {{
      "id": "segment_1", "name": "图中分段名称", "style": "构造样式", "order_left_to_right": 0,
      "geometry": "断裂几何、花状构造、破碎范围等",
      "structures": ["entities 中的断裂/储集体 id"],
      "wells": [{{"id": "well_1", "name": "井名", "evidence": "", "confidence": 0.0}}],
      "reservoir_characteristics": "该分段储集空间和富集样式",
      "evidence": "", "confidence": 0.0
    }}
  ],
  "petroleum_system": {{
    "name": "本图油气成藏系统",
    "source_rocks": [{{"id": "对象id", "evidence": "", "confidence": 0.0}}],
    "reservoirs": [{{"id": "对象id", "evidence": "", "confidence": 0.0}}],
    "seals": [{{"id": "对象id", "evidence": "", "confidence": 0.0}}],
    "migration_paths": [{{"id": "对象id", "evidence": "", "confidence": 0.0}}],
    "accumulation_zones": [{{"id": "对象id", "evidence": "", "confidence": 0.0}}],
    "evidence": ""
  }},
  "spatial_relations": [
    {{"source": "id", "type": "关系类型", "target": "id", "coordinate_frame": "section/stratigraphic/structural/flow_path", "explicit": true, "evidence": "", "confidence": 0.0}}
  ],
  "temporal_relations": [
    {{"source": "事件或对象id", "type": "before/after/contemporaneous_with/older_than/younger_than", "target": "id", "explicit": true, "basis": "图中箭头/正文/地层叠置律", "evidence": "", "confidence": 0.0}}
  ],
  "causal_relations": [
    {{"source": "id", "type": "causes/enables/controls/transforms_into/dolomitizes/concentrates", "target": "id", "explicit": true, "evidence": "", "confidence": 0.0}}
  ],
  "process_events": [
    {{
      "id": "event_1", "name": "过程名称", "type": "generation/reservoir_formation/migration/accumulation/enrichment/deposition/diagenesis/other",
      "participants": ["对象id"], "source": "对象id或空", "path": ["对象id"], "target": "对象id或空",
      "time_stage": "相对时间阶段", "location": "发生位置", "result": "结果",
      "explicit": true, "evidence": "图中文字、箭头或正文", "confidence": 0.0
    }}
  ],
  "uncertainties": ["无法可靠确定的内容"]
}}
""".strip()


def build_relation_audit_prompt(task: ImageExtractionTask, visual_result: dict) -> str:
    """中文说明：让文本模型审查视觉结果，只补有证据的时空、构造和因果关系。"""

    return f"""
你是石油地质知识图谱关系审查专家。下面是视觉模型对“{task.classification_type}”的结构化识别结果。
请检查层段岩性、构造分段、源储盖角色、流体路径、事件先后和因果关系是否自洽。

图题：{task.caption}
正文参考：{json.dumps(list(task.references), ensure_ascii=False)}
视觉识别：
{json.dumps(visual_result, ensure_ascii=False, indent=2)}

约束：
1. 不能创造视觉结果、图题或正文均未支持的新实体；不能修改实体或层段 id。
2. 相关性不能升级为 causes。只有明确机理、过程箭头或正文陈述才能使用因果/控制关系。
3. 地层 order_bottom_to_top 只允许推导空间叠置；推导新老关系时必须声明 normal_stratigraphic_order 假设。
4. 区分结构关系和过程关系：断裂切穿层段用 cuts_through；油气沿断裂运移用 migrates_along；聚集于储集体用 accumulates_in。
5. A08 图必须审查“供烃→成储→运移→聚集/富集”的依赖链和断裂分段差异；A09 图必须审查成岩过程的先后与物质转化。
6. added_relations 的 source/target 必须引用视觉结果中的实体、层段、构造分段、岩性或事件 id。
7. 每条新增关系必须提供 dimension、basis、evidence、explicit 和 confidence。正文补充关系 explicit=false。
8. 仅输出 JSON。

JSON 格式：
{{
  "accepted": true,
  "added_relations": [
    {{"source": "id", "type": "关系类型", "target": "id", "dimension": "spatial/temporal/causal/process/structural", "explicit": false, "basis": "", "evidence": "", "confidence": 0.0}}
  ],
  "event_dependencies": [
    {{"before": "event_id", "after": "event_id", "basis": "", "evidence": "", "confidence": 0.0}}
  ],
  "warnings": ["审查发现的不确定性"]
}}
""".strip()


def build_lithology_audit_prompt(task: ImageExtractionTask, visual_result: dict) -> str:
    """中文说明：生成动态的第二次视觉核对 Prompt，纠正任意样例的层段与岩性误匹配。"""

    legend = visual_result.get("legend_lithologies", [])
    units = visual_result.get("stratigraphic_units", [])
    return f"""
你是石油地质剖面图视觉核对专家。请再次查看原图，只审查“图例岩性”和“每个层段包含哪些岩性”。

图题：{task.caption}
图片类型：{task.classification_code} {task.classification_type}
第一次识别的图例：
{json.dumps(legend, ensure_ascii=False, indent=2)}
第一次识别的层段：
{json.dumps(units, ensure_ascii=False, indent=2)}

必须逐项执行：
1. 独立重读原图底部/侧边图例中的全部岩性名称、颜色和纹理；断裂、裂缝、井、箭头、油气显示等符号不得列入 legend_lithologies。
2. 逐段观察第一次识别出的每个层段的整个横向范围，只保留边界和标签在原图中确实可见的层段。
3. 逐字符抄写地层代号并保留原符号；重点复核寒武系“Є”与石炭系“C”、字母 y 与 v、字母 l 与数字 1、yj 与 y1，不得把相似字形自动替换。
4. 对每一种岩性说明它位于层段的哪一横向区、构造分段或局部位置；用 lateral_zones 保留横向差异。
5. 层段旁明确写出的“泥质岩”等文字优先于颜色判断。“烃源岩”是油气系统角色，不是具体岩性；若图中没有写页岩/泥岩等名称，不得据常识猜测。
6. 薄层、夹层、局部透镜体、断裂破碎带充填和角砾岩体不能写成整个层段的“主要岩性”。
7. 一个层段存在多种填充时必须全部列出，并标记主要、次要、夹层、局部或不确定。
8. 岩性必须按纹理符号匹配，颜色只作辅助；evidence 要描述颜色、纹理、所在层段和横向位置，不能只写“根据图例”。
9. 保持第一次结果中的层段 id 和 order_bottom_to_top，不得新增层段；若某个层段标签或边界无法辨认，保留它并降低 confidence、写入 uncertainties。
10. 岩性 id 优先复用复核后的 legend_lithologies id。仅输出 JSON。

JSON 格式：
{{
  "legend_lithologies": [
    {{"id": "岩性id", "name": "岩性", "visual_pattern": "颜色/纹理", "evidence": "图例位置", "confidence": 0.0}}
  ],
  "stratigraphic_units": [
    {{
      "id": "原层段id", "name": "层段名", "rank": "组/段/亚段/层", "order_bottom_to_top": 0,
      "geometry": "图中位置和横向变化",
      "lithologies": [
        {{
          "lithology_id": "复核图例岩性id", "name": "岩性", "role": "主要/次要/夹层/局部/不确定",
          "lateral_variation": "横向变化概述",
          "lateral_zones": [{{"zone": "左/中/右或构造分段id", "role": "", "evidence": ""}}],
          "evidence": "颜色/纹理、层段和横向位置", "confidence": 0.0
        }}
      ],
      "evidence": "层段标签位置", "confidence": 0.0
    }}
  ],
  "corrections": [{{"unit_id": "", "previous": "", "corrected": "", "reason": ""}}],
  "uncertainties": [""]
}}
""".strip()
