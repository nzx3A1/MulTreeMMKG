"""地图空间抽取器三阶段 VLM 提示词。

三个提示词分别负责图例、实体与语义候选，地点方向不交给 VLM 生成。
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from ..schema_models import ImageExtractionTask


_FOCUS_BY_CODE = {
    "A01": "区域位置、盆地或构造分区、边界、井位与油气田",
    "A02": "岩性、沉积微相、古地理单元、相带边界与横向过渡",
    "A07": "储层参数、单位、等值线、数值分区、井位与空间趋势",
    "A18": "井、油气田、有利区、预测边界、评价属性与空间展布",
}


def _task_context(task: ImageExtractionTask) -> str:
    """生成三个阶段共享的图片上下文。"""

    return (
        f"图片ID：{task.image_id}\n"
        f"分类：{task.classification_code or '未提供'} {task.classification_type or ''}\n"
        f"图题：{task.caption or '无'}\n"
        f"章节：{task.section_title or '无'}\n"
        f"正文参考：{json.dumps(list(task.references), ensure_ascii=False)}\n"
        f"判读重点：{_FOCUS_BY_CODE.get(task.classification_code, '地图实体、图例与空间结构')}"
    )


def build_legend_prompt(task: ImageExtractionTask) -> str:
    """构造第一阶段图例解析提示词。"""

    return f"""
你是石油地质地图图例解析专家。当前只执行阶段1“图例解析”，不要抽取图中实例，不要输出实体关系。

{_task_context(task)}

要求：
1. 先识别图名、地图类型、比例尺、指北信息和地图主题。
2. 逐项读取真实可见图例，区分填色、纹理、点符号和线型；颜色相近不能臆测为同类。
3. semantic_type 使用小写 snake_case，优先使用 lithology、sedimentary_microfacies、well、place、
   paleogeographic_landmass_class、geological_boundary、reservoir_property、parameter_contour_set。
4. 图例只是类别，不是图中实例。程序会把每个图例物化为独立图例节点，再与图中实例关联；未知语义类型写 UNMAPPED。
5. 岩性、沉积微相、古陆类别等有知识意义的图例概念设置 emit_entity=true；井位、地名等符号图例设为 false，程序会用 legend_class 类型保存。
6. 只输出一个 JSON 对象，不要 Markdown。

JSON：
{{
  "schema_version": "map_spatial.intermediate.v1",
  "map": {{
    "id": "map", "name": "地图题名", "type": "map_spatial", "type_zh": "具体中文图类",
    "attributes": {{"map_type": "", "scale": "", "orientation": "", "description": "", "classification_code": "{task.classification_code}"}},
    "evidence": "图题或整幅图证据", "confidence": 0.0
  }},
  "legend_items": [
    {{
      "legend_id": "legend_001", "label": "图例原文", "semantic_type": "类型",
      "type_zh": "类型中文名", "visual_encoding": {{"fill_color": "", "pattern": "", "symbol": "", "line_style": ""}},
      "emit_entity": false, "binding_relation": "可选关系名", "evidence": "", "confidence": 0.0
    }}
  ],
  "uncertainties": []
}}
""".strip()


def build_entity_prompt(task: ImageExtractionTask, legend_result: Mapping[str, Any]) -> str:
    """构造第二阶段实体、几何与图例绑定提示词。"""

    legends = json.dumps(legend_result.get("legend_items") or [], ensure_ascii=False)
    return f"""
你是石油地质地图实体定位专家。当前执行阶段2“实体抽取与图例绑定”。

{_task_context(task)}

阶段1图例：{legends}

要求：
1. 根据图例找出图中真实实例。重点包括地点、井、油气田、沉积微相、古地理/构造单元、储层参数区、剖面线和边界。
2. 每个实体必须有稳定的英文小写 id、name、type、type_zh、evidence、confidence。
3. 每个可定位实体必须给 geometry。坐标以图片左上角为 [0,0]、右下角为 [1000,1000]：
   点给 center；面给 polygon（至少3点）并可给 bbox；线给 kind=line 和 points（至少2点）。
4. legend_bindings 只引用阶段1 legend_id 与本次 entity id；绑定错误或不确定时不要强行绑定。
5. 仅将值得进入稀疏方位骨架的实体列入 spatial_mapping.candidate_entities；通常只选 place。
   若地图确实以古陆、油气田等命名空间对象为核心，可在 candidate_entity_types 中明确增加其类型。
6. spatial_anchor 选择最核心实体。不要输出任何 north_of/east_of 等方向，方向由程序计算。
7. 未知类型写 UNMAPPED，并在 attributes.raw_type 保留原始判断。只输出 JSON，不要 Markdown。

JSON：
{{
  "entities": [
    {{
      "id": "place_jingbian", "name": "靖边", "type": "place", "type_zh": "地名",
      "geometry": {{"kind": "point", "center": [481,510], "bbox": [451,490,511,530]}},
      "attributes": {{"position": "图区中部"}}, "evidence": "", "confidence": 0.0
    }}
  ],
  "legend_bindings": [{{"legend_id": "legend_place", "entity_ids": ["place_jingbian"], "confidence": 0.0}}],
  "spatial_mapping": {{
    "enabled": true,
    "candidate_entity_types": ["place"],
    "candidate_entities": [{{"entity_id": "place_jingbian", "priority": 1.0, "reason": "研究区中心地名"}}],
    "spatial_anchor": "place_jingbian"
  }},
  "uncertainties": []
}}
""".strip()


def build_relation_prompt(task: ImageExtractionTask, state: Mapping[str, Any], relation_schema: Mapping[str, Any]) -> str:
    """构造第五阶段语义关系候选提示词。"""

    entities = [
        {"id": item.get("id"), "name": item.get("name"), "type": item.get("type")}
        for item in state.get("entities") or []
        if isinstance(item, Mapping)
    ]
    direction_relations = {
        "north_of", "south_of", "east_of", "west_of",
        "northeast_of", "northwest_of", "southeast_of", "southwest_of",
    }
    pair_candidates: list[dict[str, Any]] = []
    for item in relation_schema.get("pairs") or []:
        if not isinstance(item, Mapping):
            continue
        allowed = [value for value in item.get("allowed_relations") or [] if value not in direction_relations]
        if allowed:
            pair_candidates.append({**dict(item), "allowed_relations": allowed})
    return f"""
你是石油地质地图语义关系审核专家。当前执行阶段5“关系候选判定”，程序将在之后验证几何并建图。

{_task_context(task)}

实体：{json.dumps(entities, ensure_ascii=False)}
图例绑定：{json.dumps(state.get('legend_bindings') or [], ensure_ascii=False)}
允许的具体实体对与关系：{json.dumps(pair_candidates, ensure_ascii=False)}

要求：
1. 只能从给定实体对及 allowed_relations 中选择，不能创造关系名或交换端点。
2. 图例节点与实体之间的 instance_of、has_dominant_lithology 等绑定关系由程序根据 legend_bindings 自动生成，不要重复输出。
3. 语义关系（如 has_facies、maps_property）由你判断是否值得保留。
4. surrounds、overlaps、adjacent_to 等只作为语义候选，程序将用几何确认；证据不足不要输出。
5. 井/地点落入面、剖面线穿区等可直接由几何计算，不必为了凑数输出。
6. 严禁输出地点或其他空间候选之间的八方位关系；方向由程序依据中心坐标生成。
7. 关系要稀疏、直接、无重复。只输出 JSON，不要 Markdown。

JSON：
{{
  "relation_candidates": [
    {{"source_id": "id", "type": "allowed_relation", "target_id": "id", "evidence": "", "confidence": 0.0, "attributes": {{}}}}
  ],
  "uncertainties": []
}}
""".strip()


def build_map_spatial_prompt(task: ImageExtractionTask) -> str:
    """保留旧公开入口，返回新三阶段流程的第一阶段提示词。"""

    return build_legend_prompt(task)


__all__ = [
    "build_entity_prompt",
    "build_legend_prompt",
    "build_map_spatial_prompt",
    "build_relation_prompt",
]
