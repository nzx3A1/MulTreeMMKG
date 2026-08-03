"""表格—图像嵌入混合型地层图的专用视觉提示词。"""
from __future__ import annotations

import json
from typing import Any, Mapping

from ...schema_models import ImageExtractionTask
from ..subclassifier import StratigraphicSubtypeClassification
from .ppstructure_geometry import geometry_prompt_catalog


def build_table_embedded_hybrid_prompt(
    task: ImageExtractionTask,
    classification: StratigraphicSubtypeClassification,
    geometry: Mapping[str, Any] | None = None,
) -> str:
    """中文说明：要求 VLM 只选择 PP 几何 ID 并补充语义，不允许模型生成像素坐标。"""

    catalog = geometry_prompt_catalog(geometry) if geometry is not None else {
        "coordinate_source": "PP-StructureV3 将在调用前生成",
        "tracks": [],
        "cells": [],
        "ocr_lines": [],
    }
    catalog_json = json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))

    return f"""
你正在解析一张“表格—图片嵌入混合型”地层/综合柱状/测井图片。

来源图片：{task.image_path}
图片 ID：{task.image_id}
Chunk ID：{task.chunk_id}
图题：{task.caption or '无'}
子分类：{classification.subtype.value}（{classification.subtype_name}）
分类依据：{classification.evidence}
抽取算法：ppstructurev3_geometry_semantic_vlm_depth_alignment_graph_assembly

PP-StructureV3 几何目录（其中的 ID 和像素框由程序生成；你只选择 ID，不生成坐标）：
{catalog_json}

执行边界：
1. 只读取图片中可见的表头、刻度、表格线、合并单元格、岩性柱、曲线、色条和文字；不得凭花纹猜岩性，不凭常识补值。
2. 禁止输出或估算 content_bbox、bbox、pixel_y、top_y、bottom_y；这些字段全部由 PP-StructureV3 程序生成。
3. tracks 只能从目录 tracks 逐字选择 id，再补充 role、header 和 parser；不得另造轨道 ID。
4. 区间和点图元用 geometry_refs 选择目录中的 pp_cell_* 或 pp_ocr_*；只保留 track_id、evidence、confidence 和地质语义。
5. 曲线只在刻度和单位清楚时填写数值；不清楚时保留 qualitative_response 并写入 uncertainties。
6. 关系不由模型输出。后续程序会用统一深度轴做区间重叠、层序和图谱装配。

只返回一个 JSON 对象，结构必须为：
{{
  "schema_version": "table_embedded_hybrid.v1",
  "diagram_id": "图内稳定局部ID",
  "diagram_name": "图名",
  "image_size": {{"width": 0, "height": 0}},
  "coordinate_system": {{
    "vertical_axis": {{
      "kind": "depth|thickness|relative_sequence",
      "unit": "m|无量纲",
      "increases": "downward|upward",
      "track_id": "必须来自 pp_track_*",
      "calibration_ocr_ids": ["可见连续深度刻度对应的 pp_ocr_*；厚度单元格不能冒充连续深度轴"]
    }}
  }},
  "tracks": [
    {{"id": "目录中的 pp_track_*", "role": "stratigraphy|depth|lithology|curve|text|facies|reservoir|well", "header": "表头", "parser": "专用解析器名", "evidence": "可见依据"}}
  ],
  "primitives": {{
    "stratigraphic_intervals": [{{"id":"", "name":"", "parent_id":"", "rank":"", "track_id":"pp_track_*", "geometry_refs":["pp_cell_*"], "evidence":"", "confidence":0.0}}],
    "reference_intervals": [{{"id":"", "name":"井段或相对层序段", "entity_type":"depth_interval", "track_id":"pp_track_*", "geometry_refs":["pp_cell_*"], "evidence":"", "confidence":0.0}}],
    "lithology_intervals": [{{"id":"", "name":"", "track_id":"pp_track_*", "geometry_refs":["pp_cell_*"], "evidence":"", "confidence":0.0}}],
    "facies_intervals": [{{"id":"", "name":"", "track_id":"pp_track_*", "geometry_refs":["pp_cell_*"], "evidence":"", "confidence":0.0}}],
    "curve_tracks": [{{"id":"", "name":"", "track_id":"", "scale_min":null, "scale_max":null, "unit":"", "scale_direction":"", "evidence":""}}],
    "curve_observations": [{{"id":"", "name":"", "curve_ids":[], "track_id":"pp_track_*", "geometry_refs":["pp_cell_*"], "qualitative_response":"", "evidence":"", "confidence":0.0}}],
    "reservoir_intervals": [{{"id":"", "name":"", "track_id":"pp_track_*", "geometry_refs":["pp_cell_*"], "evidence":"", "confidence":0.0}}],
    "oil_layer_intervals": [],
    "geological_feature_intervals": [],
    "point_markers": [{{"id":"", "name":"", "kind":"well", "track_id":"pp_track_*", "geometry_refs":["pp_cell_*","pp_ocr_*"], "evidence":"", "confidence":0.0}}],
    "objects": [{{"id":"", "name":"", "entity_type":"well|oil_layer_group|study_marker|geological_object", "evidence":"", "confidence":0.0}}],
    "explicit_relations": [{{"source_id":"", "relation_type":"", "target_id":"", "evidence":"", "confidence":0.0}}]
  }},
  "uncertainties": []
}}
""".strip()
