"""表格—图像嵌入混合型地层图的专用视觉提示词。"""
from __future__ import annotations

from ...schema_models import ImageExtractionTask
from ..subclassifier import StratigraphicSubtypeClassification


def build_table_embedded_hybrid_prompt(
    task: ImageExtractionTask,
    classification: StratigraphicSubtypeClassification,
) -> str:
    """中文说明：要求 VLM 只读出几何与语义图元，跨轨道关系交给确定性深度对齐生成。"""

    return f"""
你正在解析一张“表格—图片嵌入混合型”地层/综合柱状/测井图片。

来源图片：{task.image_path}
图片 ID：{task.image_id}
Chunk ID：{task.chunk_id}
图题：{task.caption or '无'}
子分类：{classification.subtype.value}（{classification.subtype_name}）
分类依据：{classification.evidence}
抽取算法：coordinate_track_specialized_depth_alignment_graph_assembly

执行边界：
1. 只读取图片中可见的表头、刻度、表格线、合并单元格、岩性柱、曲线、色条和文字；不得凭花纹猜岩性，不凭常识补值。
2. 像素坐标使用原图坐标，原点在左上角；纵轴至少给出两个可见刻度 calibration_points。
3. tracks 按从左到右列出，每个轨道必须有 id、role、header、bbox=[x0,y0,x1,y1] 和 parser。
4. 各区间图元保留 top_y、bottom_y、track_id、evidence、confidence；不要自行把横向相邻解释成地质关系。
5. 曲线只在刻度和单位清楚时填写数值；不清楚时保留 qualitative_response 并写入 uncertainties。
6. 关系不由模型输出。后续程序会用统一深度轴做区间重叠、层序和图谱装配。

只返回一个 JSON 对象，结构必须为：
{{
  "schema_version": "table_embedded_hybrid.v1",
  "diagram_id": "图内稳定局部ID",
  "diagram_name": "图名",
  "image_size": {{"width": 0, "height": 0}},
  "coordinate_system": {{
    "content_bbox": [0, 0, 0, 0],
    "vertical_axis": {{
      "kind": "depth|thickness|relative_sequence",
      "unit": "m|无量纲",
      "increases": "downward|upward",
      "calibration_points": [{{"pixel_y": 0, "value": 0, "evidence": "可见刻度"}}]
    }}
  }},
  "tracks": [
    {{"id": "track_id", "order": 0, "role": "stratigraphy|depth|lithology|curve|text|facies|reservoir|well", "header": "表头", "bbox": [0,0,0,0], "parser": "专用解析器名", "evidence": "可见依据"}}
  ],
  "primitives": {{
    "stratigraphic_intervals": [{{"id":"", "name":"", "parent_id":"", "rank":"", "track_id":"", "top_y":0, "bottom_y":0, "evidence":"", "confidence":0.0}}],
    "reference_intervals": [{{"id":"", "name":"井段或相对层序段", "entity_type":"depth_interval", "track_id":"", "top_y":0, "bottom_y":0, "evidence":"", "confidence":0.0}}],
    "lithology_intervals": [{{"id":"", "name":"", "track_id":"", "top_y":0, "bottom_y":0, "evidence":"", "confidence":0.0}}],
    "facies_intervals": [{{"id":"", "name":"", "track_id":"", "top_y":0, "bottom_y":0, "evidence":"", "confidence":0.0}}],
    "curve_tracks": [{{"id":"", "name":"", "track_id":"", "scale_min":null, "scale_max":null, "unit":"", "scale_direction":"", "evidence":""}}],
    "curve_observations": [{{"id":"", "name":"", "curve_ids":[], "track_id":"", "top_y":0, "bottom_y":0, "qualitative_response":"", "evidence":"", "confidence":0.0}}],
    "reservoir_intervals": [{{"id":"", "name":"", "track_id":"", "top_y":0, "bottom_y":0, "evidence":"", "confidence":0.0}}],
    "oil_layer_intervals": [],
    "geological_feature_intervals": [],
    "point_markers": [{{"id":"", "name":"", "kind":"well", "track_id":"", "pixel_y":0, "evidence":"", "confidence":0.0}}],
    "objects": [{{"id":"", "name":"", "entity_type":"well|oil_layer_group|study_marker|geological_object", "evidence":"", "confidence":0.0}}],
    "explicit_relations": [{{"source_id":"", "relation_type":"", "target_id":"", "evidence":"", "confidence":0.0}}]
  }},
  "uncertainties": []
}}
""".strip()
