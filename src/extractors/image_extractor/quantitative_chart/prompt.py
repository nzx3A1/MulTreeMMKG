"""定量图表 VLM 结构化抽取提示词。"""
from __future__ import annotations

import json

from ..schema_models import ImageExtractionTask


OUTPUT_SCHEMA = {
    "chart": {
        "id": "chart",
        "title": "图内标题；没有则使用图注中的标题",
        "chart_type": "scatter|line|bar|histogram|boxplot|area|heatmap|mixed|other",
        "description": "一句话客观描述",
        "confidence": 0.0,
    },
    "panels": [
        {"id": "panel_1", "label": "A", "title": "子图标题", "evidence": "图上可见证据", "confidence": 0.0}
    ],
    "axes": [
        {
            "id": "x_axis",
            "panel_id": "panel_1",
            "orientation": "x|y|secondary_x|secondary_y|color|size",
            "name": "坐标轴名称",
            "variable": "所表示变量",
            "unit": "单位；无单位为空字符串",
            "scale": "linear|log10|ln|categorical|datetime|unknown",
            "minimum": None,
            "maximum": None,
            "categories": [],
            "ticks": [],
            "evidence": "轴标题、刻度或类别标签",
            "confidence": 0.0,
        }
    ],
    "series": [
        {
            "id": "series_1",
            "panel_id": "panel_1",
            "name": "图例名称",
            "series_type": "scatter|line|bar|distribution|other",
            "visual_encoding": {"color": "", "marker": "", "line_style": ""},
            "x_axis_id": "x_axis",
            "y_axis_id": "y_axis",
            "sample_size": None,
            "data_points": [{"x": None, "y": None, "label": "", "approximate": True}],
            "statistics": {},
            "evidence": "图例、曲线、散点簇或图注",
            "confidence": 0.0,
        }
    ],
    "trends": [
        {
            "id": "trend_1",
            "series_ids": ["series_1"],
            "direction": "increasing|decreasing|stable|non_monotonic|U_shaped|inverted_U|cyclic|mixed|unknown",
            "pattern": "linear|nonlinear|step|plateau|peak|trough|clustered|scattered|other",
            "x_range": {"start": None, "end": None, "unit": ""},
            "y_range": {"start": None, "end": None, "unit": ""},
            "description": "趋势及适用范围",
            "evidence": "可在图中定位的趋势证据",
            "confidence": 0.0,
        }
    ],
    "correlations": [
        {
            "id": "correlation_1",
            "x_axis_id": "x_axis",
            "y_axis_id": "y_axis",
            "series_ids": ["series_1"],
            "direction": "positive|negative|none|nonlinear|unknown",
            "strength": "strong|moderate|weak|unknown",
            "coefficient": None,
            "method": "图中明确标注的方法；未标注为空字符串",
            "scope": "相关性成立的数据范围或分组",
            "evidence": "散点/拟合线/明确标注；不得凭正文编造系数",
            "confidence": 0.0,
        }
    ],
    "thresholds": [
        {
            "id": "threshold_1",
            "label": "阈值名称",
            "value": None,
            "unit": "",
            "axis_id": "x_axis",
            "series_ids": [],
            "condition": "<|<=|=|>=|>|range|crossing",
            "meaning": "阈值前后差异或物理含义",
            "source": "visual|caption|reference|inferred",
            "evidence": "虚线、标注、图注或正文原句摘要",
            "confidence": 0.0,
        }
    ],
    "temporal_changes": [
        {
            "id": "time_change_1",
            "series_ids": ["series_1"],
            "time_axis_id": "x_axis",
            "start_time": "",
            "end_time": "",
            "direction": "increase|decrease|stable|fluctuating|cyclic|mixed",
            "magnitude": None,
            "rate": None,
            "turning_points": [],
            "periodicity": "",
            "description": "时间变化",
            "evidence": "时间轴与序列证据",
            "confidence": 0.0,
        }
    ],
    "comparisons": [
        {
            "source_series_id": "series_1",
            "target_series_id": "series_2",
            "relation": "higher_than|lower_than|similar_to|converges_with|diverges_from|crosses|more_variable_than",
            "scope": "比较成立的区间/类别",
            "description": "定量或定性比较",
            "evidence": "图像、图注或正文证据",
            "confidence": 0.0,
        }
    ],
    "anomalies": [
        {
            "id": "anomaly_1",
            "series_ids": ["series_1"],
            "kind": "peak|trough|outlier|positive_anomaly|negative_anomaly|breakpoint|other",
            "x": None,
            "y": None,
            "label": "",
            "description": "异常特征",
            "source": "visual|caption|reference|inferred",
            "evidence": "异常证据",
            "confidence": 0.0,
        }
    ],
    "uncertainties": ["无法辨认或只能近似判断的内容"],
}


def build_quantitative_chart_prompt(task: ImageExtractionTask, *, max_points_per_series: int = 24) -> str:
    """构建严格证据约束、兼容散点/曲线/统计图的中文提示词。"""

    references = "\n".join(f"- {item}" for item in task.references) or "（无）"
    schema_text = json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)
    return f"""你是科研论文定量图表解析专家。请查看原图，抽取坐标轴、数据序列、趋势、相关性、阈值、时间变化、序列比较和异常，并只返回一个合法 JSON 对象。

图注：
{task.caption or '（无）'}

正文引用（仅作语境和辅助证据，不能覆盖原图）：
{references}

必须遵守：
1. 先识别子图、图表类型、图例、坐标轴标题、单位、刻度和尺度。特别检查对数轴、分类轴、时间轴及双 Y 轴；不得把等距绘制的 0.01/0.1/1/10 误判为线性轴。
2. 每条曲线、散点组、柱组或分布组分别建立 series。series/axis/panel 的引用必须使用同一 JSON 中存在的 id。
3. data_points 只记录图中明确标注的值或少量有代表性的可读点，每个序列最多 {max(0, int(max_points_per_series))} 个；目测值必须 approximate=true，不要伪造高精度，不要试图枚举密集散点。sample_size 只表示图注、图例或图内明确写出的 N/n 样本量；曲线上的类别数或数据点数不是 sample_size，没有 N/n 证据时必须为 null。
4. 趋势和相关性必须写明适用的序列与范围。图中没有相关系数时 coefficient=null；不能根据视觉印象编造 Pearson/Spearman 系数或显著性。
5. threshold 可来自图中明确线/标注，也可来自图注或正文，但 source 必须准确写 visual/caption/reference/inferred。仅仅存在坐标轴刻度不等于存在阈值。
6. 只有当横轴确实是时间/年代/实验阶段时才输出 temporal_changes，否则返回空数组。分类序列（例如元素名称）不是时间变化。
7. 正文中的机理解释可用于 meaning/description，但证据字段需注明“正文引用”；与图像冲突时以图像为准并写入 uncertainties。
8. 若图中存在多系列对比，输出 comparisons；若存在峰、谷、离群点、正负异常或拐点，输出 anomalies。无法可靠识别则留空，不猜测。
9. confidence 是 0~1。evidence 应短而可核验。所有缺失列表用 []，缺失对象用 {{}}，缺失数值用 null，不要用 NaN/Infinity。
10. 输出字段必须符合下列结构；可以少填对象内部的可选字段，但不得另加解释文字或 Markdown 围栏：
{schema_text}
"""


__all__ = ["OUTPUT_SCHEMA", "build_quantitative_chart_prompt"]
