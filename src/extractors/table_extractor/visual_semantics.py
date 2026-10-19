"""使用表格原图校验记录方向、表头和主题字段。"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from src.utils.llm_client import safe_json_loads

from .schema_models import RecognizedTable


def build_table_visual_semantic_prompt(table: RecognizedTable) -> str:
    """构造只要求 VLM 判断表格布局语义的受约束提示词。"""

    grid = table.grid
    html_context = {
        "row_count": grid.row_count,
        "column_count": grid.column_count,
        "html_inferred_headers": grid.header_paths,
        "expanded_matrix_preview": grid.matrix[:12],
    }
    return f"""你是表格结构分析器。请同时观察原始表格图片和下方 HTML 展开矩阵，判断记录方向、完整表头和记录主题字段。

定义：
- horizontal：顶部是列头，每一行是一条记录；subject_header 是用来给该行记录命名的列头。
- vertical：左侧是行头，每一列是一条记录；subject_header 是用来给该列记录命名的行头。
- headers 必须按数据轴顺序返回最终字段名。横向表按列返回，纵向表按行返回。
- HTML 逻辑网格共有 {grid.column_count} 列、{grid.row_count} 行；横向表必须返回恰好 {grid.column_count} 个 headers，纵向表必须返回恰好 {grid.row_count} 个 headers。即使末尾是 OCR 多识别的空列，也保留其占位字段名（例如“列9”），不要移动其他字段的索引。
- subject_index 是 subject_header 在原始 HTML 逻辑网格中的索引：横向表为列索引，纵向表为行索引，从 0 开始。
- data_start_index 是第一条记录的索引：横向表为行索引，纵向表为列索引，从 0 开始。
- 只依据图片中可见内容判断，不补写图片中不存在的字段。

HTML 辅助信息（rowspan/colspan 已展开，仅用于对齐索引，若文字与图片冲突以图片为准）：
{json.dumps(html_context, ensure_ascii=False)}

只返回一个 JSON 对象，禁止 Markdown 和解释文字：
{{
  "orientation": "horizontal 或 vertical",
  "headers": ["字段1", "字段2"],
  "subject_header": "作为记录名字的字段",
  "subject_index": 0,
  "data_start_index": 1,
  "confidence": 0.0,
  "reason": "简短视觉依据"
}}"""


def _orientation(value: Any) -> str:
    """兼容中英文方向写法并限制为两种记录方向。"""

    normalized = str(value or "").strip().lower().replace("_", "")
    if normalized in {"horizontal", "横向", "横表", "按行", "row", "rows"}:
        return "horizontal"
    if normalized in {"vertical", "纵向", "竖向", "竖表", "按列", "column", "columns"}:
        return "vertical"
    raise ValueError(f"VLM 返回了不支持的表格方向：{value!r}")


def _clean_headers(value: Any) -> list[str]:
    """把 VLM 表头结果规范为有序非空字符串列表。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("VLM headers 必须是字符串数组")
    headers = [" ".join(str(item or "").split()) for item in value]
    if not headers or any(not header for header in headers):
        raise ValueError("VLM headers 不能为空")
    return headers


def _align_headers(table: RecognizedTable, orientation: str, headers: list[str]) -> tuple[list[str], str]:
    """仅为原图表头为空的尾部虚列补齐占位名，保持 VLM 与 HTML 的列索引一致。"""

    axis_size = table.grid.column_count if orientation == "horizontal" else table.grid.row_count
    if len(headers) == axis_size:
        return headers, "exact"
    if orientation != "horizontal" or not 0 < len(headers) < axis_size:
        raise ValueError(f"VLM headers 数量 {len(headers)} 与 {orientation} 表字段轴长度 {axis_size} 不一致")
    first_row = table.grid.matrix[0] if table.grid.matrix else []
    if any(str(first_row[index] or "").strip() for index in range(len(headers), axis_size)):
        raise ValueError("VLM 漏掉了有内容的表头列，无法安全补齐索引")
    padding = [
        table.grid.header_paths[index] if index < len(table.grid.header_paths) else f"列{index + 1}"
        for index in range(len(headers), axis_size)
    ]
    return headers + padding, "trailing_blank_header_columns"


def analyze_table_visual_semantics(
    table: RecognizedTable,
    vlm_client: Any,
) -> dict[str, Any]:
    """调用一次 VLM，并返回经过边界校验的表格视觉语义。"""

    image_path = table.source.image_path
    if not image_path:
        raise ValueError(f"表格 {table.source.task_id} 没有原图，无法执行 VLM 视觉语义判断")
    response = vlm_client.describe_image(
        image_path,
        build_table_visual_semantic_prompt(table),
        task_name="表格方向、表头与主题字段识别",
    )
    payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
    if not isinstance(payload, Mapping):
        raise ValueError("VLM 表格语义响应必须是 JSON 对象")

    orientation = _orientation(payload.get("orientation"))
    headers, header_alignment = _align_headers(table, orientation, _clean_headers(payload.get("headers")))
    axis_size = table.grid.column_count if orientation == "horizontal" else table.grid.row_count
    subject_index = int(payload.get("subject_index"))
    data_start_index = int(payload.get("data_start_index"))
    if not 0 <= subject_index < axis_size:
        raise ValueError(f"VLM subject_index={subject_index} 超出逻辑网格范围 0..{axis_size - 1}")
    record_axis_size = table.grid.row_count if orientation == "horizontal" else table.grid.column_count
    if not 0 <= data_start_index < record_axis_size:
        raise ValueError(
            f"VLM data_start_index={data_start_index} 超出记录轴范围 0..{record_axis_size - 1}"
        )
    subject_header = " ".join(str(payload.get("subject_header") or "").split())
    if not subject_header:
        subject_header = headers[subject_index] if subject_index < len(headers) else ""
    confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
    return {
        "orientation": orientation,
        "headers": headers,
        "subject_header": subject_header,
        "subject_index": subject_index,
        "data_start_index": data_start_index,
        "confidence": confidence,
        "header_alignment": header_alignment,
        "reason": str(payload.get("reason") or "").strip(),
        "raw_response": dict(payload),
    }


__all__ = ["analyze_table_visual_semantics", "build_table_visual_semantic_prompt"]
