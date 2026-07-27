"""按轨道组分段调用 VLM，避免复杂地层表格单次长响应超时。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from src.utils.llm_client import safe_json_loads

from ...schema_models import ImageExtractionTask
from ..subclassifier import StratigraphicSubtypeClassification
from .pipeline import INTERVAL_FIELDS, TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION


SEGMENT_ORDER = ("layout", "stratigraphy_lithology", "facies_reservoir_wells")


def _source_image_size(task: ImageExtractionTask) -> tuple[int, int]:
    """中文说明：直接读取来源图片真实尺寸，作为所有 VLM 像素坐标的硬约束。"""

    image_path = Path(task.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"来源图片不存在：{image_path}")
    with Image.open(image_path) as image:
        return image.width, image.height


def build_segmented_table_prompt(
    task: ImageExtractionTask,
    classification: StratigraphicSubtypeClassification,
    segment: str,
    known_tracks: list[Mapping[str, Any]] | None = None,
    known_intervals: list[Mapping[str, Any]] | None = None,
) -> str:
    """中文说明：为三组互补轨道生成短响应 Prompt，所有分段共用原图像素坐标。"""

    if segment not in SEGMENT_ORDER:
        raise ValueError(f"未知表格分段：{segment}")
    image_width, image_height = _source_image_size(task)
    track_catalog = ""
    if known_tracks:
        rows = [
            {
                "id": str(track.get("id") or ""),
                "role": str(track.get("role") or ""),
                "header": str(track.get("header") or ""),
            }
            for track in known_tracks
            if str(track.get("id") or "")
        ]
        track_catalog = (
            "\nlayout 分段已经确定以下轨道；本段所有 track_id 必须逐字选用其中的 id，"
            f"不得另造别名：\n{rows}\n"
        )
    interval_catalog = ""
    if known_intervals and segment == "facies_reservoir_wells":
        rows = [
            {
                "id": str(interval.get("id") or ""),
                "name": str(interval.get("name") or ""),
                "top_y": interval.get("top_y"),
                "bottom_y": interval.get("bottom_y"),
            }
            for interval in known_intervals
            if str(interval.get("id") or "")
        ]
        interval_catalog = (
            "\n左侧 OCR 已经确定以下地层边界（均为原图真实像素）。右侧图元必须以这些行边界为锚点，"
            "不得再使用 0-800 深度值或 0-1000 归一化坐标冒充 pixel_y：\n"
            f"{json.dumps(rows, ensure_ascii=False)}\n"
        )
    common = f"""
你正在对同一张表格—图片嵌入混合型地层柱状图执行分段视觉 OCR。
图片 ID：{task.image_id}
图题：{task.caption or '无'}
已确认子类：{classification.subtype.value}
原图真实尺寸：width={image_width}，height={image_height}。

共同约束：
1. 只读取当前图片直接可见内容，不用正文常识补值；无法辨认时写入 uncertainties。
2. 坐标必须使用上述原图真实像素，原点为左上角；x 只能在 0 到 {image_width}，y 只能在 0 到 {image_height}。区间给 top_y、bottom_y，点标注给 pixel_y；禁止使用 1000 或其他归一化画布。
3. id 使用简短稳定英文或拼音，不得在不同对象间重复；evidence 必须引用图中可见文字、刻度、色条或曲线。
4. 只输出指定 segment 的 JSON，不输出 Markdown、解释或思维过程。
{track_catalog}
{interval_catalog}
""".strip()

    instructions = {
        "layout": """
本次只做坐标系重建和轨道拆分，不抽取地层区间：
- 读取原图 width/height、有效表格 content_bbox。
- layout_family 从 stratigraphic_column_table、multi_track_well_log、imaging_log_panel、stratigraphic_summary_table 中单选；无法归入时用 other_table_hybrid。
- 若存在深度或厚度轴，列出至少 3 个、优先全部可见 calibration_points；若只有离散表格行而无数值轴，则 kind=relative_sequence、unit=无量纲，并以内容顶部 value=0、底部 value=1 建立两个明确的行序锚点。
- tracks 从左到右覆盖图片实际存在的主栏，例如地层层级、代号、深度/厚度、岩性、文字描述、曲线、成像测井、相、储层、高亮区间、井号；没有的栏不要虚构。
输出：
{
  "schema_version":"table_embedded_hybrid.segment.v1",
  "segment":"layout",
  "layout_family":"stratigraphic_column_table|multi_track_well_log|imaging_log_panel|stratigraphic_summary_table|other_table_hybrid",
  "diagram_id":"",
  "diagram_name":"",
  "image_size":{"width":0,"height":0},
  "coordinate_system":{"content_bbox":[0,0,0,0],"vertical_axis":{"kind":"thickness|depth","unit":"m","increases":"downward","calibration_points":[{"pixel_y":0,"value":0,"evidence":""}]}},
  "tracks":[{"id":"","order":0,"role":"stratigraphy|depth|lithology|curve|text|facies|reservoir|well","header":"","bbox":[0,0,0,0],"parser":"","evidence":""}],
  "uncertainties":[]
}
""",
        "stratigraphy_lithology": """
本次只读取左半部的地层层级、代号、厚度轴、岩性剖面和岩性简述：
- stratigraphic_intervals 完整保留当前图片可见的系/统/组/段/亚段、测井小层或表格行层级；parent_id 必须指向已输出父层。图中没有地层层级时返回空数组。
- lithology_intervals 按图中可见岩性文字与对应纵向区间读取；不要仅凭图例花纹猜名称。
- reference_intervals 用于井段、高亮重点段、甜点段、成像测井解释段或其他明确但不属于地层单位的连续区间；没有则返回空数组。
- geological_feature_intervals 只保留图中明确标注的裂缝、孔洞、断层、顶底板等连续特征区间。
- 多轨测井或成像面板不要求创造地层单位；应优先忠实读取深度段、可见岩性和高亮区间。
输出：
{
  "schema_version":"table_embedded_hybrid.segment.v1",
  "segment":"stratigraphy_lithology",
  "primitives":{
    "stratigraphic_intervals":[{"id":"","name":"","parent_id":"","rank":"","track_id":"","top_y":0,"bottom_y":0,"evidence":"","confidence":0.0}],
    "reference_intervals":[],
    "lithology_intervals":[{"id":"","name":"","track_id":"","top_y":0,"bottom_y":0,"evidence":"","confidence":0.0}],
    "geological_feature_intervals":[]
  },
  "uncertainties":[]
}
""",
        "facies_reservoir_wells": """
本次读取其余曲线、相、储层/高亮、井号、嵌入图像面板及说明栏：
- facies_intervals 记录图片实际可见的沉积相或微相区间；没有则为空。
- curve_tracks 逐条记录可见曲线名称、单位和刻度，包括但不限于 GR、SP、AC、DEN、CNL、电阻率、TOC、矿物含量、脆性指数、含气量或海平面曲线。
- curve_observations 按可稳定对应的层段或深度段记录高低、增减、峰谷或异常响应；刻度不清时只写 qualitative_response，不虚构数值。
- reservoir_intervals 记录图中明确的储层、油气层、甜点、高亮色带或优质页岩段；没有则为空。
- point_markers 只逐个读取井号及中心 pixel_y；其他点状标签放入 objects。objects 还可保存组合、孔隙类型、成像测井面板、储层成因说明等非连续对象。
- 必须从有效内容顶部一直读取到底部，不能只抽取上半图；每个坐标都应与上面的已知地层/井段像素边界比较。
- 对成像测井双面板，应分别建立面板对象和对应深度段；对纯测井图，不得虚构沉积相、储层成因或井号。
- explicit_relations 只允许图中竖排文字或同一储层行明确表达的关系；其余跨轨关系由程序按纵轴生成。
输出：
{
  "schema_version":"table_embedded_hybrid.segment.v1",
  "segment":"facies_reservoir_wells",
  "primitives":{
    "facies_intervals":[{"id":"","name":"","track_id":"","top_y":0,"bottom_y":0,"evidence":"","confidence":0.0}],
    "curve_tracks":[{"id":"","name":"","track_id":"","scale_min":null,"scale_max":null,"unit":"","scale_direction":"","evidence":""}],
    "curve_observations":[{"id":"","name":"","curve_ids":[],"track_id":"","top_y":0,"bottom_y":0,"qualitative_response":"","evidence":"","confidence":0.0}],
    "reservoir_intervals":[{"id":"","name":"","track_id":"","top_y":0,"bottom_y":0,"evidence":"","confidence":0.0}],
    "oil_layer_intervals":[],
    "point_markers":[{"id":"","name":"","kind":"well","track_id":"","pixel_y":0,"evidence":"","confidence":0.0}],
    "objects":[{"id":"","name":"","entity_type":"oil_layer_group|pore_type|imaging_log|embedded_panel|geological_object","evidence":"","confidence":0.0}],
    "explicit_relations":[{"source_id":"","relation_type":"","target_id":"","evidence":"","confidence":0.0}]
  },
  "uncertainties":[]
}
""",
    }[segment].strip()
    return f"{common}\n\n{instructions}"


def _parse_segment(response: Any, expected_segment: str) -> dict[str, Any]:
    """中文说明：解析并校验单个真实 VLM 分段，防止错段或空响应进入合并结果。"""

    payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"表格分段 {expected_segment} 响应不是非空 JSON 对象")
    if str(payload.get("segment") or "") != expected_segment:
        raise ValueError(
            f"表格分段响应错位：expected={expected_segment}，actual={payload.get('segment')}"
        )
    return dict(payload)


def merge_segmented_table_payloads(segments: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """中文说明：按字段所有权确定性合并三段 OCR，重复字段和缺段会立即报错。"""

    missing = [segment for segment in SEGMENT_ORDER if segment not in segments]
    if missing:
        raise ValueError(f"表格分段 OCR 缺少：{missing}")
    layout = dict(segments["layout"])
    merged_primitives: dict[str, Any] = {
        **{field: [] for field in INTERVAL_FIELDS},
        "curve_tracks": [],
        "point_markers": [],
        "objects": [],
        "explicit_relations": [],
    }
    owners: dict[str, str] = {}
    uncertainties: list[str] = []
    for segment in SEGMENT_ORDER:
        payload = segments[segment]
        raw_uncertainties = payload.get("uncertainties")
        if isinstance(raw_uncertainties, list):
            for item in raw_uncertainties:
                text = str(item).strip()
                if text and text not in uncertainties:
                    uncertainties.append(text)
        primitives = payload.get("primitives")
        if not isinstance(primitives, Mapping):
            continue
        for field, values in primitives.items():
            if field not in merged_primitives:
                continue
            if field in owners:
                raise ValueError(f"分段字段 {field} 同时由 {owners[field]} 和 {segment} 输出")
            if not isinstance(values, list):
                raise ValueError(f"分段 {segment}.primitives.{field} 必须是数组")
            owners[field] = segment
            merged_primitives[field] = list(values)

    return {
        "schema_version": TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION,
        "diagram_id": str(layout.get("diagram_id") or "table_embedded_hybrid_diagram"),
        "diagram_name": str(layout.get("diagram_name") or "表格嵌入混合地层图"),
        "layout_family": str(layout.get("layout_family") or "other_table_hybrid"),
        "image_size": dict(layout.get("image_size") or {}),
        "coordinate_system": dict(layout.get("coordinate_system") or {}),
        "tracks": list(layout.get("tracks") or []),
        "primitives": merged_primitives,
        "uncertainties": uncertainties,
    }


def validate_and_repair_pixel_geometry(
    task: ImageExtractionTask,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """中文说明：以真实图片尺寸复核坐标，丢弃越界刻度并裁剪跨边界图元，避免单个坐标误差拖垮整图。"""

    image_width, image_height = _source_image_size(task)
    payload["image_size"] = {"width": image_width, "height": image_height}
    raw_uncertainties = payload.get("uncertainties")
    uncertainties = list(raw_uncertainties) if isinstance(raw_uncertainties, list) else []
    geometry_corrections: list[str] = []

    def coordinate(value: Any, *, name: str) -> float:
        """中文说明：把单个像素字段转换为有限浮点数，范围修复由各几何对象按语义处理。"""

        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 不是有效像素坐标：{value!r}") from exc
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(f"{name} 不是有限像素坐标：{value!r}")
        return number

    def clipped(value: Any, *, name: str, upper: int) -> float:
        """中文说明：把与图片边界相交的框或区间裁剪到真实画布，并记录可审计的不确定性。"""

        number = coordinate(value, name=name)
        repaired = max(0.0, min(float(upper), number))
        if repaired != number:
            geometry_corrections.append(f"{name} 从 {number:g} 裁剪为 {repaired:g}")
        return repaired

    coordinate_system = payload.get("coordinate_system")
    if not isinstance(coordinate_system, Mapping):
        raise ValueError("分段合并结果缺少 coordinate_system")
    bbox = coordinate_system.get("content_bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("coordinate_system.content_bbox 必须是四元素数组")
    x0 = clipped(bbox[0], name="content_bbox.x0", upper=image_width)
    y0 = clipped(bbox[1], name="content_bbox.y0", upper=image_height)
    x1 = clipped(bbox[2], name="content_bbox.x1", upper=image_width)
    y1 = clipped(bbox[3], name="content_bbox.y1", upper=image_height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("coordinate_system.content_bbox 必须具有正宽高")
    coordinate_system["content_bbox"] = [x0, y0, x1, y1]

    axis = coordinate_system.get("vertical_axis")
    calibration_points = axis.get("calibration_points") if isinstance(axis, Mapping) else None
    if not isinstance(calibration_points, list) or len(calibration_points) < 2:
        raise ValueError("真实 OCR 纵轴至少需要两个 calibration_points")
    valid_calibration_points: list[dict[str, Any]] = []
    for index, point in enumerate(calibration_points):
        if not isinstance(point, Mapping):
            raise ValueError(f"calibration_points[{index}] 必须是对象")
        pixel_y = coordinate(point.get("pixel_y"), name=f"calibration_points[{index}].pixel_y")
        if pixel_y < 0 or pixel_y > image_height:
            # 中文说明：越界轴刻度通常是模型按等间距外推的不可见值，必须删除而不是压到图像边缘。
            geometry_corrections.append(
                f"calibration_points[{index}] 的 pixel_y={pixel_y:g} 越界，已删除该不可见刻度"
            )
            continue
        valid_point = dict(point)
        valid_point["pixel_y"] = pixel_y
        valid_calibration_points.append(valid_point)
    if len(valid_calibration_points) < 2:
        raise ValueError("删除越界刻度后，真实 OCR 纵轴不足两个 calibration_points")
    axis["calibration_points"] = valid_calibration_points

    tracks = payload.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("真实 OCR 结果缺少 tracks")
    for index, track in enumerate(tracks):
        raw_bbox = track.get("bbox") if isinstance(track, Mapping) else None
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise ValueError(f"tracks[{index}].bbox 必须是四元素数组")
        tx0 = clipped(raw_bbox[0], name=f"tracks[{index}].x0", upper=image_width)
        ty0 = clipped(raw_bbox[1], name=f"tracks[{index}].y0", upper=image_height)
        tx1 = clipped(raw_bbox[2], name=f"tracks[{index}].x1", upper=image_width)
        ty1 = clipped(raw_bbox[3], name=f"tracks[{index}].y1", upper=image_height)
        if tx1 <= tx0 or ty1 <= ty0:
            raise ValueError(f"tracks[{index}].bbox 必须具有正宽高")
        track["bbox"] = [tx0, ty0, tx1, ty1]

    primitives = payload.get("primitives")
    if not isinstance(primitives, Mapping):
        raise ValueError("真实 OCR 结果缺少 primitives")
    known_track_ids = {
        str(track.get("id") or "")
        for track in tracks
        if isinstance(track, Mapping) and str(track.get("id") or "")
    }
    for field in INTERVAL_FIELDS:
        items = primitives.get(field)
        if not isinstance(items, list):
            raise ValueError(f"primitives.{field} 必须是数组")
        valid_items: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"primitives.{field}[{index}] 必须是对象")
            track_id = str(item.get("track_id") or "")
            if track_id and track_id not in known_track_ids:
                raise ValueError(
                    f"primitives.{field}[{index}].track_id={track_id!r} 未在 layout 轨道中定义"
                )
            raw_top_y = coordinate(item.get("top_y"), name=f"primitives.{field}[{index}].top_y")
            raw_bottom_y = coordinate(item.get("bottom_y"), name=f"primitives.{field}[{index}].bottom_y")
            if raw_bottom_y <= 0 or raw_top_y >= image_height:
                geometry_corrections.append(
                    f"primitives.{field}[{index}] 完全位于图片外，已删除"
                )
                continue
            top_y = clipped(raw_top_y, name=f"primitives.{field}[{index}].top_y", upper=image_height)
            bottom_y = clipped(
                raw_bottom_y,
                name=f"primitives.{field}[{index}].bottom_y",
                upper=image_height,
            )
            if bottom_y <= top_y:
                raise ValueError(f"primitives.{field}[{index}] 的 bottom_y 必须大于 top_y")
            valid_item = dict(item)
            valid_item["top_y"] = top_y
            valid_item["bottom_y"] = bottom_y
            valid_items.append(valid_item)
        primitives[field] = valid_items
    point_markers = primitives.get("point_markers")
    if not isinstance(point_markers, list):
        raise ValueError("primitives.point_markers 必须是数组")
    valid_point_markers: list[dict[str, Any]] = []
    for index, item in enumerate(point_markers):
        if not isinstance(item, Mapping):
            raise ValueError(f"primitives.point_markers[{index}] 必须是对象")
        track_id = str(item.get("track_id") or "")
        if track_id and track_id not in known_track_ids:
            raise ValueError(
                f"primitives.point_markers[{index}].track_id={track_id!r} 未在 layout 轨道中定义"
            )
        pixel_y = coordinate(item.get("pixel_y"), name=f"primitives.point_markers[{index}].pixel_y")
        if pixel_y < 0 or pixel_y > image_height:
            geometry_corrections.append(
                f"primitives.point_markers[{index}].pixel_y={pixel_y:g} 越界，已删除该点"
            )
            continue
        valid_item = dict(item)
        valid_item["pixel_y"] = pixel_y
        valid_point_markers.append(valid_item)
    primitives["point_markers"] = valid_point_markers
    for correction in geometry_corrections:
        uncertainty = f"geometry_repair: {correction}"
        if uncertainty not in uncertainties:
            uncertainties.append(uncertainty)
    payload["uncertainties"] = uncertainties
    return payload


def extract_segmented_table_visual(
    task: ImageExtractionTask,
    vlm_client: Any,
    classification: StratigraphicSubtypeClassification,
) -> dict[str, Any]:
    """中文说明：对同一图片依次调用三次短响应 VLM，并返回可供六阶段算法消费的完整结构。"""

    max_tokens = {
        # 中文说明：宽幅综合柱状表可能包含十余条轨道，布局 JSON 需要更高上限以避免在最后一条轨道处截断。
        "layout": int(os.getenv("STRATIGRAPHIC_TABLE_LAYOUT_MAX_TOKENS", "4096")),
        "stratigraphy_lithology": int(os.getenv("STRATIGRAPHIC_TABLE_GEOLOGY_MAX_TOKENS", "8192")),
        "facies_reservoir_wells": int(os.getenv("STRATIGRAPHIC_TABLE_RESERVOIR_MAX_TOKENS", "8192")),
    }
    segments: dict[str, dict[str, Any]] = {}
    for segment in SEGMENT_ORDER:
        response = vlm_client.describe_image(
            task.image_path,
            build_segmented_table_prompt(
                task,
                classification,
                segment,
                known_tracks=(segments.get("layout") or {}).get("tracks"),
                known_intervals=(
                    ((segments.get("stratigraphy_lithology") or {}).get("primitives") or {}).get(
                        "stratigraphic_intervals"
                    )
                ),
            ),
            task_name=(
                f"表格嵌入混合专用OCR:{segment}_v2:{task.image_id}"
                if segment == "facies_reservoir_wells"
                else f"表格嵌入混合专用OCR:{segment}:{task.image_id}"
            ),
            response_format={"type": "json_object"},
            max_tokens=max_tokens[segment],
        )
        segments[segment] = _parse_segment(response, segment)
    return validate_and_repair_pixel_geometry(task, merge_segmented_table_payloads(segments))
