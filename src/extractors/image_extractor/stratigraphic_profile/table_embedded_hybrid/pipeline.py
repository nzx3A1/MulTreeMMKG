"""表格嵌入混合图的多阶段确定性抽取流水线。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from ..subclassifier import StratigraphicProfileSubtype
from ...schema_models import ImageExtractionTask
from .layout import LinearDepthTransform, detect_rule_lines, fit_vertical_axis, rebuild_tracks


TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION = "table_embedded_hybrid.v1"
INTERVAL_FIELDS = (
    "stratigraphic_intervals",
    "reference_intervals",
    "lithology_intervals",
    "facies_intervals",
    "reservoir_intervals",
    "oil_layer_intervals",
    "geological_feature_intervals",
    "curve_observations",
)


def is_table_embedded_hybrid_payload(payload: Mapping[str, Any]) -> bool:
    """中文说明：只接受声明专用版本且包含轨道和图元的响应，旧通用结构不得回退。"""

    return (
        str(payload.get("schema_version") or "") == TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION
        and isinstance(payload.get("tracks"), list)
        and isinstance(payload.get("primitives"), Mapping)
    )


def _confidence(value: Any, default: float = 0.8) -> float:
    """中文说明：把图元置信度限制在零到一，保证输出可直接用于质量门控。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


def _image_size(image_path: str) -> tuple[int, int, str]:
    """中文说明：从真实来源图片读取尺寸，确保后续像素坐标都对应实际图像。"""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"来源图片不存在：{path}")
    with Image.open(path) as image:
        return image.width, image.height, "source_image"


def _normalize_interval(
    raw: Mapping[str, Any],
    transform: LinearDepthTransform,
    *,
    field: str,
    index: int,
) -> dict[str, Any]:
    """中文说明：把专用解析器图元统一转换为带像素范围、深度范围和证据的区间记录。"""

    local_id = str(raw.get("id") or f"{field}_{index}").strip()
    name = str(raw.get("name") or local_id).strip()
    try:
        top_y = float(raw["top_y"])
        bottom_y = float(raw["bottom_y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{field}[{index}] 缺少有效 top_y/bottom_y") from exc
    if bottom_y <= top_y:
        raise ValueError(f"{field}[{index}] 的 bottom_y 必须大于 top_y")
    top_value = transform.value_at(top_y)
    bottom_value = transform.value_at(bottom_y)
    shallow, deep = sorted((top_value, bottom_value))
    reserved = {"id", "name", "top_y", "bottom_y", "confidence", "evidence"}
    return {
        "id": local_id,
        "name": name,
        "kind": field.removesuffix("s"),
        "top_y": round(top_y, 3),
        "bottom_y": round(bottom_y, 3),
        "top_value": round(shallow, 3),
        "bottom_value": round(deep, 3),
        "vertical_unit": transform.unit,
        "evidence": str(raw.get("evidence") or f"{field} 轨道中的可见区间"),
        "confidence": _confidence(raw.get("confidence")),
        "attributes": {key: value for key, value in raw.items() if key not in reserved},
    }


def _normalize_point(
    raw: Mapping[str, Any],
    transform: LinearDepthTransform,
    *,
    index: int,
) -> dict[str, Any]:
    """中文说明：把井名等点状标注换算到公共深度轴，供后续与层段或储层条带对齐。"""

    local_id = str(raw.get("id") or f"point_marker_{index}").strip()
    name = str(raw.get("name") or local_id).strip()
    pixel_y = float(raw.get("pixel_y"))
    reserved = {"id", "name", "pixel_y", "confidence", "evidence"}
    return {
        "id": local_id,
        "name": name,
        "kind": str(raw.get("kind") or "point_marker"),
        "pixel_y": round(pixel_y, 3),
        "vertical_value": round(transform.value_at(pixel_y), 3),
        "vertical_unit": transform.unit,
        "evidence": str(raw.get("evidence") or "右侧标注与公共纵轴处于同一水平位置"),
        "confidence": _confidence(raw.get("confidence")),
        "attributes": {key: value for key, value in raw.items() if key not in reserved},
    }


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[float, float, float]:
    """中文说明：计算两个深度区间的交集和相对较短区间的覆盖比例。"""

    top = max(float(left["top_value"]), float(right["top_value"]))
    bottom = min(float(left["bottom_value"]), float(right["bottom_value"]))
    if bottom <= top:
        return top, bottom, 0.0
    left_span = float(left["bottom_value"]) - float(left["top_value"])
    right_span = float(right["bottom_value"]) - float(right["top_value"])
    ratio = (bottom - top) / min(left_span, right_span)
    return top, bottom, max(0.0, min(1.0, ratio))


def _depth_align(parsed: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """中文说明：仅以公共深度区间交集建立跨轨道边，禁止依据横向邻近猜测关联。"""

    units = [
        *parsed.get("stratigraphic_intervals", []),
        *parsed.get("reference_intervals", []),
    ]
    relation_by_field = {
        "lithology_intervals": "has_lithology",
        "facies_intervals": "has_sedimentary_facies",
        "reservoir_intervals": "contains_reservoir",
        "oil_layer_intervals": "contains_oil_layer",
        "geological_feature_intervals": "contains_geological_feature",
        "curve_observations": "characterizes",
    }
    alignments: list[dict[str, Any]] = []
    for field, relation_type in relation_by_field.items():
        for feature in parsed.get(field, []):
            candidates: list[tuple[dict[str, Any], float, float, float]] = []
            for unit in units:
                top, bottom, ratio = _overlap(feature, unit)
                if ratio < 0.15:
                    continue
                candidates.append((unit, top, bottom, ratio))
            candidate_ids = {candidate[0]["id"] for candidate in candidates}
            for unit, top, bottom, ratio in candidates:
                # 中文说明：若候选中已有该层的直接子层，则只关联更具体的子层，避免父子层重复挂接。
                has_more_specific_candidate = any(
                    str(other[0]["attributes"].get("parent_id") or "") == unit["id"]
                    for other in candidates
                    if other[0]["id"] in candidate_ids
                )
                if has_more_specific_candidate:
                    continue
                alignments.append(
                    {
                        "source_id": unit["id"] if relation_type != "characterizes" else feature["id"],
                        "relation_type": relation_type,
                        "target_id": feature["id"] if relation_type != "characterizes" else unit["id"],
                        "overlap_top_value": round(top, 3),
                        "overlap_bottom_value": round(bottom, 3),
                        "overlap_ratio": round(ratio, 3),
                        "explicit": False,
                        "basis": "shared_vertical_axis_interval_overlap",
                        "evidence": (
                            f"{feature['name']} 与 {unit['name']} 在公共纵轴 "
                            f"{top:.2f}—{bottom:.2f} {unit['vertical_unit']} 重叠"
                        ),
                        "confidence": round(min(feature["confidence"], unit["confidence"], 0.92), 3),
                    }
                )
    for point in parsed.get("point_markers", []):
        value = float(point["vertical_value"])
        matching_reservoirs = [
            item
            for item in parsed.get("reservoir_intervals", [])
            if float(item["top_value"]) <= value <= float(item["bottom_value"])
        ]
        targets = matching_reservoirs or [
            item
            for item in units
            if float(item["top_value"]) <= value <= float(item["bottom_value"])
        ]
        for target in targets[:1]:
            alignments.append(
                {
                    "source_id": point["id"],
                    "relation_type": "aligned_with",
                    "target_id": target["id"],
                    "overlap_top_value": value,
                    "overlap_bottom_value": value,
                    "overlap_ratio": 1.0,
                    "explicit": False,
                    "basis": "shared_vertical_axis_point_alignment",
                    "evidence": f"{point['name']} 标注中心与 {target['name']} 位于同一纵轴位置",
                    "confidence": round(min(point["confidence"], target["confidence"], 0.9), 3),
                }
            )
    return alignments


def _hierarchy_and_order(parsed: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """中文说明：从明确 parent_id 和同父层的深度顺序生成层级边与相邻上覆边。"""

    units = parsed.get("stratigraphic_intervals", [])
    ids = {item["id"] for item in units}
    relations: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        parent_id = str(unit["attributes"].get("parent_id") or "")
        if parent_id and parent_id in ids:
            relations.append(
                {
                    "source_id": unit["id"],
                    "relation_type": "part_of",
                    "target_id": parent_id,
                    "explicit": True,
                    "basis": "merged_stratigraphic_table_cell",
                    "evidence": unit["evidence"],
                    "confidence": unit["confidence"],
                }
            )
        grouped[parent_id].append(unit)
    for siblings in grouped.values():
        ordered = sorted(siblings, key=lambda item: float(item["top_value"]))
        for upper, lower in zip(ordered, ordered[1:]):
            gap = float(lower["top_value"]) - float(upper["bottom_value"])
            if abs(gap) > 3.0:
                continue
            relations.append(
                {
                    "source_id": upper["id"],
                    "relation_type": "directly_overlies",
                    "target_id": lower["id"],
                    "explicit": False,
                    "basis": "same_parent_vertical_order",
                    "evidence": f"公共纵轴显示 {upper['name']} 紧邻并位于 {lower['name']} 上方",
                    "confidence": round(min(upper["confidence"], lower["confidence"], 0.9), 3),
                }
            )
    return relations


def _normalize_objects(raw_objects: Any) -> list[dict[str, Any]]:
    """中文说明：规范井、油层组和研究标记等不占连续纵轴区间的可见对象。"""

    if raw_objects is None:
        return []
    if not isinstance(raw_objects, list):
        raise ValueError("primitives.objects 必须是数组")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, Mapping):
            raise ValueError(f"primitives.objects[{index}] 必须是对象")
        local_id = str(raw.get("id") or f"object_{index}").strip()
        if not local_id or local_id in seen_ids:
            raise ValueError(f"primitives.objects[{index}].id 缺失或重复：{local_id!r}")
        seen_ids.add(local_id)
        reserved = {"id", "name", "entity_type", "evidence", "confidence", "attributes"}
        attributes = {key: value for key, value in raw.items() if key not in reserved}
        if isinstance(raw.get("attributes"), Mapping):
            attributes.update(dict(raw["attributes"]))
        normalized.append(
            {
                "id": local_id,
                "name": str(raw.get("name") or local_id),
                "entity_type": str(raw.get("entity_type") or "geological_object"),
                "evidence": str(raw.get("evidence") or "图内可见对象标签"),
                "confidence": _confidence(raw.get("confidence")),
                "attributes": attributes,
            }
        )
    return normalized


def _normalize_explicit_relations(
    raw_relations: Any,
    known_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """中文说明：校验模型只连接已读取对象，无法解析的端点进入 dropped 而不静默造节点。"""

    if raw_relations is None:
        return [], []
    if not isinstance(raw_relations, list):
        raise ValueError("primitives.explicit_relations 必须是数组")
    accepted: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_relations):
        if not isinstance(raw, Mapping):
            dropped.append({"index": index, "reason": "not_an_object"})
            continue
        source_id = str(raw.get("source_id") or "")
        target_id = str(raw.get("target_id") or "")
        if source_id not in known_ids or target_id not in known_ids:
            dropped.append({**dict(raw), "reason": "unresolved_endpoint"})
            continue
        accepted.append(
            {
                "source_id": source_id,
                "relation_type": str(raw.get("relation_type") or "related_to"),
                "target_id": target_id,
                "explicit": True,
                "basis": str(raw.get("basis") or "visible_table_or_panel_annotation"),
                "evidence": str(raw.get("evidence") or "图内标签、色条或连线明确关联"),
                "confidence": _confidence(raw.get("confidence"), 0.9),
                "attributes": dict(raw.get("attributes") or {})
                if isinstance(raw.get("attributes"), Mapping)
                else {},
            }
        )
    return accepted, dropped


class TableEmbeddedHybridPipeline:
    """执行坐标重建、轨道拆分、专用解析、深度对齐和中间结果装配。"""

    def run(self, task: ImageExtractionTask, payload: Mapping[str, Any]) -> dict[str, Any]:
        """中文说明：把真实 VLM 返回的单张混合表格图元转换为确定、可审计的中间结果。"""

        if not is_table_embedded_hybrid_payload(payload):
            raise ValueError(
                f"专用响应必须使用 schema_version={TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION}"
            )
        width, height, size_source = _image_size(task.image_path)
        if width <= 0 or height <= 0:
            raise ValueError("图片尺寸必须大于零")
        coordinate_system = payload.get("coordinate_system")
        if not isinstance(coordinate_system, Mapping):
            raise ValueError("响应缺少 coordinate_system")
        vertical_axis = coordinate_system.get("vertical_axis")
        if not isinstance(vertical_axis, Mapping):
            raise ValueError("响应缺少 coordinate_system.vertical_axis")
        transform = fit_vertical_axis(vertical_axis)
        content_bbox = coordinate_system.get("content_bbox") or [0, 0, width, height]
        grid_evidence = detect_rule_lines(task.image_path, content_bbox)
        tracks = rebuild_tracks(
            payload.get("tracks"),
            image_width=width,
            image_height=height,
            detected_lines=grid_evidence,
        )
        raw_primitives = payload.get("primitives")
        if not isinstance(raw_primitives, Mapping):
            raise ValueError("响应缺少 primitives")
        parsed: dict[str, list[dict[str, Any]]] = {}
        for field in INTERVAL_FIELDS:
            raw_items = raw_primitives.get(field, [])
            if not isinstance(raw_items, list):
                raise ValueError(f"primitives.{field} 必须是数组")
            parsed[field] = [
                _normalize_interval(raw, transform, field=field, index=index)
                for index, raw in enumerate(raw_items)
                if isinstance(raw, Mapping)
            ]
        curve_tracks = raw_primitives.get("curve_tracks", [])
        parsed["curve_tracks"] = [dict(item) for item in curve_tracks if isinstance(item, Mapping)]
        point_markers = raw_primitives.get("point_markers", [])
        parsed["point_markers"] = [
            _normalize_point(raw, transform, index=index)
            for index, raw in enumerate(point_markers)
            if isinstance(raw, Mapping)
        ]
        parsed["objects"] = _normalize_objects(raw_primitives.get("objects", []))
        alignments = _depth_align(parsed)
        stratigraphic_relations = _hierarchy_and_order(parsed)
        known_ids = {
            str(item.get("id"))
            for records in parsed.values()
            for item in records
            if item.get("id")
        }
        explicit_relations, dropped_explicit_relations = _normalize_explicit_relations(
            raw_primitives.get("explicit_relations", []),
            known_ids,
        )
        track_ids = {track["id"] for track in tracks}
        unresolved_tracks = sorted(
            {
                str(item.get("track_id"))
                for records in parsed.values()
                for item in records
                if item.get("track_id") and str(item.get("track_id")) not in track_ids
            }
        )
        return {
            "schema_version": "table_embedded_hybrid.intermediate.v1",
            "subtype": StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID.value,
            "algorithm": {
                "name": "coordinate_track_specialized_depth_alignment_graph_assembly",
                "stages": [
                    "coordinate_reconstruction",
                    "track_segmentation",
                    "specialized_parsing",
                    "depth_and_sequence_alignment",
                    "structured_intermediate_result",
                    "deterministic_knowledge_graph_assembly",
                ],
            },
            "source": {
                "document_id": task.document_id,
                "chunk_id": task.chunk_id,
                "image_id": task.image_id,
                "image_index": task.image_index,
                "source_image_path": task.image_path,
                "caption": task.caption,
            },
            "diagram": {
                "id": str(payload.get("diagram_id") or task.image_id),
                "name": str(payload.get("diagram_name") or task.caption or "表格嵌入混合地层图"),
                "layout_family": str(payload.get("layout_family") or "other_table_hybrid"),
                "width": width,
                "height": height,
                "size_source": size_source,
            },
            "coordinate_system": {
                "axis_kind": str(vertical_axis.get("kind") or "depth"),
                "content_bbox": list(content_bbox),
                "vertical_transform": transform.to_dict(),
            },
            "grid_detection": grid_evidence,
            "tracks": tracks,
            "parsed": parsed,
            "alignment_relations": [
                *stratigraphic_relations,
                *alignments,
                *explicit_relations,
            ],
            "quality": {
                "track_count": len(tracks),
                "interval_count": sum(len(parsed[field]) for field in INTERVAL_FIELDS),
                "curve_track_count": len(parsed["curve_tracks"]),
                "point_marker_count": len(parsed["point_markers"]),
                "object_count": len(parsed["objects"]),
                "alignment_relation_count": len(alignments),
                "explicit_relation_count": len(explicit_relations),
                "axis_rmse": transform.rmse,
                "unresolved_track_ids": unresolved_tracks,
                "dropped_explicit_relations": dropped_explicit_relations,
                "uncertainties": [str(item) for item in payload.get("uncertainties", [])],
            },
        }
