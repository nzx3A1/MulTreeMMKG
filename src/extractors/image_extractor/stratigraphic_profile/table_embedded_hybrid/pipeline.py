"""表格嵌入混合图的多阶段确定性抽取流水线。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from ..subclassifier import StratigraphicProfileSubtype
from ...schema_models import ImageExtractionTask
from .layout import LinearDepthTransform, detect_rule_lines, fit_vertical_axis, rebuild_tracks
from .ppstructure_geometry import ensure_ppstructure_geometry
from .submember_refinement import build_submember_coverage
from .visual_track_extraction import cache_curve_track_images


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
    "track_intervals",
)
NODE_FIELDS = (*INTERVAL_FIELDS, "curve_tracks", "point_markers", "objects")


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
    reserved = {
        "id",
        "name",
        "official_name",
        "top_y",
        "bottom_y",
        "track_id",
        "geometry_refs",
        "confidence",
        "evidence",
    }
    return {
        "id": local_id,
        "name": name,
        "official_name": str(raw.get("official_name") or name),
        "kind": field.removesuffix("s"),
        "top_y": round(top_y, 3),
        "bottom_y": round(bottom_y, 3),
        "top_value": round(shallow, 3),
        "bottom_value": round(deep, 3),
        "vertical_unit": transform.unit,
        "track_id": str(raw.get("track_id") or ""),
        "geometry_refs": list(raw.get("geometry_refs") or []),
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
    reserved = {
        "id",
        "name",
        "official_name",
        "pixel_y",
        "track_id",
        "geometry_refs",
        "confidence",
        "evidence",
    }
    return {
        "id": local_id,
        "name": name,
        "official_name": str(raw.get("official_name") or name),
        "kind": str(raw.get("kind") or "point_marker"),
        "pixel_y": round(pixel_y, 3),
        "vertical_value": round(transform.value_at(pixel_y), 3),
        "vertical_unit": transform.unit,
        "track_id": str(raw.get("track_id") or ""),
        "geometry_refs": list(raw.get("geometry_refs") or []),
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
    return alignments


def _cross_track_align(
    parsed: Mapping[str, list[dict[str, Any]]],
    tracks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """中文说明：只在左右紧邻轨道之间按公共纵轴重叠建立 aligned_with，禁止跨轨跳连。"""

    track_order = {str(track.get("id") or ""): int(track.get("order", index)) for index, track in enumerate(tracks)}
    intervals = [
        item
        for field in INTERVAL_FIELDS
        for item in parsed.get(field, [])
        if item.get("track_id") in track_order
    ]
    relations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    threshold = 0.15
    for later in intervals:
        later_track = str(later.get("track_id") or "")
        later_order = track_order[later_track]
        for earlier in intervals:
            earlier_track = str(earlier.get("track_id") or "")
            if (
                not earlier_track
                or earlier_track == later_track
                or track_order[earlier_track] != later_order - 1
            ):
                continue
            top, bottom, ratio = _overlap(later, earlier)
            if ratio < threshold:
                continue
            key = (str(later.get("id") or ""), str(earlier.get("id") or ""))
            if not all(key) or key in seen:
                continue
            seen.add(key)
            relations.append(
                {
                    "source_id": key[0],
                    "relation_type": "aligned_with",
                    "target_id": key[1],
                    "overlap_top_value": round(top, 3),
                    "overlap_bottom_value": round(bottom, 3),
                    "overlap_ratio": round(ratio, 3),
                    "source_track_id": later_track,
                    "target_track_id": earlier_track,
                    "explicit": False,
                    "basis": "adjacent_track_shared_vertical_axis_overlap",
                    "evidence": (
                        f"{later.get('name')} 与左侧紧邻轨道 {earlier.get('name')} 在公共纵轴 "
                        f"{top:.3f}—{bottom:.3f} {later.get('vertical_unit') or ''} 重叠"
                    ),
                    "confidence": round(
                        min(float(later.get("confidence") or 0.8), float(earlier.get("confidence") or 0.8), 0.9),
                        3,
                    ),
                }
            )
    for point in parsed.get("point_markers", []):
        point_track = str(point.get("track_id") or "")
        if point_track not in track_order:
            continue
        value = float(point["vertical_value"])
        for interval in intervals:
            target_track = str(interval.get("track_id") or "")
            if (
                target_track not in track_order
                or track_order[target_track] != track_order[point_track] - 1
            ):
                continue
            if float(interval["top_value"]) <= value <= float(interval["bottom_value"]):
                relations.append(
                    {
                        "source_id": str(point.get("id") or ""),
                        "relation_type": "aligned_with",
                        "target_id": str(interval.get("id") or ""),
                        "overlap_top_value": value,
                        "overlap_bottom_value": value,
                        "overlap_ratio": 1.0,
                        "source_track_id": point_track,
                        "target_track_id": target_track,
                        "explicit": False,
                        "basis": "adjacent_track_shared_vertical_axis_point",
                        "evidence": (
                            f"{point.get('name')} 与左侧紧邻轨道 {interval.get('name')} "
                            "位于同一公共纵轴位置"
                        ),
                        "confidence": round(min(float(point.get("confidence") or 0.8), float(interval.get("confidence") or 0.8), 0.9), 3),
                    }
                )
    return relations


def _track_entity_coverage(
    tracks: Sequence[Mapping[str, Any]],
    parsed: Mapping[str, list[dict[str, Any]]],
    visual_quality: Mapping[str, Any],
) -> dict[str, Any]:
    """中文说明：审计每条语义轨道是否生成了对应实体，并区分结构空列与 OCR 噪声。"""

    counts: dict[str, int] = defaultdict(int)
    for values in parsed.values():
        for item in values:
            track_id = str(item.get("track_id") or "")
            if track_id:
                counts[track_id] += 1
    raw_audit = ((visual_quality.get("track_entity_coverage") or {}).get("tracks") or [])
    raw_audit_by_track = {
        str(item.get("track_id") or ""): item
        for item in raw_audit
        if isinstance(item, Mapping)
    }
    audit = []
    errors = []
    for track in tracks:
        track_id = str(track.get("id") or "")
        role = str(track.get("role") or "unknown")
        track_type = str(track.get("track_type") or "")
        raw_track_audit = raw_audit_by_track.get(track_id, {})
        text_cell_count = int(raw_track_audit.get("text_cell_count") or 0)
        node_count = counts.get(track_id, 0)
        # 中文说明：无文字不等于无内容，岩性纹理和曲线轨道仍必须生成对应实体。
        structural_blank = (
            track_type == "table_text"
            and role in {"other", "unknown"}
            and not str(track.get("header") or "").strip()
            and text_cell_count == 0
        )
        # 中文说明：PP 表线分出的无表头、无 OCR 结构空列不是内容轨道，不强制为其伪造实体。
        fallback_skipped = bool(raw_track_audit.get("fallback_skipped"))
        ocr_noise_only = bool(raw_track_audit.get("ocr_noise_only"))
        covered = (
            role == "depth"
            or node_count > 0
            or structural_blank
            or fallback_skipped
            or ocr_noise_only
        )
        audit.append(
            {
                "track_id": track_id,
                "order": track.get("order"),
                "track_type": track_type,
                "role": role,
                "header": str(track.get("header") or ""),
                "entity_count": node_count,
                "text_cell_count": text_cell_count,
                "structural_blank": structural_blank,
                "fallback_skipped": fallback_skipped,
                "ocr_noise_only": ocr_noise_only,
                "rejected_ocr_cell_count": int(raw_track_audit.get("rejected_ocr_cell_count") or 0),
                "covered": covered,
            }
        )
        if not covered:
            errors.append(f"轨道 {track_id}({track.get('header') or role}) 存在内容但未生成实体")
    return {
        "ok": not errors,
        "track_count": len(tracks),
        "covered_track_count": sum(1 for item in audit if item["covered"]),
        "tracks": audit,
        "errors": errors,
    }


def _track_type_and_slice_quality(
    tracks: Sequence[Mapping[str, Any]],
    visual_quality: Mapping[str, Any],
) -> dict[str, Any]:
    """中文说明：审计轨道分类和 legend 切片是否完整，曲线只检查轨道分类。"""

    allowed = {"table_text", "legend", "curve"}
    errors: list[str] = []
    type_counts: dict[str, int] = defaultdict(int)
    for index, track in enumerate(tracks):
        track_id = str(track.get("id") or "")
        track_type = str(track.get("track_type") or "")
        type_counts[track_type or "missing"] += 1
        if track_type not in allowed:
            errors.append(f"轨道 {track_id} 缺少有效 VLM track_type：{track_type or '<empty>'}")
        elif track_type == "legend" and len(tracks) == 1:
            errors.append(f"图例轨道 {track_id} 不存在可作为切分基准的 table_text 轨道")
    slice_quality = visual_quality.get("adjacent_slice_description")
    slice_quality = dict(slice_quality) if isinstance(slice_quality, Mapping) else {}
    recognition_order = list(slice_quality.get("track_recognition_order") or [])
    has_visual_tracks = any(
        str(track.get("track_type") or "") in {"legend", "curve"} for track in tracks
    )
    if has_visual_tracks and recognition_order != ["table_text", "legend", "curve"]:
        errors.append(f"轨道识别顺序错误：{recognition_order!r}")
    vlm_slice_track_types = list(slice_quality.get("vlm_slice_track_types") or [])
    if has_visual_tracks and vlm_slice_track_types != ["legend"]:
        errors.append(f"VLM 切片轨道类型错误：{vlm_slice_track_types!r}")
    completed_slices = [
        item
        for item in slice_quality.get("slices", [])
        if isinstance(item, Mapping) and str(item.get("status") or "") == "completed"
    ]
    non_text_sources = [
        str(item.get("slice_id") or "")
        for item in completed_slices
        if str(item.get("source_track_type") or "") != "table_text"
    ]
    if non_text_sources:
        errors.append(f"视觉切片使用了非 table_text 纵向基准：{non_text_sources}")
    stage_sequence = [int(item.get("recognition_stage") or 0) for item in completed_slices]
    if stage_sequence != sorted(stage_sequence) or any(stage != 2 for stage in stage_sequence):
        errors.append(f"VLM 切片中包含非 legend 轨道：{stage_sequence}")
    visual_track_ids = {
        str(track.get("id") or "")
        for track in tracks
        if str(track.get("track_type") or "") in {"legend", "curve"}
    }
    vlm_slice_track_ids = {
        str(track.get("id") or "")
        for track in tracks
        if str(track.get("track_type") or "") == "legend"
    }
    curve_track_ids = {
        str(track.get("id") or "")
        for track in tracks
        if str(track.get("track_type") or "") == "curve"
    }
    described_track_ids = {
        str(item.get("track_id") or "")
        for item in completed_slices
    }
    for track_id in sorted(vlm_slice_track_ids - described_track_ids):
        errors.append(f"视觉轨道 {track_id} 未生成任何相邻实体投影裁剪描述")
    errors.extend(str(item) for item in slice_quality.get("errors", []) if str(item))
    return {
        "ok": not errors,
        "allowed_track_types": sorted(allowed),
        "track_type_counts": dict(sorted(type_counts.items())),
        "visual_track_count": len(visual_track_ids),
        "vlm_slice_track_count": len(vlm_slice_track_ids),
        "described_visual_track_count": len(vlm_slice_track_ids & described_track_ids),
        "full_curve_track_count": len(curve_track_ids),
        "slice_count": int(slice_quality.get("slice_count", 0)),
        "completed_slice_count": int(slice_quality.get("completed_count", 0)),
        "errors": list(dict.fromkeys(errors)),
    }


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


def _visual_track_adjacent_order(
    parsed: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """中文说明：按原图 y 像素顺序，为同一图例轨道中相邻的 VLM 切片实体建立直接上覆链。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in parsed.get("track_intervals", []):
        attributes = item.get("attributes")
        recognition_source = (
            str(attributes.get("recognition_source") or "")
            if isinstance(attributes, Mapping)
            else ""
        )
        track_id = str(item.get("track_id") or "")
        if track_id and recognition_source == "VLM.adjacent_track_slice_description":
            grouped[track_id].append(item)
    relations: list[dict[str, Any]] = []
    for track_id, entities in grouped.items():
        ordered = sorted(
            entities,
            key=lambda item: (
                float(item["top_y"]),
                float(item["bottom_y"]),
                str(item.get("id") or ""),
            ),
        )
        for upper, lower in zip(ordered, ordered[1:]):
            relations.append(
                {
                    "source_id": str(upper["id"]),
                    "relation_type": "directly_overlies",
                    "target_id": str(lower["id"]),
                    "explicit": False,
                    "basis": "same_visual_track_adjacent_pixel_order",
                    "evidence": (
                        f"同一轨道 {track_id} 中，{upper['name']} 的像素范围 "
                        f"y={upper['top_y']:.3f}—{upper['bottom_y']:.3f} 位于相邻实体 "
                        f"{lower['name']} 的 y={lower['top_y']:.3f}—{lower['bottom_y']:.3f} 上方"
                    ),
                    "confidence": round(
                        min(
                            float(upper.get("confidence") or 0.8),
                            float(lower.get("confidence") or 0.8),
                            0.95,
                        ),
                        3,
                    ),
                    "attributes": {
                        "track_id": track_id,
                        "upper_pixel_range": dict(
                            (upper.get("attributes") or {}).get("pixel_range") or {}
                        ),
                        "lower_pixel_range": dict(
                            (lower.get("attributes") or {}).get("pixel_range") or {}
                        ),
                    },
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
        reserved = {"id", "name", "official_name", "entity_type", "evidence", "confidence", "attributes"}
        attributes = {key: value for key, value in raw.items() if key not in reserved}
        if isinstance(raw.get("attributes"), Mapping):
            attributes.update(dict(raw["attributes"]))
        normalized.append(
            {
                "id": local_id,
                "name": str(raw.get("name") or local_id),
                "official_name": str(raw.get("official_name") or raw.get("name") or local_id),
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


def _enforce_adjacent_alignment_policy(
    relations: Sequence[Mapping[str, Any]],
    parsed: Mapping[str, list[dict[str, Any]]],
    tracks: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """中文说明：显式 aligned_with 也只能连接相邻轨道，违规边进入审计而不写入图谱。"""

    track_order = {
        str(track.get("id") or ""): int(track.get("order", index))
        for index, track in enumerate(tracks)
    }
    entity_track = {
        str(item.get("id") or ""): str(item.get("track_id") or "")
        for records in parsed.values()
        for item in records
        if item.get("id")
    }
    accepted: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for raw in relations:
        relation = dict(raw)
        if str(relation.get("relation_type") or "") != "aligned_with":
            accepted.append(relation)
            continue
        source_track = entity_track.get(str(relation.get("source_id") or ""), "")
        target_track = entity_track.get(str(relation.get("target_id") or ""), "")
        if (
            source_track in track_order
            and target_track in track_order
            and abs(track_order[source_track] - track_order[target_track]) == 1
        ):
            accepted.append(relation)
            continue
        dropped.append(
            {
                **relation,
                "source_track_id": source_track,
                "target_track_id": target_track,
                "reason": "non_adjacent_track_alignment_forbidden",
            }
        )
    return accepted, dropped


class TableEmbeddedHybridPipeline:
    """执行坐标重建、轨道拆分、专用解析、深度对齐和中间结果装配。"""

    def run(self, task: ImageExtractionTask, payload: Mapping[str, Any]) -> dict[str, Any]:
        """中文说明：先确保坐标来自 PP-StructureV3，再把 VLM 语义转换为可审计中间结果。"""

        if not is_table_embedded_hybrid_payload(payload):
            raise ValueError(
                f"专用响应必须使用 schema_version={TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION}"
            )
        payload = ensure_ppstructure_geometry(payload, task.image_path)
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
        ppstructure_geometry = payload.get("ppstructure_geometry")
        ppstructure_geometry = ppstructure_geometry if isinstance(ppstructure_geometry, Mapping) else {}
        pp_rule_lines = ppstructure_geometry.get("rule_lines")
        grid_evidence = (
            dict(pp_rule_lines)
            if isinstance(pp_rule_lines, Mapping)
            else detect_rule_lines(task.image_path, content_bbox)
        )
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
        parsed["curve_tracks"] = cache_curve_track_images(
            task.image_path, tracks, curve_tracks,
        )
        point_markers = raw_primitives.get("point_markers", [])
        parsed["point_markers"] = [
            _normalize_point(raw, transform, index=index)
            for index, raw in enumerate(point_markers)
            if isinstance(raw, Mapping)
        ]
        parsed["objects"] = _normalize_objects(raw_primitives.get("objects", []))
        alignments = _depth_align(parsed)
        cross_track_alignments = _cross_track_align(parsed, tracks)
        stratigraphic_relations = _hierarchy_and_order(parsed)
        visual_track_order_relations = _visual_track_adjacent_order(parsed)
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
        explicit_relations, non_adjacent_explicit_alignments = _enforce_adjacent_alignment_policy(
            explicit_relations,
            parsed,
            tracks,
        )
        dropped_explicit_relations.extend(non_adjacent_explicit_alignments)
        track_ids = {track["id"] for track in tracks}
        unresolved_tracks = sorted(
            {
                str(item.get("track_id"))
                for records in parsed.values()
                for item in records
                if item.get("track_id") and str(item.get("track_id")) not in track_ids
            }
        )
        submember_coverage = build_submember_coverage(raw_primitives, ppstructure_geometry)
        visual_track_extraction = payload.get("visual_track_extraction")
        visual_track_extraction = (
            dict(visual_track_extraction)
            if isinstance(visual_track_extraction, Mapping)
            else {}
        )
        visual_track_extraction["curve"] = {
            "available": bool(parsed["curve_tracks"]),
            "mode": "full_track_crop",
            "node_count": len(parsed["curve_tracks"]),
            "image_count": len({item["cropped_image_cache_path"] for item in parsed["curve_tracks"]}),
        }
        track_entity_coverage = _track_entity_coverage(
            tracks,
            parsed,
            visual_track_extraction,
        )
        track_type_and_slice_quality = _track_type_and_slice_quality(
            tracks,
            visual_track_extraction,
        )
        quality_gate_errors = [
            *list(submember_coverage.get("errors") or []),
            *list(track_entity_coverage.get("errors") or []),
            *list(track_type_and_slice_quality.get("errors") or []),
        ]
        return {
            "schema_version": "table_embedded_hybrid.intermediate.v1",
            "subtype": StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID.value,
            "algorithm": {
                "name": "ppstructurev3_geometry_semantic_vlm_depth_alignment_graph_assembly",
                "stages": [
                    "ppstructurev3_layout_ocr_and_cell_geometry",
                    "vlm_track_type_classification",
                    "semantic_track_grouping_and_visual_primitive_extraction",
                    "semantic_track_and_cell_id_selection",
                    "ppstructure_pixel_coordinate_resolution",
                    "table_text_entity_projected_legend_slice_and_full_curve_track_cache",
                    "vlm_legend_slice_interpretation",
                    "same_visual_track_adjacent_pixel_order",
                    "depth_and_sequence_alignment",
                    "adjacent_track_depth_alignment",
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
                "official_name": str(
                    payload.get("diagram_official_name")
                    or payload.get("diagram_name")
                    or task.caption
                    or "表格嵌入混合地层图"
                ),
                "track_header": str(payload.get("diagram_track_header") or "整图"),
                "official_name_basis": str(
                    payload.get("diagram_official_name_basis") or "insufficient_context_keep_original"
                ),
                "official_name_confidence": _confidence(
                    payload.get("diagram_official_name_confidence"), 1.0
                ),
                "layout_family": str(payload.get("layout_family") or "other_table_hybrid"),
                "width": width,
                "height": height,
                "size_source": size_source,
            },
            "coordinate_system": {
                "axis_kind": str(vertical_axis.get("kind") or "depth"),
                "content_bbox": list(content_bbox),
                "coordinate_source": "PP-StructureV3",
                "vlm_pixel_coordinates_used": False,
                "vertical_transform": transform.to_dict(),
            },
            "ppstructure_geometry": ppstructure_geometry,
            "visual_track_extraction": visual_track_extraction,
            "legend_entries": [
                dict(item)
                for item in raw_primitives.get("legend_entries", [])
                if isinstance(item, Mapping)
            ] if isinstance(raw_primitives.get("legend_entries"), list) else [],
            "grid_detection": grid_evidence,
            "tracks": tracks,
            "parsed": parsed,
            "alignment_relations": [
                *stratigraphic_relations,
                *visual_track_order_relations,
                *alignments,
                *cross_track_alignments,
                *explicit_relations,
            ],
            "quality": {
                "track_count": len(tracks),
                "interval_count": sum(len(parsed[field]) for field in INTERVAL_FIELDS),
                "curve_track_count": len(parsed["curve_tracks"]),
                "point_marker_count": len(parsed["point_markers"]),
                "object_count": len(parsed["objects"]),
                "node_enrichment_count": 1 + sum(
                    len(parsed.get(field, [])) for field in NODE_FIELDS
                ),
                "alignment_relation_count": len(alignments),
                "cross_track_alignment_relation_count": len(cross_track_alignments),
                "visual_track_directly_overlies_count": len(visual_track_order_relations),
                "explicit_relation_count": len(explicit_relations),
                "axis_rmse": transform.rmse,
                "ppstructure_ocr_line_count": int(
                    (ppstructure_geometry.get("quality") or {}).get("ocr_line_count", 0)
                ),
                "ppstructure_cell_count": int(
                    (ppstructure_geometry.get("quality") or {}).get("cell_count", 0)
                ),
                "ppstructure_track_count": int(
                    (ppstructure_geometry.get("quality") or {}).get("track_count", 0)
                ),
                "vlm_pixel_coordinates_used": False,
                "unresolved_track_ids": unresolved_tracks,
                "dropped_explicit_relations": dropped_explicit_relations,
                "stratigraphic_cell_coverage": submember_coverage,
                "track_entity_coverage": track_entity_coverage,
                "track_type_and_slice_quality": track_type_and_slice_quality,
                "quality_gate_errors": quality_gate_errors,
                "quality_gate_ok": not quality_gate_errors,
                "uncertainties": [str(item) for item in payload.get("uncertainties", [])],
            },
        }
