"""三维地层建模图的规范化、层序推导和关系质量控制流水线。"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from ...schema_models import ImageExtractionTask
from ..subclassifier import StratigraphicProfileSubtype
from .multipass import _image_size, apply_multipass_quality_gates
from .review import apply_manual_review_corrections


THREE_DIMENSIONAL_MODEL_SCHEMA_VERSION = "three_dimensional_stratigraphic_model.v1"

ENTITY_FIELDS = {
    "lithologies": "lithology",
    "stratigraphic_units": "stratigraphic_unit",
    "wells": "well",
    "faults": "fault",
    "zones": "geological_zone",
    "fluids": "fluid_or_ion",
    "objects": "geological_object",
}


def is_three_dimensional_stratigraphic_model_payload(payload: Mapping[str, Any]) -> bool:
    """中文说明：只接受专用版本、模型对象和地层数组，禁止回退到旧通用结构。"""

    return (
        str(payload.get("schema_version") or "") == THREE_DIMENSIONAL_MODEL_SCHEMA_VERSION
        and isinstance(payload.get("model"), Mapping)
        and isinstance(payload.get("stratigraphic_units"), list)
    )


def _items(value: Any) -> list[Mapping[str, Any]]:
    """中文说明：把模型数组安全收窄为对象列表，忽略无法建图的非对象脏值。"""

    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _confidence(value: Any, default: float = 0.8) -> float:
    """中文说明：把模型置信度限制在零到一，避免异常值污染质量统计。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


def _normalize_entity(raw: Mapping[str, Any], *, field: str, index: int) -> dict[str, Any]:
    """中文说明：规范一个三维图元，并保留层序、空间和上下文证据等专用属性。"""

    local_id = str(raw.get("id") or f"{field}_{index}").strip()
    name = str(raw.get("name") or local_id).strip()
    if not local_id or not name:
        raise ValueError(f"{field}[{index}] 缺少 id 或 name")
    source_kind = str(raw.get("source_kind") or "visual").strip()
    if source_kind not in {"visual", "visual_context"}:
        source_kind = "visual"
    evidence = str(raw.get("evidence") or "").strip()
    context_evidence = str(raw.get("context_evidence") or "").strip()
    # 中文说明：声明 visual_context 却没有独立正文证据时降级为纯视觉，避免证据标签名实不符。
    if source_kind == "visual_context" and not context_evidence:
        source_kind = "visual"
    if not evidence:
        evidence = str(raw.get("visual_anchor_evidence") or "图中可见且带标签的三维图元").strip()
    reserved = {
        "id",
        "name",
        "entity_type",
        "source_kind",
        "evidence",
        "context_evidence",
        "image_region",
        "confidence",
        "attributes",
    }
    attributes = {key: value for key, value in raw.items() if key not in reserved}
    if isinstance(raw.get("attributes"), Mapping):
        attributes.update(dict(raw["attributes"]))
    if field == "stratigraphic_units":
        order = raw.get("order_bottom_to_top")
        try:
            attributes["order_bottom_to_top"] = int(order) if order is not None else None
        except (TypeError, ValueError):
            attributes["order_bottom_to_top"] = None
        attributes["column_id"] = str(raw.get("column_id") or "main_section").strip()
        attributes["parent_unit_id"] = str(raw.get("parent_unit_id") or "").strip()
        attributes["sequence_role"] = str(raw.get("sequence_role") or "layer").strip()
        raw_lithology_ids = raw.get("lithology_ids")
        attributes["lithology_ids"] = (
            [str(item) for item in raw_lithology_ids if str(item).strip()]
            if isinstance(raw_lithology_ids, list)
            else []
        )
    return {
        "id": local_id,
        "name": name,
        "entity_type": str(raw.get("entity_type") or ENTITY_FIELDS[field]),
        "source_kind": source_kind,
        "evidence": evidence,
        "context_evidence": context_evidence,
        "image_region": str(raw.get("image_region") or "").strip(),
        "confidence": _confidence(raw.get("confidence")),
        "attributes": attributes,
    }


def _normalize_relation(
    raw: Mapping[str, Any],
    *,
    evidence_scope: str,
) -> dict[str, Any]:
    """中文说明：规范视觉或上下文关系；上下文关系无条件降为非显式证据。"""

    source_id = str(raw.get("source") or raw.get("source_id") or "").strip()
    target_id = str(raw.get("target") or raw.get("target_id") or "").strip()
    relation_type = str(raw.get("type") or raw.get("relation_type") or "").strip()
    context_evidence = str(raw.get("context_evidence") or "").strip()
    visual_anchor_evidence = str(
        raw.get("visual_anchor_evidence") or raw.get("evidence") or "图中对象标签与空间位置"
    ).strip()
    evidence = context_evidence if evidence_scope == "context" else str(raw.get("evidence") or "").strip()
    reserved = {
        "source",
        "source_id",
        "target",
        "target_id",
        "type",
        "relation_type",
        "explicit",
        "basis",
        "evidence_scope",
        "evidence",
        "context_evidence",
        "visual_anchor_evidence",
        "confidence",
        "attributes",
    }
    attributes = {key: value for key, value in raw.items() if key not in reserved}
    if isinstance(raw.get("attributes"), Mapping):
        attributes.update(dict(raw["attributes"]))
    return {
        "source_id": source_id,
        "relation_type": relation_type,
        "target_id": target_id,
        "explicit": False if evidence_scope == "context" else bool(raw.get("explicit", True)),
        "basis": str(raw.get("basis") or ("reference_context" if evidence_scope == "context" else "visible_spatial_evidence")),
        "evidence_scope": evidence_scope,
        "evidence": evidence or visual_anchor_evidence,
        "context_evidence": context_evidence,
        "visual_anchor_evidence": visual_anchor_evidence,
        "confidence": _confidence(raw.get("confidence")),
        "attributes": attributes,
    }


def _vertical_relations(units: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """中文说明：仅在同柱、同父层级内按自下而上序号生成相邻上下关系。"""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        attributes = unit["attributes"]
        if attributes.get("order_bottom_to_top") is None:
            continue
        if str(attributes.get("sequence_role") or "layer") == "container":
            continue
        grouped[(str(attributes.get("column_id") or "main_section"), str(attributes.get("parent_unit_id") or ""))].append(unit)

    relations: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    for (column_id, parent_id), siblings in grouped.items():
        by_order: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for unit in siblings:
            by_order[int(unit["attributes"]["order_bottom_to_top"])].append(unit)
        ambiguous_orders = {order for order, records in by_order.items() if len(records) != 1}
        if ambiguous_orders:
            ambiguities.append(
                {
                    "column_id": column_id,
                    "parent_unit_id": parent_id,
                    "orders": sorted(ambiguous_orders),
                    "reason": "同一层序号包含多个地层，未生成可能错误的直接上下关系",
                }
            )
        comparable = [by_order[order][0] for order in sorted(by_order) if order not in ambiguous_orders]
        for lower, upper in zip(comparable, comparable[1:]):
            lower_order = int(lower["attributes"]["order_bottom_to_top"])
            upper_order = int(upper["attributes"]["order_bottom_to_top"])
            # 中文说明：若两层之间隔着含歧义序号，则不能越过歧义层强行建立直接相邻边。
            if any(lower_order < order < upper_order for order in ambiguous_orders):
                continue
            evidence = (
                f"同一三维剖面柱 {column_id} 中，{upper['name']} 的自下而上序号 {upper_order} "
                f"高于 {lower['name']} 的序号 {lower_order}；图内证据：{upper['evidence']}；{lower['evidence']}"
            )
            confidence = min(float(lower["confidence"]), float(upper["confidence"]), 0.9)
            base = {
                "explicit": False,
                "basis": "order_bottom_to_top_same_parent_and_column",
                "evidence_scope": "visual_derived",
                "evidence": evidence,
                "context_evidence": "",
                "visual_anchor_evidence": evidence,
                "confidence": round(confidence, 3),
                "attributes": {
                    "column_id": column_id,
                    "parent_unit_id": parent_id,
                    "lower_order": lower_order,
                    "upper_order": upper_order,
                },
            }
            relations.append(
                {**base, "source_id": upper["id"], "relation_type": "directly_overlies", "target_id": lower["id"]}
            )
            relations.append(
                {**base, "source_id": lower["id"], "relation_type": "directly_underlies", "target_id": upper["id"]}
            )
    return relations, ambiguities


def _promote_segment_lithologies_to_units(payload: Mapping[str, Any]) -> dict[str, Any]:
    """中文说明：将质量门通过的内部层段识别聚合到具体地层单元，并从建图协议中移除层段节点。"""

    promoted = dict(payload)
    units = [dict(item) for item in _items(payload.get("stratigraphic_units"))]
    segments = [dict(item) for item in _items(payload.get("visual_layer_segments"))]
    unit_by_id = {str(unit.get("id") or ""): unit for unit in units}
    assignment_maps: dict[str, dict[str, dict[str, Any]]] = {}
    assignment_orders: dict[str, list[str]] = {}

    for unit_id, unit in unit_by_id.items():
        assignments = {
            str(item.get("lithology_id") or ""): dict(item)
            for item in _items(unit.get("lithology_assignments"))
            if str(item.get("lithology_id") or "")
        }
        order = list(assignments)
        for raw_lithology_id in unit.get("lithology_ids", []):
            lithology_id = str(raw_lithology_id or "").strip()
            if not lithology_id or lithology_id in assignments:
                continue
            assignments[lithology_id] = {
                "lithology_id": lithology_id,
                "role": "primary",
                "evidence_source": "whole_image_unit_assignment",
                "evidence": str(unit.get("evidence") or "地层单元填充与图例匹配"),
                "confidence": unit.get("confidence", 0.0),
            }
            order.append(lithology_id)
        assignment_maps[unit_id] = assignments
        assignment_orders[unit_id] = order

    for segment in segments:
        unit_id = str(segment.get("parent_unit_id") or "")
        if unit_id not in unit_by_id:
            continue
        segment_id = str(segment.get("id") or segment.get("segment_id") or "")
        segment_role = str(segment.get("segment_role") or "vertical_layer")
        visible_regions = list(segment.get("visible_regions") or [])
        for raw_assignment in _items(segment.get("lithology_assignments")):
            lithology_id = str(raw_assignment.get("lithology_id") or "").strip()
            if not lithology_id:
                continue
            candidate = dict(raw_assignment)
            candidate.update(
                {
                    "lithology_id": lithology_id,
                    "source_segment_ids": [segment_id] if segment_id else [],
                    "source_segment_roles": [segment_role],
                    "source_visible_regions": visible_regions,
                    "visual_anchor_evidence": str(segment.get("evidence") or "可见层段边界"),
                }
            )
            current = assignment_maps[unit_id].get(lithology_id)
            if current is None:
                assignment_maps[unit_id][lithology_id] = candidate
                assignment_orders[unit_id].append(lithology_id)
                continue

            current_confidence = _confidence(current.get("confidence"), 0.0)
            candidate_confidence = _confidence(candidate.get("confidence"), 0.0)
            merged = dict(candidate if candidate_confidence > current_confidence else current)
            merged["source_segment_ids"] = list(
                dict.fromkeys([
                    *current.get("source_segment_ids", []),
                    *candidate.get("source_segment_ids", []),
                ])
            )
            merged["source_segment_roles"] = list(
                dict.fromkeys([
                    *current.get("source_segment_roles", []),
                    *candidate.get("source_segment_roles", []),
                ])
            )
            merged["source_visible_regions"] = [
                *current.get("source_visible_regions", []),
                *candidate.get("source_visible_regions", []),
            ]
            merged["source_unit_color_scan"] = bool(
                current.get("source_unit_color_scan")
                or candidate.get("source_unit_color_scan")
            )
            merged["matched_patterns"] = list(
                dict.fromkeys(
                    [
                        *(current.get("matched_patterns") or []),
                        *(candidate.get("matched_patterns") or []),
                    ]
                )
            )
            if current.get("source_color_signature") or candidate.get("source_color_signature"):
                merged["source_color_signature"] = dict(
                    current.get("source_color_signature")
                    or candidate.get("source_color_signature")
                    or {}
                )
            same_color_evidence = [
                str(current.get("same_color_region_evidence") or "").strip(),
                str(candidate.get("same_color_region_evidence") or "").strip(),
            ]
            merged["same_color_region_evidence"] = "；".join(
                dict.fromkeys(item for item in same_color_evidence if item)
            )
            evidence_parts = [
                str(current.get("evidence") or "").strip(),
                str(candidate.get("evidence") or "").strip(),
            ]
            merged["evidence"] = "；".join(dict.fromkeys(item for item in evidence_parts if item))
            assignment_maps[unit_id][lithology_id] = merged

    for unit_id, unit in unit_by_id.items():
        ordered_ids = [
            lithology_id
            for lithology_id in assignment_orders[unit_id]
            if lithology_id in assignment_maps[unit_id]
        ]
        unit["lithology_ids"] = ordered_ids
        unit["lithology_assignments"] = [assignment_maps[unit_id][item] for item in ordered_ids]
        if ordered_ids:
            unit["lithology_assignment_source"] = "direct_stratigraphic_unit_from_multipass_segments"

    multipass = promoted.get("multipass_lithology")
    if isinstance(multipass, Mapping):
        multipass = dict(multipass)
        multipass["audited_layer_segments"] = segments
        multipass["graph_projection"] = {
            "node_strategy": "stratigraphic_unit_direct_lithology",
            "visual_layer_segment_nodes_emitted": 0,
            "unit_lithology_assignment_count": sum(len(items) for items in assignment_maps.values()),
        }
        promoted["multipass_lithology"] = multipass
    promoted["stratigraphic_units"] = units
    promoted["visual_layer_segments"] = []
    return promoted


def _structural_relations(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """中文说明：仅以地层单元的 parent/lithology 字段生成层级关系和直接多岩性关系。"""

    relations: list[dict[str, Any]] = []
    for unit in units:
        attributes = unit["attributes"]
        parent_id = str(attributes.get("parent_unit_id") or "")
        if parent_id:
            relations.append(
                {
                    "source_id": unit["id"],
                    "relation_type": "part_of",
                    "target_id": parent_id,
                    "explicit": True,
                    "basis": "visible_stratigraphic_hierarchy",
                    "evidence_scope": "visual",
                    "evidence": unit["evidence"],
                    "context_evidence": "",
                    "visual_anchor_evidence": unit["evidence"],
                    "confidence": unit["confidence"],
                    "attributes": {},
                }
            )
        for lithology_id in attributes.get("lithology_ids", []):
            assignments = {
                str(item.get("lithology_id") or ""): item
                for item in _items(attributes.get("lithology_assignments"))
            }
            assignment = assignments.get(str(lithology_id), {})
            evidence_source = str(assignment.get("evidence_source") or "legend_pattern")
            context_supported = evidence_source == "reference_context"
            assignment_evidence = str(assignment.get("evidence") or unit["evidence"])
            relations.append(
                {
                    "source_id": unit["id"],
                    "relation_type": "has_lithology",
                    "target_id": lithology_id,
                    "explicit": not context_supported,
                    "basis": (
                        "reference_context_unit_lithology"
                        if context_supported
                        else "same_color_unit_pattern_scan"
                        if assignment.get("source_unit_color_scan")
                        else "multipass_layer_crop_promoted_to_stratigraphic_unit"
                        if assignment.get("source_segment_ids")
                        else "visible_fill_and_legend_match"
                    ),
                    "evidence_scope": "context" if context_supported else "visual",
                    "evidence": assignment_evidence,
                    "context_evidence": assignment_evidence if context_supported else "",
                    "visual_anchor_evidence": str(
                        assignment.get("visual_anchor_evidence") or unit["evidence"]
                    ),
                    "confidence": min(
                        unit["confidence"],
                        _confidence(assignment.get("confidence"), unit["confidence"]),
                    ),
                    "attributes": {
                        "lithology_role": assignment.get("role", "primary"),
                        "evidence_source": evidence_source,
                        "source_segment_ids": list(assignment.get("source_segment_ids") or []),
                        "source_segment_roles": list(assignment.get("source_segment_roles") or []),
                    },
                }
            )
    return relations


def _semantic_relation_error(
    relation: Mapping[str, Any],
    entity_by_id: Mapping[str, Mapping[str, Any]],
    *,
    model_id: str,
) -> str:
    """中文说明：拦截上下文中最常见的端点类型错误，避免语法合法但语义颠倒的三元组入图。"""

    source_id = str(relation.get("source_id") or "")
    target_id = str(relation.get("target_id") or "")
    relation_type = str(relation.get("relation_type") or "")
    if source_id == target_id:
        return "self_relation"
    source_type = str(entity_by_id.get(source_id, {}).get("entity_type") or "")
    target_type = str(entity_by_id.get(target_id, {}).get("entity_type") or "")
    if relation_type in {"directly_overlies", "directly_underlies"} and (
        source_type != "stratigraphic_unit" or target_type != "stratigraphic_unit"
    ):
        return "vertical_relation_requires_stratigraphic_units"
    if relation.get("evidence_scope") != "context":
        return ""
    if target_id == model_id and relation_type in {
        "acts_as_source_rock",
        "transports",
        "generates",
        "supplies_hydrocarbons_to",
        "dolomitizes",
    }:
        return "context_role_relation_cannot_target_model_root"
    if relation_type == "transports" and target_type in {
        "stratigraphic_unit",
        "lithology",
        "fault",
        "three_dimensional_stratigraphic_model",
    }:
        return "transports_target_must_be_fluid_ion_or_hydrocarbon"
    if relation_type == "generates" and target_type not in {
        "hydrocarbon",
        "oil_and_gas",
        "fluid_or_ion",
        "geological_object",
    }:
        return "generates_target_must_be_hydrocarbon"
    if relation_type == "acts_as_source_rock" and source_type not in {
        "stratigraphic_unit",
        "lithology",
    }:
        return "acts_as_source_rock_source_must_be_unit_or_lithology"
    if relation_type == "controls" and target_type == "stratigraphic_unit":
        context_evidence = str(relation.get("context_evidence") or "")
        target_name = str(entity_by_id.get(target_id, {}).get("name") or "")
        if "储层" in context_evidence and target_name not in context_evidence:
            return "controls_context_requires_explicit_reservoir_endpoint"
    if relation_type == "dolomitizes" and target_type != "lithology":
        return "dolomitizes_target_must_be_lithology"
    return ""


def _repair_context_relation(
    relation: dict[str, Any],
    entity_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """中文说明：只做证据可唯一证明的端点或关系规范化，并完整记录修复前签名。"""

    source = entity_by_id.get(str(relation.get("source_id") or ""), {})
    target = entity_by_id.get(str(relation.get("target_id") or ""), {})
    relation_type = str(relation.get("relation_type") or "")
    source_type = str(source.get("entity_type") or "")
    target_type = str(target.get("entity_type") or "")
    original = {
        "source_id": relation.get("source_id"),
        "relation_type": relation_type,
        "target_id": relation.get("target_id"),
    }
    repaired = dict(relation)
    if (
        relation_type == "acts_as_source_rock"
        and source_type in {"stratigraphic_unit", "lithology"}
        and target_type in {"hydrocarbon", "oil_and_gas", "fluid_or_ion", "geological_object"}
    ):
        target_name = str(target.get("name") or "")
        if any(keyword in target_name for keyword in ("油", "气", "烃")):
            repaired["relation_type"] = "generates"
            repaired["basis"] = f"{repaired.get('basis') or 'reference_context'}|canonicalized_role_to_generation"
            return repaired, {**original, "repair": "acts_as_source_rock_to_generates"}
    if relation_type == "dolomitizes" and target_type == "stratigraphic_unit":
        context_evidence = str(relation.get("context_evidence") or "")
        candidates = [
            record
            for record in entity_by_id.values()
            if record.get("entity_type") == "lithology"
            and str(record.get("name") or "")
            and str(record.get("name")) in context_evidence
        ]
        candidates.sort(key=lambda record: len(str(record.get("name") or "")), reverse=True)
        if candidates:
            repaired["target_id"] = str(candidates[0]["id"])
            repaired["basis"] = f"{repaired.get('basis') or 'reference_context'}|context_lithology_endpoint_match"
            repaired_attributes = dict(repaired.get("attributes") or {})
            repaired_attributes["original_target_id"] = original["target_id"]
            repaired["attributes"] = repaired_attributes
            return repaired, {
                **original,
                "repair": "unit_target_to_context_matched_lithology",
                "repaired_target_id": repaired["target_id"],
            }
    return repaired, None


class ThreeDimensionalStratigraphicModelPipeline:
    """把三维专用 VLM JSON 转为可确定性装配的中间结果。"""

    def run(self, task: ImageExtractionTask, payload: Mapping[str, Any]) -> dict[str, Any]:
        """中文说明：规范实体，推导同层级上下边，校验视觉与上下文三元组端点。"""

        if not is_three_dimensional_stratigraphic_model_payload(payload):
            raise ValueError("三维地层模型未返回 three_dimensional_stratigraphic_model.v1 结构")
        original_payload = dict(payload)
        reviewed_payload, manual_review = apply_manual_review_corrections(task, payload)
        image_width, image_height = _image_size(task.image_path)
        payload = apply_multipass_quality_gates(
            reviewed_payload,
            image_width=image_width,
            image_height=image_height,
        )
        payload = _promote_segment_lithologies_to_units(payload)
        raw_model = payload["model"]
        model_id = str(raw_model.get("id") or "model_1").strip()
        model_name = str(raw_model.get("name") or task.caption or "三维地层建模图").strip()
        model = {
            "id": model_id,
            "name": model_name,
            "entity_type": "three_dimensional_stratigraphic_model",
            "source_kind": "visual",
            "evidence": str(raw_model.get("evidence") or raw_model.get("topology_evidence") or "整张三维模型").strip(),
            "context_evidence": "",
            "image_region": str(raw_model.get("image_region") or "整图"),
            "confidence": _confidence(raw_model.get("confidence"), 0.95),
            "attributes": {
                key: value
                for key, value in raw_model.items()
                if key not in {"id", "name", "evidence", "image_region", "confidence"}
            },
        }

        entities: dict[str, list[dict[str, Any]]] = {}
        local_ids = {model_id}
        entity_by_id: dict[str, Mapping[str, Any]] = {model_id: model}
        duplicate_ids: list[str] = []
        for field in ENTITY_FIELDS:
            records: list[dict[str, Any]] = []
            for index, raw in enumerate(_items(payload.get(field))):
                record = _normalize_entity(raw, field=field, index=index)
                if record["id"] in local_ids:
                    duplicate_ids.append(record["id"])
                    continue
                local_ids.add(record["id"])
                entity_by_id[record["id"]] = record
                records.append(record)
            entities[field] = records

        visual_relations = [
            _normalize_relation(raw, evidence_scope="visual")
            for raw in _items(payload.get("relations"))
        ]
        context_relations: list[dict[str, Any]] = []
        context_relation_repairs: list[dict[str, Any]] = []
        for raw in _items(payload.get("context_relations")):
            relation, repair = _repair_context_relation(
                _normalize_relation(raw, evidence_scope="context"),
                entity_by_id,
            )
            context_relations.append(relation)
            if repair:
                context_relation_repairs.append(repair)
        vertical_relations, vertical_ambiguities = _vertical_relations(entities["stratigraphic_units"])
        structural_relations = _structural_relations(entities["stratigraphic_units"])

        # 中文说明：模型偶尔重复输出上下边；只在确定性层序已能重建同一签名时丢弃模型副本。
        vertical_signatures = {
            (relation["source_id"], relation["relation_type"], relation["target_id"])
            for relation in vertical_relations
        }
        filtered_visual_relations: list[dict[str, Any]] = []
        rebuilt_vertical_relations: list[dict[str, Any]] = []
        for relation in visual_relations:
            signature = (
                relation["source_id"],
                relation["relation_type"],
                relation["target_id"],
            )
            if signature in vertical_signatures:
                rebuilt_vertical_relations.append({**relation, "reason": "vertical_relation_rebuilt_from_order"})
                continue
            filtered_visual_relations.append(relation)

        accepted: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = rebuilt_vertical_relations
        seen_signatures: set[tuple[str, str, str]] = set()
        for relation in [*structural_relations, *filtered_visual_relations, *vertical_relations, *context_relations]:
            signature = (
                str(relation.get("source_id") or ""),
                str(relation.get("relation_type") or ""),
                str(relation.get("target_id") or ""),
            )
            if not all(signature):
                dropped.append({**relation, "reason": "missing_relation_field"})
                continue
            if signature[0] not in local_ids or signature[2] not in local_ids:
                dropped.append({**relation, "reason": "unresolved_endpoint"})
                continue
            semantic_error = _semantic_relation_error(relation, entity_by_id, model_id=model_id)
            if semantic_error:
                dropped.append({**relation, "reason": semantic_error})
                continue
            if signature in seen_signatures:
                dropped.append({**relation, "reason": "duplicate_relation"})
                continue
            seen_signatures.add(signature)
            accepted.append(relation)

        return {
            "schema_version": "three_dimensional_stratigraphic_model.intermediate.v1",
            "subtype": StratigraphicProfileSubtype.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL.value,
            "algorithm": {
                "name": "surface_topology_spatial_extraction",
                "stages": [
                    "three_dimensional_surface_and_object_recognition",
                    "dominant_unit_fill_color_mask_generation",
                    "batched_same_color_unit_lithology_enumeration",
                    "unit_color_and_layer_segment_evidence_merge",
                    "same_column_same_parent_vertical_sequence_normalization",
                    "deterministic_overlies_underlies_derivation",
                    "visual_spatial_relation_validation",
                    "context_supported_triple_validation",
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
                "references": list(task.references),
            },
            "model": model,
            "entities": entities,
            "relations": accepted,
            "quality": {
                "entity_count_without_model": sum(len(records) for records in entities.values()),
                "visual_layer_segment_count": 0,
                "lithology_assignment_count": sum(
                    len(record.get("attributes", {}).get("lithology_ids", []))
                    for record in entities["stratigraphic_units"]
                ),
                "relation_count": len(accepted),
                "vertical_relation_count": sum(
                    relation["relation_type"] in {"directly_overlies", "directly_underlies"}
                    for relation in accepted
                ),
                "context_relation_count": sum(
                    relation["evidence_scope"] == "context" for relation in accepted
                ),
                "duplicate_entity_ids": duplicate_ids,
                "vertical_ambiguities": vertical_ambiguities,
                "dropped_relations": dropped,
                "context_relation_repairs": context_relation_repairs,
                "uncertainties": [str(item) for item in payload.get("uncertainties", []) if str(item).strip()]
                if isinstance(payload.get("uncertainties"), list)
                else [],
                "manual_review": manual_review,
                "multipass_lithology_summary": (
                    dict(payload.get("multipass_lithology", {}).get("summary", {}))
                    if isinstance(payload.get("multipass_lithology"), Mapping)
                    and isinstance(payload.get("multipass_lithology", {}).get("summary"), Mapping)
                    else {}
                ),
            },
            "raw_visual_payload": original_payload,
            "reviewed_visual_payload": dict(payload),
        }
