"""地质模式图视觉结果的确定性一致性校正。"""
from __future__ import annotations

import re
from typing import Any, Mapping

from ..schema_models import ImageExtractionTask


EXPLICIT_LITHOLOGIES = (
    "泥质石灰岩",
    "灰质白云岩",
    "泥质白云岩",
    "膏质白云岩",
    "含膏白云岩",
    "泥质岩",
    "石灰岩",
    "白云岩",
    "膏盐岩",
    "角砾岩",
    "页岩",
    "泥岩",
    "砂岩",
    "灰岩",
)


def _list(value: Any) -> list[Any]:
    """中文说明：把模型的非列表字段安全降级为空列表。"""

    return value if isinstance(value, list) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    """中文说明：把模型的非字典字段安全降级为空字典。"""

    return value if isinstance(value, Mapping) else {}


def _explicit_lithology_from_label(unit: Mapping[str, Any]) -> str | None:
    """中文说明：只从层段名称或带“标签/标注”的证据中读取明确岩性，避免把解释文字误当标签。"""

    name = str(unit.get("name") or "")
    evidence = str(unit.get("evidence") or "")
    for lithology in EXPLICIT_LITHOLOGIES:
        if lithology in name:
            return lithology
        if re.search(rf"(?:标签|标注)[^，；。]{{0,24}}{re.escape(lithology)}", evidence):
            return lithology
    return None


def _ensure_lithology(visual: dict[str, Any], name: str) -> str:
    """中文说明：为层段标签中明确写出的岩性复用或补建图例外岩性实体。"""

    legend = visual.setdefault("legend_lithologies", [])
    for raw in _list(legend):
        item = _mapping(raw)
        if str(item.get("name") or "") == name:
            return str(item.get("id") or name)
    local_id = "lith_explicit_" + re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", name)
    legend.append(
        {
            "id": local_id,
            "name": name,
            "visual_pattern": "层段文字直接标注，未强制归入图例纹理",
            "evidence": f"层段标签明确写出{name}",
            "confidence": 0.99,
        }
    )
    return local_id


def _replace_references(value: Any, remap: Mapping[str, str]) -> Any:
    """中文说明：递归替换已合并实体的引用，不改动普通说明文本。"""

    if isinstance(value, str):
        return remap.get(value, value)
    if isinstance(value, list):
        return [_replace_references(item, remap) for item in value]
    if isinstance(value, Mapping):
        return {key: _replace_references(item, remap) for key, item in value.items()}
    return value


def _normalize_well_derived_faults(visual: dict[str, Any], corrections: list[dict[str, Any]]) -> None:
    """中文说明：当模型把同一分段的井号复制成断裂名时，合并为不带井号的分段主断裂系统。"""

    entities = [item for item in _list(visual.get("entities")) if isinstance(item, Mapping)]
    entity_by_id = {str(item.get("id")): item for item in entities if item.get("id")}
    remap: dict[str, str] = {}
    canonical_names: dict[str, str] = {}
    for raw_segment in _list(visual.get("structural_segments")):
        segment = _mapping(raw_segment)
        well_bases = {
            re.sub(r"井$", "", str(_mapping(well).get("name") or _mapping(well).get("id") or well))
            for well in _list(segment.get("wells"))
        }
        fault_ids = [
            str(reference)
            for reference in _list(segment.get("structures"))
            if str(_mapping(entity_by_id.get(str(reference))).get("type") or "") == "fault"
        ]
        copied_fault_ids = [
            fault_id
            for fault_id in fault_ids
            if re.sub(r"断裂$", "", str(entity_by_id[fault_id].get("name") or "")) in well_bases
        ]
        if not copied_fault_ids or len(copied_fault_ids) != len(fault_ids):
            continue
        canonical_id = copied_fault_ids[0]
        segment_name = str(segment.get("name") or segment.get("style") or "该分段")
        canonical_names[canonical_id] = f"{segment_name}主走滑断裂"
        for duplicate_id in copied_fault_ids[1:]:
            remap[duplicate_id] = canonical_id
        corrections.append(
            {
                "type": "well_fault_disambiguation",
                "segment_id": segment.get("id"),
                "merged_ids": copied_fault_ids,
                "canonical_id": canonical_id,
                "reason": "井号标签只表示井位，不能作为断裂名称",
            }
        )
    if not canonical_names:
        return
    for entity in entities:
        entity_id = str(entity.get("id") or "")
        if entity_id in canonical_names:
            entity["name"] = canonical_names[entity_id]
            entity["evidence"] = f"{entity.get('evidence', '')}；按所在构造分段规范命名"
    if remap:
        visual["entities"] = [item for item in entities if str(item.get("id") or "") not in remap]
        for key in list(visual):
            if key != "entities":
                visual[key] = _replace_references(visual[key], remap)


def _normalize_relation_explicitness(visual: dict[str, Any], corrections: list[dict[str, Any]]) -> None:
    """中文说明：将正文推导、因果解释和穿切律时间判断降为非显式关系。"""

    for group_name in ("spatial_relations", "temporal_relations", "causal_relations"):
        for raw in _list(visual.get(group_name)):
            relation = _mapping(raw)
            evidence = f"{relation.get('evidence', '')} {relation.get('basis', '')}"
            relation_type = str(relation.get("type") or "")
            inferred = "正文" in evidence and not any(marker in evidence for marker in ("图中箭头", "图内标注", "图中文字"))
            inferred = inferred or (group_name == "causal_relations" and not any(marker in evidence for marker in ("箭头", "明确标注")))
            inferred = inferred or (relation_type in {"older_than", "younger_than"} and "切穿" in evidence)
            if inferred and bool(relation.get("explicit", False)):
                relation["explicit"] = False
                corrections.append(
                    {
                        "type": "explicitness_downgrade",
                        "relation": f"{relation.get('source')}|{relation_type}|{relation.get('target')}",
                        "reason": "关系来自正文、因果解释或地质规律，不是图中直接陈述",
                    }
                )


def _expand_a08_event_targets(visual: dict[str, Any], corrections: list[dict[str, Any]]) -> None:
    """中文说明：依据已识别的油气系统角色，为作用于多个构造分段的 A08 事件补全全部目标。"""

    system = _mapping(visual.get("petroleum_system"))
    target_fields = {
        "migration": "reservoirs",
        "reservoir_formation": "reservoirs",
        "accumulation": "accumulation_zones",
        "enrichment": "accumulation_zones",
    }
    for raw in _list(visual.get("process_events")):
        event = _mapping(raw)
        evidence = str(event.get("evidence") or "")
        if "正文" in evidence and "图中" not in evidence:
            event["explicit"] = False
        field = target_fields.get(str(event.get("type") or ""))
        targets = [str(_mapping(item).get("id") or item) for item in _list(system.get(field))] if field else []
        targets = list(dict.fromkeys(target for target in targets if target))
        if len(targets) > 1:
            event["targets"] = targets
            corrections.append(
                {
                    "type": "multi_segment_event_targets",
                    "event_id": event.get("id"),
                    "targets": targets,
                    "reason": "油气系统已识别多个分段目标，不能只保留列表首项",
                }
            )


def normalize_geological_visual_result(task: ImageExtractionTask, visual: dict[str, Any]) -> list[dict[str, Any]]:
    """中文说明：执行不依赖模型的通用一致性规则，并返回可追溯的校正记录。"""

    corrections: list[dict[str, Any]] = []
    uncertainties = visual.setdefault("uncertainties", [])
    for raw in _list(visual.get("stratigraphic_units")):
        unit = _mapping(raw)
        explicit_lithology = _explicit_lithology_from_label(unit)
        if explicit_lithology:
            lithology_id = _ensure_lithology(visual, explicit_lithology)
            previous = [str(_mapping(item).get("name") or "") for item in _list(unit.get("lithologies"))]
            if previous != [explicit_lithology]:
                unit["lithologies"] = [
                    {
                        "lithology_id": lithology_id,
                        "name": explicit_lithology,
                        "role": "主要",
                        "lateral_variation": "层段标签明确标注，未见横向岩性变化文字",
                        "lateral_zones": [],
                        "evidence": f"层段标签明确标注{explicit_lithology}",
                        "confidence": 0.99,
                        "validation_source": "explicit_unit_label",
                    }
                ]
                corrections.append(
                    {
                        "type": "explicit_lithology_override",
                        "unit_id": unit.get("id"),
                        "previous": previous,
                        "corrected": [explicit_lithology],
                        "reason": "层段文字优先于颜色或相似图例猜测",
                    }
                )
        elif "烃源岩" in f"{unit.get('name', '')}{unit.get('evidence', '')}":
            for raw_lithology in _list(unit.get("lithologies")):
                lithology = _mapping(raw_lithology)
                lithology["role"] = "不确定"
                try:
                    lithology["confidence"] = min(float(lithology.get("confidence", 0.55)), 0.55)
                except (TypeError, ValueError):
                    lithology["confidence"] = 0.55
                lithology["validation_source"] = "visual_pattern_only"
            message = f"{unit.get('name', '该层段')}仅明确标注为烃源岩，具体岩性不能由角色名称推出"
            if message not in uncertainties:
                uncertainties.append(message)
    if task.classification_code == "A08":
        _normalize_well_derived_faults(visual, corrections)
        _expand_a08_event_targets(visual, corrections)
    _normalize_relation_explicitness(visual, corrections)
    visual["normalization_corrections"] = corrections
    return corrections
