"""把可追溯的人工或外部证据复核结果叠加到地质模式图视觉结果。"""
from __future__ import annotations

from typing import Any, Mapping


def _merge_by_id(records: list[Any], updates: list[Any]) -> list[Any]:
    """中文说明：按局部 id 更新记录并保留原顺序，新记录追加到末尾。"""

    result = [dict(item) if isinstance(item, Mapping) else item for item in records]
    index = {
        str(item.get("id")): position
        for position, item in enumerate(result)
        if isinstance(item, Mapping) and item.get("id")
    }
    for raw in updates:
        if not isinstance(raw, Mapping) or not raw.get("id"):
            continue
        item = dict(raw)
        local_id = str(item["id"])
        if local_id in index:
            merged = dict(result[index[local_id]])
            merged.update(item)
            result[index[local_id]] = merged
        else:
            index[local_id] = len(result)
            result.append(item)
    return result


def apply_geological_review_overlay(
    visual: dict[str, Any],
    overlay: Mapping[str, Any],
    audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """中文说明：应用图例、层段、实体和事件复核，并显式记录已解决的旧审查警告。"""

    corrections: list[dict[str, Any]] = []
    field_updates = {
        "legend_lithologies": "legend_lithologies",
        "unit_updates": "stratigraphic_units",
        "entity_updates": "entities",
        "event_updates": "process_events",
        "segment_updates": "structural_segments",
    }
    for overlay_field, visual_field in field_updates.items():
        updates = overlay.get(overlay_field)
        if not isinstance(updates, list) or not updates:
            continue
        original = visual.get(visual_field)
        original_list = original if isinstance(original, list) else []
        visual[visual_field] = _merge_by_id(original_list, updates)
        corrections.append(
            {
                "type": "review_overlay",
                "field": visual_field,
                "updated_ids": [str(item.get("id")) for item in updates if isinstance(item, Mapping) and item.get("id")],
                "reason": str(overlay.get("review_basis") or "人工或外部证据复核"),
            }
        )
    added_units = overlay.get("add_units")
    if isinstance(added_units, list) and added_units:
        visual["stratigraphic_units"] = _merge_by_id(visual.get("stratigraphic_units", []), added_units)
        corrections.append(
            {
                "type": "review_overlay_add_units",
                "field": "stratigraphic_units",
                "updated_ids": [str(item.get("id")) for item in added_units if isinstance(item, Mapping) and item.get("id")],
                "reason": str(overlay.get("review_basis") or "区域裁剪 OCR 补充层段"),
            }
        )
    relation_additions = overlay.get("relation_additions")
    if isinstance(relation_additions, Mapping):
        for group_name, additions in relation_additions.items():
            if not isinstance(additions, list) or not additions:
                continue
            visual.setdefault(str(group_name), []).extend(dict(item) for item in additions if isinstance(item, Mapping))
            corrections.append(
                {
                    "type": "review_overlay_relations",
                    "field": str(group_name),
                    "count": len(additions),
                    "reason": str(overlay.get("review_basis") or "外部证据关系复核"),
                }
            )
    for raw_template in overlay.get("relation_templates", []) if isinstance(overlay.get("relation_templates"), list) else []:
        if not isinstance(raw_template, Mapping):
            continue
        group_name = str(raw_template.get("group") or "spatial_relations")
        generated = []
        for source in raw_template.get("sources", []) if isinstance(raw_template.get("sources"), list) else []:
            for target in raw_template.get("targets", []) if isinstance(raw_template.get("targets"), list) else []:
                generated.append(
                    {
                        "source": source,
                        "type": raw_template.get("type", "related_to"),
                        "target": target,
                        "dimension": raw_template.get("dimension", "structural"),
                        "coordinate_frame": raw_template.get("coordinate_frame", "stratigraphic"),
                        "explicit": bool(raw_template.get("explicit", False)),
                        "basis": raw_template.get("basis", "外部证据复核"),
                        "evidence": raw_template.get("evidence", ""),
                        "confidence": raw_template.get("confidence", 0.9),
                    }
                )
        visual.setdefault(group_name, []).extend(generated)
        if generated:
            corrections.append(
                {
                    "type": "review_overlay_relation_template",
                    "field": group_name,
                    "count": len(generated),
                    "reason": str(raw_template.get("basis") or overlay.get("review_basis") or "外部证据复核"),
                }
            )
    uncertainty_filters = overlay.get("remove_uncertainties_containing")
    if isinstance(uncertainty_filters, list) and isinstance(visual.get("uncertainties"), list):
        visual["uncertainties"] = [
            item
            for item in visual["uncertainties"]
            if not any(str(marker) in str(item) for marker in uncertainty_filters)
        ]
    visual.setdefault("normalization_corrections", []).extend(corrections)
    resolved_audit_warnings: list[str] = []
    warning_filters = overlay.get("resolved_audit_warning_contains")
    if audit is not None and isinstance(warning_filters, list) and isinstance(audit.get("warnings"), list):
        remaining_warnings = []
        for warning in audit["warnings"]:
            if any(str(marker) in str(warning) for marker in warning_filters):
                resolved_audit_warnings.append(str(warning))
            else:
                remaining_warnings.append(warning)
        audit["warnings"] = remaining_warnings
    visual["review_overlay"] = {
        "reviewer": overlay.get("reviewer", "external_review"),
        "review_basis": overlay.get("review_basis", ""),
        "evidence_sources": overlay.get("evidence_sources", []),
        "notes": overlay.get("notes", []),
        "resolved_audit_warnings": resolved_audit_warnings,
    }
    return corrections
