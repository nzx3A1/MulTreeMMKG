"""对已记录真实 VLM 响应应用小范围、可审计的专项标签复核。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from ...schema_models import ImageExtractionTask


CORRECTIONS_PATH = Path(__file__).with_name("manual_review_corrections.json")


def _load_corrections() -> dict[str, Any]:
    """中文说明：读取版本化专项复核清单；文件缺失或根结构错误时立即暴露配置问题。"""

    if not CORRECTIONS_PATH.is_file():
        return {}
    payload = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("三维专项复核清单必须是按图片文件名索引的 JSON 对象")
    return payload


def _replace_relation_aliases(payload: dict[str, Any], aliases: Mapping[str, str]) -> None:
    """中文说明：实体合并后同步修正视觉和上下文关系端点，避免产生悬空旧 ID。"""

    for field in ("relations", "context_relations"):
        records = payload.get(field)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in ("source", "source_id", "target", "target_id"):
                value = str(record.get(key) or "")
                if value in aliases:
                    record[key] = aliases[value]


def apply_manual_review_corrections(
    task: ImageExtractionTask,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """中文说明：按精确图片和精确旧名称应用合并/更名，并返回完整应用与跳过记录。"""

    corrected = deepcopy(dict(payload))
    review = _load_corrections().get(Path(task.image_path).name)
    if not isinstance(review, Mapping):
        return corrected, {"configured": False, "applied": [], "skipped": []}
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}

    for instruction in review.get("merge_entities", []):
        if not isinstance(instruction, Mapping):
            continue
        field = str(instruction.get("field") or "")
        match_names = [str(name) for name in instruction.get("match_names", [])]
        minimum_match_count = int(instruction.get("minimum_match_count") or len(set(match_names)))
        records = corrected.get(field)
        matches = [
            record
            for record in records
            if isinstance(record, dict) and str(record.get("name") or "") in match_names
        ] if isinstance(records, list) else []
        if len(matches) < minimum_match_count:
            skipped.append(
                {
                    "operation": "merge_entities",
                    "field": field,
                    "match_names": match_names,
                    "minimum_match_count": minimum_match_count,
                    "reason": "exact_match_count_changed",
                }
            )
            continue
        canonical = matches[0]
        canonical_id = str(canonical.get("id") or "")
        merged_ids = [str(record.get("id") or "") for record in matches]
        raw_lithology_ids = [
            str(lithology_id)
            for record in matches
            for lithology_id in (
                record.get("lithology_ids") if isinstance(record.get("lithology_ids"), list) else []
            )
        ]
        canonical.update(dict(instruction.get("set")) if isinstance(instruction.get("set"), Mapping) else {})
        if raw_lithology_ids:
            canonical["lithology_ids"] = list(dict.fromkeys(raw_lithology_ids))
        corrected[field] = [record for record in records if record is canonical or record not in matches]
        for merged_id in merged_ids[1:]:
            aliases[merged_id] = canonical_id
        applied.append(
            {
                "operation": "merge_entities",
                "field": field,
                "old_ids": merged_ids,
                "canonical_id": canonical_id,
                "new_name": canonical.get("name"),
            }
        )

    for instruction in review.get("update_entities", []):
        if not isinstance(instruction, Mapping):
            continue
        field = str(instruction.get("field") or "")
        match_name = str(instruction.get("match_name") or "")
        match_names = [
            str(name)
            for name in instruction.get("match_names", [match_name] if match_name else [])
        ]
        match_order = instruction.get("match_order_bottom_to_top")
        records = corrected.get(field)
        matches = [
            record
            for record in records
            if isinstance(record, dict)
            and (not match_names or str(record.get("name") or "") in match_names)
            and (
                match_order is None
                or record.get("order_bottom_to_top") == match_order
            )
        ] if isinstance(records, list) else []
        if len(matches) != 1:
            if bool(instruction.get("optional", False)) and not matches:
                continue
            skipped.append(
                {
                    "operation": "update_entity",
                    "field": field,
                    "match_names": match_names,
                    "match_order_bottom_to_top": match_order,
                    "reason": "exact_match_count_changed",
                }
            )
            continue
        before = dict(matches[0])
        matches[0].update(dict(instruction.get("set")) if isinstance(instruction.get("set"), Mapping) else {})
        applied.append(
            {
                "operation": "update_entity",
                "field": field,
                "id": matches[0].get("id"),
                "old_name": before.get("name"),
                "new_name": matches[0].get("name"),
                "old_order": before.get("order_bottom_to_top"),
                "new_order": matches[0].get("order_bottom_to_top"),
            }
        )

    _replace_relation_aliases(corrected, aliases)
    return corrected, {
        "configured": True,
        "review_source": review.get("review_source"),
        "review_date": review.get("review_date"),
        "review_scope": review.get("review_scope"),
        "review_result": dict(review.get("review_result")) if isinstance(review.get("review_result"), Mapping) else {},
        "applied": applied,
        "skipped": skipped,
    }
