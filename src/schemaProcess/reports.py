"""Schema Gap 报告构建。"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping, Sequence


def build_unmapped_entity_report(
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    """按 raw_type 汇总未映射实体，并保留逐条溯源记录。"""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    flat: list[dict[str, Any]] = []
    for entity, context in records:
        row = {
            "entity_id": entity.get("id"),
            "name": entity.get("name"),
            "raw_type": entity.get("raw_type", entity.get("type")),
            "type_zh": entity.get("type_zh"),
            "attributes": deepcopy(entity.get("attributes") or {}),
            "provenance": deepcopy(entity.get("provenance")),
            "reason": (entity.get("schema_alignment") or {}).get("reason"),
            **dict(context),
        }
        flat.append(row)
        groups[str(row["raw_type"] or "<EMPTY>")].append(row)

    grouped = []
    for raw_type, items in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        grouped.append(
            {
                "raw_type": raw_type,
                "count": len(items),
                "type_zh_values": sorted({str(item["type_zh"]) for item in items if item["type_zh"]}),
                "examples": items[:10],
            }
        )
    return {
        "_stage": "stage_05_schema_alignment",
        "_description": "未映射实体类型汇总，用于发现 Schema Gap",
        "statistics": {
            "unmapped_entity_count": len(flat),
            "unique_raw_type_count": len(grouped),
        },
        "unmapped_entity_types": grouped,
        "entities": flat,
    }


def build_unmapped_relation_report(
    records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    """按有向端点 Schema 与 raw_relation 汇总未映射关系。"""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    flat: list[dict[str, Any]] = []
    for relation, context in records:
        row = {
            "relation_id": relation.get("id"),
            "raw_relation": relation.get("raw_relation", relation.get("type")),
            "relation_name": relation.get("relation_name"),
            "type_zh": relation.get("type_zh"),
            "source_id": relation.get("source_id"),
            "source_name": relation.get("source_name"),
            "source_schema": relation.get("source_schema"),
            "target_id": relation.get("target_id"),
            "target_name": relation.get("target_name"),
            "target_schema": relation.get("target_schema"),
            "attributes": deepcopy(relation.get("attributes") or {}),
            "provenance": deepcopy(relation.get("provenance")),
            "reason": (relation.get("schema_alignment") or {}).get("reason"),
            **dict(context),
        }
        flat.append(row)
        key = (
            str(row["source_schema"] or "<UNMAPPED>"),
            str(row["raw_relation"] or "<EMPTY>"),
            str(row["target_schema"] or "<UNMAPPED>"),
        )
        groups[key].append(row)

    grouped = []
    for key, items in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        grouped.append(
            {
                "source_schema": key[0],
                "raw_relation": key[1],
                "target_schema": key[2],
                "count": len(items),
                "examples": items[:10],
            }
        )
    return {
        "_stage": "stage_05_schema_alignment",
        "_description": "未映射关系汇总，用于发现缺失的 Schema Relation",
        "statistics": {
            "unmapped_relation_count": len(flat),
            "unique_relation_pattern_count": len(grouped),
        },
        "unmapped_relation_patterns": grouped,
        "relations": flat,
    }


__all__ = ["build_unmapped_entity_report", "build_unmapped_relation_report"]
