"""构建 Stage 06 实体映射、未决候选和冲突报告。

报告与主图分离，便于批量运行后人工抽查，同时保证每个自动融合决定都可以追溯。
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .models import EntityMention, PairDecision, SourceKey


def decision_to_dict(decision: PairDecision) -> dict[str, Any]:
    """将内部实体对判定转换为 JSON 字典。"""

    return {
        "pair_id": decision.pair_id,
        "left": decision.left.to_dict(),
        "right": decision.right.to_dict(),
        "decision": decision.decision,
        "confidence": decision.confidence,
        "method": decision.method,
        "reason": decision.reason,
        "score": decision.score,
        "signals": dict(decision.signals),
    }


def _mention_summary(mention: EntityMention) -> dict[str, Any]:
    """构造报告中紧凑但足够复核的实体摘要。"""

    return {
        **mention.key.to_dict(),
        "name": mention.name,
        "aliases": list(mention.aliases),
        "schema_type": mention.schema_type,
        "schema_type_zh": mention.schema_type_zh or None,
        "raw_type": mention.raw_type,
        "scope": mention.scope,
        "modality": mention.modality,
        "attributes": mention.attributes,
        "provenance": mention.provenance,
    }


def build_entity_alignment_report(
    mapping_rows: Sequence[Mapping[str, Any]],
    global_entities: Sequence[Mapping[str, Any]],
    decisions: Sequence[PairDecision],
    source_file: str,
) -> dict[str, Any]:
    """构建逐 mention 到全局实体的映射报告。"""

    decision_counts = Counter(decision.decision for decision in decisions)
    method_counts = Counter(
        decision.method for decision in decisions if decision.decision == "SAME"
    )
    return {
        "_stage": "stage_06_multimodal_alignment",
        "_description": "原始实体 mention 到文档级全局实体的完整映射",
        "source_file": source_file,
        "statistics": {
            "source_entity_count": len(mapping_rows),
            "global_entity_count": len(global_entities),
            "fused_source_entity_count": len(mapping_rows) - len(global_entities),
            "multi_mention_global_entity_count": sum(
                int((entity.get("metadata") or {}).get("mention_count", 0)) > 1
                for entity in global_entities
            ),
            "candidate_decision_counts": dict(sorted(decision_counts.items())),
            "same_decision_method_counts": dict(sorted(method_counts.items())),
        },
        "mappings": list(mapping_rows),
    }


def build_unresolved_report(
    decisions: Sequence[PairDecision],
    mentions_by_key: Mapping[SourceKey, EntityMention],
    source_mapping: Mapping[SourceKey, str],
    source_file: str,
) -> dict[str, Any]:
    """输出仍位于不同全局实体簇的 UNCERTAIN 候选。"""

    rows: list[dict[str, Any]] = []
    for decision in decisions:
        if decision.decision != "UNCERTAIN":
            continue
        if source_mapping.get(decision.left) == source_mapping.get(decision.right):
            continue
        row = decision_to_dict(decision)
        row["left_entity"] = _mention_summary(mentions_by_key[decision.left])
        row["right_entity"] = _mention_summary(mentions_by_key[decision.right])
        rows.append(row)
    rows.sort(key=lambda item: (-float(item["score"]), str(item["pair_id"])))
    return {
        "_stage": "stage_06_multimodal_alignment",
        "_description": "自动规则与 LLM 均未能可靠裁决的候选实体对",
        "source_file": source_file,
        "statistics": {"unresolved_candidate_count": len(rows)},
        "candidates": rows,
    }


def build_conflict_report(
    global_entities: Sequence[Mapping[str, Any]],
    rejected_cluster_merges: Sequence[Mapping[str, Any]],
    relation_issues: Mapping[str, Sequence[Mapping[str, Any]]],
    unresolved_event_participants: Sequence[Mapping[str, Any]],
    source_file: str,
) -> dict[str, Any]:
    """汇总实体多值属性、簇拒绝、关系和事件端点异常。"""

    attribute_conflicts: list[dict[str, Any]] = []
    type_variations: list[dict[str, Any]] = []
    for entity in global_entities:
        for claim in entity.get("attribute_claims") or []:
            if claim.get("status") == "MULTI_VALUE":
                attribute_conflicts.append(
                    {
                        "global_entity_id": entity.get("id"),
                        "entity_name": entity.get("name"),
                        **dict(claim),
                    }
                )
        type_candidates = entity.get("schema_type_candidates") or []
        nonempty_types = {
            (candidate.get("schema_type_zh"), candidate.get("schema_type"))
            for candidate in type_candidates
            if candidate.get("schema_type_zh") or candidate.get("schema_type")
        }
        if len(nonempty_types) > 1:
            type_variations.append(
                {
                    "global_entity_id": entity.get("id"),
                    "entity_name": entity.get("name"),
                    "selected_schema_type_zh": entity.get("schema_type_zh"),
                    "schema_type_candidates": type_candidates,
                }
            )

    relation_sections = {
        key: list(value)
        for key, value in relation_issues.items()
    }
    statistics = {
        "attribute_conflict_count": len(attribute_conflicts),
        "type_variation_count": len(type_variations),
        "rejected_cluster_merge_count": len(rejected_cluster_merges),
        "unresolved_event_participant_count": len(unresolved_event_participants),
    }
    statistics.update(
        {f"{key}_count": len(value) for key, value in relation_sections.items()}
    )
    return {
        "_stage": "stage_06_multimodal_alignment",
        "_description": "融合过程中的多值、方向、自环、悬空端点和簇级约束记录",
        "source_file": source_file,
        "statistics": statistics,
        "attribute_conflicts": attribute_conflicts,
        "type_variations": type_variations,
        "rejected_cluster_merges": list(rejected_cluster_merges),
        **relation_sections,
        "unresolved_event_participants": list(unresolved_event_participants),
    }


__all__ = [
    "build_conflict_report",
    "build_entity_alignment_report",
    "build_unresolved_report",
    "decision_to_dict",
]
