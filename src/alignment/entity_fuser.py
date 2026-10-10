"""把受约束实体簇物化为文档级全局实体。

全局实体保存统一名称和主类型，同时把全部原始实体、属性声明和 Schema 类型候选
作为可追溯证据保留下来，避免融合造成信息覆盖。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .models import EntityMention, PairDecision, SourceKey
from .normalizer import stable_id, stable_json


_STATUS_WEIGHTS = {
    "POST_SCHEMA_MAPPED": 5.0,
    "RULE_MAPPED": 4.0,
    "SEMANTIC_MAPPED": 3.0,
    "SKIPPED": 2.0,
    "UNMAPPED": 1.0,
}


def _source_reference(mention: EntityMention) -> dict[str, Any]:
    """构造实体属性和类型候选共用的来源引用。"""

    return {
        **mention.key.to_dict(),
        "modality": mention.modality,
    }


def _canonical_name(cluster: Sequence[EntityMention]) -> str:
    """按规范化名称频次、正式名称和信息量选择全局实体名称。"""

    normalized_counts = Counter(mention.normalized_name for mention in cluster if mention.normalized_name)
    if not normalized_counts:
        return sorted((mention.name for mention in cluster), key=lambda value: (-len(value), value))[0]
    dominant_normalized = sorted(
        normalized_counts,
        key=lambda value: (-normalized_counts[value], -len(value), value),
    )[0]
    candidates: set[str] = set()
    for mention in cluster:
        if mention.normalized_name != dominant_normalized:
            continue
        official_name = str(mention.original.get("official_name") or "").strip()
        candidates.add(official_name or mention.name)
    return sorted(candidates, key=lambda value: (-len(value), value))[0]


def _type_candidates(cluster: Sequence[EntityMention]) -> list[dict[str, Any]]:
    """按中文 Schema 类型汇总带质量权重的候选类型。"""

    groups: dict[tuple[str, str], list[EntityMention]] = defaultdict(list)
    for mention in cluster:
        groups[(mention.schema_type_zh, mention.schema_type)].append(mention)
    result: list[dict[str, Any]] = []
    for (type_zh, schema_type), mentions in groups.items():
        weighted_support = sum(
            _STATUS_WEIGHTS.get(mention.schema_alignment_status, 1.0)
            for mention in mentions
        )
        result.append(
            {
                "schema_type_zh": type_zh or None,
                "schema_type": schema_type or None,
                "mention_count": len(mentions),
                "weighted_support": weighted_support,
                "alignment_statuses": sorted(
                    {mention.schema_alignment_status for mention in mentions}
                ),
                "sources": [_source_reference(mention) for mention in mentions],
            }
        )
    return sorted(
        result,
        key=lambda item: (
            -float(item["weighted_support"]),
            -int(item["mention_count"]),
            str(item["schema_type_zh"] or ""),
            str(item["schema_type"] or ""),
        ),
    )


def _attribute_claims(cluster: Sequence[EntityMention]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """汇总所有属性值，仅把无冲突的单值属性写入主 attributes。"""

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    original_keys: dict[str, str] = {}
    for mention in cluster:
        for key, value in mention.attributes.items():
            normalized_key = str(key).strip().lower()
            original_keys.setdefault(normalized_key, str(key))
            value_token = stable_json(value)
            claim = grouped[normalized_key].setdefault(
                value_token,
                {"value": deepcopy(value), "sources": []},
            )
            claim["sources"].append(_source_reference(mention))

    merged_attributes: dict[str, Any] = {}
    claims: list[dict[str, Any]] = []
    for normalized_key in sorted(grouped):
        values = grouped[normalized_key]
        ordered_values = sorted(values.values(), key=lambda item: stable_json(item["value"]))
        if len(ordered_values) == 1:
            merged_attributes[original_keys[normalized_key]] = deepcopy(ordered_values[0]["value"])
        claims.append(
            {
                "attribute": original_keys[normalized_key],
                "status": "CONSISTENT" if len(ordered_values) == 1 else "MULTI_VALUE",
                "values": ordered_values,
            }
        )
    return merged_attributes, claims


def _decision_payload(decision: PairDecision) -> dict[str, Any]:
    """把实体对判定转换为报告和全局实体可复用的字典。"""

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


def build_global_entities(
    clusters: Sequence[Sequence[EntityMention]],
    decisions: Sequence[PairDecision],
) -> tuple[list[dict[str, Any]], dict[SourceKey, str], list[dict[str, Any]]]:
    """物化全局实体，并返回来源映射和逐 mention 对齐记录。"""

    decisions_by_member: dict[SourceKey, list[PairDecision]] = defaultdict(list)
    for decision in decisions:
        if decision.decision != "SAME":
            continue
        decisions_by_member[decision.left].append(decision)
        decisions_by_member[decision.right].append(decision)

    global_entities: list[dict[str, Any]] = []
    source_mapping: dict[SourceKey, str] = {}
    mapping_rows: list[dict[str, Any]] = []
    for cluster in clusters:
        members = sorted(cluster, key=lambda item: item.key)
        document_id = members[0].document_id
        global_id = stable_id(
            "ge",
            document_id,
            [mention.key.token for mention in members],
        )
        for mention in members:
            source_mapping[mention.key] = global_id

        candidates = _type_candidates(members)
        primary_type = candidates[0]
        canonical_name = _canonical_name(members)
        merged_attributes, attribute_claims = _attribute_claims(members)
        member_keys = {mention.key for mention in members}
        merge_decisions = sorted(
            {
                decision.pair_id: decision
                for mention in members
                for decision in decisions_by_member.get(mention.key, [])
                if decision.left in member_keys and decision.right in member_keys
            }.values(),
            key=lambda item: item.pair_id,
        )
        aliases = sorted(
            {
                value
                for mention in members
                for value in (mention.name, *mention.aliases)
                if value and value != canonical_name
            }
        )
        modalities = sorted({mention.modality for mention in members})
        chunks = sorted({mention.chunk_id for mention in members})
        global_entities.append(
            {
                "id": global_id,
                "name": canonical_name,
                "official_name": canonical_name,
                "type": primary_type["schema_type"],
                "type_zh": primary_type["schema_type_zh"],
                "schema_type": primary_type["schema_type"],
                "schema_type_zh": primary_type["schema_type_zh"],
                "aliases": aliases,
                "attributes": merged_attributes,
                "attribute_claims": attribute_claims,
                "schema_type_candidates": candidates,
                "mentions": [
                    {
                        **_source_reference(mention),
                        "scope": mention.scope,
                        "original_entity": deepcopy(mention.original),
                    }
                    for mention in members
                ],
                "metadata": {
                    "document_id": document_id,
                    "scope": members[0].scope,
                    "mention_count": len(members),
                    "source_chunks": chunks,
                    "source_modalities": modalities,
                    "cross_modal": len(modalities) > 1,
                    "fusion_status": "FUSED" if len(members) > 1 else "SINGLETON",
                    "merge_decisions": [_decision_payload(item) for item in merge_decisions],
                },
            }
        )

        for mention in members:
            related = sorted(decisions_by_member.get(mention.key, []), key=lambda item: item.pair_id)
            mapping_rows.append(
                {
                    **_source_reference(mention),
                    "name": mention.name,
                    "schema_type": mention.schema_type,
                    "schema_type_zh": mention.schema_type_zh or None,
                    "scope": mention.scope,
                    "global_entity_id": global_id,
                    "global_entity_name": canonical_name,
                    "cluster_size": len(members),
                    "mapping_status": "FUSED" if len(members) > 1 else "SINGLETON",
                    "supporting_decisions": [_decision_payload(item) for item in related],
                }
            )

    global_entities.sort(key=lambda item: (item["metadata"]["document_id"], item["id"]))
    mapping_rows.sort(
        key=lambda item: (
            item["document_id"],
            item["chunk_id"],
            item["original_entity_id"],
        )
    )
    return global_entities, source_mapping, mapping_rows


__all__ = ["build_global_entities"]
