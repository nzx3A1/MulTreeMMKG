"""重写原始关系端点并将重复事实汇集到融合实体上。

关系融合不会丢弃 ``UNMAPPED`` 或 ``SKIPPED`` 数据；相同事实合并 evidence，
不同关系或不同限定条件继续作为独立边连接到新的全局节点。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .models import SourceKey
from .normalizer import normalize_relation, stable_id, stable_json
from .scope_rules import AlignmentRules
from .source_collector import graph_source, require_mapping, require_sequence


def _relation_semantics(relation: Mapping[str, Any]) -> tuple[str, str]:
    """返回关系的规范化比较键和首选显示名称。"""

    schema_relation = str(relation.get("schema_relation") or "").strip()
    raw_relation = str(
        relation.get("raw_relation")
        or relation.get("type")
        or relation.get("relation_name")
        or relation.get("type_zh")
        or ""
    ).strip()
    display = schema_relation or raw_relation or "UNSPECIFIED_RELATION"
    return normalize_relation(display), display


def _relation_qualifiers(
    relation: Mapping[str, Any],
    rules: AlignmentRules,
) -> dict[str, Any]:
    """从关系属性中剔除置信度等证据字段，保留事实限定条件。"""

    attributes = relation.get("attributes")
    if not isinstance(attributes, Mapping):
        return {}
    return {
        str(key): deepcopy(value)
        for key, value in attributes.items()
        if str(key).strip().lower() not in rules.relation_evidence_attribute_keys
        and value not in (None, "", [], {})
    }


def _evidence_row(
    relation: Mapping[str, Any],
    document_id: str,
    chunk_id: str,
    modality: str,
) -> dict[str, Any]:
    """构造一条不丢字段的原始关系证据记录。"""

    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "modality": modality,
        "original_relation_id": relation.get("id"),
        "raw_relation": relation.get("raw_relation", relation.get("type")),
        "schema_relation": relation.get("schema_relation"),
        "type_zh": relation.get("type_zh"),
        "attributes": deepcopy(relation.get("attributes") or {}),
        "provenance": deepcopy(relation.get("provenance")),
        "metadata": deepcopy(relation.get("metadata") or {}),
        "schema_alignment": deepcopy(relation.get("schema_alignment") or {}),
    }


def _most_common_text(values: Sequence[Any]) -> str | None:
    """选择出现次数最多的非空文本，并以字典序稳定打破平局。"""

    counts = Counter(str(value).strip() for value in values if str(value or "").strip())
    if not counts:
        return None
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _detect_direction_conflicts(
    relations: Sequence[Mapping[str, Any]],
    rules: AlignmentRules,
) -> list[dict[str, Any]]:
    """检测相同关系和限定条件在同一对节点间出现相反方向的情况。"""

    index: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for relation in relations:
        normalized = normalize_relation(relation.get("schema_relation") or relation.get("type"))
        if normalized.lower() in rules.symmetric_relations:
            continue
        source_id = str(relation.get("source_id") or "")
        target_id = str(relation.get("target_id") or "")
        qualifier_token = stable_json(relation.get("qualifiers") or {})
        key = (str(relation.get("document_id") or ""), normalized, source_id, target_id)
        inverse_key = (key[0], key[1], target_id, source_id)
        inverse = index.get((*inverse_key[:2], inverse_key[2], inverse_key[3], qualifier_token))
        if inverse is not None:
            pair = tuple(sorted((str(relation.get("id")), str(inverse.get("id")))))
            if pair not in seen:
                seen.add(pair)
                conflicts.append(
                    {
                        "kind": "OPPOSITE_DIRECTION",
                        "relation": normalized,
                        "left_global_relation_id": pair[0],
                        "right_global_relation_id": pair[1],
                        "source_id": source_id,
                        "target_id": target_id,
                    }
                )
        index[(*key, qualifier_token)] = relation
    return conflicts


def fuse_relations(
    payload: Mapping[str, Any],
    source_mapping: Mapping[SourceKey, str],
    global_entities: Sequence[Mapping[str, Any]],
    rules: AlignmentRules,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], int]:
    """重写所有关系端点、融合重复事实并返回异常分类。"""

    entity_by_id = {str(entity["id"]): entity for entity in global_entities}
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    issues: dict[str, list[dict[str, Any]]] = {
        "dangling_relations": [],
        "absorbed_self_loops": [],
        "retained_self_loops": [],
        "direction_conflicts": [],
    }
    processed_count = 0
    graphs = require_sequence(payload.get("graphs"), "stage_05.graphs")
    for graph_index, raw_graph in enumerate(graphs):
        graph = require_mapping(raw_graph, f"graphs[{graph_index}]")
        document_id, chunk_id, modality, _ = graph_source(graph, graph_index)
        relations = require_sequence(graph.get("relations") or [], f"graphs[{graph_index}].relations")
        for relation_index, raw_relation in enumerate(relations):
            relation = require_mapping(
                raw_relation,
                f"graphs[{graph_index}].relations[{relation_index}]",
            )
            processed_count += 1
            source_key = SourceKey(document_id, chunk_id, str(relation.get("source_id") or ""))
            target_key = SourceKey(document_id, chunk_id, str(relation.get("target_id") or ""))
            source_global_id = source_mapping.get(source_key)
            target_global_id = source_mapping.get(target_key)
            evidence = _evidence_row(relation, document_id, chunk_id, modality)
            if source_global_id is None or target_global_id is None:
                issues["dangling_relations"].append(
                    {
                        "reason": "原始关系端点未找到实体映射",
                        "missing_source": source_global_id is None,
                        "missing_target": target_global_id is None,
                        "source_key": source_key.to_dict(),
                        "target_key": target_key.to_dict(),
                        "evidence": evidence,
                    }
                )
                continue

            relation_key, display = _relation_semantics(relation)
            qualifiers = _relation_qualifiers(relation, rules)
            row = {
                "document_id": document_id,
                "source_global_id": source_global_id,
                "target_global_id": target_global_id,
                "relation_key": relation_key,
                "display": display,
                "qualifiers": qualifiers,
                "relation": relation,
                "evidence": evidence,
            }
            if source_global_id == target_global_id and relation_key.lower() in rules.absorb_self_loop_relations:
                issues["absorbed_self_loops"].append(
                    {
                        "global_entity_id": source_global_id,
                        "relation": relation_key,
                        "reason": "同义或共指关系已被实体融合吸收",
                        "evidence": evidence,
                    }
                )
                continue
            if source_global_id == target_global_id:
                issues["retained_self_loops"].append(
                    {
                        "global_entity_id": source_global_id,
                        "relation": relation_key,
                        "reason": "实体融合后产生自环，关系保留并等待复核",
                        "evidence": evidence,
                    }
                )
            group_key = (
                document_id,
                source_global_id,
                relation_key,
                target_global_id,
                stable_json(qualifiers),
            )
            groups[group_key].append(row)

    global_relations: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        document_id, source_id, relation_key, target_id, _ = group_key
        rows = groups[group_key]
        evidences = sorted(
            (row["evidence"] for row in rows),
            key=lambda item: (
                str(item["chunk_id"]),
                str(item["original_relation_id"]),
            ),
        )
        relations = [row["relation"] for row in rows]
        schema_relation = _most_common_text(
            [relation.get("schema_relation") for relation in relations]
        )
        display_type = schema_relation or _most_common_text([row["display"] for row in rows])
        type_zh = _most_common_text(
            [
                (relation.get("schema_alignment") or {}).get("relation_zh")
                if isinstance(relation.get("schema_alignment"), Mapping)
                else None
                for relation in relations
            ]
            + [relation.get("type_zh") for relation in relations]
        )
        statuses = sorted(
            {
                str((relation.get("schema_alignment") or {}).get("status") or "UNKNOWN")
                for relation in relations
                if isinstance(relation, Mapping)
            }
        )
        modalities = sorted({str(evidence["modality"]) for evidence in evidences})
        chunks = sorted({str(evidence["chunk_id"]) for evidence in evidences})
        global_relation_id = stable_id("gr", *group_key)
        source_entity = entity_by_id[source_id]
        target_entity = entity_by_id[target_id]
        global_relations.append(
            {
                "id": global_relation_id,
                "document_id": document_id,
                "type": display_type,
                "relation_name": display_type,
                "type_zh": type_zh,
                "source_id": source_id,
                "source_name": source_entity.get("name"),
                "source_schema": source_entity.get("schema_type"),
                "target_id": target_id,
                "target_name": target_entity.get("name"),
                "target_schema": target_entity.get("schema_type"),
                "schema_relation": schema_relation,
                "normalized_relation": relation_key,
                "attributes": deepcopy(rows[0]["qualifiers"]),
                "qualifiers": deepcopy(rows[0]["qualifiers"]),
                "raw_relation_variants": sorted(
                    {
                        str(relation.get("raw_relation") or relation.get("type") or "")
                        for relation in relations
                        if relation.get("raw_relation") or relation.get("type")
                    }
                ),
                "schema_alignment_statuses": statuses,
                "evidence": evidences,
                "metadata": {
                    "fusion_status": "FUSED" if len(evidences) > 1 else "SINGLETON",
                    "evidence_count": len(evidences),
                    "source_chunks": chunks,
                    "source_modalities": modalities,
                    "cross_modal": len(modalities) > 1,
                    "self_loop_after_fusion": source_id == target_id,
                },
            }
        )

    global_relations.sort(key=lambda item: (item["document_id"], item["id"]))
    issues["direction_conflicts"] = _detect_direction_conflicts(global_relations, rules)
    return global_relations, issues, processed_count


def rewrite_events(
    payload: Mapping[str, Any],
    source_mapping: Mapping[SourceKey, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """保留全部原始事件，并把参与者重写为融合后的全局实体 ID。"""

    rewritten: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    graphs = require_sequence(payload.get("graphs"), "stage_05.graphs")
    for graph_index, raw_graph in enumerate(graphs):
        graph = require_mapping(raw_graph, f"graphs[{graph_index}]")
        document_id, chunk_id, modality, _ = graph_source(graph, graph_index)
        events = require_sequence(graph.get("events") or [], f"graphs[{graph_index}].events")
        for event_index, raw_event in enumerate(events):
            event = require_mapping(raw_event, f"graphs[{graph_index}].events[{event_index}]")
            original_id = str(event.get("id") or f"event_{event_index}")
            original_participants = [str(item) for item in event.get("participants") or []]
            participant_mapping: list[dict[str, Any]] = []
            global_participants: list[str] = []
            for participant in original_participants:
                source_key = SourceKey(document_id, chunk_id, participant)
                global_id = source_mapping.get(source_key)
                participant_mapping.append(
                    {
                        **source_key.to_dict(),
                        "global_entity_id": global_id,
                    }
                )
                if global_id is not None and global_id not in global_participants:
                    global_participants.append(global_id)
                elif global_id is None:
                    unresolved.append(
                        {
                            "document_id": document_id,
                            "chunk_id": chunk_id,
                            "original_event_id": original_id,
                            "missing_participant": participant,
                        }
                    )
            result = deepcopy(dict(event))
            result["id"] = stable_id("gev", document_id, chunk_id, original_id)
            result["original_event_id"] = original_id
            result["document_id"] = document_id
            result["chunk_id"] = chunk_id
            result["modality"] = modality
            result["original_participants"] = original_participants
            result["participants"] = global_participants
            result["participant_mapping"] = participant_mapping
            rewritten.append(result)
    rewritten.sort(key=lambda item: (item["document_id"], item["chunk_id"], item["id"]))
    return rewritten, unresolved


__all__ = ["fuse_relations", "rewrite_events"]
