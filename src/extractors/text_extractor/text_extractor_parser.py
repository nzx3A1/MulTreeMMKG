"""将文本模型的候选结果解析为统一的 ``Graph`` 对象。"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from model import Entity, Event, Graph, Relation, SourceModality

from .schema_models import RelevantSchema


EVENT_TYPES = {
    "geological_process", "experiment", "observation", "charging",
    "migration", "accumulation", "other",
}


def _stable_id(prefix: str, *parts: Any) -> str:
    """根据文档及候选内容生成可复现 ID，支持跨 Chunk 实体去重。"""

    raw = "|".join(str(part or "").strip().lower() for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"


def _items(payload: Any, key: str) -> list[Mapping[str, Any]]:
    """从宽松的模型响应中安全读取指定候选数组。"""

    if not isinstance(payload, Mapping) or not isinstance(payload.get(key), list):
        return []
    return [item for item in payload[key] if isinstance(item, Mapping)]


def _has_evidence(text: str, provenance: Any) -> bool:
    """只接受能够在当前正文中逐字定位的非空证据。"""

    evidence = str(provenance or "").strip()
    return bool(evidence and evidence in text)


def parse_extraction_payload(
    payload: Mapping[str, Any],
    *,
    chunk: Mapping[str, Any],
    schema: RelevantSchema,
) -> Graph:
    """按局部 Schema、正文证据和引用完整性解析三类候选结果。"""

    text = str(chunk.get("text") or "")
    document_id = str(chunk.get("document_id") or "")
    chunk_id = str(chunk.get("id") or "")
    concept_map = schema.concept_map
    rejected: list[dict[str, str]] = []
    accepted_count = 0
    entities: list[Entity] = []
    temp_to_stable: dict[str, str] = {}

    for index, candidate in enumerate(_items(payload, "entities"), start=1):
        temp_id = str(candidate.get("temp_id") or candidate.get("id") or "").strip()
        name = str(candidate.get("name") or "").strip()
        entity_type = str(candidate.get("type") or "other").strip() or "other"
        validation_errors: list[str] = []
        if not name:
            validation_errors.append("missing_name")
        if entity_type not in concept_map:
            validation_errors.append("type_not_in_schema")
        if not _has_evidence(text, candidate.get("provenance")):
            validation_errors.append("evidence_not_in_text")

        stable_id = _stable_id("ent", document_id, name or temp_id or index, entity_type)
        if temp_id:
            temp_to_stable[temp_id] = stable_id
        concept = concept_map.get(entity_type)
        metadata = dict(candidate.get("metadata") or {})
        metadata["validation"] = {
            "passed": not validation_errors,
            "errors": validation_errors,
        }
        if validation_errors:
            rejected.append({
                "kind": "entity", "temp_id": temp_id,
                "reason": ",".join(validation_errors),
            })
        else:
            accepted_count += 1
        entities.append(Entity(
            id=stable_id, name=name, official_name=candidate.get("official_name"),
            type=entity_type,
            type_zh=(concept.zh_name if concept else None) or candidate.get("type_zh"),
            aliases=list(candidate.get("aliases") or []),
            attributes=dict(candidate.get("attributes") or {}),
            provenance=str(candidate.get("provenance") or ""), normalized_id=candidate.get("normalized_id"),
            metadata=metadata,
        ))

    relation_rules = {
        (item.source_schema, item.relation_en.upper(), item.target_schema): item
        for item in schema.relations
    }
    entity_types = {entity.id: entity.type for entity in entities}
    relations: list[Relation] = []
    for index, candidate in enumerate(_items(payload, "relations"), start=1):
        source_ref = str(candidate.get("source_id") or "").strip()
        target_ref = str(candidate.get("target_id") or "").strip()
        source_id = temp_to_stable.get(source_ref)
        target_id = temp_to_stable.get(target_ref)
        relation_type = str(candidate.get("type") or "other").strip().upper() or "OTHER"
        rule = relation_rules.get((entity_types.get(source_id, ""), relation_type, entity_types.get(target_id, "")))
        validation_errors: list[str] = []
        if not source_id:
            validation_errors.append("source_entity_not_found")
        if not target_id:
            validation_errors.append("target_entity_not_found")
        if not rule:
            validation_errors.append("relation_not_in_schema")
        if not _has_evidence(text, candidate.get("provenance")):
            validation_errors.append("evidence_not_in_text")

        # 悬空端点使用可复现占位 ID，使无效关系仍能进入 Graph 并被后续检查定位。
        resolved_source_id = source_id or _stable_id("missing_ent", document_id, source_ref or "source", index)
        resolved_target_id = target_id or _stable_id("missing_ent", document_id, target_ref or "target", index)
        metadata = dict(candidate.get("metadata") or {})
        metadata["validation"] = {
            "passed": not validation_errors,
            "errors": validation_errors,
            "original_source_id": source_ref,
            "original_target_id": target_ref,
        }
        temp_id = str(candidate.get("temp_id") or candidate.get("id") or "").strip()
        if validation_errors:
            rejected.append({
                "kind": "relation", "temp_id": temp_id,
                "reason": ",".join(validation_errors),
            })
        else:
            accepted_count += 1
        relations.append(Relation(
            id=_stable_id("rel", document_id, resolved_source_id, relation_type, resolved_target_id),
            type=relation_type,
            relation_name=candidate.get("relation_name") or candidate.get("official_name"),
            type_zh=(rule.relation_zh if rule else None) or candidate.get("type_zh"),
            source_id=resolved_source_id, target_id=resolved_target_id,
            attributes=dict(candidate.get("attributes") or {}), provenance=str(candidate.get("provenance") or ""),
            metadata=metadata,
        ))

    events: list[Event] = []
    for candidate in _items(payload, "events"):
        event_type = str(candidate.get("type") or "").strip().lower()
        name = str(candidate.get("name") or "").strip()
        if event_type not in EVENT_TYPES or not name or not _has_evidence(text, candidate.get("provenance")):
            rejected.append({"kind": "event", "temp_id": str(candidate.get("temp_id") or candidate.get("id") or ""), "reason": "invalid_type_name_or_evidence"})
            continue
        participants = list(dict.fromkeys(
            stable for item in candidate.get("participants") or []
            if (stable := temp_to_stable.get(str(item)))
        ))
        events.append(Event(
            id=_stable_id("evt", document_id, name, event_type), type=event_type, name=name,
            participants=participants, time=candidate.get("time"), location=candidate.get("location"),
            attributes=dict(candidate.get("attributes") or {}), provenance=str(candidate.get("provenance") or ""),
            metadata=dict(candidate.get("metadata") or {}),
        ))
        accepted_count += 1

    graph = Graph.from_chunk(document_id, chunk_id, SourceModality.TEXT, entities, relations, events,
                             raw_response=dict(payload), stage="stage_04_text_extraction_front")
    graph.metadata.extra["validation"] = {
        "passed": not rejected,
        "accepted_count": accepted_count,
        "rejected_count": len(rejected), "rejected": rejected,
        "retained_invalid_count": len(rejected),
    }
    graph.metadata.extra["schema_selection"] = schema.to_dict()
    return graph


__all__ = ["parse_extraction_payload"]
