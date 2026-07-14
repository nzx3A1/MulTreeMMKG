"""将文本模型的候选结果解析为统一的 ``Graph`` 对象。"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from model import Entity, Event, Graph, Relation, SourceModality

from .schema_models import RelevantSchema


def _stable_id(prefix: str, *parts: Any) -> str:
    """根据文档及候选内容生成可复现 ID，支持跨 Chunk 实体去重。"""

    raw = "|".join(str(part or "").strip().lower() for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"


def _items(payload: Any, key: str) -> list[Mapping[str, Any]]:
    """从宽松的模型响应中安全读取指定候选数组。"""

    if not isinstance(payload, Mapping) or not isinstance(payload.get(key), list):
        return []
    return [item for item in payload[key] if isinstance(item, Mapping)]


def parse_extraction_payload(
    payload: Mapping[str, Any],
    *,
    chunk: Mapping[str, Any],
    schema: RelevantSchema,
) -> Graph:
    """只按字段格式和对象引用解析候选，不校验类型是否属于 Schema 白名单。"""

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
        raw_entity_type = candidate.get("type")
        entity_type = str(raw_entity_type or "").strip() or "other"
        validation_errors: list[str] = []
        if not name:
            validation_errors.append("missing_name")
        if not str(raw_entity_type or "").strip():
            validation_errors.append("missing_type")
        if candidate.get("aliases") is not None and not isinstance(candidate.get("aliases"), list):
            validation_errors.append("invalid_aliases_format")
        if candidate.get("attributes") is not None and not isinstance(candidate.get("attributes"), Mapping):
            validation_errors.append("invalid_attributes_format")
        if candidate.get("metadata") is not None and not isinstance(candidate.get("metadata"), Mapping):
            validation_errors.append("invalid_metadata_format")
        if candidate.get("provenance") is not None and not isinstance(candidate.get("provenance"), str):
            validation_errors.append("invalid_provenance_format")

        stable_id = _stable_id("ent", document_id, name or temp_id or index, entity_type)
        if temp_id:
            temp_to_stable[temp_id] = stable_id
        concept = concept_map.get(entity_type)
        metadata = dict(candidate.get("metadata") or {}) if isinstance(candidate.get("metadata") or {}, Mapping) else {}
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
            aliases=list(candidate.get("aliases") or []) if isinstance(candidate.get("aliases") or [], list) else [],
            attributes=dict(candidate.get("attributes") or {}) if isinstance(candidate.get("attributes") or {}, Mapping) else {},
            provenance=str(candidate.get("provenance") or ""), normalized_id=candidate.get("normalized_id"),
            metadata=metadata,
        ))

    relation_rules = {
        (item.source_schema, item.relation_en.upper(), item.target_schema): item
        for item in schema.relations
    }
    entities_by_id = {entity.id: entity for entity in entities}
    relations: list[Relation] = []
    for index, candidate in enumerate(_items(payload, "relations"), start=1):
        source_ref = str(candidate.get("source_id") or "").strip()
        target_ref = str(candidate.get("target_id") or "").strip()
        source_id = temp_to_stable.get(source_ref)
        target_id = temp_to_stable.get(target_ref)
        raw_relation_type = candidate.get("type")
        relation_type = str(raw_relation_type or "").strip().upper() or "OTHER"
        # 中文说明：已知 Schema 关系只用于补充中文名，不参与候选通过与否的判断。
        rule = next(
            (item for key, item in relation_rules.items() if key[1] == relation_type),
            None,
        )
        validation_errors: list[str] = []
        if not source_id:
            validation_errors.append("source_entity_not_found")
        if not target_id:
            validation_errors.append("target_entity_not_found")
        if not str(raw_relation_type or "").strip():
            validation_errors.append("missing_type")
        for field_name in ("source_name", "source_type", "target_name", "target_type"):
            field_value = candidate.get(field_name)
            if field_value is None or not str(field_value).strip():
                validation_errors.append(f"missing_{field_name}")
            elif not isinstance(field_value, str):
                validation_errors.append(f"invalid_{field_name}_format")
        if candidate.get("attributes") is not None and not isinstance(candidate.get("attributes"), Mapping):
            validation_errors.append("invalid_attributes_format")
        if candidate.get("metadata") is not None and not isinstance(candidate.get("metadata"), Mapping):
            validation_errors.append("invalid_metadata_format")
        if candidate.get("provenance") is not None and not isinstance(candidate.get("provenance"), str):
            validation_errors.append("invalid_provenance_format")

        # 悬空端点使用可复现占位 ID，使无效关系仍能进入 Graph 并被后续检查定位。
        resolved_source_id = source_id or _stable_id("missing_ent", document_id, source_ref or "source", index)
        resolved_target_id = target_id or _stable_id("missing_ent", document_id, target_ref or "target", index)
        source_entity = entities_by_id.get(resolved_source_id)
        target_entity = entities_by_id.get(resolved_target_id)
        # 中文说明：最终关系中的端点名称和类型统一取自已解析实体，保证四个冗余字段与 ID 同步。
        source_name = source_entity.name if source_entity else str(candidate.get("source_name") or "")
        source_type = source_entity.type if source_entity else str(candidate.get("source_type") or "")
        target_name = target_entity.name if target_entity else str(candidate.get("target_name") or "")
        target_type = target_entity.type if target_entity else str(candidate.get("target_type") or "")
        metadata = dict(candidate.get("metadata") or {}) if isinstance(candidate.get("metadata") or {}, Mapping) else {}
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
            source_id=resolved_source_id, source_name=source_name, source_type=source_type,
            target_id=resolved_target_id, target_name=target_name, target_type=target_type,
            attributes=dict(candidate.get("attributes") or {}) if isinstance(candidate.get("attributes") or {}, Mapping) else {},
            provenance=str(candidate.get("provenance") or ""),
            metadata=metadata,
        ))

    events: list[Event] = []
    for candidate in _items(payload, "events"):
        raw_event_type = candidate.get("type")
        event_type = str(raw_event_type or "").strip().lower() or "other"
        name = str(candidate.get("name") or "").strip()
        validation_errors: list[str] = []
        if not name:
            validation_errors.append("missing_name")
        if not str(raw_event_type or "").strip():
            validation_errors.append("missing_type")
        if candidate.get("participants") is not None and not isinstance(candidate.get("participants"), list):
            validation_errors.append("invalid_participants_format")
        if candidate.get("attributes") is not None and not isinstance(candidate.get("attributes"), Mapping):
            validation_errors.append("invalid_attributes_format")
        if candidate.get("metadata") is not None and not isinstance(candidate.get("metadata"), Mapping):
            validation_errors.append("invalid_metadata_format")
        if candidate.get("provenance") is not None and not isinstance(candidate.get("provenance"), str):
            validation_errors.append("invalid_provenance_format")
        participants = list(dict.fromkeys(
            stable for item in candidate.get("participants") or []
            if (stable := temp_to_stable.get(str(item)))
        )) if isinstance(candidate.get("participants") or [], list) else []
        metadata = dict(candidate.get("metadata") or {}) if isinstance(candidate.get("metadata") or {}, Mapping) else {}
        metadata["validation"] = {
            "passed": not validation_errors,
            "errors": validation_errors,
        }
        temp_id = str(candidate.get("temp_id") or candidate.get("id") or "")
        if validation_errors:
            rejected.append({
                "kind": "event",
                "temp_id": temp_id,
                "reason": ",".join(validation_errors),
            })
        else:
            accepted_count += 1
        events.append(Event(
            id=_stable_id("evt", document_id, name, event_type), type=event_type, name=name,
            participants=participants, time=candidate.get("time"), location=candidate.get("location"),
            attributes=dict(candidate.get("attributes") or {}) if isinstance(candidate.get("attributes") or {}, Mapping) else {},
            provenance=str(candidate.get("provenance") or ""), metadata=metadata,
        ))

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
