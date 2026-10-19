"""图片抽取结果后处理：按同一 Chunk 内的实体名称合并节点并修复引用。"""
from __future__ import annotations

from copy import deepcopy

from model import Entity, Graph


def merge_same_name_entities(graph: Graph) -> Graph:
    """中文说明：保留同名实体的首个节点，汇集信息并重定向关系和事件引用。"""

    seen_names: set[str] = set()
    has_duplicate = False
    for entity in graph.entities:
        name = entity.name.strip()
        if name and name in seen_names:
            has_duplicate = True
            break
        if name:
            seen_names.add(name)
    if not has_duplicate:
        return graph

    names: dict[str, str] = {}
    canonical_entities: dict[str, Entity] = {}
    id_remap: dict[str, str] = {}
    merged = deepcopy(graph)

    for entity in merged.entities:
        name = entity.name.strip()
        if not name:
            canonical_entities[entity.id] = entity
            continue
        canonical_id = names.get(name)
        if canonical_id is None:
            names[name] = entity.id
            canonical_entities[entity.id] = entity
            continue

        canonical = canonical_entities[canonical_id]
        id_remap[entity.id] = canonical_id
        sources = canonical.metadata.setdefault("same_name_merge", {"source_entities": []})["source_entities"]
        if not sources:
            sources.append(canonical.to_dict(exclude={"metadata": {"same_name_merge"}}))
        sources.append(entity.to_dict())
        canonical.aliases = list(dict.fromkeys([*canonical.aliases, *entity.aliases]))
        if not canonical.official_name:
            canonical.official_name = entity.official_name
        canonical.provenance = "\n".join(
            dict.fromkeys(value for value in (canonical.provenance, entity.provenance) if value)
        )
        for key, value in entity.attributes.items():
            canonical.attributes.setdefault(key, value)

    if not id_remap:
        return graph

    merged.entities = list(canonical_entities.values())
    for relation in merged.relations:
        relation.source_id = id_remap.get(relation.source_id, relation.source_id)
        relation.target_id = id_remap.get(relation.target_id, relation.target_id)
        source = canonical_entities.get(relation.source_id)
        target = canonical_entities.get(relation.target_id)
        if source is not None:
            relation.source_name = source.name
            relation.source_type = source.type
        if target is not None:
            relation.target_name = target.name
            relation.target_type = target.type
    for event in merged.events:
        event.participants = list(dict.fromkeys(id_remap.get(item, item) for item in event.participants))
    merged.metadata.extra["same_name_merged_entity_count"] = len(id_remap)
    merged.metadata.extra["same_name_entity_id_map"] = id_remap
    return merged
