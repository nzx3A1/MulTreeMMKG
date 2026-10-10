"""从 Stage 05 子图收集实体 mention、上下文和原始邻接结构。

所有实体使用 ``document_id + chunk_id + entity_id`` 建立来源键，杜绝不同 Chunk
复用局部 ID 时发生错误关联。
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .models import EntityMention, NeighborSignature, SourceKey
from .normalizer import normalize_name, normalize_relation
from .scope_rules import AlignmentRules


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """校验输入值为 JSON 对象，并在异常中保留字段位置。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 JSON 对象")
    return value


def require_sequence(value: Any, label: str) -> Sequence[Any]:
    """校验输入值为 JSON 数组，排除字符串等伪序列。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} 必须是 JSON 数组")
    return value


def graph_source(graph: Mapping[str, Any], graph_index: int) -> tuple[str, str, str, Mapping[str, Any]]:
    """读取一个子图必需的文档、Chunk 和模态来源字段。"""

    metadata = require_mapping(graph.get("metadata"), f"graphs[{graph_index}].metadata")
    document_id = str(metadata.get("document_id") or "").strip()
    chunk_id = str(metadata.get("chunk_id") or "").strip()
    modality = str(metadata.get("modality") or metadata.get("source_modality") or "").strip().lower()
    if not document_id:
        raise ValueError(f"graphs[{graph_index}].metadata 缺少 document_id")
    if not chunk_id:
        raise ValueError(f"graphs[{graph_index}].metadata 缺少 chunk_id")
    if not modality:
        raise ValueError(f"graphs[{graph_index}].metadata 缺少 modality")
    return document_id, chunk_id, modality, metadata


def _context_text(metadata: Mapping[str, Any], rules: AlignmentRules) -> str:
    """拼接正文、图表标题和引用文本，生成上下文比较字符串。"""

    values: list[str] = []
    for key in ("text", "caption", "references"):
        value = metadata.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values.extend(str(item) for item in value if item is not None)
        elif value is not None:
            values.append(str(value))
    return normalize_name(" ".join(values), rules.name_replacements)


def _aliases(entity: Mapping[str, Any]) -> tuple[str, ...]:
    """从 aliases 和 official_name 收集去重后的原始别名。"""

    values: list[str] = []
    raw_aliases = entity.get("aliases") or []
    if isinstance(raw_aliases, Sequence) and not isinstance(raw_aliases, (str, bytes, bytearray)):
        values.extend(str(item).strip() for item in raw_aliases if str(item).strip())
    official_name = str(entity.get("official_name") or "").strip()
    name = str(entity.get("name") or "").strip()
    if official_name and official_name != name:
        values.append(official_name)
    return tuple(sorted(set(values)))


def collect_entity_mentions(
    payload: Mapping[str, Any],
    rules: AlignmentRules,
) -> tuple[list[EntityMention], dict[tuple[int, str], SourceKey]]:
    """收集所有实体 mention，并从原始关系回填邻接签名。"""

    graphs = require_sequence(payload.get("graphs"), "stage_05.graphs")
    mentions: list[EntityMention] = []
    graph_keys: dict[tuple[int, str], SourceKey] = {}
    mentions_by_key: dict[SourceKey, EntityMention] = {}

    for graph_index, raw_graph in enumerate(graphs):
        graph = require_mapping(raw_graph, f"graphs[{graph_index}]")
        document_id, chunk_id, modality, metadata = graph_source(graph, graph_index)
        context_text = _context_text(metadata, rules)
        entities = require_sequence(graph.get("entities") or [], f"graphs[{graph_index}].entities")
        for entity_index, raw_entity in enumerate(entities):
            entity = require_mapping(
                raw_entity,
                f"graphs[{graph_index}].entities[{entity_index}]",
            )
            entity_id = str(entity.get("id") or "").strip()
            name = str(entity.get("name") or entity.get("official_name") or "").strip()
            if not entity_id:
                raise ValueError(f"graphs[{graph_index}].entities[{entity_index}] 缺少 id")
            if not name:
                raise ValueError(f"graphs[{graph_index}].entities[{entity_index}] 缺少 name")
            if (graph_index, entity_id) in graph_keys:
                raise ValueError(f"graphs[{graph_index}] 中实体 id 重复：{entity_id}")
            key = SourceKey(document_id, chunk_id, entity_id)
            if key in mentions_by_key:
                raise ValueError(f"实体来源键重复：{key.token}")
            aliases = _aliases(entity)
            alignment = entity.get("schema_alignment")
            alignment_status = (
                str(alignment.get("status") or "UNKNOWN")
                if isinstance(alignment, Mapping)
                else "UNKNOWN"
            )
            mention = EntityMention(
                key=key,
                graph_index=graph_index,
                entity_index=entity_index,
                document_id=document_id,
                chunk_id=chunk_id,
                modality=modality,
                scope=rules.entity_scope(entity),
                name=name,
                normalized_name=normalize_name(name, rules.name_replacements),
                aliases=aliases,
                normalized_aliases=tuple(
                    sorted(
                        {
                            normalize_name(alias, rules.name_replacements)
                            for alias in aliases
                            if normalize_name(alias, rules.name_replacements)
                        }
                    )
                ),
                schema_type=str(entity.get("schema_type") or entity.get("type") or "").strip(),
                schema_type_zh=str(entity.get("schema_type_zh") or "").strip(),
                raw_type=str(entity.get("raw_type") or entity.get("type") or "").strip(),
                attributes=deepcopy(dict(entity.get("attributes") or {})),
                provenance=deepcopy(entity.get("provenance")),
                context_text=context_text,
                schema_alignment_status=alignment_status,
                original=deepcopy(dict(entity)),
            )
            mentions.append(mention)
            graph_keys[(graph_index, entity_id)] = key
            mentions_by_key[key] = mention

    for graph_index, raw_graph in enumerate(graphs):
        graph = require_mapping(raw_graph, f"graphs[{graph_index}]")
        relations = require_sequence(graph.get("relations") or [], f"graphs[{graph_index}].relations")
        for relation_index, raw_relation in enumerate(relations):
            relation = require_mapping(
                raw_relation,
                f"graphs[{graph_index}].relations[{relation_index}]",
            )
            source_key = graph_keys.get((graph_index, str(relation.get("source_id") or "")))
            target_key = graph_keys.get((graph_index, str(relation.get("target_id") or "")))
            if source_key is None or target_key is None:
                continue
            source = mentions_by_key[source_key]
            target = mentions_by_key[target_key]
            relation_label = normalize_relation(
                relation.get("schema_relation")
                or relation.get("raw_relation")
                or relation.get("type")
                or relation.get("type_zh")
            )
            source.neighbors.append(
                NeighborSignature(
                    relation=relation_label,
                    neighbor_key=target_key,
                    neighbor_name=target.normalized_name,
                    neighbor_type_zh=target.schema_type_zh,
                )
            )
            target.neighbors.append(
                NeighborSignature(
                    relation=f"INVERSE:{relation_label}",
                    neighbor_key=source_key,
                    neighbor_name=source.normalized_name,
                    neighbor_type_zh=source.schema_type_zh,
                )
            )

    for mention in mentions:
        mention.neighbors.sort(
            key=lambda item: (
                item.relation,
                item.neighbor_name,
                item.neighbor_type_zh,
                item.neighbor_key,
            )
        )
    return sorted(mentions, key=lambda item: item.key), graph_keys


__all__ = [
    "collect_entity_mentions",
    "graph_source",
    "require_mapping",
    "require_sequence",
]
