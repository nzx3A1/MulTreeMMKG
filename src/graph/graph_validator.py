"""统一 Graph 结构与 Schema 约束校验器。"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model import Graph


def validate_graph(graph: Graph | Mapping[str, Any], graph_schema: Any | None = None) -> dict[str, Any]:
    """校验引用、ID、Schema 类型三元组和来源信息，返回错误与警告报告。"""

    errors: list[str] = []
    warnings: list[str] = []
    try:
        graph_model = graph if isinstance(graph, Graph) else Graph(**dict(graph))
    except Exception as exc:
        return {"ok": False, "errors": [f"Graph 数据结构不合法：{exc}"], "warnings": []}

    errors.extend(graph_model.validate_references())
    _check_unique_ids(graph_model, errors)
    _check_provenance(graph_model, warnings)
    if graph_schema is not None:
        _check_schema(graph_model, graph_schema, errors)
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def _check_unique_ids(graph: Graph, errors: list[str]) -> None:
    """检查实体、关系和事件各自 ID 是否重复。"""

    for label, values in (
        ("实体", [entity.id for entity in graph.entities]),
        ("关系", [relation.id for relation in graph.relations]),
        ("事件", [event.id for event in graph.events]),
    ):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            errors.append(f"{label} ID 重复：{', '.join(duplicates)}")


def _check_provenance(graph: Graph, warnings: list[str]) -> None:
    """提醒缺少原文证据字符串的抽取结果。"""

    for label, items in (
        ("实体", graph.entities),
        ("关系", graph.relations),
        ("事件", graph.events),
    ):
        for item in items:
            if not item.provenance:
                warnings.append(f"{label} {item.id} 缺少 provenance")


def _check_schema(graph: Graph, graph_schema: Any, errors: list[str]) -> None:
    """在传入 RelevantSchema 时检查实体类型和关系方向三元组。"""

    concepts = getattr(graph_schema, "concepts", ())
    relations = getattr(graph_schema, "relations", ())
    allowed_types = {concept.schema for concept in concepts}
    allowed_relations = {
        (relation.source_schema, relation.relation_en, relation.target_schema) for relation in relations
    }
    entity_by_id = {entity.id: entity for entity in graph.entities}
    for entity in graph.entities:
        if allowed_types and entity.type not in allowed_types:
            errors.append(f"实体 {entity.id} 的类型不在局部 Schema：{entity.type}")
    for relation in graph.relations:
        source = entity_by_id.get(relation.source_id)
        target = entity_by_id.get(relation.target_id)
        if source is None or target is None:
            continue
        triple = (source.type, relation.type, target.type)
        if allowed_relations and triple not in allowed_relations:
            errors.append(f"关系 {relation.id} 不符合 Schema 方向三元组：{triple}")
