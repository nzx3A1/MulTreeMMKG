"""文本抽取 Graph 的最终一致性校验。"""
from __future__ import annotations

from model import Graph


def validate_graph(graph: Graph, text: str) -> list[str]:
    """检查引用完整性，以及实体和关系证据是否确实来自当前正文。"""

    errors = list(graph.validate_references())
    for entity in graph.entities:
        if not entity.provenance or entity.provenance not in text:
            errors.append(f"实体 {entity.id} 的 provenance 不在当前正文中")
    for relation in graph.relations:
        if not relation.provenance or relation.provenance not in text:
            errors.append(f"关系 {relation.id} 的 provenance 不在当前正文中")
    return errors


def ensure_valid_graph(graph: Graph, text: str) -> Graph:
    """执行最终一致性校验，将错误写入 Graph 元数据但不丢弃候选。"""

    errors = validate_graph(graph, text)
    validation = graph.metadata.extra.setdefault("validation", {})
    validation["final_consistency_passed"] = not errors
    validation["final_consistency_errors"] = errors
    return graph


__all__ = ["ensure_valid_graph", "validate_graph"]
