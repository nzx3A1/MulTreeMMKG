"""文本抽取 Graph 的最终一致性校验。"""
from __future__ import annotations

from model import Graph


def validate_graph(graph: Graph, text: str) -> list[str]:
    """只检查 Graph 的关系端点和事件参与者引用格式是否完整。"""

    # 中文说明：保留 text 参数兼容现有调用接口，但不再校验证据内容或 Schema 白名单。
    _ = text
    return list(graph.validate_references())


def ensure_valid_graph(graph: Graph, text: str) -> Graph:
    """执行最终结构一致性校验，将引用格式错误写入 Graph 元数据。"""

    errors = validate_graph(graph, text)
    validation = graph.metadata.extra.setdefault("validation", {})
    validation["final_consistency_passed"] = not errors
    validation["final_consistency_errors"] = errors
    return graph


__all__ = ["ensure_valid_graph", "validate_graph"]
