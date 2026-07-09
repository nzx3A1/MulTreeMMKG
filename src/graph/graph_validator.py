"""图谱校验器。

对融合后的图做一致性 / 完整性校验：
    - 节点引用完整性（所有关系 src/dst 必须存在）
    - schema 一致性（节点标签 / 关系类型必须符合 graph_schema）
    - 属性类型 / 必填项校验
    - 报告并返回通过 / 失败项列表

对应阶段：12 增强图校验（在写入 Neo4j 前执行）
"""
from __future_ import annotations


def validate_graph(graph: dict, graph_schema) -> dict:
    """校验图谱，返回包含 ok / errors / warnings 的报告。"""
    raise NotImplementedError
