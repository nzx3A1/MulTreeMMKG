"""Neo4j 写入器。

将最终图谱（final_graph.json）批量写入 Neo4j：
    - 创建 driver / session
    - 执行 schema_graph.cypher 创建约束和索引
    - 使用 UNWIND + MERGE 批量写入节点 / 关系
    - 支持事务回滚与重试

最终落地数据存储在 Neo4j 中供查询与分析。
"""
from __future__ import annotations


def write_to_neo4j(graph: dict, neo4j_cfg) -> None:
    """把图谱批量写入 Neo4j。"""
    raise NotImplementedError
