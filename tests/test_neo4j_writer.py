"""src/graph/neo4j_writer 的集成测试。

使用 neo4j 测试容器或本地测试库验证：
    - 节点 / 关系被正确写入
    - 约束与索引按 schema_graph.cypher 创建
    - 写入失败能回滚
"""
from __future__ import annotations

import pytest


def test_neo4j_writer_writes_nodes_and_relations():
    """图谱节点与关系应被正确写入 Neo4j。"""
    pytest.skip("阶段六 Neo4j 写入层尚未实现；阶段一只验证基础设施。")
