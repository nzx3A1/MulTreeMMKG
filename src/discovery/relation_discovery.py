"""关系发现器。

在已有 schema 之外，利用 LLM 从文档全文/摘要中挖掘潜在新关系：
    - 候选关系挖掘（开放式）
    - 关系归一与频次统计
    - 是否允许并入 schema 由 schema_config 控制

输出：stage_11_relation_discovery.json
对应阶段：11 关系发现
"""
from __future_ import annotations


def discover_relations(
    paper: dict,
    summary: dict,
    aligned_graph: dict,
    llm_client,
    schema_cfg,
) -> list[dict]:
    """从全文上下文中挖掘新关系，返回候选关系列表。"""
    raise NotImplementedError
