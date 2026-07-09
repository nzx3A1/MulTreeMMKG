"""关系对齐器。

在实体对齐基础上，把各模态抽取出的关系做归一化：
    - 主语 / 宾语 映射到对齐后的实体 cluster
    - 关系谓语归一化到 relation_schema
    - 合并、冲突解决（如同一对实体的多个关系取置信度最高）

对应阶段：10 跨模态对齐
"""
from __future_ import annotations


def align_relations(
    relations: list[dict],
    aligned_entities: list[dict],
    schema_cfg,
) -> list[dict]:
    """对关系做归一化与冲突消解。"""
    raise NotImplementedError
