"""图谱融合器。

将骨架图、多模态抽取结果、关系发现结果融合为一张统一的多模态图：
    - 实体 / 关系节点合并
    - 与文档骨架（Paper/Section/Chunk）建立溯源边
    - 输出 stage_12_enhanced_graph.json

对应阶段：12 增强融合图
"""
from __future_ import annotations


def merge_graph(
    skeleton: dict,
    aligned: dict,
    discovered_relations: list[dict],
) -> dict:
    """融合多源图为一张增强图。"""
    raise NotImplementedError
