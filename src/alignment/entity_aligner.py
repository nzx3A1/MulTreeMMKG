"""实体对齐器。

把文本/表格/图像/公式四路抽取出的实体做跨模态对齐：
    - 名称归一化、别名合并
    - Embedding 相似度匹配
    - LLM 判定（同义/上下位/不同实体）

输出整合后的实体列表，供去重器与图谱融合使用。
对应阶段：10 跨模态对齐
"""
from __future__ import annotations


def align_entities(
    text_entities: list[dict],
    table_entities: list[dict],
    image_entities: list[dict],
    formula_entities: list[dict],
    embedding_client,
    llm_client,
) -> list[dict]:
    """四路实体对齐，返回全局统一实体列表（含 cluster_id）。"""
    raise NotImplementedError
