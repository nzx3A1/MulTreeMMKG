"""文本实体/关系抽取器。

针对每个 chunk 的文本部分，使用 schema 约束的 LLM 调用抽取出
实体（Entity）与实体间关系（Relation），供后续对齐与图谱融合。

输出：stage_06_text_extraction.json
对应阶段：06 文本抽取
"""
from __future__ import annotations


def extract_from_text(chunks: list[dict], llm_client, schema_cfg) -> list[dict]:
    """从文本 chunk 中抽取实体和关系。"""
    raise NotImplementedError
