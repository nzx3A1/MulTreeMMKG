"""表格抽取器。

将 MinerU 解析出的表格转为结构化行/列表示（必要时调用 LLM 理解表头与单元格），
抽取出表内涉及的实体与关系（如“油田-产层-产能”三元组）。

输出：stage_07_table_extraction.json
对应阶段：07 表格抽取
"""
from __future__ import annotations


def extract_from_tables(tables: list[dict], llm_client, schema_cfg) -> list[dict]:
    """从表格列表中抽取实体与关系。"""
    raise NotImplementedError
