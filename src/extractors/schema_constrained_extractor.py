"""Schema 约束抽取器（统一封装）。

为文本/表格/图像/公式各模态抽取提供统一的 schema 约束：
    - 根据 entity_schema / relation_schema 生成 prompt 片段
    - 解析 LLM/VLM 返回的 JSON
    - 用 jsonschema 做校验与兜底

其它抽取器内部通常会调用本模块以保证输出结构一致。
"""
from __future__ import annotations


def constrained_extract(
    content: str,
    modality: str,
    llm_or_vlm,
    schema_cfg,
) -> list[dict]:
    """在 schema 约束下抽取实体/关系，返回符合 schema 的统一 dict 列表。"""
    raise NotImplementedError
