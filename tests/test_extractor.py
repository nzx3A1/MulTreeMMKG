"""src/extractors 模块的单元测试。

覆盖：
    - text_extractor / table_extractor / image_extractor / formula_extractor
    - schema_constrained_extractor 的输出 schema 校验
"""
from __future__ import annotations

import pytest


def test_text_extractor_returns_schema_valid_entities():
    """文本抽取器返回的实体应通过 entity_schema 校验。"""
    pytest.skip("阶段三/四抽取层尚未实现；阶段一只验证基础设施。")
