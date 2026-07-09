"""src/alignment 模块的单元测试。

覆盖：
    - entity_aligner 跨模态对齐
    - relation_aligner 冲突消解
    - deduplicator 重复检测
"""
from __future__ import annotations

import pytest


def test_entity_aligner_merges_synonyms():
    """同义实体应被合并到同一 cluster。"""
    pytest.skip("阶段五对齐层尚未实现；阶段一只验证基础设施。")
