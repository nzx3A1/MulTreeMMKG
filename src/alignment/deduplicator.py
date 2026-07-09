"""去重器。

针对实体和关系做精确 / 模糊去重，避免重复节点/边进入最终图谱。
- 实体：name + type 完全相同视为同一候选；embedding 相似度 > 阈值视为同一候选
- 关系：(src_id, type, dst_id) 三元组相同视为重复

输出：stage_10_modal_alignment_dedup.json
对应阶段：10 跨模态对齐去重
"""
from __future_ import annotations


def deduplicate(aligned: dict) -> dict:
    """对实体与关系执行去重，返回去重后的统一结构。"""
    raise NotImplementedError
