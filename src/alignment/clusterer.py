"""将 SAME 实体对聚合为满足硬约束的文档级实体簇。

聚类使用受约束并查集：即使 A≈B、B≈C，也只有 A 与 C 不存在类型、作用域或
身份属性冲突时才允许传递合并。
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .entity_matcher import EntityPairMatcher
from .models import EntityMention, PairDecision, SourceKey


class _UnionFind:
    """实现根节点稳定可重复的轻量并查集。"""

    def __init__(self, keys: Sequence[SourceKey]) -> None:
        """为每个来源实体创建独立集合。"""

        self.parent = {key: key for key in keys}

    def find(self, key: SourceKey) -> SourceKey:
        """查找集合根节点并执行路径压缩。"""

        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: SourceKey, right: SourceKey) -> SourceKey:
        """合并两个集合，并固定选择字典序较小的根节点。"""

        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        root, child = sorted((left_root, right_root))
        self.parent[child] = root
        return root


class ConstrainedEntityClusterer:
    """按照已确认 SAME 的实体对执行完整簇级约束检查。"""

    def __init__(self, matcher: EntityPairMatcher) -> None:
        """保存提供硬冲突检查能力的实体匹配器。"""

        self.matcher = matcher

    def _cluster_members(
        self,
        union_find: _UnionFind,
        mentions_by_key: Mapping[SourceKey, EntityMention],
    ) -> dict[SourceKey, list[EntityMention]]:
        """按照当前并查集根节点收集实体成员。"""

        result: dict[SourceKey, list[EntityMention]] = {}
        for key in sorted(mentions_by_key):
            result.setdefault(union_find.find(key), []).append(mentions_by_key[key])
        return result

    def _can_merge(
        self,
        left_members: Sequence[EntityMention],
        right_members: Sequence[EntityMention],
    ) -> str | None:
        """逐对检查两个实体簇是否存在不可合并冲突。"""

        for left in left_members:
            for right in right_members:
                conflict = self.matcher.hard_conflict(left, right)
                if conflict:
                    return f"{left.key.token} 与 {right.key.token}：{conflict}"
        return None

    def cluster(
        self,
        mentions: Sequence[EntityMention],
        decisions: Sequence[PairDecision],
    ) -> tuple[list[list[EntityMention]], list[dict[str, object]]]:
        """生成实体簇，并返回因簇级约束被拒绝的 SAME 判定。"""

        mentions_by_key = {mention.key: mention for mention in mentions}
        union_find = _UnionFind(list(mentions_by_key))
        rejected: list[dict[str, object]] = []
        same_decisions = sorted(
            (decision for decision in decisions if decision.decision == "SAME"),
            key=lambda item: (-item.confidence, -item.score, item.pair_id),
        )
        for decision in same_decisions:
            left_root = union_find.find(decision.left)
            right_root = union_find.find(decision.right)
            if left_root == right_root:
                continue
            current = self._cluster_members(union_find, mentions_by_key)
            conflict = self._can_merge(current[left_root], current[right_root])
            if conflict:
                rejected.append(
                    {
                        "pair_id": decision.pair_id,
                        "left": decision.left.to_dict(),
                        "right": decision.right.to_dict(),
                        "reason": conflict,
                        "original_method": decision.method,
                        "original_confidence": decision.confidence,
                    }
                )
                continue
            union_find.union(left_root, right_root)

        clusters = list(self._cluster_members(union_find, mentions_by_key).values())
        for cluster in clusters:
            cluster.sort(key=lambda item: item.key)
        clusters.sort(key=lambda items: items[0].key)
        return clusters, rejected

    @staticmethod
    def temporary_mapping(
        clusters: Sequence[Sequence[EntityMention]],
    ) -> dict[SourceKey, str]:
        """为图结构二次召回生成无需持久化的稳定簇标识。"""

        result: dict[SourceKey, str] = {}
        for cluster in clusters:
            cluster_id = cluster[0].key.token
            for mention in cluster:
                result[mention.key] = cluster_id
        return result


__all__ = ["ConstrainedEntityClusterer"]
