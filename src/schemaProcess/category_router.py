"""Stage 05 Schema 一级类别路由：在向量召回前缩小概念候选范围。"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import SchemaSnapshot


def _normalize_route_text(value: Any) -> str:
    """将类别路由文本归一为便于中英文关键词匹配的形式。"""

    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())


@dataclass(frozen=True)
class CategoryRoute:
    """记录一个实体的一级类别路由结果及可解释依据。"""

    category: str | None
    score: int
    matched_hints: tuple[str, ...] = ()
    reason: str = ""


class SchemaCategoryRouter:
    """使用原始类型、中文类型、名称和属性对实体做轻量一级分类。"""

    def __init__(
        self,
        snapshot: SchemaSnapshot,
        rules_path: str | Path,
        min_score: int = 2,
    ) -> None:
        """加载配置中的类别关键词，并仅保留当前 Schema 存在的类别。"""

        if min_score <= 0:
            raise ValueError("类别路由 min_score 必须大于 0")
        self.snapshot = snapshot
        self.rules_path = Path(rules_path)
        self.min_score = min_score
        payload = json.loads(self.rules_path.read_text(encoding="utf-8"))
        available_categories = {concept.category for concept in snapshot.concepts if concept.category}
        self.routes: dict[str, tuple[str, ...]] = {}
        for item in payload.get("category_routes", []):
            if not isinstance(item, Mapping):
                continue
            category = str(item.get("category") or "").strip()
            if category not in available_categories:
                continue
            hints = tuple(
                str(hint).strip()
                for hint in item.get("hints", [])
                if str(hint).strip()
            )
            if hints:
                self.routes[category] = hints
        self.route_counts: Counter[str] = Counter()

    @staticmethod
    def _entity_fields(entity: Mapping[str, Any]) -> dict[str, str]:
        """提取不同权重的实体文本字段，属性只作为弱辅助信号。"""

        attributes = entity.get("attributes")
        try:
            attributes_text = json.dumps(attributes, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            attributes_text = str(attributes or "")
        return {
            "type": _normalize_route_text(
                f"{entity.get('raw_type', entity.get('type')) or ''} {entity.get('type_zh') or ''}"
            ),
            "name": _normalize_route_text(
                entity.get("name") or entity.get("official_name") or ""
            ),
            "attributes": _normalize_route_text(attributes_text),
        }

    def route(self, entity: Mapping[str, Any]) -> CategoryRoute:
        """返回唯一高分一级类别；无命中或并列时返回全 Schema 回退标记。"""

        fields = self._entity_fields(entity)
        scores: dict[str, int] = {}
        matched_by_category: dict[str, list[str]] = {}
        for category, hints in self.routes.items():
            score = 0
            matched: list[str] = []
            for hint in hints:
                normalized_hint = _normalize_route_text(hint)
                if not normalized_hint:
                    continue
                if normalized_hint in fields["type"]:
                    score += 4
                    matched.append(hint)
                elif normalized_hint in fields["name"]:
                    score += 2
                    matched.append(hint)
                elif normalized_hint in fields["attributes"]:
                    score += 1
                    matched.append(hint)
            if score:
                scores[category] = score
                matched_by_category[category] = matched

        if not scores:
            self.route_counts["FULL_SCHEMA_FALLBACK"] += 1
            return CategoryRoute(None, 0, reason="未命中可靠的一级类别关键词，回退全 Schema")
        best_score = max(scores.values())
        winners = sorted(category for category, score in scores.items() if score == best_score)
        if best_score < self.min_score or len(winners) != 1:
            self.route_counts["FULL_SCHEMA_FALLBACK"] += 1
            reason = "一级类别证据不足" if best_score < self.min_score else "一级类别最高分并列"
            return CategoryRoute(None, best_score, reason=f"{reason}，回退全 Schema")

        category = winners[0]
        matched_hints = tuple(dict.fromkeys(matched_by_category[category]))
        self.route_counts[category] += 1
        return CategoryRoute(
            category=category,
            score=best_score,
            matched_hints=matched_hints,
            reason=f"命中类别关键词：{'、'.join(matched_hints)}",
        )

    def route_many(self, entities: Sequence[Mapping[str, Any]]) -> list[CategoryRoute]:
        """按输入顺序批量执行一级类别路由。"""

        return [self.route(entity) for entity in entities]


__all__ = ["CategoryRoute", "SchemaCategoryRouter"]
