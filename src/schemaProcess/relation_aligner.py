"""在实体 Schema 已确定后，对开放关系执行端点约束映射。"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .llm_selector import CandidateSelector
from .models import SchemaRelation, SchemaSnapshot, SelectionDecision


GENERIC_RELATIONS = {"RELATED_TO"}
WEAK_RELATION_NAMES = {
    "RELATED",
    "RELATED_TO",
    "ASSOCIATED",
    "ASSOCIATED_WITH",
    "CORRELATED",
    "CORRELATED_WITH",
}


def normalize_relation_name(value: Any) -> str:
    """把英文关系名归一为大写下划线形式。"""

    text = str(value or "").strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").upper()


def is_weak_relation(relation: Mapping[str, Any]) -> bool:
    """只有原始语义明确为弱‘相关’时，才允许通用关系。"""

    for field in ("raw_relation", "type", "relation_name"):
        if normalize_relation_name(relation.get(field)) in WEAK_RELATION_NAMES:
            return True
    chinese = " ".join(str(relation.get(field) or "") for field in ("type_zh", "relation_name"))
    return any(token in chinese for token in ("相关", "有关", "关联"))


class RelationAligner:
    """按方向查询候选；单候选直映射，多候选由规则/LLM 消歧。"""

    def __init__(
        self,
        snapshot: SchemaSnapshot,
        selector: CandidateSelector,
        min_llm_confidence: float = 0.72,
    ) -> None:
        self.snapshot = snapshot
        self.selector = selector
        self.min_llm_confidence = min_llm_confidence

    @staticmethod
    def _base_result(
        relation: Mapping[str, Any],
        source_schema: str | None,
        target_schema: str | None,
    ) -> dict[str, Any]:
        result = deepcopy(dict(relation))
        result["raw_relation"] = relation.get("raw_relation", relation.get("type"))
        result["source_schema"] = source_schema
        result["target_schema"] = target_schema
        return result

    def _unmapped(
        self,
        relation: Mapping[str, Any],
        source_schema: str | None,
        target_schema: str | None,
        reason: str,
        confidence: float = 0.0,
    ) -> dict[str, Any]:
        result = self._base_result(relation, source_schema, target_schema)
        result["schema_relation"] = None
        result["schema_alignment"] = {
            "status": "UNMAPPED",
            "method": "none",
            "confidence": round(max(0.0, min(1.0, confidence)), 6),
            "reason": reason,
        }
        return result

    def _mapped(
        self,
        relation: Mapping[str, Any],
        source_schema: str,
        target_schema: str,
        candidate: SchemaRelation,
        status: str,
        method: str,
        confidence: float,
        reason: str,
    ) -> dict[str, Any]:
        result = self._base_result(relation, source_schema, target_schema)
        result["schema_relation"] = candidate.relation_en
        result["schema_alignment"] = {
            "status": status,
            "method": method,
            "confidence": round(max(0.0, min(1.0, confidence)), 6),
            "reason": reason,
            "relation_zh": candidate.relation_zh,
        }
        return result

    @staticmethod
    def _exact_candidate(
        relation: Mapping[str, Any],
        candidates: Sequence[SchemaRelation],
    ) -> SchemaRelation | None:
        raw_en = {
            normalize_relation_name(relation.get(field))
            for field in ("raw_relation", "type", "relation_name")
            if relation.get(field)
        }
        raw_zh = {
            str(relation.get(field) or "").strip()
            for field in ("type_zh", "relation_name")
            if relation.get(field)
        }
        matches = [
            candidate
            for candidate in candidates
            if normalize_relation_name(candidate.relation_en) in raw_en
            or (candidate.relation_zh and candidate.relation_zh in raw_zh)
        ]
        return matches[0] if len(matches) == 1 else None

    def align_many(
        self,
        requests: Sequence[
            tuple[Mapping[str, Any], str | None, str | None]
        ],
    ) -> list[dict[str, Any]]:
        aligned: list[dict[str, Any] | None] = [None] * len(requests)
        llm_indices: list[int] = []
        llm_requests: list[tuple[Mapping[str, Any], str, str, Sequence[SchemaRelation]]] = []

        for index, (relation, source_schema, target_schema) in enumerate(requests):
            if not source_schema or not target_schema:
                aligned[index] = self._unmapped(
                    relation,
                    source_schema,
                    target_schema,
                    "至少一个端点实体未映射到 Schema，无法查询合法关系候选",
                )
                continue

            candidates = self.snapshot.relation_candidates(source_schema, target_schema)
            if not candidates:
                aligned[index] = self._unmapped(
                    relation,
                    source_schema,
                    target_schema,
                    f"概念图谱不存在 {source_schema} -> {target_schema} 的 SCHEMA_RELATION",
                )
                continue

            exact = self._exact_candidate(relation, candidates)
            if exact is not None:
                if exact.relation_en in GENERIC_RELATIONS and not is_weak_relation(relation):
                    aligned[index] = self._unmapped(
                        relation,
                        source_schema,
                        target_schema,
                        "原始关系具有明确语义，不允许降级为通用 RELATED_TO",
                    )
                else:
                    aligned[index] = self._mapped(
                        relation,
                        source_schema,
                        target_schema,
                        exact,
                        "SEMANTIC_MAPPED" if len(candidates) > 1 else "DIRECT_MAPPED",
                        "exact_relation_rule",
                        0.99,
                        "原始关系标识/中文名与端点候选精确一致",
                    )
                continue

            if len(candidates) == 1:
                candidate = candidates[0]
                if candidate.relation_en in GENERIC_RELATIONS and not is_weak_relation(relation):
                    aligned[index] = self._unmapped(
                        relation,
                        source_schema,
                        target_schema,
                        "端点仅有通用 RELATED_TO，但原始关系不是弱‘相关’语义",
                    )
                else:
                    aligned[index] = self._mapped(
                        relation,
                        source_schema,
                        target_schema,
                        candidate,
                        "DIRECT_MAPPED",
                        "single_endpoint_candidate",
                        0.98,
                        "有向端点类型在概念图谱中只有一个合法关系候选",
                    )
                continue

            llm_indices.append(index)
            llm_requests.append((relation, source_schema, target_schema, candidates))

        if llm_requests:
            decisions = self.selector.select_relations(llm_requests)
            if len(decisions) != len(llm_requests):
                raise RuntimeError("LLM 关系判定数量与请求数量不一致")
            for index, request, decision in zip(llm_indices, llm_requests, decisions):
                relation, source_schema, target_schema, candidates = request
                aligned[index] = self._apply_llm_decision(
                    relation, source_schema, target_schema, candidates, decision
                )

        return [item for item in aligned if item is not None]

    def _apply_llm_decision(
        self,
        relation: Mapping[str, Any],
        source_schema: str,
        target_schema: str,
        candidates: Sequence[SchemaRelation],
        decision: SelectionDecision,
    ) -> dict[str, Any]:
        if decision.selected is None:
            return self._unmapped(
                relation, source_schema, target_schema, decision.reason, decision.confidence
            )
        candidate = next(
            (item for item in candidates if item.relation_en == decision.selected),
            None,
        )
        if candidate is None:
            return self._unmapped(
                relation, source_schema, target_schema, "LLM 选择不在关系候选集合中"
            )
        if candidate.relation_en in GENERIC_RELATIONS and not is_weak_relation(relation):
            return self._unmapped(
                relation,
                source_schema,
                target_schema,
                "LLM 选择了 RELATED_TO，但原始关系具有更明确语义",
                decision.confidence,
            )
        if decision.confidence < self.min_llm_confidence:
            return self._unmapped(
                relation,
                source_schema,
                target_schema,
                f"LLM 置信度 {decision.confidence:.3f} 低于阈值 {self.min_llm_confidence:.3f}；{decision.reason}",
                decision.confidence,
            )
        return self._mapped(
            relation,
            source_schema,
            target_schema,
            candidate,
            "SEMANTIC_MAPPED",
            "candidate_llm",
            decision.confidence,
            decision.reason,
        )


__all__ = [
    "GENERIC_RELATIONS",
    "RelationAligner",
    "is_weak_relation",
    "normalize_relation_name",
]
