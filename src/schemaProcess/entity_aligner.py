"""实体规则映射、向量召回和 LLM 判定的编排。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .entity_rules import EntityRuleMapper
from .llm_selector import CandidateSelector
from .models import SelectionDecision, SemanticCandidate
from .semantic_retriever import VectorSemanticRetriever


class EntityAligner:
    """保守地生成实体 ``raw_type/schema_type/schema_alignment`` 字段。"""

    def __init__(
        self,
        rule_mapper: EntityRuleMapper,
        retriever: VectorSemanticRetriever,
        selector: CandidateSelector,
        min_llm_confidence: float = 0.72,
        min_vector_similarity: float = 0.25,
    ) -> None:
        self.rule_mapper = rule_mapper
        self.retriever = retriever
        self.selector = selector
        self.min_llm_confidence = min_llm_confidence
        self.min_vector_similarity = min_vector_similarity

    @staticmethod
    def _unmapped(entity: Mapping[str, Any], reason: str, confidence: float = 0.0) -> dict[str, Any]:
        result = deepcopy(dict(entity))
        result["raw_type"] = entity.get("raw_type", entity.get("type"))
        result["schema_type"] = None
        result["schema_alignment"] = {
            "status": "UNMAPPED",
            "method": "none",
            "confidence": round(max(0.0, min(1.0, confidence)), 6),
            "reason": reason,
        }
        return result

    def align_many(self, entities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        aligned: list[dict[str, Any] | None] = [None] * len(entities)
        unresolved_indices: list[int] = []
        unresolved_entities: list[Mapping[str, Any]] = []

        for index, entity in enumerate(entities):
            rule_match = self.rule_mapper.map_entity(entity)
            if rule_match is None:
                unresolved_indices.append(index)
                unresolved_entities.append(entity)
                continue
            result = deepcopy(dict(entity))
            result["raw_type"] = entity.get("raw_type", entity.get("type"))
            result["schema_type"] = rule_match.schema_type
            result["schema_alignment"] = {
                "status": "RULE_MAPPED",
                "method": "suffix_rule",
                "confidence": round(rule_match.confidence, 6),
                "reason": rule_match.reason,
                "rule_id": rule_match.rule_id,
            }
            aligned[index] = result

        if unresolved_entities:
            candidate_groups = self.retriever.retrieve_many(unresolved_entities)
            requests = list(zip(unresolved_entities, candidate_groups))
            decisions = self.selector.select_entities(requests)
            if len(decisions) != len(unresolved_entities):
                raise RuntimeError("LLM 实体判定数量与请求数量不一致")
            for original_index, entity, candidates, decision in zip(
                unresolved_indices, unresolved_entities, candidate_groups, decisions
            ):
                aligned[original_index] = self._apply_semantic_decision(entity, candidates, decision)

        return [item for item in aligned if item is not None]

    def _apply_semantic_decision(
        self,
        entity: Mapping[str, Any],
        candidates: Sequence[SemanticCandidate],
        decision: SelectionDecision,
    ) -> dict[str, Any]:
        if decision.selected is None:
            return self._unmapped(entity, decision.reason, decision.confidence)

        chosen = next(
            (item for item in candidates if item.concept.schema == decision.selected),
            None,
        )
        if chosen is None:
            return self._unmapped(entity, "LLM 选择不在向量候选集合中")
        if decision.confidence < self.min_llm_confidence:
            return self._unmapped(
                entity,
                f"LLM 置信度 {decision.confidence:.3f} 低于阈值 {self.min_llm_confidence:.3f}；{decision.reason}",
                decision.confidence,
            )
        if chosen.similarity < self.min_vector_similarity:
            return self._unmapped(
                entity,
                f"候选向量相似度 {chosen.similarity:.3f} 低于阈值 {self.min_vector_similarity:.3f}；{decision.reason}",
                min(decision.confidence, max(0.0, chosen.similarity)),
            )

        confidence = 0.6 * decision.confidence + 0.4 * max(0.0, min(1.0, chosen.similarity))
        result = deepcopy(dict(entity))
        result["raw_type"] = entity.get("raw_type", entity.get("type"))
        result["schema_type"] = chosen.concept.schema
        result["schema_alignment"] = {
            "status": "SEMANTIC_MAPPED",
            "method": "vector_llm",
            "confidence": round(confidence, 6),
            "reason": (
                f"{decision.reason}（向量相似度={chosen.similarity:.3f}，"
                f"LLM置信度={decision.confidence:.3f}）"
            ),
            "vector_similarity": round(chosen.similarity, 6),
        }
        return result


__all__ = ["EntityAligner"]
