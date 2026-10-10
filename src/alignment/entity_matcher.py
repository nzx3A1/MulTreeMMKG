"""实体候选对的规则评分、硬冲突校验和可选 LLM 裁决。

规则层负责可以确定的 SAME/DIFFERENT；只有证据处于灰区的候选才提交给 LLM，
且模型只能返回判定，不能修改原始实体或 Schema。
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .models import CandidatePair, EntityMention, PairDecision, SourceKey
from .normalizer import normalized_value
from .scope_rules import AlignmentRules


logger = logging.getLogger(__name__)


_IDENTIFIER_PATTERN = re.compile(r"\d+(?:\s*[-—–~至]\s*\d+)?|[零〇一二三四五六七八九十百]+")


class EntityPairMatcher:
    """使用名称、类型、上下文、邻接结构和属性判断实体是否共指。"""

    def __init__(
        self,
        rules: AlignmentRules,
        auto_merge_threshold: float = 0.78,
        uncertain_threshold: float = 0.58,
    ) -> None:
        """保存实体判定阈值。"""

        if not 0.0 <= uncertain_threshold <= auto_merge_threshold <= 1.0:
            raise ValueError("实体对齐阈值必须满足 0 <= uncertain <= auto_merge <= 1")
        self.rules = rules
        self.auto_merge_threshold = auto_merge_threshold
        self.uncertain_threshold = uncertain_threshold

    def hard_conflict(self, left: EntityMention, right: EntityMention) -> str | None:
        """检测作用域、类型和身份属性中的不可合并条件。"""

        if left.document_id != right.document_id:
            return "实体来自不同文档"
        if left.scope != right.scope:
            return "实体作用域不同"
        if left.scope == "chunk_local" and left.chunk_id != right.chunk_id:
            return "Chunk 局部结构实体来自不同 Chunk"
        type_relation = self.rules.type_compatibility(
            left.schema_type_zh,
            right.schema_type_zh,
            left.schema_type,
            right.schema_type,
            left.raw_type,
            right.raw_type,
        )
        if type_relation == "incompatible":
            return "schema_type_zh 属于显式互斥类型"
        for neighbor in left.neighbors:
            relation = neighbor.relation.removeprefix("INVERSE:").lower()
            if (
                neighbor.neighbor_key == right.key
                and relation not in self.rules.absorb_self_loop_relations
            ):
                return f"两个实体之间存在非共指关系 {neighbor.relation}"
        for neighbor in right.neighbors:
            relation = neighbor.relation.removeprefix("INVERSE:").lower()
            if (
                neighbor.neighbor_key == left.key
                and relation not in self.rules.absorb_self_loop_relations
            ):
                return f"两个实体之间存在非共指关系 {neighbor.relation}"
        type_values = {
            left.schema_type_zh.strip().lower(),
            right.schema_type_zh.strip().lower(),
        }
        if type_values & self.rules.identifier_schema_types_zh:
            left_identifiers = tuple(_IDENTIFIER_PATTERN.findall(left.name))
            right_identifiers = tuple(_IDENTIFIER_PATTERN.findall(right.name))
            if (
                left_identifiers
                and right_identifiers
                and left_identifiers != right_identifiers
            ):
                return "名称中的井号、层段号或范围标识冲突"
        for key in self.rules.identity_attribute_keys:
            left_values = {
                normalized_value(value)
                for attribute_key, value in left.attributes.items()
                if str(attribute_key).strip().lower() == key and value not in (None, "")
            }
            right_values = {
                normalized_value(value)
                for attribute_key, value in right.attributes.items()
                if str(attribute_key).strip().lower() == key and value not in (None, "")
            }
            if left_values and right_values and left_values.isdisjoint(right_values):
                return f"身份属性 {key} 冲突"
        return None

    def _safe_abbreviation(self, left: EntityMention, right: EntityMention) -> bool:
        """判断两个名称是否仅相差一个配置允许的实体类型后缀。"""

        left_name = left.normalized_name
        right_name = right.normalized_name
        if not left_name or not right_name or left_name == right_name:
            return False
        for suffix in self.rules.abbreviation_suffixes:
            normalized_suffix = suffix.replace("/", "").strip().lower()
            if not normalized_suffix:
                continue
            left_core = left_name[: -len(normalized_suffix)] if left_name.endswith(normalized_suffix) else left_name
            right_core = right_name[: -len(normalized_suffix)] if right_name.endswith(normalized_suffix) else right_name
            if left_core == right_core and len(left_core) >= 2:
                return True
        return False

    @staticmethod
    def _attribute_score(left: EntityMention, right: EntityMention) -> float:
        """计算两实体同名属性的一致比例，无共同属性时返回中性值。"""

        left_attributes = {str(key).strip().lower(): value for key, value in left.attributes.items()}
        right_attributes = {str(key).strip().lower(): value for key, value in right.attributes.items()}
        common_keys = sorted(set(left_attributes) & set(right_attributes))
        if not common_keys:
            return 0.5
        matches = sum(
            normalized_value(left_attributes[key]) == normalized_value(right_attributes[key])
            for key in common_keys
        )
        return matches / len(common_keys)

    @staticmethod
    def _weighted_score(signals: Mapping[str, float], attribute_score: float) -> float:
        """按固定权重汇总实体候选的多源证据。"""

        embedding_available = bool(signals.get("embedding_available"))
        if embedding_available:
            score = (
                0.32 * signals.get("name", 0.0)
                + 0.20 * signals.get("type", 0.0)
                + 0.10 * signals.get("context", 0.5)
                + 0.13 * signals.get("neighbor", 0.5)
                + 0.10 * attribute_score
                + 0.15 * signals.get("embedding", 0.5)
            )
        else:
            score = (
                0.42 * signals.get("name", 0.0)
                + 0.23 * signals.get("type", 0.0)
                + 0.10 * signals.get("context", 0.5)
                + 0.15 * signals.get("neighbor", 0.5)
                + 0.10 * attribute_score
            )
        return max(0.0, min(1.0, score))

    def decide(
        self,
        candidate: CandidatePair,
        mentions_by_key: Mapping[SourceKey, EntityMention],
    ) -> PairDecision:
        """对单个候选实体对生成可审计的规则判定。"""

        left = mentions_by_key[candidate.left]
        right = mentions_by_key[candidate.right]
        conflict = self.hard_conflict(left, right)
        attribute_score = self._attribute_score(left, right)
        signals = dict(candidate.signals)
        signals["attributes"] = attribute_score
        score = self._weighted_score(signals, attribute_score)
        if left.modality != right.modality:
            score = min(1.0, score + 0.03)
            signals["cross_modality_bonus"] = 0.03

        if conflict:
            return PairDecision(
                left=candidate.left,
                right=candidate.right,
                decision="DIFFERENT",
                confidence=1.0,
                method="hard_constraint",
                reason=conflict,
                score=score,
                signals=signals,
            )

        exact_or_alias = bool(signals.get("exact_name") or signals.get("alias_name"))
        if exact_or_alias and signals.get("type", 0.0) >= 0.35:
            return PairDecision(
                left=candidate.left,
                right=candidate.right,
                decision="SAME",
                confidence=0.98 if signals.get("type") == 1.0 else 0.90,
                method="exact_or_alias_name",
                reason="规范化名称或别名一致，且类型不存在硬冲突",
                score=max(score, 0.90),
                signals=signals,
            )

        if self._safe_abbreviation(left, right) and signals.get("type", 0.0) >= 0.75:
            return PairDecision(
                left=candidate.left,
                right=candidate.right,
                decision="SAME",
                confidence=max(0.88, score),
                method="safe_type_suffix_abbreviation",
                reason="名称仅相差配置允许的实体类型后缀，且 schema_type_zh 相同或兼容",
                score=score,
                signals=signals,
            )

        strong_non_context_evidence = (
            bool(signals.get("embedding_available"))
            and signals.get("embedding", 0.0) >= 0.92
            and signals.get("neighbor", 0.0) >= 0.65
            and attribute_score >= 0.75
        )
        if (
            score >= self.auto_merge_threshold
            and signals.get("name", 0.0) >= 0.90
            and signals.get("type", 0.0) >= 0.75
            and strong_non_context_evidence
        ):
            return PairDecision(
                left=candidate.left,
                right=candidate.right,
                decision="SAME",
                confidence=score,
                method="weighted_rules",
                reason="名称、类型、上下文、邻接结构和属性的综合得分达到自动融合阈值",
                score=score,
                signals=signals,
            )
        if score < self.uncertain_threshold:
            return PairDecision(
                left=candidate.left,
                right=candidate.right,
                decision="DIFFERENT",
                confidence=1.0 - score,
                method="weighted_rules",
                reason="综合证据不足以支持同一事物",
                score=score,
                signals=signals,
            )
        return PairDecision(
            left=candidate.left,
            right=candidate.right,
            decision="UNCERTAIN",
            confidence=score,
            method="weighted_rules",
            reason="规则证据位于灰区，需要 LLM 或人工复核",
            score=score,
            signals=signals,
        )

    def decide_many(
        self,
        candidates: Sequence[CandidatePair],
        mentions_by_key: Mapping[SourceKey, EntityMention],
    ) -> list[PairDecision]:
        """按稳定顺序批量判断候选实体对。"""

        return [self.decide(candidate, mentions_by_key) for candidate in candidates]


class LLMEntityPairResolver:
    """批量让 LLM 只裁决规则无法确定的实体对。"""

    def __init__(
        self,
        llm_client: Any,
        batch_size: int = 12,
        min_same_confidence: float = 0.82,
        max_candidates: int = 200,
    ) -> None:
        """保存 LLM 客户端、批量大小和可接受置信度。"""

        self.llm_client = llm_client
        self.batch_size = max(1, int(batch_size))
        self.min_same_confidence = min_same_confidence
        self.max_candidates = int(max_candidates)

    @staticmethod
    def _mention_payload(mention: EntityMention) -> dict[str, Any]:
        """构造限制长度的实体证据，避免 Prompt 被长正文淹没。"""

        return {
            "source_key": mention.key.token,
            "name": mention.name,
            "aliases": list(mention.aliases),
            "schema_type": mention.schema_type,
            "schema_type_zh": mention.schema_type_zh,
            "raw_type": mention.raw_type,
            "attributes": mention.attributes,
            "chunk_id": mention.chunk_id,
            "modality": mention.modality,
            "provenance": str(mention.provenance or "")[:800],
            "neighbors": [
                {
                    "relation": item.relation,
                    "name": item.neighbor_name,
                    "schema_type_zh": item.neighbor_type_zh,
                }
                for item in mention.neighbors[:20]
            ],
        }

    def _resolve_batch(
        self,
        decisions: Sequence[PairDecision],
        mentions_by_key: Mapping[SourceKey, EntityMention],
    ) -> dict[str, PairDecision]:
        """调用一次 LLM 并校验该批次的结构化判定。"""

        cases = [
            {
                "pair_id": decision.pair_id,
                "left": self._mention_payload(mentions_by_key[decision.left]),
                "right": self._mention_payload(mentions_by_key[decision.right]),
                "rule_score": decision.score,
                "signals": decision.signals,
            }
            for decision in decisions
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是地学多模态知识图谱实体共指裁决器。判断同一文档中的两个 mention "
                    "是否指向同一个现实事物。类型标签可能存在粒度或映射误差；相关、包含或上下位"
                    "不等于同一事物。只能返回 JSON：{\"decisions\":[{\"pair_id\":...,"
                    "\"decision\":\"SAME|DIFFERENT|UNCERTAIN\",\"confidence\":0到1,"
                    "\"reason\":...}]}。不要创造或修改实体。"
                ),
            },
            {
                "role": "user",
                "content": "请逐一裁决以下候选实体对：\n"
                + json.dumps(cases, ensure_ascii=False),
            },
        ]
        response = self.llm_client.chat_json(
            messages,
            temperature=0.0,
            max_tokens=max(1200, 360 * len(cases)),
        )
        rows = response.get("decisions", []) if isinstance(response, Mapping) else []
        original_by_id = {decision.pair_id: decision for decision in decisions}
        resolved: dict[str, PairDecision] = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, Mapping):
                continue
            pair_id = str(row.get("pair_id") or "")
            original = original_by_id.get(pair_id)
            if original is None:
                continue
            decision_value = str(row.get("decision") or "UNCERTAIN").strip().upper()
            if decision_value not in {"SAME", "DIFFERENT", "UNCERTAIN"}:
                decision_value = "UNCERTAIN"
            try:
                confidence = max(0.0, min(1.0, float(row.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            if decision_value == "SAME" and confidence < self.min_same_confidence:
                decision_value = "UNCERTAIN"
            resolved[pair_id] = PairDecision(
                left=original.left,
                right=original.right,
                decision=decision_value,
                confidence=confidence,
                method="llm_pair_resolution",
                reason=str(row.get("reason") or "LLM 未提供理由"),
                score=original.score,
                signals=original.signals,
            )
        return resolved

    def resolve(
        self,
        decisions: Sequence[PairDecision],
        mentions_by_key: Mapping[SourceKey, EntityMention],
    ) -> list[PairDecision]:
        """按规则得分优先级裁决有限数量的 UNCERTAIN 候选。"""

        uncertain = sorted(
            (decision for decision in decisions if decision.decision == "UNCERTAIN"),
            key=lambda item: (-item.score, item.pair_id),
        )
        if self.max_candidates >= 0:
            uncertain = uncertain[: self.max_candidates]
        replacements: dict[str, PairDecision] = {}
        started_at = time.perf_counter()
        total_batches = (len(uncertain) + self.batch_size - 1) // self.batch_size
        for start in range(0, len(uncertain), self.batch_size):
            batch = uncertain[start : start + self.batch_size]
            batch_number = start // self.batch_size + 1
            try:
                replacements.update(self._resolve_batch(batch, mentions_by_key))
            except Exception as exc:
                logger.warning("LLM 实体对裁决失败，本批保留 UNCERTAIN：%s", exc)
            elapsed = time.perf_counter() - started_at
            completed = min(len(uncertain), start + len(batch))
            eta = elapsed / completed * (len(uncertain) - completed) if completed else 0.0
            logger.info(
                "[Stage 06 LLM] 批次 %s/%s，完成=%s/%s，耗时=%.1f 秒，ETA=%.1f 秒",
                batch_number,
                total_batches,
                completed,
                len(uncertain),
                elapsed,
                eta,
            )
        return [replacements.get(decision.pair_id, decision) for decision in decisions]


def decisions_by_status(decisions: Sequence[PairDecision]) -> dict[str, list[PairDecision]]:
    """按 SAME、DIFFERENT 和 UNCERTAIN 对判定结果分组。"""

    result: dict[str, list[PairDecision]] = defaultdict(list)
    for decision in decisions:
        result[decision.decision].append(decision)
    return dict(result)


__all__ = ["EntityPairMatcher", "LLMEntityPairResolver", "decisions_by_status"]
