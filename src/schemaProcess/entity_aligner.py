"""实体规则映射、向量召回和 LLM 判定的编排。"""
from __future__ import annotations

import logging
import time
from collections import Counter
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .category_router import CategoryRoute, SchemaCategoryRouter
from .entity_rules import EntityRuleMapper
from .entity_cache import EntityMappingCache, build_entity_mapping_key
from .llm_selector import CandidateSelector
from .mapping_filter import is_predefined_non_schema_type
from .models import SelectionDecision, SemanticCandidate
from .semantic_retriever import VectorSemanticRetriever


logger = logging.getLogger(__name__)


class EntityAligner:
    """保守地生成实体类型及其 Schema 中英文映射字段。"""

    def __init__(
        self,
        rule_mapper: EntityRuleMapper,
        retriever: VectorSemanticRetriever,
        selector: CandidateSelector,
        min_llm_confidence: float = 0.72,
        min_vector_similarity: float = 0.25,
        vector_auto_accept_threshold: float = 0.92,
        vector_auto_accept_margin: float = 0.12,
        mapping_cache: EntityMappingCache | None = None,
        category_router: SchemaCategoryRouter | None = None,
    ) -> None:
        """初始化规则、向量、LLM 门控和实体映射缓存。"""

        self.rule_mapper = rule_mapper
        self.retriever = retriever
        self.selector = selector
        self.min_llm_confidence = min_llm_confidence
        self.min_vector_similarity = min_vector_similarity
        self.vector_auto_accept_threshold = vector_auto_accept_threshold
        self.vector_auto_accept_margin = vector_auto_accept_margin
        self.mapping_cache = mapping_cache
        self.category_router = category_router
        self.cache_hits = 0

    @staticmethod
    def _unmapped(entity: Mapping[str, Any], reason: str, confidence: float = 0.0) -> dict[str, Any]:
        """保留实体原始字段，并生成保守的未映射结果。"""

        result = deepcopy(dict(entity))
        result["raw_type"] = entity.get("raw_type", entity.get("type"))
        result["schema_type"] = None
        result["schema_type_zh"] = None
        result["schema_alignment"] = {
            "status": "UNMAPPED",
            "method": "none",
            "confidence": round(max(0.0, min(1.0, confidence)), 6),
            "reason": reason,
        }
        return result

    @staticmethod
    def _skipped(entity: Mapping[str, Any], source_modality: str | None) -> dict[str, Any]:
        """保留预定义结构节点，但明确标记为无需 Schema 映射。"""

        result = deepcopy(dict(entity))
        raw_type = entity.get("raw_type", entity.get("type"))
        result["raw_type"] = raw_type
        result["schema_type"] = None
        result["schema_type_zh"] = None
        result["schema_alignment"] = {
            "status": "SKIPPED",
            "method": "predefined_type_filter",
            "confidence": 1.0,
            "reason": f"{source_modality or 'unknown'} 模态预定义结构类型 {raw_type!r}，无需 Schema 映射",
        }
        return result

    @staticmethod
    def _mapping_template(result: Mapping[str, Any]) -> dict[str, Any]:
        """提取可缓存的 Schema 映射字段，不缓存实体原始属性。"""

        return {
            "schema_type": result.get("schema_type"),
            "schema_type_zh": result.get("schema_type_zh"),
            "schema_alignment": deepcopy(result.get("schema_alignment") or {}),
        }

    @staticmethod
    def _apply_mapping_template(
        entity: Mapping[str, Any],
        template: Mapping[str, Any],
        cache_hit: bool = False,
    ) -> dict[str, Any]:
        """将缓存的映射字段广播到当前实体并保留其全部原始内容。"""

        result = deepcopy(dict(entity))
        result["raw_type"] = entity.get("raw_type", entity.get("type"))
        result["schema_type"] = template.get("schema_type")
        result["schema_type_zh"] = template.get("schema_type_zh")
        result["schema_alignment"] = deepcopy(template.get("schema_alignment") or {})
        if cache_hit:
            result["schema_alignment"]["cache_hit"] = True
        return result

    @staticmethod
    def _attach_category_route(
        result: dict[str, Any],
        route: CategoryRoute | None,
    ) -> dict[str, Any]:
        """把一级类别路由依据附加到语义映射结果，便于审计召回范围。"""

        if route is None:
            return result
        result["schema_alignment"]["category_route"] = route.category
        result["schema_alignment"]["category_route_score"] = route.score
        result["schema_alignment"]["category_route_reason"] = route.reason
        return result

    def align_many(
        self,
        entities: Sequence[Mapping[str, Any]],
        source_modalities: Sequence[str | None] | None = None,
    ) -> list[dict[str, Any]]:
        """对实体去重后执行规则、缓存、向量直连和批量 LLM 映射。"""

        started_at = time.perf_counter()
        logger.info("[实体映射 1/4] 开始处理 %s 个实体", len(entities))
        if source_modalities is not None and len(source_modalities) != len(entities):
            raise ValueError("实体数量与来源模态数量不一致")

        aligned: list[dict[str, Any] | None] = [None] * len(entities)
        unresolved_keys: list[str] = []
        unresolved_entities: dict[str, Mapping[str, Any]] = {}
        unresolved_indices: dict[str, list[int]] = {}

        for index, entity in enumerate(entities):
            metadata = entity.get("metadata") if isinstance(entity.get("metadata"), Mapping) else {}
            requested_modality = source_modalities[index] if source_modalities is not None else None
            source_modality = (
                str(requested_modality or metadata.get("source_modality") or "").strip() or None
            )
            raw_type = entity.get("raw_type", entity.get("type"))
            if is_predefined_non_schema_type(raw_type, source_modality):
                aligned[index] = self._skipped(entity, source_modality)
                continue

            mapping_key = build_entity_mapping_key(entity)
            if self.mapping_cache is not None:
                cached = self.mapping_cache.get(mapping_key)
                if cached is not None and self._valid_cached_template(cached):
                    aligned[index] = self._apply_mapping_template(entity, cached, cache_hit=True)
                    self.cache_hits += 1
                    continue

            rule_match = self.rule_mapper.map_entity(entity)
            if rule_match is None:
                if mapping_key not in unresolved_entities:
                    unresolved_keys.append(mapping_key)
                    unresolved_entities[mapping_key] = entity
                    unresolved_indices[mapping_key] = []
                unresolved_indices[mapping_key].append(index)
                continue
            result = deepcopy(dict(entity))
            result["raw_type"] = entity.get("raw_type", entity.get("type"))
            result["schema_type"] = rule_match.schema_type
            result["schema_type_zh"] = self.rule_mapper.snapshot.concepts_by_schema[
                rule_match.schema_type
            ].zh_name
            result["schema_alignment"] = {
                "status": "RULE_MAPPED",
                "method": "suffix_rule",
                "confidence": round(rule_match.confidence, 6),
                "reason": rule_match.reason,
                "rule_id": rule_match.rule_id,
            }
            aligned[index] = result
            if self.mapping_cache is not None:
                self.mapping_cache.set(mapping_key, self._mapping_template(result))

        skipped_count = sum(
            (item.get("schema_alignment") or {}).get("status") == "SKIPPED"
            for item in aligned
            if item is not None
        )
        rule_count = sum(
            (item.get("schema_alignment") or {}).get("status") == "RULE_MAPPED"
            for item in aligned
            if item is not None
        )
        unresolved_instance_count = sum(len(indices) for indices in unresolved_indices.values())
        logger.info(
            "[实体映射 1/4] 预处理完成：SKIPPED=%s，规则映射=%s，缓存命中=%s，"
            "待语义映射=%s（去重后=%s）",
            skipped_count,
            rule_count,
            self.cache_hits,
            unresolved_instance_count,
            len(unresolved_entities),
        )

        if unresolved_entities:
            unique_entities = [unresolved_entities[key] for key in unresolved_keys]
            routes = (
                self.category_router.route_many(unique_entities)
                if self.category_router is not None
                else [None] * len(unique_entities)
            )
            categories = [route.category if route is not None else None for route in routes]
            route_counts = Counter(category or "FULL_SCHEMA_FALLBACK" for category in categories)
            logger.info(
                "[实体映射 2/4] 一级分类完成：%s",
                dict(sorted(route_counts.items())),
            )
            candidate_groups = (
                self.retriever.retrieve_many(unique_entities, categories)
                if self.category_router is not None
                else self.retriever.retrieve_many(unique_entities)
            )
            llm_keys: list[str] = []
            llm_entities: list[Mapping[str, Any]] = []
            llm_candidates: list[Sequence[SemanticCandidate]] = []
            llm_routes: list[CategoryRoute | None] = []
            vector_auto_count = 0
            for key, entity, candidates, route in zip(
                unresolved_keys, unique_entities, candidate_groups, routes
            ):
                auto_mapped = self._vector_auto_result(entity, candidates)
                if auto_mapped is not None:
                    vector_auto_count += 1
                    auto_mapped = self._attach_category_route(auto_mapped, route)
                    template = self._mapping_template(auto_mapped)
                    if self.mapping_cache is not None:
                        self.mapping_cache.set(key, template)
                    for original_index in unresolved_indices[key]:
                        aligned[original_index] = self._apply_mapping_template(
                            entities[original_index], template
                        )
                    continue
                llm_keys.append(key)
                llm_entities.append(entity)
                llm_candidates.append(candidates)
                llm_routes.append(route)

            logger.info(
                "[实体映射 3/4] 向量门控完成：高置信直连=%s，进入 LLM 消歧=%s",
                vector_auto_count,
                len(llm_entities),
            )
            requests = list(zip(llm_entities, llm_candidates))
            if requests:
                logger.info(
                    "[实体映射 4/4] 开始批量 LLM 消歧，共 %s 个去重实体",
                    len(requests),
                )
            decisions = self.selector.select_entities(requests) if requests else []
            if len(decisions) != len(llm_entities):
                raise RuntimeError("LLM 实体判定数量与请求数量不一致")
            for key, entity, candidates, decision, route in zip(
                llm_keys, llm_entities, llm_candidates, decisions, llm_routes
            ):
                mapped = self._apply_semantic_decision(entity, candidates, decision)
                mapped = self._attach_category_route(mapped, route)
                template = self._mapping_template(mapped)
                if self.mapping_cache is not None:
                    self.mapping_cache.set(key, template)
                for original_index in unresolved_indices[key]:
                    aligned[original_index] = self._apply_mapping_template(
                        entities[original_index], template
                    )

        if self.mapping_cache is not None:
            self.mapping_cache.save()

        results = [item for item in aligned if item is not None]
        status_counts = Counter(
            str((item.get("schema_alignment") or {}).get("status") or "UNKNOWN")
            for item in results
        )
        logger.info(
            "[实体映射] 全部完成：%s，总耗时 %.2f 秒",
            dict(sorted(status_counts.items())),
            time.perf_counter() - started_at,
        )
        return results

    def _valid_cached_template(self, template: Mapping[str, Any]) -> bool:
        """校验缓存映射仍指向当前 Schema，避免坏缓存阻断本次对齐。"""

        schema_type = template.get("schema_type")
        status = (template.get("schema_alignment") or {}).get("status")
        return status == "UNMAPPED" or schema_type in self.rule_mapper.snapshot.concepts_by_schema

    def _vector_auto_result(
        self,
        entity: Mapping[str, Any],
        candidates: Sequence[SemanticCandidate],
    ) -> dict[str, Any] | None:
        """对 Top-1 高相似且有明显间隔的候选直接映射，跳过 LLM。"""

        if not candidates:
            return None
        top1_candidate = candidates[0]
        top1 = max(0.0, min(1.0, top1_candidate.similarity))
        top2 = max(0.0, min(1.0, candidates[1].similarity)) if len(candidates) > 1 else 0.0
        margin = top1 - top2 if len(candidates) > 1 else 1.0
        if top1 < self.vector_auto_accept_threshold or margin < self.vector_auto_accept_margin:
            return None
        result = deepcopy(dict(entity))
        result["raw_type"] = entity.get("raw_type", entity.get("type"))
        result["schema_type"] = top1_candidate.concept.schema
        result["schema_type_zh"] = top1_candidate.concept.zh_name
        result["schema_alignment"] = {
            "status": "SEMANTIC_MAPPED",
            "method": "vector_auto_accept",
            "confidence": round(top1, 6),
            "reason": (
                f"Top-1 向量相似度={top1:.3f}，与第二候选的间隔={margin:.3f}，"
                "达到自动接受阈值"
            ),
            "vector_similarity": round(top1, 6),
            "vector_margin": round(margin, 6),
        }
        return result

    def _apply_semantic_decision(
        self,
        entity: Mapping[str, Any],
        candidates: Sequence[SemanticCandidate],
        decision: SelectionDecision,
    ) -> dict[str, Any]:
        """校验 LLM 选择及阈值，并生成语义映射或未映射结果。"""

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
        result["schema_type_zh"] = chosen.concept.zh_name
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
