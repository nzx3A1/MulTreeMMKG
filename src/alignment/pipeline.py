"""Stage 06 多模态实体和关系对齐总编排。

流程依次执行来源收集、候选召回、规则/图结构/LLM 判同、受约束聚类、全局实体
物化、关系端点重写、重复事实融合、事件参与者重写和审计报告生成。
"""
from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.embedding_client import EmbeddingClient
from src.utils.json_io import read_json, write_json
from src.utils.llm_client import LLMClient

from .candidate_generator import EntityCandidateGenerator
from .clusterer import ConstrainedEntityClusterer
from .entity_fuser import build_global_entities
from .entity_matcher import EntityPairMatcher, LLMEntityPairResolver
from .models import EntityMention, PairDecision, SourceKey
from .relation_fuser import fuse_relations, rewrite_events
from .reports import (
    build_conflict_report,
    build_entity_alignment_report,
    build_unresolved_report,
)
from .scope_rules import AlignmentRules, DEFAULT_RULES_PATH
from .source_collector import collect_entity_mentions, graph_source, require_mapping, require_sequence


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "stage_05_schema_aligned_graph.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "stage_06_multimodal_aligned_graph.json"
DEFAULT_ENTITY_REPORT_PATH = PROJECT_ROOT / "output" / "stage_06_entity_alignment_report.json"
DEFAULT_UNRESOLVED_PATH = PROJECT_ROOT / "output" / "stage_06_unresolved_candidates.json"
DEFAULT_CONFLICT_PATH = PROJECT_ROOT / "output" / "stage_06_conflicts.json"


def _embedding_text(mention: EntityMention) -> str:
    """构造强调名称、中文 Schema 类型和局部证据的实体向量文本。"""

    return "\n".join(
        (
            f"name: {mention.name}",
            f"aliases: {', '.join(mention.aliases)}",
            f"schema_type_zh: {mention.schema_type_zh}",
            f"schema_type: {mention.schema_type}",
            f"raw_type: {mention.raw_type}",
            f"provenance: {str(mention.provenance or '')[:500]}",
        )
    )


def _merge_refined_decisions(
    original: Sequence[PairDecision],
    refined: Sequence[PairDecision],
) -> list[PairDecision]:
    """以图结构确认的 SAME 覆盖非硬冲突旧判定，并保留稳定顺序。"""

    indexed = {decision.pair_id: decision for decision in original}
    for decision in refined:
        previous = indexed.get(decision.pair_id)
        if previous is None:
            indexed[decision.pair_id] = decision
        elif decision.decision == "SAME" and previous.method != "hard_constraint":
            indexed[decision.pair_id] = decision
    return [indexed[key] for key in sorted(indexed)]


class MultimodalAlignmentPipeline:
    """把 Stage 05 子图转换为按文档聚合的多模态融合图谱。"""

    def __init__(
        self,
        rules_path: str | Path = DEFAULT_RULES_PATH,
        embedding_client: Any = None,
        llm_client: Any = None,
        enable_embedding: bool = True,
        enable_llm: bool = True,
        min_name_similarity: float = 0.58,
        min_embedding_similarity: float = 0.84,
        auto_merge_threshold: float = 0.78,
        uncertain_threshold: float = 0.58,
        llm_batch_size: int = 12,
        min_llm_same_confidence: float = 0.82,
        max_llm_candidates: int = 200,
    ) -> None:
        """保存规则、外部服务开关和实体对齐阈值。"""

        self.rules_path = Path(rules_path)
        self.rules = AlignmentRules.load(self.rules_path)
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.enable_embedding = enable_embedding
        self.enable_llm = enable_llm
        self.candidate_generator = EntityCandidateGenerator(
            rules=self.rules,
            min_name_similarity=min_name_similarity,
            min_embedding_similarity=min_embedding_similarity,
        )
        self.matcher = EntityPairMatcher(
            rules=self.rules,
            auto_merge_threshold=auto_merge_threshold,
            uncertain_threshold=uncertain_threshold,
        )
        self.clusterer = ConstrainedEntityClusterer(self.matcher)
        self.llm_batch_size = llm_batch_size
        self.min_llm_same_confidence = min_llm_same_confidence
        self.max_llm_candidates = max_llm_candidates

    def _encode_mentions(
        self,
        mentions: Sequence[EntityMention],
    ) -> dict[SourceKey, Sequence[float]]:
        """批量编码实体 mention；服务不可用时安全退化到无向量模式。"""

        if not self.enable_embedding or not mentions:
            return {}
        client = self.embedding_client or EmbeddingClient()
        texts = [_embedding_text(mention) for mention in mentions]
        try:
            embeddings = client.encode(texts)
        except Exception as exc:
            logger.warning("实体 Embedding 生成失败，将继续使用规则和 LLM：%s", exc)
            return {}
        if len(embeddings) != len(mentions):
            logger.warning(
                "实体 Embedding 数量不匹配：期望=%s，实际=%s，将忽略本次向量",
                len(mentions),
                len(embeddings),
            )
            return {}
        return {
            mention.key: embedding
            for mention, embedding in zip(mentions, embeddings)
            if embedding
        }

    def _resolve_with_llm(
        self,
        decisions: Sequence[PairDecision],
        mentions_by_key: Mapping[SourceKey, EntityMention],
    ) -> list[PairDecision]:
        """按配置调用 LLM 裁决灰区候选，并在禁用时原样返回。"""

        if not self.enable_llm or not any(
            decision.decision == "UNCERTAIN" for decision in decisions
        ):
            return list(decisions)
        resolver = LLMEntityPairResolver(
            llm_client=self.llm_client or LLMClient(),
            batch_size=self.llm_batch_size,
            min_same_confidence=self.min_llm_same_confidence,
            max_candidates=self.max_llm_candidates,
        )
        return resolver.resolve(decisions, mentions_by_key)

    @staticmethod
    def _document_graphs(
        payload: Mapping[str, Any],
        global_entities: Sequence[Mapping[str, Any]],
        global_relations: Sequence[Mapping[str, Any]],
        events: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """把融合对象按 document_id 组织为与既有阶段相近的 Graph 数组。"""

        source_metadata: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: {"chunks": set(), "modalities": set()}
        )
        graphs = require_sequence(payload.get("graphs"), "stage_05.graphs")
        for graph_index, raw_graph in enumerate(graphs):
            graph = require_mapping(raw_graph, f"graphs[{graph_index}]")
            document_id, chunk_id, modality, _ = graph_source(graph, graph_index)
            source_metadata[document_id]["chunks"].add(chunk_id)
            source_metadata[document_id]["modalities"].add(modality)

        entities_by_document: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        relations_by_document: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        events_by_document: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for entity in global_entities:
            entities_by_document[str((entity.get("metadata") or {}).get("document_id"))].append(entity)
        for relation in global_relations:
            relations_by_document[str(relation.get("document_id"))].append(relation)
        for event in events:
            events_by_document[str(event.get("document_id"))].append(event)

        result: list[dict[str, Any]] = []
        for document_id in sorted(source_metadata):
            result.append(
                {
                    "entities": list(entities_by_document.get(document_id, [])),
                    "relations": list(relations_by_document.get(document_id, [])),
                    "events": list(events_by_document.get(document_id, [])),
                    "metadata": {
                        "document_id": document_id,
                        "stage": "stage_06_multimodal_alignment",
                        "scope": "document_global_graph",
                        "source_chunks": sorted(source_metadata[document_id]["chunks"]),
                        "source_modalities": sorted(source_metadata[document_id]["modalities"]),
                    },
                }
            )
        return result

    def align_payload(
        self,
        payload: Mapping[str, Any],
        source_name: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """在内存中执行完整 Stage 06，并返回主图及三类审计报告。"""

        started_at = time.perf_counter()
        logger.info("[Stage 06 1/8] 收集实体来源、上下文和原始邻接结构")
        mentions, _ = collect_entity_mentions(payload, self.rules)
        mentions_by_key = {mention.key: mention for mention in mentions}

        logger.info("[Stage 06 2/8] 生成可选实体向量，实体=%s", len(mentions))
        embeddings = self._encode_mentions(mentions)

        logger.info("[Stage 06 3/8] 召回候选实体对")
        candidates = self.candidate_generator.generate(mentions, embeddings)
        decisions = self.matcher.decide_many(candidates, mentions_by_key)
        logger.info("[Stage 06 3/8] 候选实体对=%s", len(candidates))

        logger.info("[Stage 06 4/8] 首轮聚类并用已融合邻居补充判同")
        first_clusters, _ = self.clusterer.cluster(mentions, decisions)
        temporary_mapping = self.clusterer.temporary_mapping(first_clusters)
        refined_candidates = self.candidate_generator.generate_graph_refinement(
            mentions,
            temporary_mapping,
            existing_pair_ids=set(),
        )
        refined_decisions = self.matcher.decide_many(refined_candidates, mentions_by_key)
        decisions = _merge_refined_decisions(decisions, refined_decisions)
        logger.info(
            "[Stage 06 4/8] 首轮实体簇=%s，图结构补充候选=%s",
            len(first_clusters),
            len(refined_candidates),
        )

        logger.info("[Stage 06 5/8] 裁决规则灰区候选")
        decisions = self._resolve_with_llm(decisions, mentions_by_key)
        clusters, rejected_cluster_merges = self.clusterer.cluster(mentions, decisions)
        logger.info(
            "[Stage 06 5/8] 最终实体簇=%s，簇级拒绝=%s",
            len(clusters),
            len(rejected_cluster_merges),
        )

        logger.info("[Stage 06 6/8] 物化全局实体并生成 mention 映射")
        global_entities, source_mapping, mapping_rows = build_global_entities(
            clusters,
            decisions,
        )

        logger.info("[Stage 06 7/8] 重写关系端点、融合重复事实并迁移事件参与者")
        global_relations, relation_issues, processed_relation_count = fuse_relations(
            payload,
            source_mapping,
            global_entities,
            self.rules,
        )
        events, unresolved_event_participants = rewrite_events(payload, source_mapping)
        source_event_count = sum(
            len(
                require_sequence(
                    require_mapping(graph, f"graphs[{graph_index}]").get("events") or [],
                    f"graphs[{graph_index}].events",
                )
            )
            for graph_index, graph in enumerate(
                require_sequence(payload.get("graphs"), "stage_05.graphs")
            )
        )

        logger.info("[Stage 06 8/8] 生成主图、统计和审计报告")
        output_graphs = self._document_graphs(
            payload,
            global_entities,
            global_relations,
            events,
        )
        relation_evidence_count = sum(
            int((relation.get("metadata") or {}).get("evidence_count", 0))
            for relation in global_relations
        )
        absorbed_count = len(relation_issues["absorbed_self_loops"])
        dangling_count = len(relation_issues["dangling_relations"])
        entity_source_count = sum(
            int((entity.get("metadata") or {}).get("mention_count", 0))
            for entity in global_entities
        )
        decision_counts = Counter(decision.decision for decision in decisions)
        statistics = {
            "source_graph_count": len(require_sequence(payload.get("graphs"), "stage_05.graphs")),
            "document_graph_count": len(output_graphs),
            "source_entity_count": len(mentions),
            "global_entity_count": len(global_entities),
            "fused_source_entity_count": len(mentions) - len(global_entities),
            "multi_mention_global_entity_count": sum(
                int((entity.get("metadata") or {}).get("mention_count", 0)) > 1
                for entity in global_entities
            ),
            "cross_modal_global_entity_count": sum(
                bool((entity.get("metadata") or {}).get("cross_modal"))
                for entity in global_entities
            ),
            "source_relation_count": processed_relation_count,
            "global_relation_count": len(global_relations),
            "merged_relation_evidence_count": sum(
                max(0, int((relation.get("metadata") or {}).get("evidence_count", 0)) - 1)
                for relation in global_relations
            ),
            "cross_modal_global_relation_count": sum(
                bool((relation.get("metadata") or {}).get("cross_modal"))
                for relation in global_relations
            ),
            "source_event_count": source_event_count,
            "output_event_count": len(events),
            "candidate_pair_count": len(decisions),
            "candidate_decision_counts": dict(sorted(decision_counts.items())),
            "rejected_cluster_merge_count": len(rejected_cluster_merges),
            "dangling_relation_count": dangling_count,
            "absorbed_self_loop_relation_count": absorbed_count,
        }
        validation = {
            "entity_source_conserved": entity_source_count == len(mentions),
            "relation_source_conserved": (
                relation_evidence_count + absorbed_count + dangling_count
                == processed_relation_count
            ),
            "event_source_conserved": source_event_count == len(events),
            "no_dangling_relation_endpoints": dangling_count == 0,
            "no_unresolved_event_participants": not unresolved_event_participants,
        }
        main_output = {
            "_status": "completed",
            "_stage": "stage_06_multimodal_alignment",
            "_description": "同一文档内实体共指融合、关系端点重写与多模态证据汇集",
            "source_file": source_name,
            "source_statistics": deepcopy(payload.get("statistics") or {}),
            "statistics": statistics,
            "validation": validation,
            "graphs": output_graphs,
            "entity_mapping": mapping_rows,
            "_alignment": {
                "rules_path": str(self.rules_path.resolve()),
                "embedding_enabled": self.enable_embedding,
                "embedding_available_count": len(embeddings),
                "llm_enabled": self.enable_llm,
                "min_name_similarity": self.candidate_generator.min_name_similarity,
                "min_embedding_similarity": self.candidate_generator.min_embedding_similarity,
                "auto_merge_threshold": self.matcher.auto_merge_threshold,
                "uncertain_threshold": self.matcher.uncertain_threshold,
            },
        }
        logger.info(
            "Stage 06 内存对齐完成，总耗时 %.2f 秒",
            time.perf_counter() - started_at,
        )
        entity_report = build_entity_alignment_report(
            mapping_rows,
            global_entities,
            decisions,
            source_name,
        )
        unresolved_report = build_unresolved_report(
            decisions,
            mentions_by_key,
            source_mapping,
            source_name,
        )
        conflict_report = build_conflict_report(
            global_entities,
            rejected_cluster_merges,
            relation_issues,
            unresolved_event_participants,
            source_name,
        )
        return main_output, entity_report, unresolved_report, conflict_report

    def run(
        self,
        input_path: str | Path = DEFAULT_INPUT_PATH,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
        entity_report_path: str | Path = DEFAULT_ENTITY_REPORT_PATH,
        unresolved_path: str | Path = DEFAULT_UNRESOLVED_PATH,
        conflict_path: str | Path = DEFAULT_CONFLICT_PATH,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """读取 Stage 05 文件、执行融合并原子约定地写出四个 JSON 结果。"""

        source_path = Path(input_path)
        payload = require_mapping(read_json(source_path), "Stage 05 输入")
        results = self.align_payload(payload, str(source_path.resolve()))
        for path, result in zip(
            (output_path, entity_report_path, unresolved_path, conflict_path),
            results,
        ):
            write_json(path, result)
        return results


__all__ = [
    "DEFAULT_CONFLICT_PATH",
    "DEFAULT_ENTITY_REPORT_PATH",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_UNRESOLVED_PATH",
    "MultimodalAlignmentPipeline",
]
