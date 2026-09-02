"""Stage 05 Schema 对齐总编排与文件输出。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.embedding_client import EmbeddingClient
from src.utils.json_io import read_json, write_json
from src.utils.llm_client import LLMClient

from .entity_aligner import EntityAligner
from .entity_rules import DEFAULT_RULES_PATH, EntityRuleMapper
from .llm_selector import CandidateSelector, SchemaLLMSelector
from .relation_aligner import RelationAligner
from .reports import build_unmapped_entity_report, build_unmapped_relation_report
from .schema_repository import Neo4jSchemaRepository, SchemaRepository
from .semantic_retriever import VectorSemanticRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "stage_04_merged_extraction.json"
DEFAULT_ALIGNED_OUTPUT_PATH = PROJECT_ROOT / "output" / "schema_aligned_graph.json"
DEFAULT_UNMAPPED_ENTITY_PATH = PROJECT_ROOT / "output" / "unmapped_entity_types.json"
DEFAULT_UNMAPPED_RELATION_PATH = PROJECT_ROOT / "output" / "unmapped_relations.json"


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} 必须是 JSON 数组")
    return value


def _alignment_counts(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str((item.get("schema_alignment") or {}).get("status") or "UNKNOWN")
        for item in items
    )
    return dict(sorted(counts.items()))


class SchemaAlignmentPipeline:
    """加载一次 Schema 快照，先对齐全部实体，再对齐全部关系。"""

    def __init__(
        self,
        repository: SchemaRepository | None = None,
        embedding_client: EmbeddingClient | None = None,
        llm_client: LLMClient | None = None,
        selector: CandidateSelector | None = None,
        rules_path: str | Path = DEFAULT_RULES_PATH,
        top_k: int = 5,
        embedding_batch_size: int = 32,
        llm_batch_size: int = 8,
        min_llm_confidence: float = 0.72,
        min_vector_similarity: float = 0.25,
    ) -> None:
        self.repository = repository or Neo4jSchemaRepository()
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.selector = selector
        self.rules_path = Path(rules_path)
        self.top_k = top_k
        self.embedding_batch_size = embedding_batch_size
        self.llm_batch_size = llm_batch_size
        self.min_llm_confidence = min_llm_confidence
        self.min_vector_similarity = min_vector_similarity

    def align_payload(
        self,
        payload: Mapping[str, Any],
        source_name: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """对齐内存中的 Stage 04 payload，并返回主图及两类 Gap 报告。"""

        raw_graphs = _require_sequence(payload.get("graphs"), "stage_04.graphs")
        snapshot = self.repository.load()
        selector = self.selector or SchemaLLMSelector(
            llm_client=self.llm_client,
            batch_size=self.llm_batch_size,
        )
        rule_mapper = EntityRuleMapper(snapshot=snapshot, rules_path=self.rules_path)
        retriever = VectorSemanticRetriever(
            snapshot=snapshot,
            embedding_client=self.embedding_client,
            top_k=self.top_k,
            batch_size=self.embedding_batch_size,
        )
        entity_aligner = EntityAligner(
            rule_mapper=rule_mapper,
            retriever=retriever,
            selector=selector,
            min_llm_confidence=self.min_llm_confidence,
            min_vector_similarity=self.min_vector_similarity,
        )
        relation_aligner = RelationAligner(
            snapshot=snapshot,
            selector=selector,
            min_llm_confidence=self.min_llm_confidence,
        )

        result = deepcopy(dict(payload))
        result_graphs = _require_sequence(result.get("graphs"), "stage_04.graphs")

        entity_locations: list[tuple[int, int]] = []
        all_entities: list[Mapping[str, Any]] = []
        for graph_index, graph in enumerate(result_graphs):
            if not isinstance(graph, Mapping):
                raise TypeError(f"graphs[{graph_index}] 必须是 JSON 对象")
            entities = _require_sequence(graph.get("entities", []), f"graphs[{graph_index}].entities")
            for entity_index, entity in enumerate(entities):
                if not isinstance(entity, Mapping):
                    raise TypeError(f"graphs[{graph_index}].entities[{entity_index}] 必须是对象")
                entity_locations.append((graph_index, entity_index))
                all_entities.append(entity)

        aligned_entities = entity_aligner.align_many(all_entities)
        for (graph_index, entity_index), entity in zip(entity_locations, aligned_entities):
            result_graphs[graph_index]["entities"][entity_index] = entity

        relation_locations: list[tuple[int, int]] = []
        relation_requests: list[tuple[Mapping[str, Any], str | None, str | None]] = []
        for graph_index, graph in enumerate(result_graphs):
            entities = _require_sequence(graph.get("entities", []), f"graphs[{graph_index}].entities")
            endpoint_map: dict[str, Mapping[str, Any]] = {}
            for entity in entities:
                entity_id = str(entity.get("id") or "").strip()
                if not entity_id:
                    raise ValueError(f"graphs[{graph_index}] 中存在缺少 id 的实体")
                if entity_id in endpoint_map:
                    raise ValueError(f"graphs[{graph_index}] 中实体 id 重复：{entity_id}")
                endpoint_map[entity_id] = entity

            relations = _require_sequence(graph.get("relations", []), f"graphs[{graph_index}].relations")
            for relation_index, relation in enumerate(relations):
                if not isinstance(relation, Mapping):
                    raise TypeError(f"graphs[{graph_index}].relations[{relation_index}] 必须是对象")
                source = endpoint_map.get(str(relation.get("source_id") or ""))
                target = endpoint_map.get(str(relation.get("target_id") or ""))
                source_schema = source.get("schema_type") if source else None
                target_schema = target.get("schema_type") if target else None
                relation_locations.append((graph_index, relation_index))
                relation_requests.append((relation, source_schema, target_schema))

        aligned_relations = relation_aligner.align_many(relation_requests)
        for (graph_index, relation_index), relation in zip(relation_locations, aligned_relations):
            result_graphs[graph_index]["relations"][relation_index] = relation

        unmapped_entities: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        unmapped_relations: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for graph_index, graph in enumerate(result_graphs):
            metadata = graph.get("metadata") if isinstance(graph.get("metadata"), Mapping) else {}
            context = {
                "graph_index": graph_index,
                "chunk_id": metadata.get("chunk_id"),
                "modality": metadata.get("modality"),
            }
            entities = _require_sequence(graph.get("entities", []), f"graphs[{graph_index}].entities")
            relations = _require_sequence(graph.get("relations", []), f"graphs[{graph_index}].relations")
            for entity in entities:
                if (entity.get("schema_alignment") or {}).get("status") == "UNMAPPED":
                    unmapped_entities.append((entity, context))
            for relation in relations:
                if (relation.get("schema_alignment") or {}).get("status") == "UNMAPPED":
                    unmapped_relations.append((relation, context))

            graph_metadata = deepcopy(dict(metadata))
            graph_metadata["schema_alignment"] = {
                "entity_status_counts": _alignment_counts(entities),
                "relation_status_counts": _alignment_counts(relations),
            }
            graph["metadata"] = graph_metadata

        result["_stage"] = "stage_05_schema_alignment"
        result["_schema_alignment"] = {
            "source_file": source_name,
            "schema_source": snapshot.source,
            "schema_concept_count": len(snapshot.concepts),
            "schema_relation_count": len(snapshot.relations),
            "top_k": self.top_k,
            "min_llm_confidence": self.min_llm_confidence,
            "min_vector_similarity": self.min_vector_similarity,
            "entity_status_counts": _alignment_counts(aligned_entities),
            "relation_status_counts": _alignment_counts(aligned_relations),
        }
        return (
            result,
            build_unmapped_entity_report(unmapped_entities),
            build_unmapped_relation_report(unmapped_relations),
        )

    def run(
        self,
        input_path: str | Path = DEFAULT_INPUT_PATH,
        aligned_output_path: str | Path = DEFAULT_ALIGNED_OUTPUT_PATH,
        unmapped_entity_path: str | Path = DEFAULT_UNMAPPED_ENTITY_PATH,
        unmapped_relation_path: str | Path = DEFAULT_UNMAPPED_RELATION_PATH,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """读取 Stage 04，完成对齐并原子化语义地写出三个独立 JSON。"""

        input_path = Path(input_path)
        payload = read_json(input_path)
        if not isinstance(payload, Mapping):
            raise TypeError("stage_04_merged_extraction.json 顶层必须是对象")
        aligned, entity_report, relation_report = self.align_payload(
            payload,
            source_name=str(input_path.resolve()),
        )
        write_json(aligned_output_path, aligned)
        write_json(unmapped_entity_path, entity_report)
        write_json(unmapped_relation_path, relation_report)
        return aligned, entity_report, relation_report


def align_schema_graph(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    aligned_output_path: str | Path = DEFAULT_ALIGNED_OUTPUT_PATH,
    unmapped_entity_path: str | Path = DEFAULT_UNMAPPED_ENTITY_PATH,
    unmapped_relation_path: str | Path = DEFAULT_UNMAPPED_RELATION_PATH,
    **pipeline_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """供其他流水线直接调用的便利函数。"""

    return SchemaAlignmentPipeline(**pipeline_kwargs).run(
        input_path=input_path,
        aligned_output_path=aligned_output_path,
        unmapped_entity_path=unmapped_entity_path,
        unmapped_relation_path=unmapped_relation_path,
    )


__all__ = [
    "DEFAULT_ALIGNED_OUTPUT_PATH",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_UNMAPPED_ENTITY_PATH",
    "DEFAULT_UNMAPPED_RELATION_PATH",
    "SchemaAlignmentPipeline",
    "align_schema_graph",
]
