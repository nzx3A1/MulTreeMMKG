"""ImageChunk 到专用图片抽取器的统一调度流水线。"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from model import Graph
from model.base import SourceModality
from model.graph import GraphMetadata
from src.utils.json_io import write_json
from src.utils.logger import get_logger

from .classification import (
    ImageClassificationProvider,
    InlineImageClassificationProvider,
    VLMImageClassificationProvider,
)
from .factory import build_default_registry
from .registry import ImageExtractorRegistry
from .router import ImageExtractorRouter
from .schema_models import (
    ImageExtractionContext,
    ImageExtractionTask,
    ImageExtractorKind,
    as_string_tuple,
)


logger = get_logger("extractors.image_extractor.pipeline")


def write_image_extraction_result(
    path: str | Path,
    graphs: Sequence[Graph],
    *,
    status: str,
) -> None:
    """按其他第四阶段模态的统一结构保存图片 Graph 与统计信息。"""

    result = {
        "_status": status,
        "statistics": {
            "graph_count": len(graphs),
            "completed_image_chunk_count": len(graphs),
            "entity_count": sum(len(graph.entities) for graph in graphs),
            "relation_count": sum(len(graph.relations) for graph in graphs),
            "event_count": sum(len(graph.events) for graph in graphs),
        },
        # 生产结果只保留结构化图谱；原始模型响应体积大且不参与后续阶段。
        "graphs": [
            graph.to_dict(exclude={"metadata": {"raw_response"}})
            for graph in graphs
        ],
    }
    write_json(path, result)
    logger.info(
        f"[图片抽取/保存] 结果文件已写入：Graph={len(graphs)}，"
        f"实体={result['statistics']['entity_count']}，关系={result['statistics']['relation_count']}，"
        f"事件={result['statistics']['event_count']}，path={Path(path).resolve()}"
    )


def build_image_tasks(
    chunk: Mapping[str, Any],
    classification_provider: ImageClassificationProvider | None = None,
) -> list[ImageExtractionTask]:
    """中文说明：把一个可含多路径的 ImageChunk 展开为单图片任务列表。"""

    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
    if not chunk_id:
        raise ValueError("ImageChunk 缺少 id/chunk_id")
    paths = as_string_tuple(chunk.get("image_path"))
    references = as_string_tuple(chunk.get("references"))
    provider = classification_provider or InlineImageClassificationProvider()
    classifications = []
    for index, image_path in enumerate(paths):
        logger.info(
            f"[图片抽取/分类] 开始：chunk={chunk_id}，image={index + 1}/{len(paths)}，path={image_path}"
        )
        classification_started_at = perf_counter()
        try:
            classification = provider.resolve(chunk, image_path, index)
        except Exception:
            logger.exception(
                f"[图片抽取/分类] 失败：chunk={chunk_id}，image_index={index}，path={image_path}，"
                f"耗时={perf_counter() - classification_started_at:.2f}s"
            )
            raise
        classifications.append(classification)
        logger.info(
            f"[图片抽取/分类] 完成：chunk={chunk_id}，image_index={index}，"
            f"code={classification.code or '未分类'}，type={classification.type_name or '未知'}，"
            f"耗时={perf_counter() - classification_started_at:.2f}s"
        )
    return [
        ImageExtractionTask(
            document_id=str(chunk.get("document_id") or chunk_id.split(":section:", 1)[0]),
            chunk_id=chunk_id,
            image_id=f"{chunk_id}:image:{index}",
            image_index=index,
            image_path=image_path,
            caption=str(chunk.get("caption") or ""),
            references=references,
            classification_code=classifications[index].code,
            classification_type=classifications[index].type_name,
            section_id=str(chunk.get("section_id") or ""),
            section_title=str(chunk.get("section_title") or ""),
            raw_chunk=chunk,
        )
        for index, image_path in enumerate(paths)
    ]


def _merge_task_graphs(chunk: Mapping[str, Any], task_graphs: Sequence[Graph]) -> Graph:
    """中文说明：把同一 ImageChunk 下的单图结果重新汇总为一个来源明确的 Graph。"""

    merged = Graph.merge(task_graphs)
    task_routes = [
        {
            "image_id": graph.metadata.extra.get("image_id"),
            "image_path": graph.metadata.extra.get("image_path"),
            "extractor_kind": graph.metadata.extra.get("extractor_kind"),
            "status": graph.metadata.extra.get("status"),
            "model_called": graph.metadata.extra.get("model_called", False),
            "classification_code": graph.metadata.extra.get("classification_code"),
            "classification_type": graph.metadata.extra.get("classification_type"),
            "vlm_call_count": int(graph.metadata.extra.get("vlm_call_count") or 0),
            "vlm_stages": list(graph.metadata.extra.get("vlm_stages") or []),
            "stratigraphic_subtype": graph.metadata.extra.get("stratigraphic_subtype"),
            "stratigraphic_subtype_name": graph.metadata.extra.get("stratigraphic_subtype_name"),
            "stratigraphic_subtype_confidence": graph.metadata.extra.get("stratigraphic_subtype_confidence"),
            "stratigraphic_subtype_evidence": graph.metadata.extra.get("stratigraphic_subtype_evidence"),
            # 中文说明：保留单图抽取器的底层错误，避免批量聚合后只剩下笼统的 model_error 状态。
            "model_errors": list(graph.metadata.extra.get("model_errors") or []),
        }
        for graph in task_graphs
    ]
    statuses = [str(route.get("status") or "unknown") for route in task_routes]
    if statuses and all(status == "completed" for status in statuses):
        aggregate_status = "completed"
    elif statuses and all(status.startswith("skipped_") for status in statuses):
        aggregate_status = "skipped_non_target"
    elif any(status == "model_error" for status in statuses):
        aggregate_status = "model_error" if all(status == "model_error" for status in statuses) else "partial"
    else:
        aggregate_status = "not_implemented"
    model_called = any(bool(route.get("model_called")) for route in task_routes)
    map_relation_details = [
        graph.metadata.extra.get("relation_generation")
        for graph in task_graphs
        if isinstance(graph.metadata.extra.get("relation_generation"), Mapping)
    ]
    map_legend_details = [
        {
            "image_id": graph.metadata.extra.get("image_id"),
            "legend_items": list(graph.metadata.extra.get("legend_items") or []),
            "legend_bindings": list(graph.metadata.extra.get("legend_bindings") or []),
        }
        for graph in task_graphs
        if graph.metadata.extra.get("extractor_kind") == "map_spatial"
    ]
    relation_generation: dict[str, Any] | None = None
    if len(map_relation_details) == 1:
        relation_generation = dict(map_relation_details[0])
    elif map_relation_details:
        # 中文说明：多图 Chunk 合并白名单与空间链审计信息，避免只保留最后一张图的关系生成记录。
        schema_groups: list[dict[str, Any]] = []
        seen_groups: set[tuple[str, str, tuple[str, ...]]] = set()
        candidate_entities: list[str] = []
        generated_edges: list[dict[str, Any]] = []
        for detail in map_relation_details:
            for group in detail.get("relation_schema_candidates") or []:
                if not isinstance(group, Mapping):
                    continue
                key = (
                    str(group.get("source_type") or ""),
                    str(group.get("target_type") or ""),
                    tuple(str(value) for value in group.get("allowed_relations") or []),
                )
                if key not in seen_groups:
                    seen_groups.add(key)
                    schema_groups.append(dict(group))
            spatial = detail.get("spatial_direction_mapping")
            if isinstance(spatial, Mapping):
                candidate_entities.extend(str(value) for value in spatial.get("candidate_entities") or [])
                generated_edges.extend(dict(value) for value in spatial.get("generated_edges") or [] if isinstance(value, Mapping))
        relation_generation = {
            "legend_driven": True,
            "deterministic_mapping": True,
            "relation_schema_candidates": schema_groups,
            "legend_binding_mapping": {
                "enabled": True,
                "generated_by_program": True,
                "binding_count": sum(
                    int((detail.get("legend_binding_mapping") or {}).get("binding_count") or 0)
                    for detail in map_relation_details
                ),
                "generated_relation_count": sum(
                    int((detail.get("legend_binding_mapping") or {}).get("generated_relation_count") or 0)
                    for detail in map_relation_details
                ),
            },
            "spatial_direction_mapping": {
                "enabled": True,
                "candidate_selection_by_vlm": True,
                "direction_calculated_by_program": True,
                "directed": True,
                "max_out_degree": 1,
                "max_in_degree": 1,
                "allow_inverse_duplicate": False,
                "allow_cycle": False,
                "candidate_entities": list(dict.fromkeys(candidate_entities)),
                "generated_edges": generated_edges,
                "generated_edge_count": len(generated_edges),
                "per_image": map_relation_details,
            },
        }
    extra = {
        "status": aggregate_status,
        "image_task_count": len(task_graphs),
        "routes": task_routes,
        "model_called": model_called,
        "vlm_call_count": sum(int(route.get("vlm_call_count") or 0) for route in task_routes),
    }
    if relation_generation is not None:
        extra["relation_generation"] = relation_generation
    if len(map_legend_details) == 1:
        extra["legend_items"] = map_legend_details[0]["legend_items"]
        extra["legend_bindings"] = map_legend_details[0]["legend_bindings"]
    elif map_legend_details:
        # 中文说明：多图 Chunk 为每条图例记录补 image_id，防止局部 legend_id 在合并后冲突。
        extra["legend_items"] = [
            {"image_id": detail["image_id"], **dict(item)}
            for detail in map_legend_details
            for item in detail["legend_items"]
            if isinstance(item, Mapping)
        ]
        extra["legend_bindings"] = [
            {"image_id": detail["image_id"], **dict(item)}
            for detail in map_legend_details
            for item in detail["legend_bindings"]
            if isinstance(item, Mapping)
        ]
    return Graph(
        entities=merged.entities,
        relations=merged.relations,
        events=merged.events,
        metadata=GraphMetadata(
            document_id=str(chunk.get("document_id") or "") or None,
            chunk_id=str(chunk.get("id") or chunk.get("chunk_id") or "") or None,
            modality=SourceModality.IMAGE,
            stage="stage_04_image_extraction",
            raw_response=[graph.metadata.raw_response for graph in task_graphs if graph.metadata.raw_response is not None],
            extra=extra,
        ),
    )


def _skipped_task_graph(
    task: ImageExtractionTask,
    extractor_kind: ImageExtractorKind,
    *,
    reason: str,
    model_called: bool,
) -> Graph:
    """中文说明：为统一入口中不在目标范围的图片生成无节点、无关系、无事件的可追踪结果。"""

    graph = Graph.from_chunk(
        document_id=task.document_id,
        chunk_id=task.chunk_id,
        modality=SourceModality.IMAGE,
        stage="stage_04_image_extraction_skipped",
    )
    graph.metadata.extra.update(
        {
            "status": "skipped_non_target_main_classification",
            "skip_reason": reason,
            "extractor_kind": extractor_kind.value,
            "image_id": task.image_id,
            "image_index": task.image_index,
            "image_path": task.image_path,
            "source_image_path": task.image_path,
            "classification_code": task.classification_code,
            "classification_type": task.classification_type,
            "model_called": model_called,
            "events_extracted": False,
        }
    )
    return graph


def extract_from_images(
    image_chunks: Sequence[Mapping[str, Any]],
    llm_client: Any | None = None,
    vlm_client: Any | None = None,
    *,
    registry: ImageExtractorRegistry | None = None,
    router: ImageExtractorRouter | None = None,
    classification_provider: ImageClassificationProvider | None = None,
    context_options: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    show_progress: bool = True,
) -> list[Graph]:
    """中文说明：接收 extractor_init 的图片 Chunk，完成展开、路由和骨架 Graph 汇总。"""

    active_registry = registry or build_default_registry()
    active_router = router or ImageExtractorRouter()
    context = ImageExtractionContext(
        llm_client=llm_client,
        vlm_client=vlm_client,
        options=dict(context_options or {}),
    )
    allowed_kinds = {
        str(value)
        for value in context.options.get("allowed_extractor_kinds", [])
    }
    results: list[Graph] = []
    total_chunks = len(image_chunks)
    total_images = sum(len(as_string_tuple(chunk.get("image_path"))) for chunk in image_chunks)
    pipeline_started_at = perf_counter()
    logger.info(
        f"[图片抽取/管线] 启动：Chunk={total_chunks}，图片={total_images}，"
        f"限制抽取器={sorted(allowed_kinds) if allowed_kinds else '无'}"
    )

    for chunk_index, chunk in enumerate(image_chunks, start=1):
        chunk_started_at = perf_counter()
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "<missing>")
        section_title = str(chunk.get("section_title") or "未标注章节")
        logger.info(
            f"[图片抽取/Chunk] 开始 {chunk_index}/{total_chunks}：chunk={chunk_id}，section={section_title}"
        )
        tasks = build_image_tasks(chunk, classification_provider=classification_provider)
        if not tasks:
            logger.warning(f"[图片抽取/Chunk] 未找到图片路径，将生成空 Graph：chunk={chunk_id}")
        task_graphs: list[Graph] = []
        for task in tasks:
            kind = active_router.route(task)
            task_started_at = perf_counter()
            logger.info(
                f"[图片抽取/路由] chunk={task.chunk_id}，image_index={task.image_index}，"
                f"classification={task.classification_code or '未分类'}({task.classification_type or '未知'})，"
                f"extractor={kind.value}，path={task.image_path}"
            )
            if allowed_kinds and kind.value not in allowed_kinds:
                task_graph = _skipped_task_graph(
                    task,
                    kind,
                    reason="main_classification_not_stratigraphic_profile",
                    model_called=bool(context.options.get("classification_model_called", False)),
                )
            else:
                try:
                    task_graph = active_registry.get(kind).extract(task, context)
                except Exception:
                    logger.exception(
                        f"[图片抽取/单图] 失败：chunk={task.chunk_id}，image_index={task.image_index}，"
                        f"extractor={kind.value}，path={task.image_path}，"
                        f"耗时={perf_counter() - task_started_at:.2f}s"
                    )
                    raise
            task_graphs.append(task_graph)
            task_status = str(task_graph.metadata.extra.get("status") or "unknown")
            logger.info(
                f"[图片抽取/单图] 完成：chunk={task.chunk_id}，image_index={task.image_index}，"
                f"extractor={kind.value}，status={task_status}，"
                f"实体={len(task_graph.entities)}，关系={len(task_graph.relations)}，事件={len(task_graph.events)}，"
                f"耗时={perf_counter() - task_started_at:.2f}s"
            )
            model_errors = list(task_graph.metadata.extra.get("model_errors") or [])
            if model_errors:
                logger.warning(
                    f"[图片抽取/单图] 模型调用存在错误：chunk={task.chunk_id}，"
                    f"image_index={task.image_index}，status={task_status}，errors={model_errors}"
                )
        merged_graph = _merge_task_graphs(chunk, task_graphs)
        results.append(merged_graph)
        logger.info(
            f"[图片抽取/Chunk] 完成 {chunk_index}/{total_chunks}：chunk={chunk_id}，"
            f"status={merged_graph.metadata.extra.get('status', 'unknown')}，单图任务={len(tasks)}，"
            f"实体={len(merged_graph.entities)}，关系={len(merged_graph.relations)}，事件={len(merged_graph.events)}，"
            f"耗时={perf_counter() - chunk_started_at:.2f}s"
        )
        if show_progress:
            print(f"图片抽取进度：{chunk_index}/{total_chunks}，chunk={chunk_id}，单图任务={len(tasks)}")

    if output_path is not None:
        write_image_extraction_result(output_path, results, status="completed")
    logger.info(
        f"[图片抽取/管线] 完成：Graph={len(results)}，"
        f"实体={sum(len(graph.entities) for graph in results)}，"
        f"关系={sum(len(graph.relations) for graph in results)}，"
        f"事件={sum(len(graph.events) for graph in results)}，"
        f"耗时={perf_counter() - pipeline_started_at:.2f}s"
    )
    return results


def extract_table_embedded_hybrid_only(
    image_chunks: Sequence[Mapping[str, Any]],
    vlm_client: Any,
    *,
    classification_provider: ImageClassificationProvider | None = None,
    output_path: str | Path | None = None,
    show_progress: bool = True,
) -> list[Graph]:
    """中文说明：统一入口先真实执行大类/子分类，只抽取表格嵌入混合型，其余图片明确跳过。"""

    provider = classification_provider or VLMImageClassificationProvider(vlm_client)
    return extract_from_images(
        image_chunks,
        llm_client=None,
        vlm_client=vlm_client,
        classification_provider=provider,
        context_options={
            "allowed_extractor_kinds": [ImageExtractorKind.STRATIGRAPHIC_PROFILE.value],
            "allowed_stratigraphic_subtypes": ["table_embedded_hybrid"],
            "classification_model_called": True,
            "enable_relation_audit": False,
        },
        output_path=output_path,
        show_progress=show_progress,
    )
