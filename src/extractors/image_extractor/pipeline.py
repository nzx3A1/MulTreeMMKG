"""ImageChunk 到专用图片抽取器的统一调度流水线。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from model import Graph
from model.base import SourceModality
from model.graph import GraphMetadata
from src.utils.json_io import write_json

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
    classifications = [provider.resolve(chunk, image_path, index) for index, image_path in enumerate(paths)]
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
            extra={
                "status": aggregate_status,
                "image_task_count": len(task_graphs),
                "routes": task_routes,
                "model_called": model_called,
            },
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

    for chunk_index, chunk in enumerate(image_chunks, start=1):
        tasks = build_image_tasks(chunk, classification_provider=classification_provider)
        task_graphs: list[Graph] = []
        for task in tasks:
            kind = active_router.route(task)
            if allowed_kinds and kind.value not in allowed_kinds:
                task_graphs.append(
                    _skipped_task_graph(
                        task,
                        kind,
                        reason="main_classification_not_stratigraphic_profile",
                        model_called=bool(context.options.get("classification_model_called", False)),
                    )
                )
                continue
            task_graphs.append(active_registry.get(kind).extract(task, context))
        results.append(_merge_task_graphs(chunk, task_graphs))
        if show_progress:
            print(f"图片抽取架构路由：{chunk_index}/{len(image_chunks)}，单图任务={len(tasks)}")

    if output_path is not None:
        write_json(output_path, [graph.to_dict() for graph in results])
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
