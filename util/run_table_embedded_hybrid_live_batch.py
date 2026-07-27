"""真实调用 VLM 批量分类地层图片，并只抽取表格嵌入混合型图片。"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping


# 中文说明：允许从 util 目录直接执行，并始终复用项目统一图片入口和模型配置。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_config import settings
from src.extractors.image_extractor import (
    ImageExtractionTask,
    ImageExtractorKind,
    ImageExtractorRouter,
    VLMImageClassificationProvider,
    build_image_tasks,
    extract_table_embedded_hybrid_only,
)
from src.extractors.image_extractor.schema_models import as_string_tuple
from src.extractors.image_extractor.stratigraphic_profile import (
    StratigraphicProfileSubtype,
    VLMStratigraphicProfileSubtypeClassifier,
)
from src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid import (
    TableEmbeddedHybridPipeline,
    is_table_embedded_hybrid_payload,
    validate_and_repair_pixel_geometry,
)
from src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid.batch import (
    _validate_provenance,
)
from src.utils.json_io import read_json, write_json
from src.utils.vlm_client import VLMClient
from util.run_table_embedded_hybrid_live_single import (
    RecordingVLMClient,
    _find_extraction_payload,
)


DEFAULT_SOURCE = PROJECT_ROOT / "src" / "extractors" / "image_extractor" / "stratigraphic_profile" / "testImage" / "image_chunks_stratigraphic_profile.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "extractors" / "image_extractor" / "stratigraphic_profile" / "three_dimensional_stratigraphic_model"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "table_embedded_hybrid_live_batch_results.json"
DEFAULT_NODES_OUTPUT = DEFAULT_OUTPUT_DIR / "table_embedded_hybrid_live_batch_nodes.json"
DEFAULT_RELATIONS_OUTPUT = DEFAULT_OUTPUT_DIR / "table_embedded_hybrid_live_batch_relations.json"
# 中文说明：修改分类边界、坐标修复或图谱语义后递增，防止整图断点绕过新的确定性算法。
BATCH_PIPELINE_REVISION = "2026-07-27.3"


def _load_chunks(source_path: Path) -> list[dict[str, Any]]:
    """中文说明：读取根数组清单，校验每条 Chunk 都有唯一、存在的来源图片。"""

    payload = read_json(source_path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"地层图片 Chunk 文件必须是非空根数组：{source_path}")
    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise TypeError(f"第 {index} 条 Chunk 不是对象")
        chunk_id = str(raw.get("id") or raw.get("chunk_id") or "")
        paths = as_string_tuple(raw.get("image_path"))
        if not chunk_id or len(paths) != 1:
            raise ValueError(f"第 {index} 条 Chunk 必须具有 id 和唯一 image_path")
        image_path = str(Path(paths[0]).resolve())
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"第 {index} 条来源图片不存在：{image_path}")
        if chunk_id in seen_ids or image_path.casefold() in seen_paths:
            raise ValueError(f"清单出现重复 Chunk 或图片：{chunk_id}")
        seen_ids.add(chunk_id)
        seen_paths.add(image_path.casefold())
        chunk = dict(raw)
        chunk["image_path"] = [image_path]
        chunk["manifest_index"] = index
        chunks.append(chunk)
    return chunks


def _task_from_classification(
    chunk: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> ImageExtractionTask:
    """中文说明：用真实大类分类结果重建单图任务，供中间结果和溯源质量校验使用。"""

    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
    image_path = as_string_tuple(chunk.get("image_path"))[0]
    return ImageExtractionTask(
        document_id=str(chunk.get("document_id") or chunk_id.split(":section:", 1)[0]),
        chunk_id=chunk_id,
        image_id=f"{chunk_id}:image:0",
        image_index=0,
        image_path=image_path,
        caption=str(chunk.get("caption") or ""),
        references=as_string_tuple(chunk.get("references")),
        classification_code=str(classification.get("primary_code") or ""),
        classification_type=str(classification.get("primary_type") or ""),
        section_id=str(chunk.get("section_id") or ""),
        section_title=str(chunk.get("section_title") or ""),
        raw_chunk=chunk,
    )


def _existing_state(output_path: Path) -> tuple[dict[int, dict[str, Any]], list[Mapping[str, Any]]]:
    """中文说明：读取批量断点，返回逐清单索引结果和所有可按 Prompt 指纹复用的真实响应。"""

    if not output_path.is_file():
        return {}, []
    try:
        payload = read_json(output_path)
    except Exception:
        return {}, []
    if not isinstance(payload, Mapping) or not isinstance(payload.get("results"), list):
        return {}, []
    by_index: dict[int, dict[str, Any]] = {}
    calls: list[Mapping[str, Any]] = []
    for raw in payload["results"]:
        if not isinstance(raw, Mapping):
            continue
        index = raw.get("manifest_index")
        if isinstance(index, int):
            by_index[index] = dict(raw)
        raw_calls = raw.get("vlm_api_calls")
        if isinstance(raw_calls, list):
            calls.extend(call for call in raw_calls if isinstance(call, Mapping))
    return by_index, calls


def _summary(results: list[Mapping[str, Any]], source_count: int) -> dict[str, Any]:
    """中文说明：汇总分类、子分类、状态、实体关系和真实 API 调用统计。"""

    main_codes: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    entity_types: Counter[str] = Counter()
    relation_types: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    entity_count = relation_count = event_count = 0
    actual_requests = reused_calls = recorded_real_requests = 0
    for result in results:
        status_counts[str(result.get("status") or "unknown")] += 1
        classification = result.get("classification")
        if isinstance(classification, Mapping):
            main_codes[str(classification.get("primary_code") or "unknown")] += 1
        subtype = result.get("stratigraphic_subtype")
        if isinstance(subtype, Mapping) and subtype.get("subtype"):
            subtype_counts[str(subtype["subtype"])] += 1
        graph = result.get("graph")
        if isinstance(graph, Mapping):
            entities = graph.get("entities") if isinstance(graph.get("entities"), list) else []
            relations = graph.get("relations") if isinstance(graph.get("relations"), list) else []
            events = graph.get("events") if isinstance(graph.get("events"), list) else []
            entity_count += len(entities)
            relation_count += len(relations)
            event_count += len(events)
            entity_types.update(str(item.get("type") or "unknown") for item in entities if isinstance(item, Mapping))
            relation_types.update(str(item.get("type") or "unknown") for item in relations if isinstance(item, Mapping))
        calls = result.get("vlm_api_calls")
        if isinstance(calls, list):
            actual_requests += sum(bool(call.get("actual_request_this_run")) for call in calls if isinstance(call, Mapping))
            reused_calls += sum(bool(call.get("reused_from_previous_result")) for call in calls if isinstance(call, Mapping))
            # 中文说明：最终刷新可能全部命中缓存，单独统计有真实服务时间戳和原始响应的历史 API 调用证据。
            recorded_real_requests += sum(bool(call.get("real_api_called")) for call in calls if isinstance(call, Mapping))
    completed = status_counts.get("completed", 0)
    skipped = status_counts.get("skipped_non_target", 0)
    failed = sum(count for status, count in status_counts.items() if status in {"failed", "model_error", "partial"})
    return {
        "source_chunk_count": source_count,
        "result_count": len(results),
        "table_target_count": completed,
        "completed_count": completed,
        "skipped_non_target_count": skipped,
        "failed_count": failed,
        "entity_count": entity_count,
        "relation_count": relation_count,
        "event_count": event_count,
        "actual_api_request_count_this_run": actual_requests,
        "recorded_real_api_call_count": recorded_real_requests,
        "reused_api_call_count": reused_calls,
        "main_classification_counts": dict(sorted(main_codes.items())),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "entity_types": dict(sorted(entity_types.items())),
        "relation_types": dict(sorted(relation_types.items())),
    }


def _write_state(
    output_path: Path,
    source_path: Path,
    source_count: int,
    results_by_index: Mapping[int, Mapping[str, Any]],
    *,
    run_mode: str,
) -> dict[str, Any]:
    """中文说明：按原清单顺序原子式重建批量 JSON，并同步导出节点和关系侧车文件。"""

    results = [dict(results_by_index[index]) for index in sorted(results_by_index)]
    summary = _summary(results, source_count)
    terminal_count = summary["completed_count"] + summary["skipped_non_target_count"]
    payload = {
        "schema_version": "table_embedded_hybrid.live-batch-output.v1",
        "pipeline_revision": BATCH_PIPELINE_REVISION,
        "status": (
            "completed"
            if run_mode == "full" and terminal_count == source_count and summary["failed_count"] == 0
            else "classification_completed"
            if run_mode == "classification_only" and len(results) == source_count and summary["failed_count"] == 0
            else "partial"
        ),
        "run_mode": run_mode,
        "source_chunk_path": str(source_path.resolve()),
        "target_subtype": StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID.value,
        "events_extracted": False,
        "model": settings.vlm.model,
        "base_url": settings.vlm.base_url,
        "algorithm": "真实大类分类 → 真实地层子分类 → 非目标跳过 / 表格目标三段OCR → 坐标与轨道重建 → 深度对齐 → 确定性图谱装配",
        "summary": summary,
        "results": results,
    }
    write_json(output_path, payload)

    if run_mode == "full":
        nodes: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        for result in results:
            if result.get("status") != "completed" or not isinstance(result.get("graph"), Mapping):
                continue
            source = {
                "source_image_filename": result.get("source_image_filename"),
                "source_image_path": result.get("source_image_path"),
            }
            nodes.extend({**source, **item} for item in result["graph"].get("entities", []) if isinstance(item, Mapping))
            relations.extend({**source, **item} for item in result["graph"].get("relations", []) if isinstance(item, Mapping))
        write_json(DEFAULT_NODES_OUTPUT, nodes)
        write_json(DEFAULT_RELATIONS_OUTPUT, relations)
    return payload


def _classification_only_result(
    chunk: Mapping[str, Any],
    recorder: RecordingVLMClient,
) -> dict[str, Any]:
    """中文说明：只执行大类与必要的地层子分类，用于低成本确认目标图片集合。"""

    call_start = len(recorder.calls)
    provider = VLMImageClassificationProvider(recorder)
    tasks = build_image_tasks(chunk, classification_provider=provider)
    if len(tasks) != 1:
        raise ValueError("当前批量清单要求每条 Chunk 恰好展开为一个单图任务")
    task = tasks[0]
    classification = provider.records[-1]
    subtype_payload: dict[str, Any] = {}
    status = "classified_non_target"
    if classification.get("status") == "completed" and ImageExtractorRouter().route(task) is ImageExtractorKind.STRATIGRAPHIC_PROFILE:
        subtype = VLMStratigraphicProfileSubtypeClassifier().classify(task, recorder)
        subtype_payload = subtype.to_dict()
        status = (
            "classified_table_candidate"
            if subtype.subtype is StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID
            else "classified_non_target"
        )
    return {
        "manifest_index": int(chunk["manifest_index"]),
        "pipeline_revision": BATCH_PIPELINE_REVISION,
        "run_mode": "classification_only",
        "status": status,
        "chunk_id": task.chunk_id,
        "image_id": task.image_id,
        "source_image_path": task.image_path,
        "source_image_filename": Path(task.image_path).name,
        "classification": classification,
        "stratigraphic_subtype": subtype_payload,
        "target_selected": status == "classified_table_candidate",
        "vlm_api_calls": recorder.calls[call_start:],
        "graph": {"entities": [], "relations": [], "events": []},
        "validation": {"event_count": 0},
    }


def _full_result(
    chunk: Mapping[str, Any],
    recorder: RecordingVLMClient,
) -> dict[str, Any]:
    """中文说明：通过统一图片入口重跑单个 Chunk，并组装目标抽取或非目标跳过结果。"""

    call_start = len(recorder.calls)
    provider = VLMImageClassificationProvider(recorder)
    graph = extract_table_embedded_hybrid_only(
        [chunk],
        recorder,
        classification_provider=provider,
        show_progress=False,
    )[0]
    classification = provider.records[-1]
    task = _task_from_classification(chunk, classification)
    route = graph.metadata.extra.get("routes", [{}])[0]
    route = route if isinstance(route, Mapping) else {}
    graph_status = str(graph.metadata.extra.get("status") or "unknown")
    calls = recorder.calls[call_start:]
    subtype_payload = {
        "subtype": route.get("stratigraphic_subtype") or "",
        "subtype_name": route.get("stratigraphic_subtype_name") or "",
        "confidence": route.get("stratigraphic_subtype_confidence") or 0.0,
        "evidence": route.get("stratigraphic_subtype_evidence") or "",
        "source": "vlm_visual_classification" if route.get("stratigraphic_subtype") else "",
    }
    extraction_payload: dict[str, Any] = {}
    intermediate: dict[str, Any] = {}
    reference_errors = graph.validate_references()
    provenance_errors: list[str] = []
    dropped_relations: list[dict[str, Any]] = []
    if graph_status == "completed":
        extraction_payload = validate_and_repair_pixel_geometry(
            task,
            _find_extraction_payload(calls),
        )
        if not is_table_embedded_hybrid_payload(extraction_payload):
            raise ValueError("完成状态缺少可复原的 table_embedded_hybrid.v1 真实响应")
        intermediate = TableEmbeddedHybridPipeline().run(task, extraction_payload)
        provenance_errors = _validate_provenance(graph, task)
        dropped_relations = list(intermediate.get("quality", {}).get("dropped_explicit_relations") or [])
    event_count = len(graph.events)
    status = "completed" if graph_status == "completed" else "skipped_non_target" if graph_status == "skipped_non_target" else "failed"
    if reference_errors or provenance_errors or dropped_relations or event_count:
        status = "failed"
    return {
        "manifest_index": int(chunk["manifest_index"]),
        "pipeline_revision": BATCH_PIPELINE_REVISION,
        "run_mode": "full",
        "status": status,
        "chunk_id": task.chunk_id,
        "image_id": task.image_id,
        "source_image_path": task.image_path,
        "source_image_filename": Path(task.image_path).name,
        "classification": classification,
        "stratigraphic_subtype": subtype_payload,
        "target_selected": graph_status == "completed",
        "real_vlm_called": bool(calls),
        "vlm_api_calls": calls,
        "ocr_and_content_extraction": extraction_payload,
        "structured_intermediate_result": intermediate,
        "graph": graph.to_dict(),
        "validation": {
            "reference_errors": reference_errors,
            "provenance_errors": provenance_errors,
            "event_count": event_count,
            "dropped_explicit_relations": dropped_relations,
        },
    }


def run_batch(
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    classification_only: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """中文说明：按清单顺序执行可恢复批处理，每张完成后立即刷新主结果和节点关系侧车。"""

    chunks = _load_chunks(source_path)
    existing, cached_calls = _existing_state(output_path) if resume else ({}, [])
    recorder = RecordingVLMClient(VLMClient(config=settings.vlm), cached_calls=cached_calls)
    results_by_index: dict[int, dict[str, Any]] = dict(existing)
    run_mode = "classification_only" if classification_only else "full"
    for sequence, chunk in enumerate(chunks, start=1):
        index = int(chunk["manifest_index"])
        previous = results_by_index.get(index)
        if (
            previous
            and previous.get("run_mode") == run_mode
            and previous.get("pipeline_revision") == BATCH_PIPELINE_REVISION
            and (
                previous.get("status") in {"classified_table_candidate", "classified_non_target"}
                if classification_only
                else previous.get("status") in {"completed", "skipped_non_target"}
            )
        ):
            print(f"批量复用整图：{sequence}/{len(chunks)} {Path(chunk['image_path'][0]).name}", flush=True)
            continue
        call_start = len(recorder.calls)
        try:
            result = (
                _classification_only_result(chunk, recorder)
                if classification_only
                else _full_result(chunk, recorder)
            )
        except Exception as exc:
            paths = as_string_tuple(chunk.get("image_path"))
            result = {
                "manifest_index": index,
                "pipeline_revision": BATCH_PIPELINE_REVISION,
                "run_mode": run_mode,
                "status": "failed",
                "chunk_id": str(chunk.get("id") or ""),
                "image_id": f"{chunk.get('id')}:image:0",
                "source_image_path": paths[0] if paths else "",
                "source_image_filename": Path(paths[0]).name if paths else "",
                "error": str(exc),
                "vlm_api_calls": recorder.calls[call_start:],
                "graph": {"entities": [], "relations": [], "events": []},
                "validation": {"event_count": 0},
            }
        results_by_index[index] = result
        payload = _write_state(
            output_path,
            source_path,
            len(chunks),
            results_by_index,
            run_mode=run_mode,
        )
        print(
            f"批量进度：{sequence}/{len(chunks)}，状态={result['status']}，"
            f"目标完成={payload['summary']['completed_count']}，失败={payload['summary']['failed_count']}",
            flush=True,
        )
    return _write_state(
        output_path,
        source_path,
        len(chunks),
        results_by_index,
        run_mode=run_mode,
    )


def main() -> None:
    """中文说明：解析分类预检、断点续跑和输出参数并打印最终批量统计。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="真实 VLM 批量抽取 table_embedded_hybrid 图片")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="地层图片 Chunk 根数组")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="批量完整结果 JSON")
    parser.add_argument("--classification-only", action="store_true", help="只执行大类和子分类预检")
    parser.add_argument("--no-resume", action="store_true", help="忽略现有断点并全部重新请求")
    args = parser.parse_args()

    payload = run_batch(
        args.source,
        args.output,
        classification_only=args.classification_only,
        resume=not args.no_resume,
    )
    print(f"批量状态：{payload['status']}")
    print(f"统计：{payload['summary']}")
    print(f"输出：{args.output.resolve()}")


if __name__ == "__main__":
    main()
