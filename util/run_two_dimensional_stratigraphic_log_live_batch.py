"""真实调用 VLM 批量抽取用户清单中的全部二维平面地层—测井图。"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping


# 中文说明：允许从 util 目录直接运行，并始终复用项目模型配置和地层统一调度器。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_config import settings
from src.extractors.image_extractor.schema_models import ImageExtractionContext
from src.extractors.image_extractor.stratigraphic_profile import StratigraphicProfileExtractor
from src.extractors.image_extractor.stratigraphic_profile.two_dimensional_stratigraphic_log import (
    TwoDimensionalStratigraphicLogPipeline,
    build_two_dimensional_stratigraphic_log_graph,
    build_two_dimensional_task,
    classification_from_chunk,
    is_two_dimensional_stratigraphic_log_payload,
    load_two_dimensional_target_chunks,
    validate_two_dimensional_graph,
)
from src.extractors.image_extractor.stratigraphic_profile.two_dimensional_stratigraphic_log.batch import (
    TARGET_SUBTYPE,
)
from src.utils.json_io import read_json, write_json
from src.utils.vlm_client import VLMClient
from util.run_table_embedded_hybrid_live_single import RecordingVLMClient


DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "src"
    / "extractors"
    / "image_extractor"
    / "stratigraphic_profile"
    / "testImage"
    / "stratigraphic_profile_subtype_mock.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "src"
    / "extractors"
    / "image_extractor"
    / "stratigraphic_profile"
    / "two_dimensional_stratigraphic_log"
    / "result"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "two_dimensional_stratigraphic_log_live_batch_results.json"
DEFAULT_NODES_OUTPUT = DEFAULT_OUTPUT_DIR / "two_dimensional_stratigraphic_log_live_batch_nodes.json"
DEFAULT_RELATIONS_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "two_dimensional_stratigraphic_log_live_batch_relations.json"
)
BATCH_PIPELINE_REVISION = "2026-07-27.4"


class _FixedSubtypeClassifier:
    """向统一调度器提供源清单中已确认的单图子分类，不重复调用 VLM 分类。"""

    uses_vlm = False

    def __init__(self, classification: Any) -> None:
        """中文说明：保存当前 Chunk 的人工视觉子分类结果。"""

        self.classification = classification

    def classify(self, task: Any, vlm_client: Any = None) -> Any:
        """中文说明：返回固定子分类；参数只为兼容统一分类接口。"""

        _ = task, vlm_client
        return self.classification


def _existing_state(
    output_path: Path,
) -> tuple[dict[int, dict[str, Any]], list[Mapping[str, Any]]]:
    """中文说明：读取逐目标断点，并收集可按 Prompt 指纹复用的历史真实响应。"""

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
        target_index = raw.get("target_index")
        if isinstance(target_index, int):
            by_index[target_index] = dict(raw)
        raw_calls = raw.get("vlm_api_calls")
        if isinstance(raw_calls, list):
            calls.extend(call for call in raw_calls if isinstance(call, Mapping))
    return by_index, calls


def _visual_payload(calls: list[Mapping[str, Any]]) -> dict[str, Any]:
    """中文说明：从单图调用审计中定位二维专用响应，拒绝混入分类或其他子类型结果。"""

    for call in reversed(calls):
        parsed = call.get("parsed_response")
        if (
            "二维平面地层测井视觉抽取" in str(call.get("task_name") or "")
            and isinstance(parsed, Mapping)
            and str(parsed.get("schema_version") or "")
            == "two_dimensional_stratigraphic_log.v1"
        ):
            return dict(parsed)
    return {}


def _intermediate_from_graph(graph: Any) -> dict[str, Any]:
    """中文说明：从 Graph 原始响应中取回已校验并补齐空间关系的中间结果。"""

    raw_response = graph.metadata.raw_response
    if not isinstance(raw_response, Mapping):
        return {}
    intermediate = raw_response.get("structured_intermediate_result")
    return dict(intermediate) if isinstance(intermediate, Mapping) else {}


def _run_target(
    chunk: Mapping[str, Any],
    recorder: RecordingVLMClient,
) -> dict[str, Any]:
    """中文说明：复用源子分类，通过统一调度器抽取单个二维目标并执行来源和关系质量门控。"""

    call_start = len(recorder.calls)
    task = build_two_dimensional_task(chunk)
    classification = classification_from_chunk(chunk)
    extractor = StratigraphicProfileExtractor(
        subtype_classifier=_FixedSubtypeClassifier(classification)
    )
    graph = extractor.extract(
        task,
        ImageExtractionContext(
            vlm_client=recorder,
            llm_client=None,
            options={"allowed_stratigraphic_subtypes": [TARGET_SUBTYPE.value]},
        ),
    )
    calls = recorder.calls[call_start:]
    return _finalize_target_result(
        chunk,
        graph,
        calls,
        _visual_payload(calls),
        visual_reused_from_previous_result=False,
    )


def _finalize_target_result(
    chunk: Mapping[str, Any],
    graph: Any,
    calls: list[Mapping[str, Any]],
    visual: Mapping[str, Any],
    *,
    visual_reused_from_previous_result: bool,
) -> dict[str, Any]:
    """中文说明：统一完成新请求或本地重建图的质量门控、审计字段和批量记录。"""

    task = build_two_dimensional_task(chunk)
    classification = classification_from_chunk(chunk)
    validation = validate_two_dimensional_graph(graph, task)
    intermediate = _intermediate_from_graph(graph)
    dropped = list((intermediate.get("quality") or {}).get("dropped_relations") or [])
    deprecated_position_types = {"above", "below", "left_of", "right_of", "adjacent_to"}
    dropped_explicit = [
        item
        for item in dropped
        if isinstance(item, Mapping)
        and (item.get("source_id") or item.get("source"))
        and (item.get("target_id") or item.get("target"))
        and bool(item.get("explicit", True))
        and not (
            item.get("reason") == "unsupported_relation_type"
            and item.get("type") in deprecated_position_types
        )
    ]
    graph_status = str(graph.metadata.extra.get("status") or "unknown")
    validation_failed = any(
        validation[key]
        for key in ("reference_errors", "provenance_errors", "quality_errors")
    )
    status = (
        "completed"
        if graph_status == "completed" and not validation_failed and not dropped_explicit
        else "failed"
    )
    return {
        "manifest_index": int(chunk["manifest_index"]),
        "target_index": int(chunk["target_index"]),
        "pipeline_revision": BATCH_PIPELINE_REVISION,
        "status": status,
        "chunk_id": task.chunk_id,
        "image_id": task.image_id,
        "source_image_path": task.image_path,
        "source_image_filename": Path(task.image_path).name,
        "caption": task.caption,
        "stratigraphic_subtype": classification.to_dict(),
        "subtype_source_reused_from_manifest": True,
        "real_vlm_called": bool(calls),
        "visual_reused_from_previous_result": visual_reused_from_previous_result,
        "vlm_api_calls": calls,
        "visual_extraction": dict(visual),
        "structured_intermediate_result": intermediate,
        "graph": graph.to_dict(),
        "validation": {
            **validation,
            "dropped_explicit_relations": dropped_explicit,
            "graph_dispatch_status": graph_status,
        },
    }


def _rebuild_target_from_previous(
    chunk: Mapping[str, Any],
    previous: Mapping[str, Any],
) -> dict[str, Any]:
    """中文说明：使用结果文件中已保存的真实视觉 JSON 重跑关系规则和孤立节点裁剪，不发起 API。"""

    visual = previous.get("visual_extraction")
    if not isinstance(visual, Mapping) or not is_two_dimensional_stratigraphic_log_payload(
        visual
    ):
        raise ValueError("既有结果缺少可本地重建的二维专用视觉响应")
    task = build_two_dimensional_task(chunk)
    classification = classification_from_chunk(chunk)
    intermediate = TwoDimensionalStratigraphicLogPipeline().run(task, visual)
    graph = build_two_dimensional_stratigraphic_log_graph(task, intermediate)
    graph.metadata.extra.update(
        {
            "model_called": True,
            "vlm_called": True,
            "vlm_call_count": 1,
            "subtype_classification_vlm_called": False,
            "stratigraphic_subtype": classification.subtype.value,
            "stratigraphic_subtype_name": classification.subtype_name,
            "stratigraphic_subtype_confidence": classification.confidence,
            "stratigraphic_subtype_evidence": classification.evidence,
            "stratigraphic_subtype_source": classification.source,
            "subtype_extraction_algorithm": "axis_layer_correlation_extraction",
        }
    )
    raw_calls = previous.get("vlm_api_calls")
    calls = [dict(call) for call in raw_calls if isinstance(call, Mapping)] if isinstance(raw_calls, list) else []
    return _finalize_target_result(
        chunk,
        graph,
        calls,
        visual,
        visual_reused_from_previous_result=True,
    )


def _summary(
    results: list[Mapping[str, Any]],
    source_count: int,
    target_count: int,
) -> dict[str, Any]:
    """中文说明：汇总目标完成率、实体关系类型、位置关系和真实 API 调用证据。"""

    status_counts: Counter[str] = Counter()
    entity_types: Counter[str] = Counter()
    relation_types: Counter[str] = Counter()
    entity_count = relation_count = event_count = 0
    vertical_count = horizontal_count = semantic_position_count = 0
    actual_requests = reused_calls = recorded_real_requests = 0
    for result in results:
        status_counts[str(result.get("status") or "unknown")] += 1
        graph = result.get("graph")
        if isinstance(graph, Mapping):
            entities = graph.get("entities") if isinstance(graph.get("entities"), list) else []
            relations = graph.get("relations") if isinstance(graph.get("relations"), list) else []
            events = graph.get("events") if isinstance(graph.get("events"), list) else []
            entity_count += len(entities)
            relation_count += len(relations)
            event_count += len(events)
            entity_types.update(
                str(item.get("type") or "unknown")
                for item in entities
                if isinstance(item, Mapping)
            )
            relation_types.update(
                str(item.get("type") or "unknown")
                for item in relations
                if isinstance(item, Mapping)
            )
        validation = result.get("validation")
        if isinstance(validation, Mapping):
            vertical_count += int(validation.get("vertical_relation_count") or 0)
            horizontal_count += int(validation.get("horizontal_relation_count") or 0)
            semantic_position_count += int(
                validation.get("semantic_position_relation_count") or 0
            )
        calls = result.get("vlm_api_calls")
        if isinstance(calls, list):
            actual_requests += sum(
                bool(call.get("actual_request_this_run"))
                for call in calls
                if isinstance(call, Mapping)
            )
            reused_calls += sum(
                bool(call.get("reused_from_previous_result"))
                for call in calls
                if isinstance(call, Mapping)
            )
            recorded_real_requests += sum(
                bool(call.get("real_api_called"))
                for call in calls
                if isinstance(call, Mapping)
            )
    return {
        "source_chunk_count": source_count,
        "target_chunk_count": target_count,
        "result_count": len(results),
        "completed_count": status_counts.get("completed", 0),
        "failed_count": status_counts.get("failed", 0),
        "entity_count": entity_count,
        "relation_count": relation_count,
        "event_count": event_count,
        "vertical_relation_count": vertical_count,
        "horizontal_relation_count": horizontal_count,
        "semantic_position_relation_count": semantic_position_count,
        "actual_api_request_count_this_run": actual_requests,
        "recorded_real_api_call_count": recorded_real_requests,
        "reused_api_call_count": reused_calls,
        "status_counts": dict(sorted(status_counts.items())),
        "entity_types": dict(sorted(entity_types.items())),
        "relation_types": dict(sorted(relation_types.items())),
    }


def _write_state(
    output_path: Path,
    source_path: Path,
    source_count: int,
    target_count: int,
    results_by_index: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """中文说明：按目标顺序刷新主结果，并同步导出完成图的节点和关系侧车。"""

    results = [dict(results_by_index[index]) for index in sorted(results_by_index)]
    summary = _summary(results, source_count, target_count)
    payload = {
        "schema_version": "two_dimensional_stratigraphic_log.live-batch-output.v1",
        "pipeline_revision": BATCH_PIPELINE_REVISION,
        "status": (
            "completed"
            if summary["result_count"] == target_count
            and summary["completed_count"] == target_count
            and summary["failed_count"] == 0
            else "partial"
        ),
        "source_chunk_path": str(source_path.resolve()),
        "target_subtype": TARGET_SUBTYPE.value,
        "events_extracted": False,
        "model": settings.vlm.model,
        "base_url": settings.vlm.base_url,
        "algorithm": (
            "复用清单子分类 → 真实二维视觉图元抽取 → "
            "层序上下关系与横向位置确定性装配 → 来源和引用质量校验"
        ),
        "summary": summary,
        "results": results,
    }
    write_json(output_path, payload)

    nodes: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") != "completed" or not isinstance(result.get("graph"), Mapping):
            continue
        source = {
            "source_image_filename": result.get("source_image_filename"),
            "source_image_path": result.get("source_image_path"),
        }
        graph = result["graph"]
        nodes.extend(
            {**source, **item}
            for item in graph.get("entities", [])
            if isinstance(item, Mapping)
        )
        relations.extend(
            {**source, **item}
            for item in graph.get("relations", [])
            if isinstance(item, Mapping)
        )
    write_json(DEFAULT_NODES_OUTPUT, nodes)
    write_json(DEFAULT_RELATIONS_OUTPUT, relations)
    return payload


def run_batch(
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """中文说明：顺序处理全部二维目标，每张结束即保存断点并支持复用同 Prompt 的真实响应。"""

    source_count, targets = load_two_dimensional_target_chunks(source_path)
    existing, cached_calls = _existing_state(output_path) if resume else ({}, [])
    recorder = RecordingVLMClient(VLMClient(config=settings.vlm), cached_calls=cached_calls)
    results_by_index: dict[int, dict[str, Any]] = dict(existing)
    for sequence, chunk in enumerate(targets, start=1):
        target_index = int(chunk["target_index"])
        previous = results_by_index.get(target_index)
        if (
            previous
            and previous.get("pipeline_revision") == BATCH_PIPELINE_REVISION
            and previous.get("status") == "completed"
        ):
            print(
                f"批量复用整图：{sequence}/{len(targets)} "
                f"{Path(chunk['image_path'][0]).name}",
                flush=True,
            )
            continue
        call_start = len(recorder.calls)
        try:
            previous_visual = previous.get("visual_extraction") if previous else None
            if isinstance(previous_visual, Mapping) and is_two_dimensional_stratigraphic_log_payload(
                previous_visual
            ):
                result = _rebuild_target_from_previous(chunk, previous)
                print(
                    f"本地重建：{sequence}/{len(targets)} "
                    f"{Path(chunk['image_path'][0]).name}",
                    flush=True,
                )
            else:
                result = _run_target(chunk, recorder)
        except Exception as exc:
            task = build_two_dimensional_task(chunk)
            result = {
                "manifest_index": int(chunk["manifest_index"]),
                "target_index": target_index,
                "pipeline_revision": BATCH_PIPELINE_REVISION,
                "status": "failed",
                "chunk_id": task.chunk_id,
                "image_id": task.image_id,
                "source_image_path": task.image_path,
                "source_image_filename": Path(task.image_path).name,
                "caption": task.caption,
                "stratigraphic_subtype": classification_from_chunk(chunk).to_dict(),
                "subtype_source_reused_from_manifest": True,
                "error": str(exc),
                "vlm_api_calls": recorder.calls[call_start:],
                "visual_extraction": {},
                "structured_intermediate_result": {},
                "graph": {"entities": [], "relations": [], "events": []},
                "validation": {"event_count": 0, "quality_errors": [str(exc)]},
            }
        results_by_index[target_index] = result
        payload = _write_state(
            output_path,
            source_path,
            source_count,
            len(targets),
            results_by_index,
        )
        print(
            f"批量进度：{sequence}/{len(targets)}，状态={result['status']}，"
            f"完成={payload['summary']['completed_count']}，"
            f"失败={payload['summary']['failed_count']}",
            flush=True,
        )
    return _write_state(
        output_path,
        source_path,
        source_count,
        len(targets),
        results_by_index,
    )


def main() -> None:
    """中文说明：解析源文件、输出和断点参数，并打印最终批量统计。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="真实 VLM 批量抽取 two_dimensional_stratigraphic_log 图片"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="含子分类的根数组 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="批量完整结果 JSON")
    parser.add_argument("--no-resume", action="store_true", help="忽略既有断点并重新请求")
    args = parser.parse_args()
    payload = run_batch(args.source, args.output, resume=not args.no_resume)
    print(f"批量状态：{payload['status']}")
    print(f"统计：{payload['summary']}")
    print(f"输出：{args.output.resolve()}")


if __name__ == "__main__":
    main()
