"""从子分类 JSON 过滤全部三维地层建模图并批量执行真实 VLM 抽取。"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping


# 中文说明：支持直接以模块运行，同时始终复用项目统一模型配置和 Graph 数据模型。
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_config import settings
from model import Graph
from src.extractors.image_extractor.schema_models import (
    ImageExtractionContext,
    ImageExtractionTask,
    as_string_tuple,
)
from src.extractors.image_extractor.stratigraphic_profile.extractor import StratigraphicProfileExtractor
from src.extractors.image_extractor.stratigraphic_profile.subclassifier import (
    StratigraphicProfileSubtype,
    StratigraphicProfileSubtypeClassifier,
)
from src.utils.json_io import read_json, write_json
from src.utils.vlm_client import VLMClient


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = PACKAGE_DIR / "testData" / "stratigraphic_profile_subtype_mock.json"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "result"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "three_dimensional_stratigraphic_model_batch_results.json"


def _load_target_chunks(source_path: Path) -> list[dict[str, Any]]:
    """中文说明：读取根数组并只保留人工子分类明确为三维模型且图片真实存在的 Chunk。"""

    payload = read_json(source_path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"子分类文件必须是非空根数组：{source_path}")
    targets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for manifest_index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise TypeError(f"第 {manifest_index} 条 Chunk 不是对象")
        subtype = raw.get("stratigraphic_subtype")
        if not isinstance(subtype, Mapping):
            raise ValueError(f"第 {manifest_index} 条 Chunk 缺少 stratigraphic_subtype")
        if str(subtype.get("subtype") or "") != StratigraphicProfileSubtype.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL.value:
            continue
        chunk_id = str(raw.get("id") or "").strip()
        paths = as_string_tuple(raw.get("image_path"))
        if not chunk_id or len(paths) != 1:
            raise ValueError(f"三维目标第 {manifest_index} 条必须具有 id 和唯一 image_path")
        image_path = str(Path(paths[0]).resolve())
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"三维目标来源图片不存在：{image_path}")
        if chunk_id in seen_ids or image_path.casefold() in seen_paths:
            raise ValueError(f"三维目标清单出现重复 Chunk 或图片：{chunk_id}")
        seen_ids.add(chunk_id)
        seen_paths.add(image_path.casefold())
        chunk = dict(raw)
        chunk["image_path"] = [image_path]
        chunk["manifest_index"] = manifest_index
        targets.append(chunk)
    if not targets:
        raise ValueError("子分类文件中没有 three_dimensional_stratigraphic_model 目标")
    return targets


def _build_task(chunk: Mapping[str, Any]) -> ImageExtractionTask:
    """中文说明：把一个单图 Chunk 转换为保留图题、正文参考和来源字段的规范任务。"""

    chunk_id = str(chunk.get("id") or "")
    image_path = as_string_tuple(chunk.get("image_path"))[0]
    return ImageExtractionTask(
        document_id=str(chunk.get("document_id") or chunk_id.split(":section:", 1)[0]),
        chunk_id=chunk_id,
        image_id=f"{chunk_id}:image:0",
        image_index=0,
        image_path=image_path,
        caption=str(chunk.get("caption") or ""),
        references=as_string_tuple(chunk.get("references")),
        classification_code="A04",
        classification_type="三维地层建模图",
        section_id=str(chunk.get("section_id") or ""),
        section_title=str(chunk.get("section_title") or ""),
        raw_chunk=chunk,
    )


def _validate_provenance(graph: Graph, task: ImageExtractionTask) -> list[str]:
    """中文说明：逐节点和逐关系检查图片、Chunk、图内证据及上下文证据边界。"""

    errors: list[str] = []
    expected = {
        "source_modality": "image",
        "source_image_path": task.image_path,
        "source_image_id": task.image_id,
        "source_chunk_id": task.chunk_id,
        "image_index": task.image_index,
    }
    for kind, records in (("实体", graph.entities), ("关系", graph.relations)):
        for record in records:
            if not str(record.provenance or "").strip():
                errors.append(f"{kind} {record.id} 缺少 provenance")
            for key, expected_value in expected.items():
                if record.metadata.get(key) != expected_value:
                    errors.append(f"{kind} {record.id} 的 {key} 不匹配当前来源任务")
            if not str(record.metadata.get("visual_evidence") or "").strip():
                errors.append(f"{kind} {record.id} 缺少 visual_evidence")
            if (
                kind == "关系"
                and record.attributes.get("evidence_scope") == "context"
                and not str(record.metadata.get("context_evidence") or "").strip()
            ):
                errors.append(f"上下文关系 {record.id} 缺少 context_evidence")
    return errors


def _result_record(chunk: Mapping[str, Any], task: ImageExtractionTask, graph: Graph) -> dict[str, Any]:
    """中文说明：将单图 Graph、上下关系和多轮逐层岩性质量整理为可独立审计的结果对象。"""

    reference_errors = graph.validate_references()
    provenance_errors = _validate_provenance(graph, task) if graph.entities or graph.relations else []
    vertical_relations = [
        relation
        for relation in graph.relations
        if relation.type in {"directly_overlies", "directly_underlies"}
    ]
    context_relations = [
        relation
        for relation in graph.relations
        if relation.attributes.get("evidence_scope") == "context"
    ]
    layer_segments = [entity for entity in graph.entities if entity.type == "visual_layer_segment"]
    stratigraphic_unit_ids = {
        entity.id for entity in graph.entities if entity.type == "stratigraphic_unit"
    }
    lithology_assignments = [
        relation
        for relation in graph.relations
        if relation.type == "has_lithology" and relation.source_id in stratigraphic_unit_ids
    ]
    quality = graph.metadata.extra.get("quality")
    multipass_summary = (
        dict(quality.get("multipass_lithology_summary") or {})
        if isinstance(quality, Mapping)
        else {}
    )
    graph_status = str(graph.metadata.extra.get("status") or "unknown")
    status = "completed" if graph_status == "completed" else "failed"
    if reference_errors or provenance_errors or graph.events:
        status = "failed"
    subtype = chunk.get("stratigraphic_subtype")
    return {
        "manifest_index": int(chunk["manifest_index"]),
        "status": status,
        "chunk_id": task.chunk_id,
        "image_id": task.image_id,
        "source_image_path": task.image_path,
        "source_image_filename": Path(task.image_path).name,
        "caption": task.caption,
        "references": list(task.references),
        "stratigraphic_subtype": dict(subtype) if isinstance(subtype, Mapping) else {},
        "graph": graph.to_dict(),
        "validation": {
            "reference_errors": reference_errors,
            "provenance_errors": provenance_errors,
            "event_count": len(graph.events),
            "vertical_relation_count": len(vertical_relations),
            "context_relation_count": len(context_relations),
            "visual_layer_segment_count": len(layer_segments),
            "lithology_assignment_count": len(lithology_assignments),
            "multipass_lithology_summary": multipass_summary,
            "vlm_call_count": int(graph.metadata.extra.get("vlm_call_count") or 0),
            "model_errors": list(graph.metadata.extra.get("model_errors") or []),
        },
    }


def _summarize(results: list[Mapping[str, Any]], target_count: int) -> dict[str, Any]:
    """中文说明：汇总全部目标、上下文关系以及多轮逐层岩性识别的覆盖和复核数量。"""

    status_counts: Counter[str] = Counter()
    entity_types: Counter[str] = Counter()
    relation_types: Counter[str] = Counter()
    entity_count = relation_count = event_count = vertical_count = context_count = 0
    segment_count = lithology_assignment_count = multipass_vlm_call_count = 0
    reviewed_segment_count = unknown_segment_count = 0
    for result in results:
        status_counts[str(result.get("status") or "unknown")] += 1
        graph = result.get("graph")
        if not isinstance(graph, Mapping):
            continue
        entities = graph.get("entities") if isinstance(graph.get("entities"), list) else []
        relations = graph.get("relations") if isinstance(graph.get("relations"), list) else []
        events = graph.get("events") if isinstance(graph.get("events"), list) else []
        entity_count += len(entities)
        relation_count += len(relations)
        event_count += len(events)
        entity_types.update(str(item.get("type") or "unknown") for item in entities if isinstance(item, Mapping))
        relation_types.update(str(item.get("type") or "unknown") for item in relations if isinstance(item, Mapping))
        validation = result.get("validation")
        if isinstance(validation, Mapping):
            vertical_count += int(validation.get("vertical_relation_count") or 0)
            context_count += int(validation.get("context_relation_count") or 0)
            segment_count += int(validation.get("visual_layer_segment_count") or 0)
            lithology_assignment_count += int(validation.get("lithology_assignment_count") or 0)
            multipass = validation.get("multipass_lithology_summary")
            if isinstance(multipass, Mapping):
                multipass_vlm_call_count += int(multipass.get("vlm_call_count") or 0)
                reviewed_segment_count += int(multipass.get("reviewed_segment_count") or 0)
                unknown_segment_count += int(multipass.get("unknown_segment_count") or 0)
    return {
        "target_chunk_count": target_count,
        "result_count": len(results),
        "completed_count": status_counts.get("completed", 0),
        "failed_count": status_counts.get("failed", 0),
        "entity_count": entity_count,
        "relation_count": relation_count,
        "event_count": event_count,
        "vertical_relation_count": vertical_count,
        "context_relation_count": context_count,
        "visual_layer_segment_count": segment_count,
        "lithology_assignment_count": lithology_assignment_count,
        "multipass_vlm_call_count": multipass_vlm_call_count,
        "reviewed_segment_count": reviewed_segment_count,
        "unknown_segment_count": unknown_segment_count,
        "status_counts": dict(sorted(status_counts.items())),
        "entity_types": dict(sorted(entity_types.items())),
        "relation_types": dict(sorted(relation_types.items())),
    }


def _write_batch_output(
    output_path: Path,
    source_path: Path,
    results: list[Mapping[str, Any]],
    target_count: int,
    *,
    run_mode: str,
) -> dict[str, Any]:
    """中文说明：刷新批量主文件，并把每张图写成唯一命名的独立 JSON 结果。"""

    summary = _summarize(results, target_count)
    payload = {
        "schema_version": "three_dimensional_stratigraphic_model.batch-output.v1",
        "status": "completed" if summary["completed_count"] == target_count and not summary["failed_count"] else "partial",
        "run_mode": run_mode,
        "source_chunk_path": str(source_path.resolve()),
        "target_subtype": StratigraphicProfileSubtype.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL.value,
        "model": settings.vlm.model,
        "base_url": settings.vlm.base_url,
        "events_extracted": False,
        "algorithm": "三维表面与对象识别 → 层界冻结 → 图例目录 → 地层底色签名与同色掩膜 → 地层单元分批多岩性识别 → 逐层岩性识别 → 低置信度盲审/裁决 → 联合全图一致性审查 → 同柱同父层级上下层序 → 空间与上下文三元组 → Graph 装配",
        "summary": summary,
        "results": [dict(result) for result in results],
    }
    write_json(output_path, payload)
    for sequence, result in enumerate(results, start=1):
        filename = str(result.get("source_image_filename") or f"image_{sequence}")
        stem = Path(filename).stem
        write_json(output_path.parent / f"{sequence:02d}_{stem}_extraction.json", dict(result))
    return payload


def run_batch(
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    vlm_client: Any | None = None,
    run_mode: str = "live_vlm",
) -> dict[str, Any]:
    """中文说明：对清单中全部三维目标逐图真实抽取，单图失败不阻断其余目标。"""

    targets = _load_target_chunks(source_path)
    classifier = StratigraphicProfileSubtypeClassifier(source_path)
    extractor = StratigraphicProfileExtractor(subtype_classifier=classifier)
    client = vlm_client or VLMClient(config=settings.vlm)
    results: list[dict[str, Any]] = []
    for sequence, chunk in enumerate(targets, start=1):
        task = _build_task(chunk)
        graph = extractor.extract(
            task,
            ImageExtractionContext(
                vlm_client=client,
                llm_client=None,
                options={
                    "allowed_stratigraphic_subtypes": [
                        StratigraphicProfileSubtype.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL.value
                    ]
                },
            ),
        )
        result = _result_record(chunk, task, graph)
        results.append(result)
        payload = _write_batch_output(
            output_path,
            source_path,
            results,
            len(targets),
            run_mode=run_mode,
        )
        print(
            f"三维抽取进度：{sequence}/{len(targets)}，状态={result['status']}，"
            f"上下关系={result['validation']['vertical_relation_count']}，"
            f"上下文关系={result['validation']['context_relation_count']}，"
            f"累计完成={payload['summary']['completed_count']}",
            flush=True,
        )
    return _write_batch_output(
        output_path,
        source_path,
        results,
        len(targets),
        run_mode=run_mode,
    )


class _RecordedPayloadVLM:
    """按图片文件名返回已记录真实 VLM 响应的只读重建客户端。"""

    def __init__(self, payload_by_filename: Mapping[str, Mapping[str, Any]]) -> None:
        """中文说明：复制已记录响应索引，确定性重建期间不建立任何网络连接。"""

        self.payload_by_filename = {
            str(filename): dict(payload) for filename, payload in payload_by_filename.items()
        }

    def describe_image(self, image_path: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """中文说明：返回当前图片上一轮真实响应，让新质量规则可离线重放。"""

        _ = prompt, kwargs
        filename = Path(image_path).name
        if filename not in self.payload_by_filename:
            raise KeyError(f"已有批量结果缺少图片 {filename} 的真实 VLM 响应")
        return dict(self.payload_by_filename[filename])


def rebuild_from_recorded_live_output(
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """中文说明：复用主结果中记录的真实响应，应用最新版确定性规则而不重复请求模型。"""

    existing = read_json(output_path)
    if not isinstance(existing, Mapping) or not isinstance(existing.get("results"), list):
        raise ValueError(f"没有可重建的批量主结果：{output_path}")
    payload_by_filename: dict[str, Mapping[str, Any]] = {}
    for result in existing["results"]:
        if not isinstance(result, Mapping):
            continue
        graph = result.get("graph")
        metadata = graph.get("metadata") if isinstance(graph, Mapping) else None
        raw_response = metadata.get("raw_response") if isinstance(metadata, Mapping) else None
        visual = raw_response.get("visual") if isinstance(raw_response, Mapping) else None
        filename = str(result.get("source_image_filename") or "")
        if filename and isinstance(visual, Mapping):
            payload_by_filename[filename] = visual
    return run_batch(
        source_path,
        output_path,
        vlm_client=_RecordedPayloadVLM(payload_by_filename),
        run_mode="deterministic_rebuild_from_recorded_live_payload",
    )


def main() -> None:
    """中文说明：解析命令行参数，运行全部三维目标并在质量未全部通过时返回非零状态。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="批量抽取子分类 JSON 中全部三维地层建模图")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="地层子分类 Chunk JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="批量抽取主结果 JSON")
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        help="不请求模型，复用主结果内记录的真实 VLM JSON 应用最新版确定性规则",
    )
    args = parser.parse_args()
    result = (
        rebuild_from_recorded_live_output(args.source, args.output)
        if args.rebuild_existing
        else run_batch(args.source, args.output)
    )
    summary = result["summary"]
    print(f"目标/完成/失败：{summary['target_chunk_count']}/{summary['completed_count']}/{summary['failed_count']}")
    print(f"实体/关系/事件：{summary['entity_count']}/{summary['relation_count']}/{summary['event_count']}")
    print(f"上下关系/上下文关系：{summary['vertical_relation_count']}/{summary['context_relation_count']}")
    print(f"输出文件：{args.output.resolve()}")
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
