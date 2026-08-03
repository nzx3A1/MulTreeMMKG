"""真实调用视觉 API 跑通一张表格嵌入混合地层图的完整抽取流程。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time
from typing import Any, Mapping


# 中文说明：支持从 util 目录直接运行，同时只复用项目已有配置、客户端和抽取器。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_config import settings
from src.extractors.image_extractor.router import ImageExtractorRouter
from src.extractors.image_extractor.schema_models import (
    ImageExtractionContext,
    ImageExtractionTask,
    ImageExtractorKind,
    as_string_tuple,
)
from src.extractors.image_extractor.stratigraphic_profile import StratigraphicProfileExtractor
from src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid.batch import _validate_provenance
from src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid.pipeline import (
    TableEmbeddedHybridPipeline,
    is_table_embedded_hybrid_payload,
)
from src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid.segmented_vlm import (
    NODE_ENRICHMENT_SCHEMA_VERSION,
    SEGMENT_ORDER,
    apply_node_enrichment,
    merge_segmented_table_payloads,
    validate_and_repair_pixel_geometry,
)
from src.extractors.image_extractor.vlm_classification import VLMImageClassifier
from src.utils.json_io import write_json
from src.utils.llm_client import safe_json_loads
from src.utils.vlm_client import VLMClient


DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "mineru_output" / "鄂尔多斯盆地奥陶系马家沟组白云岩储层特征及成因机制_吴东旭" / "images" / "787873a7863480cdb8cd2b32373fe10780afb8365cc9153ebd6aa6e0ebcea629.jpg"
# 中文说明：单图结果默认写入当前子类型自己的 result 目录，避免与三维地层模型结果混放。
DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "extractors" / "image_extractor" / "stratigraphic_profile" / "table_embedded_hybrid" / "result" / "table_embedded_hybrid_live_api_single_extraction_result.json"

# 中文说明：该常量逐字段保存用户指定的唯一测试 Chunk，避免测试时误取清单中的其他图片。
LIVE_TEST_CHUNK: dict[str, Any] = {
    "id": "鄂尔多斯盆地奥陶系马家沟组白云岩储层特征及成因机制_吴东旭:section:9:image:0:asset:0",
    "order": 1,
    "modality": "image",
    "image_path": [str(DEFAULT_IMAGE_PATH)],
    "caption": "图2 鄂尔多斯盆地奥陶系白云岩储层分布柱状图\nFig. 2 The stratigraphic column showing of the Ordovician dolomite reservoir distribution in the Ordos Basin",
    "references": [
        "盆地奥陶系地层由下而上依次为冶里组，亮甲山组和马家沟组。冶里组和亮甲山组主要岩性为含燧石结核或条带白云岩、泥质白云岩和竹叶状白云岩，层厚一般不超过 100 m。与之相比，马家沟组厚度在 100～900 m 之间，主要为碳酸盐岩和蒸发岩沉积，是鄂尔多斯盆地白云岩储层发育的主要层位。马家沟组地层从下到上可以分为马一—马六总共 6 个段。其中马一、马三和马五为相对海退期，以碳酸盐岩局限台地和蒸发台地相为主。岩性主要为白云岩，且发育了三大套的膏盐岩沉积。马二、马四和马六段为相对海侵期，主要为碳酸盐岩开阔台地和局限台地相，以白云岩和灰岩沉积为主。马五段根据相对海平面升降，从上到下又可以分为马五 —马五 十个亚段，其中马五亚段为马五段主要的蒸发岩发育期（图2）。"
    ],
}


class RecordingVLMClient:
    """包装真实客户端，只记录阶段响应和耗时，不保存 API Key 或图片 base64。"""

    # 中文说明：通知地层抽取器使用三段短响应 OCR，而不是容易超时的单次长响应。
    supports_segmented_table_extraction = True

    def __init__(
        self,
        client: VLMClient,
        *,
        classification_model: str | None = None,
        extraction_model: str | None = None,
        cached_calls: list[Mapping[str, Any]] | None = None,
    ) -> None:
        """中文说明：默认复用配置模型，并可加载同一结果文件中已完成的阶段以支持断点续跑。"""

        self.client = client
        self.classification_model = classification_model or client.model_name_vl
        self.extraction_model = extraction_model or client.model_name_vl
        self.calls: list[dict[str, Any]] = []
        self.cached_calls = [dict(call) for call in (cached_calls or [])]
        self.actual_request_count = 0

    def describe_image(self, image_path: str | Path, prompt: str, **kwargs: Any) -> str:
        """中文说明：透传真实请求，记录模型、时间、原始文本与可解析 JSON，证明每阶段确实调用 API。"""

        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        task_name = str(kwargs.get("task_name") or "")
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        request_model = (
            self.extraction_model
            if "视觉抽取" in task_name or "专用OCR" in task_name or "官方名规范化" in task_name
            else self.classification_model
        )
        kwargs.setdefault("model", request_model)
        # 中文说明：思考开关由 VLMClient 按具体模型协议选择，避免阶段脚本重复维护服务差异。
        for index, cached in enumerate(self.cached_calls):
            parsed = cached.get("parsed_response")
            if (
                cached.get("status") == "completed"
                and str(cached.get("task_name") or "") == task_name
                and str(cached.get("model") or "") == request_model
                and str(cached.get("prompt_sha256") or "") == prompt_sha256
                and isinstance(parsed, Mapping)
                and bool(parsed)
                and str(cached.get("raw_response") or "")
            ):
                reused = dict(self.cached_calls.pop(index))
                reused.update(
                    {
                        "sequence": len(self.calls) + 1,
                        "reused_from_previous_result": True,
                        "actual_request_this_run": False,
                        "prompt_sha256": prompt_sha256,
                    }
                )
                self.calls.append(reused)
                print(f"API 复用：{task_name}，模型={request_model}", flush=True)
                return str(reused["raw_response"])

        print(f"API 开始：{task_name}，模型={request_model}", flush=True)
        self.actual_request_count += 1
        try:
            response = self.client.describe_image(image_path, prompt, **kwargs)
        except Exception as exc:
            self.calls.append(
                {
                    "sequence": len(self.calls) + 1,
                    "task_name": task_name,
                    "status": "failed",
                    "real_api_called": True,
                    "actual_request_this_run": True,
                    "model": request_model,
                    "base_url": self.client.config.base_url,
                    "started_at_utc": started_at.isoformat(),
                    "prompt_sha256": prompt_sha256,
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "error": str(exc),
                }
            )
            print(f"API 失败：{task_name}，错误={exc}", flush=True)
            raise
        raw_response = str(response or "")
        parsed: Any = None
        parse_error = ""
        try:
            parsed = safe_json_loads(raw_response)
        except Exception as exc:
            parse_error = str(exc)
        self.calls.append(
            {
                "sequence": len(self.calls) + 1,
                "task_name": task_name,
                "status": "completed",
                "real_api_called": True,
                "actual_request_this_run": True,
                "model": request_model,
                "base_url": self.client.config.base_url,
                "started_at_utc": started_at.isoformat(),
                "prompt_sha256": prompt_sha256,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "response_character_count": len(raw_response),
                "raw_response": raw_response,
                "parsed_response": parsed,
                "parse_error": parse_error,
            }
        )
        print(
            f"API 完成：{task_name}，耗时={self.calls[-1]['duration_seconds']} 秒，字符={len(raw_response)}",
            flush=True,
        )
        return raw_response


def _task_from_chunk(chunk: Mapping[str, Any]) -> ImageExtractionTask:
    """中文说明：将用户指定 Chunk 转成单图任务，并在大类 API 返回前保持分类字段为空。"""

    image_paths = as_string_tuple(chunk.get("image_path"))
    chunk_id = str(chunk.get("id") or "")
    if not chunk_id or len(image_paths) != 1:
        raise ValueError("实时单图测试要求 Chunk 具有 id 和唯一 image_path")
    image_path = Path(image_paths[0])
    if not image_path.is_file():
        raise FileNotFoundError(f"实时测试图片不存在：{image_path}")
    return ImageExtractionTask(
        document_id=chunk_id.split(":section:", 1)[0],
        chunk_id=chunk_id,
        image_id=f"{chunk_id}:image:0",
        image_index=0,
        image_path=str(image_path.resolve()),
        caption=str(chunk.get("caption") or ""),
        references=as_string_tuple(chunk.get("references")),
        raw_chunk=dict(chunk),
    )


def _find_extraction_payload(calls: list[Mapping[str, Any]]) -> dict[str, Any]:
    """中文说明：从真实调用审计中定位专用 OCR/内容抽取响应，拒绝把分类响应误当抽取结果。"""

    segments: dict[str, Mapping[str, Any]] = {}
    for call in calls:
        task_name = str(call.get("task_name") or "")
        if "视觉抽取" not in task_name and "专用OCR" not in task_name:
            continue
        payload = call.get("parsed_response")
        if isinstance(payload, Mapping) and is_table_embedded_hybrid_payload(payload):
            return dict(payload)
        if isinstance(payload, Mapping):
            segment = str(payload.get("segment") or "")
            if segment in SEGMENT_ORDER:
                segments[segment] = payload
    if len(segments) == len(SEGMENT_ORDER):
        return merge_segmented_table_payloads(segments)
    return {}


def _find_node_enrichment(calls: list[Mapping[str, Any]]) -> dict[str, Any]:
    """中文说明：从调用审计中读取第 4 次节点官方名响应，供断点结果确定性重建。"""

    for call in reversed(calls):
        payload = call.get("parsed_response")
        if (
            isinstance(payload, Mapping)
            and str(payload.get("schema_version") or "") == NODE_ENRICHMENT_SCHEMA_VERSION
        ):
            return dict(payload)
    return {}


def _load_cached_calls(output_path: Path) -> list[Mapping[str, Any]]:
    """中文说明：从同一单图结果读取可复用 API 记录，损坏或异图文件一律视为无缓存。"""

    if not output_path.is_file():
        return []
    try:
        from src.utils.json_io import read_json

        payload = read_json(output_path)
    except Exception:
        return []
    if not isinstance(payload, Mapping):
        return []
    results = payload.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], Mapping):
        return []
    if str(results[0].get("chunk_id") or "") != str(LIVE_TEST_CHUNK["id"]):
        return []
    calls = results[0].get("vlm_api_calls")
    return list(calls) if isinstance(calls, list) else []


def run_live_pipeline(
    chunk: Mapping[str, Any] = LIVE_TEST_CHUNK,
    *,
    cached_calls: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """中文说明：依次执行大类 API、子分类 API、专用 OCR/抽取 API 和确定性图谱装配。"""

    task = _task_from_chunk(chunk)
    recorder = RecordingVLMClient(VLMClient(config=settings.vlm), cached_calls=cached_calls)

    classification = VLMImageClassifier().classify(task, recorder)
    classified_task = replace(
        task,
        classification_code=classification.primary_code,
        classification_type=classification.primary_type,
    )
    routed_kind = ImageExtractorRouter().route(classified_task)
    if routed_kind is not ImageExtractorKind.STRATIGRAPHIC_PROFILE:
        raise ValueError(
            f"真实大类分类路由到 {routed_kind.value}，不是地层抽取器：{classification.to_dict()}"
        )

    graph = StratigraphicProfileExtractor().extract(
        classified_task,
        ImageExtractionContext(
            vlm_client=recorder,
            llm_client=None,
            options={"enable_relation_audit": False},
        ),
    )
    extraction_payload = _find_extraction_payload(recorder.calls)
    intermediate: dict[str, Any] = {}
    if is_table_embedded_hybrid_payload(extraction_payload):
        extraction_payload = validate_and_repair_pixel_geometry(classified_task, extraction_payload)
        node_enrichment = _find_node_enrichment(recorder.calls)
        if node_enrichment:
            extraction_payload = apply_node_enrichment(extraction_payload, node_enrichment)
        intermediate = TableEmbeddedHybridPipeline().run(classified_task, extraction_payload)

    graph_payload = graph.to_dict()
    reference_errors = graph.validate_references()
    provenance_errors = _validate_provenance(graph, classified_task)
    event_count = len(graph.events)
    model_errors = list(graph.metadata.extra.get("model_errors") or [])
    dropped_relations = (
        list(intermediate.get("quality", {}).get("dropped_explicit_relations") or [])
        if intermediate
        else []
    )
    completed = bool(
        extraction_payload
        and intermediate
        and graph.entities
        and not model_errors
        and not reference_errors
        and not provenance_errors
        and not dropped_relations
        and event_count == 0
    )
    entity_types = Counter(entity.type for entity in graph.entities)
    relation_types = Counter(relation.type for relation in graph.relations)
    subtype = {
        "subtype": graph.metadata.extra.get("stratigraphic_subtype", ""),
        "subtype_name": graph.metadata.extra.get("stratigraphic_subtype_name", ""),
        "confidence": graph.metadata.extra.get("stratigraphic_subtype_confidence", 0.0),
        "evidence": graph.metadata.extra.get("stratigraphic_subtype_evidence", ""),
        "source": graph.metadata.extra.get("stratigraphic_subtype_source", ""),
    }
    return {
        "schema_version": "table_embedded_hybrid.live-api-output.v1",
        "status": "completed" if completed else "failed_quality_gate",
        "source_chunk": dict(chunk),
        "target_subtype": "table_embedded_hybrid",
        "events_extracted": False,
        "algorithm": "视觉大类分类 → 视觉子分类 → PP-StructureV3 像素几何 → 三段 VLM 语义 ID 选择 → OCR 深度轴/相对层序 → 结构化中间结果 → 确定性知识图谱装配",
        "api_execution": {
            "real_api_called": bool(recorder.calls),
            "configured_model": settings.vlm.model,
            "classification_model": recorder.classification_model,
            "extraction_model": recorder.extraction_model,
            "base_url": settings.vlm.base_url,
            "call_count": len(recorder.calls),
            "actual_request_count_this_run": recorder.actual_request_count,
            "reused_call_count": sum(
                bool(call.get("reused_from_previous_result")) for call in recorder.calls
            ),
            "completed_call_count": sum(call.get("status") == "completed" for call in recorder.calls),
            "failed_call_count": sum(call.get("status") == "failed" for call in recorder.calls),
        },
        "summary": {
            "target_image_count": 1,
            "completed_count": int(completed),
            "failed_count": int(not completed),
            "entity_count": len(graph.entities),
            "relation_count": len(graph.relations),
            "event_count": event_count,
            "reference_error_count": len(reference_errors),
            "provenance_error_count": len(provenance_errors),
            "dropped_explicit_relation_count": len(dropped_relations),
            "entity_types": dict(sorted(entity_types.items())),
            "relation_types": dict(sorted(relation_types.items())),
        },
        "results": [
            {
                "sequence": 1,
                "status": "completed" if completed else "failed_quality_gate",
                "chunk_id": classified_task.chunk_id,
                "image_id": classified_task.image_id,
                "source_image_path": classified_task.image_path,
                "source_image_filename": Path(classified_task.image_path).name,
                "real_vlm_called": bool(recorder.calls),
                "real_ocr_called": bool(intermediate.get("ppstructure_geometry")),
                "ppstructure_runtime": dict(
                    intermediate.get("ppstructure_geometry", {}).get("runtime", {})
                ) if intermediate else {},
                "classification": classification.to_dict(),
                "extractor_route": routed_kind.value,
                "stratigraphic_subtype": subtype,
                "vlm_api_calls": recorder.calls,
                "ocr_and_content_extraction": extraction_payload,
                "structured_intermediate_result": intermediate,
                "graph": graph_payload,
                "validation": {
                    "model_errors": model_errors,
                    "reference_errors": reference_errors,
                    "provenance_errors": provenance_errors,
                    "event_count": event_count,
                    "dropped_explicit_relations": dropped_relations,
                },
            }
        ],
    }


def main() -> None:
    """中文说明：运行用户指定单图的完整实时链路，并无论质量是否通过都写出可审计结果。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="真实调用 API 测试一张表格嵌入混合地层图片")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="完整抽取结果 JSON")
    parser.add_argument("--no-resume", action="store_true", help="忽略已有结果中的成功阶段并全部重新请求")
    args = parser.parse_args()

    cached_calls = [] if args.no_resume else _load_cached_calls(args.output)
    result = run_live_pipeline(cached_calls=cached_calls)
    write_json(args.output, result)
    summary = result["summary"]
    first = result["results"][0]
    print(f"状态：{result['status']}")
    print(f"真实 API 调用：{result['api_execution']['call_count']} 次")
    print(f"大类：{first['classification']['primary_code']} {first['classification']['primary_type']}")
    print(f"子类：{first['stratigraphic_subtype']['subtype']}")
    print(f"节点/关系/事件：{summary['entity_count']}/{summary['relation_count']}/{summary['event_count']}")
    print(f"输出文件：{args.output.resolve()}")
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
