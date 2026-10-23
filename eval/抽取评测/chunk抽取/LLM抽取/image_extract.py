"""逐图像 Chunk 提示词抽取，支持仅重跑需要子图切分的样本并汇总回原图。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from text_extract import DATA_DIR, call_model, image_paths, read_samples, run, to_graph
from model import Graph
from src.extractors.image_extractor.panel_splitter import VLMPanelSplitter
from src.extractors.image_extractor.pipeline import (
    _attach_panel_provenance, _merge_task_graphs, _summarize_source_images,
)
from src.extractors.image_extractor.postprocess import merge_same_name_entities
from src.extractors.image_extractor.schema_models import ImageExtractionTask, as_string_tuple
from src.extractors.image_extractor.vlm_classification import VLMImageClassifier
from src.utils.llm_client import LLMClient
from src.utils.vlm_client import VLMClient


class CountedVLM(VLMClient):
    """统计每个样本的逻辑视觉请求次数，不将 SDK 重试重复计数。"""

    def __init__(self) -> None:
        """初始化独立客户端和线程内请求计数。"""
        super().__init__()
        self.request_count = 0

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """统一记录检测、分类与提示词抽取发出的请求。"""
        self.request_count += 1
        return super().chat(messages, **kwargs)


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """使用同目录临时文件原子写入检查点，避免中断破坏已有结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def extract_split_sample(sample: dict[str, Any], input_path: Path, panel_dir: Path) -> dict[str, Any]:
    """检测单个样本，仅在需要拆分或缺少旧结果时逐叶抽取，保持提示词评测方法。"""
    vlm = CountedVLM()
    report: dict[str, Any] = {"sample_id": sample["sample_id"], "status": "failed", "sources": [],
                              "original_image_paths": sample.get("_original_image_paths", sample["content"]["image_path"])}
    try:
        tasks = []
        sample_id = sample["sample_id"]
        for index, path in enumerate(image_paths(sample, "image", input_path)):
            source = ImageExtractionTask(
                document_id=sample_id.split("::", 1)[0], chunk_id=sample_id,
                image_id=f"{sample_id}:image:{index}", image_index=index, image_path=str(path),
                caption=str(sample["content"].get("caption") or ""),
                references=as_string_tuple(sample["content"].get("references")),
                section_title=str(sample["section"].get("section_title") or ""),
            )
            prepared = VLMPanelSplitter(vlm, panel_dir).prepare(source)
            tasks.extend(prepared.tasks)
            report["sources"].append({"source_image_id": source.image_id, "source_image_path": str(path),
                                      "audit": prepared.audit,
                                      "panels": [{"image_id": task.image_id, "image_path": task.image_path,
                                                  **dict(task.panel_metadata)} for task in prepared.tasks]})
        failures = [task.panel_metadata.get("error") for task in tasks if task.panel_metadata.get("status") != "verified_single"]
        if failures:
            raise ValueError("子图检测/复核失败：" + "；".join(str(error) for error in failures))
        has_split = any(task.panel_metadata.get("panel_path") for task in tasks)
        report["split_required"] = has_split
        report["panel_count"] = len(tasks)
        if not has_split and not sample.get("_missing_previous_graph"):
            report["status"] = "preserved_unsplit"
            return {"report": report, "graph": None}
        graphs = []
        classifier = VLMImageClassifier()
        llm = LLMClient()
        for task in tasks:
            classification = classifier.classify(task, vlm)
            if classification.secondary_codes:
                raise ValueError(f"子图分类仍包含多个类型：{task.image_id}")
            task = replace(task, classification_code=classification.primary_code,
                           classification_type=classification.primary_type, panel_metadata={
                               **dict(task.panel_metadata), "classification_vlm_call_count": 1,
                           })
            leaf = {**sample, "sample_id": task.image_id, "content": {
                **sample["content"], "image_path": [task.image_path],
                "caption": f"当前独立子图：{task.panel_metadata.get('panel_label', '单图')}；"
                           f"类型：{classification.primary_type}。父图题仅供定位，禁止从其他子图推断：{task.caption}",
            }}
            response = call_model(leaf, "image", input_path, llm, vlm)
            graph = Graph(**to_graph(leaf, "image", response))
            graph.metadata.chunk_id = sample_id
            graph.metadata.extra.update({
                "status": "completed", "image_id": task.image_id, "image_path": task.image_path,
                "image_index": task.image_index, "extractor_kind": "llm_only",
                "classification_code": classification.primary_code,
                "classification_type": classification.primary_type, "vlm_call_count": 1, "model_called": True,
            })
            _attach_panel_provenance(graph, task)
            graphs.append(graph)
            print(f"子图抽取 {task.image_id}：实体={len(graph.entities)}，关系={len(graph.relations)}", flush=True)
        chunk = {"id": sample_id, "document_id": sample_id.split("::", 1)[0]}
        merged = merge_same_name_entities(_merge_task_graphs(chunk, graphs))
        _summarize_source_images(merged, graphs)
        merged.metadata.stage = "stage_04_image_llm_only"
        merged.metadata.extra.update({"extraction_method": "llm_only_with_panel_isolation",
                                      "panel_preprocessing_version": 1, "vlm_call_count": vlm.request_count})
        if merged.validate_references():
            raise ValueError("子图汇总后存在无效实体引用")
        report["status"] = "completed"
        return {"report": report, "graph": merged.to_dict(exclude={"metadata": {"raw_response"}})}
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return {"report": report, "graph": None}
    finally:
        report["model_request_count"] = vlm.request_count


def rerun_split_images(
    input_path: Path, output_path: Path, *, workers: int = 3, image_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """备份旧结果，并行检测样本，只替换需切分样本；按完成样本持续保存且可断点续跑。"""
    samples = read_samples(input_path, "image")
    previous = json.loads(output_path.read_text(encoding="utf-8-sig")) if output_path.is_file() else {}
    if previous and previous.get("input") != str(input_path.resolve()):
        raise ValueError("已有结果的输入路径不一致")
    graphs = {graph["metadata"]["chunk_id"]: graph for graph in previous.get("graphs", [])}
    sample_ids = [sample["sample_id"] for sample in samples]
    manifest_path = output_path.with_name(output_path.stem + ".panel_rerun.json")
    fingerprint = hashlib.sha256(input_path.read_bytes())
    for sample in samples:
        for value in sample["content"]["image_path"]:
            path = Path(value)
            path = path if path.is_absolute() else input_path.parent / path
            # 中文说明：缺失图片交给单样本报告，不能阻塞其他图片的检测与重跑。
            fingerprint.update(path.read_bytes() if path.is_file() else ("missing:" + str(path)).encode("utf-8"))
    signature = fingerprint.hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if manifest and manifest.get("input_sha256") != signature:
        raise ValueError("输入或原图已变化，请归档旧的 panel_rerun 清单后重新运行")
    if not manifest:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
        backup = output_path.with_name(f"{output_path.stem}.previous-{stamp}{output_path.suffix}")
        if output_path.is_file():
            shutil.copy2(output_path, backup)
        manifest = {"version": 1, "input_sha256": signature, "backup": str(backup.resolve()),
                    "base_model_request_count": previous.get("statistics", {}).get("model_request_count", 0), "samples": {}}
        write_atomic(manifest_path, manifest)
    saved_overrides = manifest.get("image_overrides", {})
    overrides = saved_overrides if image_overrides is None else image_overrides
    changed_paths = {path for path in set(saved_overrides) | set(overrides)
                     if saved_overrides.get(path) != overrides.get(path)}
    for sample in samples:
        if changed_paths.intersection(sample["content"]["image_path"]):
            old_report = manifest["samples"].pop(sample["sample_id"], None)
            if old_report:
                manifest["base_model_request_count"] += old_report["model_request_count"]
    manifest["image_overrides"] = overrides

    def checkpoint(status: str) -> dict[str, Any]:
        """把本轮状态与原始 Graph 结构一同保存，失败样本明确标记旧结果待替换。"""
        reports = manifest["samples"]
        failed = [sample_id for sample_id, record in reports.items() if record["status"] == "failed"]
        pending = [sample_id for sample_id in sample_ids if sample_id not in reports or reports[sample_id]["status"] == "failed"]
        records = [graphs[sample_id] for sample_id in sample_ids if sample_id in graphs]
        result = {**previous, "_status": status, "input": str(input_path.resolve()),
                  "pending_sample_ids": pending, "review_issue": {sample_id: reports[sample_id]["error"] for sample_id in failed} or None,
                  "panel_rerun": {"manifest": str(manifest_path.resolve()), "backup": manifest["backup"],
                                  "rerun_sample_ids": [key for key, value in reports.items() if value["status"] == "completed"],
                                  "preserved_sample_ids": [key for key, value in reports.items() if value["status"] == "preserved_unsplit"],
                                  "stale_previous_sample_ids": failed},
                  "statistics": {**previous.get("statistics", {}), "chunk_count": len(samples), "graph_count": len(records),
                                 "entity_count": sum(len(graph["entities"]) for graph in records),
                                 "relation_count": sum(len(graph["relations"]) for graph in records),
                                 "model_request_count": manifest["base_model_request_count"] + sum(value["model_request_count"] for value in reports.values()),
                                 "model_error_count": len(failed), "invalid_response_count": 0,
                                 "rerun_chunk_count": sum(value["status"] == "completed" for value in reports.values()),
                                 "preserved_chunk_count": sum(value["status"] == "preserved_unsplit" for value in reports.values())},
                  "graphs": records}
        write_atomic(manifest_path, manifest)
        write_atomic(output_path, result)
        return result

    pending = [sample for sample in samples if manifest["samples"].get(sample["sample_id"], {}).get("status") not in {"completed", "preserved_unsplit"}]
    checkpoint("running")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(extract_split_sample, {
                                       **sample, "_missing_previous_graph": sample["sample_id"] not in graphs,
                                       "_original_image_paths": sample["content"]["image_path"],
                                       "content": {**sample["content"], "image_path": [overrides.get(path, path) for path in sample["content"]["image_path"]]},
                                   },
                                   input_path, output_path.parent / "image_panels"): sample["sample_id"] for sample in pending}
        for future in as_completed(futures):
            sample_id = futures[future]
            result = future.result()
            report = result["report"]
            # 中文说明：失败重试的历史请求仍计入总数，避免续跑后请求统计回退。
            report["model_request_count"] += manifest["samples"].get(sample_id, {}).get("model_request_count", 0)
            manifest["samples"][sample_id] = report
            if result["graph"] is not None:
                graphs[sample_id] = result["graph"]
            checkpoint("running")
            print(f"样本 {sample_id}：{report['status']}，子图={report.get('panel_count', 0)}，错误={report.get('error', '无')}", flush=True)
    failed = any(value["status"] == "failed" for value in manifest["samples"].values())
    return checkpoint("partial_panel_error" if failed else "completed")


def main() -> int:
    """解析图像 Chunk 的输入输出路径并执行视觉模型抽取。"""
    parser = argparse.ArgumentParser(description="逐图像 Chunk 仅用提示词和项目视觉模型抽取")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "image_chunk.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "image_extraction.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-split-images", action="store_true", help="检测子图，仅重跑需要切分的样本并汇总回原样本")
    parser.add_argument("--workers", type=int, default=3, help="独立样本并发数")
    parser.add_argument("--image-overrides", type=Path, help="可追溯的旧图路径到现存原图路径映射 JSON")
    args = parser.parse_args()
    result = (
        rerun_split_images(args.input, args.output, workers=args.workers,
                          image_overrides=json.loads(args.image_overrides.read_text(encoding="utf-8"))["overrides"] if args.image_overrides else None)
        if args.rerun_split_images and not args.dry_run
        else run(args.input, args.output, "image", dry_run=args.dry_run)
    )
    print(f"状态={result['_status']}；{args.output.resolve()}")
    return 0 if result["_status"] in {"completed", "validated_no_extraction"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
