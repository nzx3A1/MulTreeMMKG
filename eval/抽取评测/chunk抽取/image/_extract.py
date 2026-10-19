"""从图片 Chunk 评测样本调用项目图片分类与知识抽取流水线。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
CHUNK_EXTRACT_DIR = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CHUNK_EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CHUNK_EXTRACT_DIR))

from image_extract import inspect_images, load_chunks
from src.extractors.image_extractor.classification import VLMImageClassificationProvider
from src.extractors.image_extractor.pipeline import extract_from_images
from src.utils.llm_client import LLMClient
from src.utils.vlm_client import VLMClient


def _load_checkpoint(output_path: Path) -> tuple[dict[str, dict], list[dict], list[dict]]:
    """读取逐 Chunk 检查点，返回已完成图谱、分类记录和历史错误。"""
    if not output_path.is_file():
        return {}, [], []
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"已有检查点无法读取：{output_path}：{exc}") from exc
    graphs = payload.get("graphs") if isinstance(payload, dict) else None
    if not isinstance(graphs, list):
        raise ValueError(f"已有检查点缺少 graphs 数组：{output_path}")
    completed = {
        str(graph["metadata"]["chunk_id"]): graph
        for graph in graphs
        if isinstance(graph, dict)
        and isinstance(graph.get("metadata"), dict)
        and graph["metadata"].get("chunk_id")
    }
    classifications = payload.get("classification_records", [])
    failures = payload.get("failures", [])
    return (
        completed,
        classifications if isinstance(classifications, list) else [],
        failures if isinstance(failures, list) else [],
    )


def _write_checkpoint(
    output_path: Path,
    *,
    input_path: Path,
    chunks: list[dict],
    completed: dict[str, dict],
    classification_records: list[dict],
    failures: list[dict],
    status: str,
) -> dict:
    """原子写入当前批次统计和图谱，确保中断后可按 Chunk 续跑。"""
    image_count = sum(len(chunk["image_path"]) for chunk in chunks)
    entity_count = sum(len(graph.get("entities", [])) for graph in completed.values())
    relation_count = sum(len(graph.get("relations", [])) for graph in completed.values())
    result = {
        "_status": status,
        "input": str(input_path.resolve()),
        "statistics": {
            "chunk_count": len(chunks),
            "image_count": image_count,
            "graph_count": len(completed),
            "pending_chunk_count": max(0, len(chunks) - len(completed)),
            "entity_count": entity_count,
            "relation_count": relation_count,
            "classification_failure_count": sum(item.get("status") == "failed" for item in classification_records),
            "failed_chunk_count": len(failures),
        },
        "pending_sample_ids": [chunk["id"] for chunk in chunks if chunk["id"] not in completed],
        "chunk_statuses": {
            chunk_id: graph.get("metadata", {}).get("extra", {}).get("status", "unknown")
            for chunk_id, graph in completed.items()
        },
        "classification_records": classification_records,
        "failures": failures,
        "graphs": list(completed.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return result


def run(input_path: Path, output_path: Path, *, dry_run: bool = False) -> dict:
    """逐 Chunk 调用图片分类和抽取器，并在每个 Chunk 后写入检查点。"""
    chunks = load_chunks(input_path)
    inspections = inspect_images(chunks)
    missing = [
        image["path"]
        for item in inspections
        for image in item["image_paths"]
        if not image["exists"]
    ]
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 张评测图片，首个路径：{missing[0]}")

    completed, classification_records, failures = _load_checkpoint(output_path)
    if dry_run:
        return _write_checkpoint(
            output_path,
            input_path=input_path,
            chunks=chunks,
            completed={},
            classification_records=[],
            failures=[],
            status="validated_no_extraction",
        )

    vlm_client = VLMClient()
    llm_client = LLMClient()
    provider = VLMImageClassificationProvider(vlm_client)
    provider.records.extend(classification_records)
    _write_checkpoint(
        output_path,
        input_path=input_path,
        chunks=chunks,
        completed=completed,
        classification_records=classification_records,
        failures=failures,
        status="running",
    )
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = str(chunk["id"])
        if chunk_id in completed:
            print(f"跳过已完成 Chunk：{chunk_id}", flush=True)
            continue
        failures = [item for item in failures if item.get("sample_id") != chunk_id]
        print(f"开始图片抽取 {index}/{len(chunks)}：{chunk_id}", flush=True)
        try:
            graphs = extract_from_images(
                [chunk],
                llm_client,
                vlm_client,
                classification_provider=provider,
                show_progress=True,
            )
            for graph in graphs:
                completed[str(graph.metadata.chunk_id)] = graph.to_dict(
                    exclude={"metadata": {"raw_response"}}
                )
        except Exception as exc:
            failures.append({"sample_id": chunk_id, "error": f"{type(exc).__name__}: {exc}"})
        classification_records = list(provider.records)
        status = "running" if index < len(chunks) else "completed"
        result = _write_checkpoint(
            output_path,
            input_path=input_path,
            chunks=chunks,
            completed=completed,
            classification_records=classification_records,
            failures=failures,
            status=status,
        )
        stats = result["statistics"]
        print(
            f"已保存检查点：Graph={stats['graph_count']}/{stats['chunk_count']}，"
            f"失败={stats['failed_chunk_count']}，输出={output_path.resolve()}",
            flush=True,
        )
    final_status = "completed" if len(completed) == len(chunks) and not failures else "partial"
    return _write_checkpoint(
        output_path,
        input_path=input_path,
        chunks=chunks,
        completed=completed,
        classification_records=classification_records,
        failures=failures,
        status=final_status,
    )


def main() -> int:
    """读取图片 Chunk 参数并启动可续跑的实际抽取任务。"""
    parser = argparse.ArgumentParser(description="调用项目图片抽取器处理评测 ImageChunk")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "eval/dataset/dev/chunk/image_chunk.json",
        help="图片 Chunk 评测样本 JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "image_extraction.json",
        help="图片抽取结果 JSON",
    )
    parser.add_argument("--classifications", type=Path, help="已有逐图片分类 JSON；省略时调用 VLM 分类")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入图片，不调用模型")
    args = parser.parse_args()

    # 中文说明：图片抽取器的模型调用和结果结构复用 chunk_extract/image_extract.py。
    if args.classifications:
        parser.error("检查点版本会实时保存本轮 VLM 分类结果，不接受预先分类文件")
    result = run(args.input, args.output, dry_run=args.dry_run)
    stats = result["statistics"]
    print(
        f"图片抽取状态={result['_status']}，Chunk={stats['chunk_count']}，"
        f"Graph={stats['graph_count']}，实体={stats['entity_count']}，关系={stats['relation_count']}，"
        f"输出={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
