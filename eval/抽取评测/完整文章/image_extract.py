"""对评测图片 Chunk 调用现有图片任务展开和路由算法，不触发视觉模型。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.image_extractor.pipeline import build_image_tasks
from src.extractors.image_extractor.pipeline import extract_from_images
from src.extractors.image_extractor.classification import VLMImageClassificationProvider
from src.extractors.image_extractor.router import ImageExtractorRouter
from src.utils.llm_client import LLMClient
from src.utils.vlm_client import VLMClient

EVAL_DIR = Path(__file__).resolve().parent


def load_chunks(path: Path) -> list[dict]:
    """把图片评测样本适配为现有 ImageChunk 抽取任务格式。"""
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("图片样本文件必须是 JSON 数组")
    chunks = []
    seen: set[str] = set()
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if "::" not in sample_id or sample_id in seen:
            raise ValueError(f"样本 ID 无效或重复：{sample_id}")
        seen.add(sample_id)
        document_id, _ = sample_id.split("::", 1)
        section = sample["section"]
        chunks.append({
            "id": sample_id,
            "document_id": document_id,
            "modality": "image",
            "order": sample["order"],
            "section_id": section["section_id"],
            "section_title": section["section_title"],
            **sample["content"],
        })
    return chunks


def run(input_path: Path, output_path: Path, *, execute: bool = False) -> dict:
    """默认仅展开和路由图片；显式 execute 时才调用视觉抽取算法。"""
    chunks = load_chunks(input_path)
    if execute:
        vlm_client = VLMClient()
        graphs = extract_from_images(
            chunks,
            LLMClient(),
            vlm_client,
            classification_provider=VLMImageClassificationProvider(vlm_client),
            show_progress=True,
        )
        result = {
            "_status": "vlm_attempted",
            "input": str(input_path.resolve()),
            "statistics": {
                "chunk_count": len(chunks),
                "graph_count": len(graphs),
                "entity_count": sum(len(graph.entities) for graph in graphs),
                "relation_count": sum(len(graph.relations) for graph in graphs),
            },
            "graphs": [graph.to_dict() for graph in graphs],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    router = ImageExtractorRouter()
    results = []
    for chunk in chunks:
        tasks = build_image_tasks(chunk)
        images = []
        for task in tasks:
            exists = Path(task.image_path).is_file()
            kind = router.route(task).value if task.classification_code else None
            images.append({
                "image_id": task.image_id,
                "image_path": task.image_path,
                "image_exists": exists,
                "classification_code": task.classification_code or None,
                "extractor_kind": kind,
                "status": "missing_image" if not exists else "pending_vlm_classification" if not kind else "pending_vlm_extraction",
            })
        results.append({
            "sample_id": chunk["id"],
            "document_id": chunk["document_id"],
            "section_id": chunk["section_id"],
            "caption": chunk.get("caption"),
            "images": images,
        })
    image_count = sum(len(item["images"]) for item in results)
    missing_count = sum(not image["image_exists"] for item in results for image in item["images"])
    result = {
        "_status": "prepared_no_vlm",
        "input": str(input_path.resolve()),
        "statistics": {"chunk_count": len(chunks), "image_count": image_count, "missing_image_count": missing_count, "model_call_count": 0, "extracted_entity_count": None},
        "chunks": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    """解析路径参数并保存图片任务与抽取器路由检查结果。"""
    parser = argparse.ArgumentParser(description="图片 Chunk 无 VLM 任务展开与路由检查")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "eval/dataset/dev/chunk/image_chunk.json")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "image_extraction.json")
    parser.add_argument("--execute", action="store_true", help="显式启用真实 LLM/VLM 分类与抽取；默认不调用模型")
    args = parser.parse_args()
    result = run(args.input, args.output, execute=args.execute)
    print(f"图片结果：{result['statistics']['chunk_count']} 个 Chunk，状态={result['_status']}，{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
