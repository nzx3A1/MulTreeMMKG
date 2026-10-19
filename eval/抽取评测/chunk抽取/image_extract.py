"""读取 dev 图像 Chunk，调用现有分类、路由和图像抽取算法。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.image_extractor.classification import (
    RecordImageClassificationProvider,
    VLMImageClassificationProvider,
)
from src.extractors.image_extractor.pipeline import extract_from_images
from src.utils.llm_client import LLMClient
from src.utils.vlm_client import VLMClient

EVAL_DIR = Path(__file__).resolve().parent


def resolve_image_path(value: str, input_path: Path) -> str:
    """将样本中的绝对或相对图片路径统一为绝对路径。"""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = input_path.parent / candidate
    return str(candidate.resolve())


def load_chunks(path: Path) -> list[dict]:
    """校验图像样本并转换成项目图片抽取器所需的 Chunk。"""
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("图像样本文件必须是 JSON 数组")
    chunks: list[dict] = []
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("图像样本必须是 JSON 对象")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or "::" not in sample_id or sample_id in seen:
            raise ValueError(f"样本 ID 无效或重复：{sample_id}")
        seen.add(sample_id)
        section = sample.get("section")
        content = sample.get("content")
        if not isinstance(section, dict) or not isinstance(content, dict):
            raise ValueError(f"图像样本结构无效：{sample_id}")
        paths = content.get("image_path")
        if not isinstance(paths, list) or not paths or any(not isinstance(item, str) or not item.strip() for item in paths):
            raise ValueError(f"图像样本缺少有效 image_path：{sample_id}")
        document_id, _ = sample_id.split("::", 1)
        chunks.append({
            "id": sample_id,
            "document_id": document_id,
            "modality": "image",
            "order": sample["order"],
            "section_id": section["section_id"],
            "section_title": section["section_title"],
            **content,
            "image_path": [resolve_image_path(item, path) for item in paths],
        })
    return chunks


def inspect_images(chunks: list[dict]) -> list[dict]:
    """列出每个样本的图片路径及文件存在状态。"""
    return [
        {
            "sample_id": chunk["id"],
            "image_paths": [
                {"path": image_path, "exists": Path(image_path).is_file()}
                for image_path in chunk["image_path"]
            ],
        }
        for chunk in chunks
    ]


def run(
    input_path: Path,
    output_path: Path,
    *,
    dry_run: bool = False,
    classifications_path: Path | None = None,
) -> dict:
    """调用图像抽取器；检查模式只验证文件，不生成虚假的图谱。"""
    chunks = load_chunks(input_path)
    inspections = inspect_images(chunks)
    missing = [
        image["path"]
        for item in inspections
        for image in item["image_paths"]
        if not image["exists"]
    ]
    if missing and not dry_run:
        raise FileNotFoundError(f"缺少 {len(missing)} 张评测图片，首个路径：{missing[0]}")
    graphs = []
    classification_records: list[dict] = []
    if not dry_run:
        # 中文说明：脚本只调用抽取管线，分类和语义抽取中的请求均由 src/extractors 发起。
        vlm_client = VLMClient()
        provider = (
            RecordImageClassificationProvider.from_json(classifications_path)
            if classifications_path
            else VLMImageClassificationProvider(vlm_client)
        )
        graphs = extract_from_images(
            chunks,
            LLMClient(),
            vlm_client,
            classification_provider=provider,
            show_progress=True,
        )
        classification_records = list(getattr(provider, "records", []))
    records = [graph.to_dict(exclude={"metadata": {"raw_response"}}) for graph in graphs]
    statuses = {
        record["metadata"]["chunk_id"]: record["metadata"].get("extra", {}).get("status", "unknown")
        for record in records
    }
    result = {
        "_status": "validated_no_extraction" if dry_run else "extraction_attempted",
        "input": str(input_path.resolve()),
        "classifications_input": str(classifications_path.resolve()) if classifications_path else None,
        "statistics": {
            "chunk_count": len(chunks),
            "image_count": sum(len(chunk["image_path"]) for chunk in chunks),
            "missing_image_count": len(missing),
            "graph_count": len(records),
            "entity_count": None if dry_run else sum(len(record["entities"]) for record in records),
            "relation_count": None if dry_run else sum(len(record["relations"]) for record in records),
            "classification_failure_count": sum(item.get("status") == "failed" for item in classification_records),
        },
        "pending_sample_ids": [chunk["id"] for chunk in chunks] if dry_run else [],
        "chunk_statuses": statuses,
        "image_checks": inspections if dry_run else [],
        "classification_records": classification_records,
        "graphs": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    """解析参数并运行图像抽取或输入检查。"""
    parser = argparse.ArgumentParser(description="用项目图像抽取器处理 dev 图像 Chunk")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "eval/dataset/dev/chunk/image_chunk.json")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "image_extraction.json")
    parser.add_argument("--classifications", type=Path, help="已有逐图片分类文件；省略时由项目分类器分类")
    parser.add_argument("--dry-run", action="store_true", help="只检查图片文件，不执行分类或抽取")
    args = parser.parse_args()
    result = run(args.input, args.output, dry_run=args.dry_run, classifications_path=args.classifications)
    print(f"图像结果：{result['statistics']['graph_count']}/{result['statistics']['chunk_count']} 个 Graph，状态={result['_status']}，{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
