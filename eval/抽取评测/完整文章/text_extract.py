"""对评测文本 Chunk 运行现有抽取器的无模型流程，并保存可追溯结果。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.text_extractor.text_extractor import extract_from_text
from src.utils.llm_client import DisabledLLMClient, LLMClient

EVAL_DIR = Path(__file__).resolve().parent


def load_chunks(path: Path) -> dict[str, list[dict]]:
    """校验文本样本并转换成抽取器需要的文档及 Chunk 字段。"""
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("文本样本文件必须是 JSON 数组")
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if "::" not in sample_id or sample_id in seen:
            raise ValueError(f"样本 ID 无效或重复：{sample_id}")
        seen.add(sample_id)
        document_id, _ = sample_id.split("::", 1)
        section = sample["section"]
        grouped[document_id].append({
            "id": sample_id,
            "document_id": document_id,
            "modality": "text",
            "order": sample["order"],
            "section_id": section["section_id"],
            "section_title": section["section_title"],
            "text": sample["content"]["text"],
        })
    return grouped


def run(input_path: Path, output_path: Path, *, execute: bool = False) -> dict:
    """默认禁用模型检查流程；显式 execute 时才执行真实文本抽取。"""
    grouped = load_chunks(input_path)
    records = []
    client = LLMClient() if execute else DisabledLLMClient()
    for document_id, chunks in grouped.items():
        graphs = extract_from_text(chunks, client, max_workers=1, show_progress=False)
        for graph in graphs:
            if not execute:
                graph.metadata.extra["status"] = "pending_llm"
                graph.metadata.extra["model_called"] = False
                graph.metadata.extra["note"] = "禁用模型的流程检查；空实体和空关系不代表真实抽取结果"
            records.append(graph.to_dict())
        print(f"{document_id}: 已检查 {len(graphs)} 个文本 Chunk")
    result = {
        "_status": "llm_attempted" if execute else "dry_run_no_llm",
        "input": str(input_path.resolve()),
        "statistics": {
            "chunk_count": len(records),
            "model_call_count": None if execute else 0,
            "extracted_entity_count": sum(len(graph["entities"]) for graph in records) if execute else None,
        },
        "graphs": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    """解析路径参数并运行文本 Chunk 无模型评测。"""
    parser = argparse.ArgumentParser(description="文本 Chunk 抽取器无 LLM 流程检查")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "eval/dataset/dev/chunk/text_chunk.json")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "text_extraction.json")
    parser.add_argument("--execute", action="store_true", help="显式启用真实 LLM 抽取；默认不调用模型")
    args = parser.parse_args()
    result = run(args.input, args.output, execute=args.execute)
    print(f"文本结果：{result['statistics']['chunk_count']} 个 Chunk，状态={result['_status']}，{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
