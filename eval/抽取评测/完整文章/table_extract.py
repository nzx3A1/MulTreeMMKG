"""对评测表格 Chunk 调用本地 OCR、表格解析与规则建图，不请求模型。"""
from __future__ import annotations

import argparse
import json
import site
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 中文说明：优先使用所选 Conda 环境依赖，避免用户级 PyTorch DLL 与环境内 OCR 库冲突。
USER_SITE = Path(site.getusersitepackages()).resolve()
sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != USER_SITE]

from src.extractors.table_extractor.table_extractor import extract_from_tables
from src.utils.llm_client import DisabledLLMClient

EVAL_DIR = Path(__file__).resolve().parent


def load_chunks(path: Path) -> list[dict]:
    """将表格样本转换为原有表格识别器可处理的 Chunk 格式。"""
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("表格样本文件必须是 JSON 数组")
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
            "modality": "table",
            "order": sample["order"],
            "section_id": section["section_id"],
            "section_title": section["section_title"],
            **sample["content"],
        })
    return chunks


def run(input_path: Path, output_path: Path, work_dir: Path, *, engine: str) -> dict:
    """运行本地表格抽取，并列出未生成 Graph 的样本供人工排查。"""
    chunks = load_chunks(input_path)
    graphs = extract_from_tables(
        chunks,
        DisabledLLMClient(),
        output_path=None,
        work_dir=work_dir,
        engine=engine,
        device="cpu",
        ocr_device="cpu",
        use_vlm_semantic=False,
        use_llm_semantic=False,
        show_progress=True,
    )
    completed = {str(graph.metadata.chunk_id) for graph in graphs}
    empty_graph_ids = [str(graph.metadata.chunk_id) for graph in graphs if not graph.entities and not graph.relations]
    result = {
        "_status": "completed_local" if len(completed) == len(chunks) else "partial_local",
        "input": str(input_path.resolve()),
        "work_dir": str(work_dir.resolve()),
        "statistics": {
            "chunk_count": len(chunks),
            "graph_count": len(graphs),
            "unprocessed_count": len(chunks) - len(completed),
            "empty_graph_count": len(empty_graph_ids),
            "entity_count": sum(len(graph.entities) for graph in graphs),
            "relation_count": sum(len(graph.relations) for graph in graphs),
            "model_call_count": 0,
        },
        "unprocessed_sample_ids": [chunk["id"] for chunk in chunks if chunk["id"] not in completed],
        "empty_graph_sample_ids": empty_graph_ids,
        "graphs": [graph.to_dict() for graph in graphs],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    """解析本地识别参数并保存表格抽取结果。"""
    parser = argparse.ArgumentParser(description="表格 Chunk 本地 OCR 与规则抽取")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "eval/dataset/dev/chunk/table_chunk.json")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "table_extraction.json")
    parser.add_argument("--work-dir", type=Path, default=EVAL_DIR / "table_artifacts")
    parser.add_argument("--engine", choices=("rapidtable", "mineru"), default="rapidtable", help="默认仅使用本地 RapidTable；mineru 会请求远程服务")
    args = parser.parse_args()
    result = run(args.input, args.output, args.work_dir, engine=args.engine)
    print(f"表格结果：{result['statistics']['graph_count']}/{result['statistics']['chunk_count']} 个 Graph，{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
