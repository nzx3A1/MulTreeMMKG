"""读取 dev 表格 Chunk，调用现有本地识别、质量门和规则建图流程。"""
from __future__ import annotations

import argparse
import json
import site
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 中文说明：排除用户级依赖路径，避免与所选 Conda 环境中的 OCR 库发生冲突。
USER_SITE = Path(site.getusersitepackages()).resolve()
sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != USER_SITE]

from src.extractors.table_extractor.table_extractor import extract_from_tables
from src.extractors.table_extractor.table_parse import adapt_table_chunk

EVAL_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def resolve_source_path(value: str, input_path: Path) -> str:
    """将表格图片的绝对或相对路径统一为绝对路径。"""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = input_path.parent / candidate
    return str(candidate.resolve())


def load_chunks(path: Path) -> list[dict]:
    """校验表格样本并转换成项目表格抽取器的 Chunk 格式。"""
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("表格样本文件必须是 JSON 数组")
    chunks: list[dict] = []
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("表格样本必须是 JSON 对象")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or "::" not in sample_id or sample_id in seen:
            raise ValueError(f"样本 ID 无效或重复：{sample_id}")
        seen.add(sample_id)
        section = sample.get("section")
        content = sample.get("content")
        if not isinstance(section, dict) or not isinstance(content, dict):
            raise ValueError(f"表格样本结构无效：{sample_id}")
        document_id, _ = sample_id.split("::", 1)
        normalized = dict(content)
        paths = normalized.get("table_path") or []
        if not isinstance(paths, list) or any(not isinstance(item, str) or not item.strip() for item in paths):
            raise ValueError(f"table_path 必须是有效路径列表：{sample_id}")
        normalized["table_path"] = [resolve_source_path(item, path) for item in paths]
        markdown = normalized.get("markdown")
        if isinstance(markdown, str) and "\n" not in markdown and Path(markdown).suffix.lower() in IMAGE_SUFFIXES:
            normalized["markdown"] = resolve_source_path(markdown, path)
        chunks.append({
            "id": sample_id,
            "document_id": document_id,
            "modality": "table",
            "order": sample["order"],
            "section_id": section["section_id"],
            "section_title": section["section_title"],
            **normalized,
        })
    return chunks


def prepare_sources(chunks: list[dict]) -> list:
    """展开逐表任务，并在抽取前确认图片文件和任务都有效。"""
    sources = []
    for chunk in chunks:
        markdown = chunk.get("markdown")
        if isinstance(markdown, str) and "\n" not in markdown and Path(markdown).suffix.lower() in IMAGE_SUFFIXES:
            if not Path(markdown).is_file():
                raise FileNotFoundError(f"表格图片不存在：{chunk['id']}，{markdown}")
        for image_path in chunk.get("table_path") or []:
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"表格图片不存在：{chunk['id']}，{image_path}")
        tasks = adapt_table_chunk(chunk)
        if not tasks:
            raise ValueError(f"表格样本未生成识别任务：{chunk['id']}")
        sources.extend(tasks)
    return sources


def run(input_path: Path, output_path: Path, work_dir: Path, *, dry_run: bool = False, minimum_score: float = 0.55) -> dict:
    """调用项目表格抽取算法，并保留质量门未通过的样本 ID。"""
    chunks = load_chunks(input_path)
    sources = prepare_sources(chunks)
    graphs = []
    if not dry_run:
        # 中文说明：只调用 src/extractors 的本地表格入口；不启用 LLM/VLM 语义补充。
        graphs = extract_from_tables(
            sources,
            output_path=None,
            work_dir=work_dir,
            engine="rapidtable",
            device="cpu",
            ocr_device="cpu",
            use_vlm_semantic=False,
            use_llm_semantic=False,
            minimum_score=minimum_score,
            show_progress=True,
        )
    completed = {str(graph.metadata.chunk_id) for graph in graphs}
    empty_graph_ids = [str(graph.metadata.chunk_id) for graph in graphs if not graph.entities and not graph.relations]
    result = {
        "_status": "validated_no_extraction" if dry_run else "completed_local" if len(completed) == len(chunks) else "partial_local",
        "input": str(input_path.resolve()),
        "work_dir": str(work_dir.resolve()),
        "statistics": {
            "chunk_count": len(chunks),
            "table_task_count": len(sources),
            "graph_count": len(graphs),
            "unprocessed_count": len(chunks) - len(completed) if not dry_run else len(chunks),
            "empty_graph_count": len(empty_graph_ids),
            "entity_count": None if dry_run else sum(len(graph.entities) for graph in graphs),
            "relation_count": None if dry_run else sum(len(graph.relations) for graph in graphs),
            "llm_vlm_call_count": 0,
        },
        "unprocessed_sample_ids": [chunk["id"] for chunk in chunks if chunk["id"] not in completed],
        "empty_graph_sample_ids": empty_graph_ids,
        "note": "本地 RapidTable/OCR 会加载本地识别模型；未调用 LLM/VLM 服务。" if not dry_run else "仅校验输入，未执行识别或抽取。",
        "graphs": [graph.to_dict() for graph in graphs],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    """解析参数并运行表格抽取或输入检查。"""
    parser = argparse.ArgumentParser(description="用项目本地表格抽取器处理 dev 表格 Chunk")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "eval/dataset/dev/chunk/table_chunk.json")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "table_extraction.json")
    parser.add_argument("--work-dir", type=Path, default=EVAL_DIR / "table_artifacts")
    parser.add_argument("--minimum-score", type=float, default=0.55, help="表格质量门最低分")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入，不运行本地识别或抽取")
    args = parser.parse_args()
    result = run(args.input, args.output, args.work_dir, dry_run=args.dry_run, minimum_score=args.minimum_score)
    print(f"表格结果：{result['statistics']['graph_count']}/{result['statistics']['chunk_count']} 个 Graph，状态={result['_status']}，{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
