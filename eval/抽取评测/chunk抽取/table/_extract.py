"""调用项目表格抽取器处理表格 Chunk，以原图 VLM 辅助判断主键并保存结果。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CHUNK_EVAL_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CHUNK_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(CHUNK_EVAL_DIR))

from table_extract import load_chunks, prepare_sources
from src.extractors.table_extractor.table_extractor import extract_from_tables


def run_extraction(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    *,
    minimum_score: float = 0.55,
    dry_run: bool = False,
    ocr_backend: str = "onnxruntime",
    use_vlm_semantic: bool = True,
) -> dict:
    """运行 OCR、可选的视觉语义判断和建图，并将结果及实际调用情况写入 JSON。"""
    chunks = load_chunks(input_path)
    sources = prepare_sources(chunks)
    graphs = []
    if not dry_run:
        # 中文说明：OCR 和建图仍在本地执行，VLM 仅辅助判断记录方向、表头与主键。
        graphs = extract_from_tables(
            sources,
            output_path=None,
            work_dir=work_dir,
            engine="rapidtable",
            device="cpu",
            ocr_device="cpu",
            ocr_backend=ocr_backend,
            use_vlm_semantic=use_vlm_semantic,
            use_llm_semantic=False,
            minimum_score=minimum_score,
            show_progress=True,
        )

    completed_ids = {str(graph.metadata.chunk_id) for graph in graphs}
    empty_ids = [
        str(graph.metadata.chunk_id)
        for graph in graphs
        if not graph.entities and not graph.relations
    ]
    visual_results = [graph.metadata.extra.get("visual_semantics") for graph in graphs]
    vlm_attempt_count = sum(isinstance(item, dict) for item in visual_results)
    vlm_success_count = sum(isinstance(item, dict) and item.get("status") == "success" for item in visual_results)
    vlm_accepted_count = sum(
        graph.metadata.extra.get("semantic_plan", {}).get("analysis_source") == "vlm"
        for graph in graphs
    )
    generated_key_graph_count = sum(
        bool(graph.metadata.extra.get("semantic_plan", {}).get("generated_key"))
        for graph in graphs
    )
    generated_key_entity_count = sum(
        int(graph.metadata.extra.get("generated_key_entity_count", 0))
        for graph in graphs
    )
    result = {
        "_status": "validated_no_extraction" if dry_run else "completed_local" if len(completed_ids) == len(chunks) else "partial_local",
        "input": str(input_path.resolve()),
        "work_dir": str(work_dir.resolve()),
        "statistics": {
            "chunk_count": len(chunks),
            "table_task_count": len(sources),
            "graph_count": len(graphs),
            "unprocessed_count": len(chunks) - len(completed_ids) if not dry_run else len(chunks),
            "empty_graph_count": len(empty_ids),
            "entity_count": None if dry_run else sum(len(graph.entities) for graph in graphs),
            "relation_count": None if dry_run else sum(len(graph.relations) for graph in graphs),
            "llm_vlm_call_count": vlm_attempt_count,
            "vlm_success_count": vlm_success_count,
            "vlm_failed_count": vlm_attempt_count - vlm_success_count,
            "vlm_accepted_count": vlm_accepted_count,
            "vlm_rejected_count": vlm_success_count - vlm_accepted_count,
            "generated_key_graph_count": generated_key_graph_count,
            "generated_key_entity_count": generated_key_entity_count,
        },
        "unprocessed_sample_ids": [chunk["id"] for chunk in chunks if chunk["id"] not in completed_ids],
        "empty_graph_sample_ids": empty_ids,
        "ocr_backend": ocr_backend,
        "note": (
            "仅校验输入，未执行识别或抽取。" if dry_run else
            "使用本地 RapidTable/OCR 建图，VLM 辅助判断原文主键；无可靠主键时按记录行列生成稳定键。" if use_vlm_semantic else
            "使用本地 RapidTable/OCR 与规则建图；无可靠主键时按记录行列生成稳定键，未调用 LLM/VLM 服务。"
        ),
        "graphs": [graph.to_dict() for graph in graphs],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    """解析输入输出及视觉语义开关并执行表格 Chunk 抽取。"""
    parser = argparse.ArgumentParser(description="调用项目表格抽取算法处理 dev 表格 Chunk")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "eval/dataset/dev/chunk/table_chunk.json",
        help="表格 Chunk JSON 文件",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "table_extraction.json",
        help="抽取结果 JSON 文件",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=OUTPUT_DIR / "table_artifacts",
        help="RapidTable 中间产物目录",
    )
    parser.add_argument("--minimum-score", type=float, default=0.55, help="表格质量门最低分")
    parser.add_argument("--ocr-backend", choices=("onnxruntime", "torch"), default="onnxruntime", help="本地 OCR 执行后端")
    parser.add_argument("--no-vlm-semantic", action="store_true", help="关闭原图 VLM 主键判断，仅使用 HTML 规则")
    parser.add_argument("--dry-run", action="store_true", help="仅校验输入和图片，不运行识别")
    args = parser.parse_args()

    # 中文说明：复用评测入口中与 src/extractors/table_extractor 对接的本地识别和规则建图流程。
    result = run_extraction(
        args.input,
        args.output,
        args.work_dir,
        dry_run=args.dry_run,
        minimum_score=args.minimum_score,
        ocr_backend=args.ocr_backend,
        use_vlm_semantic=not args.no_vlm_semantic,
    )
    stats = result["statistics"]
    print(
        f"表格抽取结束：状态={result['_status']}，"
        f"Chunk={stats['graph_count']}/{stats['chunk_count']}，"
        f"表格任务={stats['table_task_count']}，"
        f"实体={stats['entity_count']}，关系={stats['relation_count']}，"
        f"结果={args.output.resolve()}"
    )
    return 0 if result["_status"] in {"completed_local", "validated_no_extraction"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
