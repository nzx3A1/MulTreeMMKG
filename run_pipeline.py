"""石油地质多模态知识图谱构建流水线 —— 主入口脚本。

按阶段依次执行：
    01 MinerU 解析  →  02 章节树  →  03 文档骨架图  →  04 摘要
    → 05 多模态 chunk  →  06~09 多模态抽取
    → 10 对齐去重  →  11 关系发现  →  12 增强图  →  final_graph

使用方式：
    python run_pipeline.py --input data/raw_pdf
    python run_pipeline.py --input data/raw_pdf/xxx.pdf --paper-id paper_001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config.app_config import settings
from src.parser.markdown_parser import parse_markdown
from src.parser.mineru_loader import load_paper
from src.parser.section_splitter import split_sections
from src.skeleton.chunk_builder import build_modal_chunks
from src.skeleton.document_skeleton_builder import build_skeleton
from src.summarizer.bottom_up_summarizer import summarize_bottom_up
from src.utils.json_io import write_json


def _load_mineru_index() -> list[dict[str, Any]]:
    """读取 MinerU 输出索引，供 CLI 根据 PDF 文件名定位 paper_id。"""

    index_path = settings.mineru_output_dir / "index.json"
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text(encoding="utf-8"))


def _available_paper_ids() -> list[str]:
    """列出已经具备 MinerU 输出的 paper_id。"""

    paper_ids = []
    for path in sorted(settings.mineru_output_dir.glob("paper_*")):
        if path.is_dir() and (path / "full.md").exists():
            paper_ids.append(path.name)
    return paper_ids


def _resolve_paper_ids(input_path: Path, explicit_paper_id: str | None) -> list[str]:
    """根据命令行输入解析需要处理的论文 ID。"""

    if explicit_paper_id:
        return [explicit_paper_id]
    if input_path.is_dir():
        index = _load_mineru_index()
        if index:
            names = {path.name for path in input_path.glob("*.pdf")}
            ids = [row["paper_id"] for row in index if row.get("source_pdf") in names]
            if ids:
                return ids
        return _available_paper_ids()
    if input_path.is_file():
        for row in _load_mineru_index():
            if row.get("source_pdf") == input_path.name:
                return [row["paper_id"]]
        candidate = input_path.stem
        if (settings.mineru_output_dir / candidate / "full.md").exists():
            return [candidate]
    raise FileNotFoundError(f"无法根据输入定位 MinerU 输出: {input_path}")


def _run_stage_1_to_5(paper_id: str) -> dict[str, Any]:
    """运行单篇论文的阶段 1 到阶段 5，并返回所有中间结果。"""

    paper = load_paper(paper_id)
    paper["markdown_blocks"] = parse_markdown(paper.get("raw_markdown") or "")
    section_tree = split_sections(paper)
    skeleton = build_skeleton(section_tree)
    chunks = build_modal_chunks(skeleton, paper)
    summary = summarize_bottom_up(chunks)
    return {
        "paper": paper,
        "section_tree": section_tree,
        "skeleton": skeleton,
        "summary": summary,
        "chunks": chunks,
    }


def _write_stage_outputs(results: list[dict[str, Any]]) -> None:
    """把阶段 1 到 5 的聚合结果写入 output 目录。"""

    papers = [row["paper"] for row in results]
    section_trees = [row["section_tree"] for row in results]
    skeletons = [row["skeleton"] for row in results]
    summaries = [row["summary"] for row in results]
    chunks = [chunk for row in results for chunk in row["chunks"]]

    write_json(settings.output_dir / "stage_01_mineru_parse.json", {
        "_stage": 1,
        "_description": "MinerU 解析结果归一化输出。",
        "_produced_by": "run_pipeline.py::_write_stage_outputs",
        "paper_ids": [paper["paper_id"] for paper in papers],
        "paper_id": papers[0]["paper_id"] if len(papers) == 1 else None,
        "papers": papers,
        "blocks": [block for paper in papers for block in paper["blocks"]],
    })
    write_json(settings.output_dir / "stage_02_section_tree.json", {
        "_stage": 2,
        "_description": "文档章节树（章 → 节 → 小节）。",
        "_produced_by": "run_pipeline.py::_write_stage_outputs",
        "paper_ids": [tree["paper_id"] for tree in section_trees],
        "paper_id": section_trees[0]["paper_id"] if len(section_trees) == 1 else None,
        "documents": section_trees,
        "sections": [section for tree in section_trees for section in tree["flat_sections"]],
    })
    write_json(settings.output_dir / "stage_03_document_skeleton_graph.json", {
        "_stage": 3,
        "_description": "文档骨架图：Paper / Section / Subsection / Chunk / Figure / Table / Formula 节点及结构关系。",
        "_produced_by": "run_pipeline.py::_write_stage_outputs",
        "paper_ids": [skeleton["paper_id"] for skeleton in skeletons],
        "documents": skeletons,
        "nodes": [node for skeleton in skeletons for node in skeleton["nodes"]],
        "edges": [edge for skeleton in skeletons for edge in skeleton["edges"]],
    })
    write_json(settings.output_dir / "stage_04_section_summary.json", {
        "_stage": 4,
        "_description": "自底向上生成的各级章节摘要。",
        "_produced_by": "run_pipeline.py::_write_stage_outputs",
        "documents": summaries,
        "summaries": [item for summary in summaries for item in summary["summaries"]],
        "document_summaries": [item for summary in summaries for item in summary["document_summaries"]],
    })
    write_json(settings.output_dir / "stage_05_modal_chunks.json", {
        "_stage": 5,
        "_description": "多模态 chunk 列表（文本 + 表格 + 图像 + 公式）。",
        "_produced_by": "run_pipeline.py::_write_stage_outputs",
        "paper_ids": [paper["paper_id"] for paper in papers],
        "chunks": chunks,
        "statistics": {
            "document_count": len(papers),
            "chunk_count": len(chunks),
        },
    })


def main() -> None:
    """命令行入口：解析参数后串行调用各阶段。"""

    parser = argparse.ArgumentParser(description="石油地质多模态知识图谱构建流水线")
    parser.add_argument("--input", required=True, help="原始 PDF 文件或 PDF 目录；需已存在对应 MinerU 输出")
    parser.add_argument("--paper-id", default=None, help="显式指定 data/mineru_output 下的 paper_id")
    parser.add_argument("--from-stage", type=int, default=1, help="起始阶段，目前阶段二支持 1")
    parser.add_argument("--to-stage", type=int, default=5, help="结束阶段，目前阶段二支持到 5")
    args = parser.parse_args()

    if args.from_stage != 1 or args.to_stage > 5:
        raise NotImplementedError("当前已实现阶段 1-5；阶段 6 之后将在后续阶段补齐。")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = settings.project_root / input_path
    paper_ids = _resolve_paper_ids(input_path, args.paper_id)
    results = [_run_stage_1_to_5(paper_id) for paper_id in paper_ids]
    _write_stage_outputs(results)
    print(f"完成阶段 1-5：{len(results)} 篇论文，输出目录 {settings.output_dir}")


if __name__ == "__main__":
    main()
