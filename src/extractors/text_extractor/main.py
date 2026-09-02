"""第三阶段章节总结 JSON 到文本知识图谱的独立执行入口。

本模块只负责文件读取、章节递归展开、文本模态筛选和命令行参数处理；实体关系
抽取、JSONL 检查点以及最终 JSON 写入均复用同目录中已经实现的无 Schema 流程。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

# 中文说明：支持直接执行本文件，将项目根目录加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Graph
from src.extractors.extractor_init import collect_chunks
from src.extractors.text_extractor.pipeline import extract_text_chunks_to_file, write_text_extraction_result
from src.utils.json_io import read_json
from src.utils.llm_client import LLMClient


DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "stage_03_document_summary.json"
"""第三阶段章节总结 JSON 的默认输入路径。"""

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "stage_04_text_extraction.json"
"""文本模态 Graph 抽取结果的默认输出路径。"""

TEXT_EXTRACTION_WORKERS = 4
"""文本抽取默认工作线程数，统一由启动入口配置。"""


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """校验 JSON 顶层是对象，并给出便于定位输入问题的错误信息。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 JSON 对象，实际为 {type(value).__name__}")
    return value


def _is_text_chunk(chunk: Mapping[str, Any]) -> bool:
    """兼容不同模态字段写法，只保留可交给文本抽取器的 Chunk。"""

    value = chunk.get("modality") or chunk.get("chunk_type") or chunk.get("type") or ""
    modality = str(getattr(value, "value", value)).strip().lower()
    aliases = {"textchunk": "text"}
    return aliases.get(modality.replace("_", ""), modality) == "text"


def extract_text_from_summary_file(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    llm_client: Any | None = None,
    max_workers: int = TEXT_EXTRACTION_WORKERS,
    show_progress: bool = True,
) -> list[Graph]:
    """读取第三阶段总结文件，抽取全部章节文本窗口并保存为 JSON。

    函数会递归遍历 ``document.sections[*].children``，把章节标题、章节总结、
    文档总结和章节上下文补充到每个文本窗口，再调用现有的
    ``extract_text_chunks_to_file`` 完成逐 Chunk 文本抽取。输出文件包含
    ``graphs`` 和统计信息；有文本窗口时同时生成同名 ``.jsonl`` 文件作为断点检查点。

    Args:
        input_path: 第三阶段 ``stage_03_document_summary.json`` 文件。
        output_path: 文本抽取结果 JSON 文件。
        llm_client: 可选的文本模型客户端，未传入时创建默认 ``LLMClient``。
        show_progress: 是否显示逐 Chunk 进度条。

    Returns:
        按文本窗口顺序返回抽取完成的 ``Graph`` 列表。
    """

    # 中文说明：先读取并校验第三阶段文件，再由公共收集器递归展开章节树。
    payload = _as_mapping(read_json(input_path), "第三阶段总结输入")
    chunks = [chunk for chunk in collect_chunks(payload) if _is_text_chunk(chunk)]

    # 中文说明：空文本输入也写出标准结果 JSON，保证下游无需区分“无窗口”和“未执行”。
    if not chunks:
        write_text_extraction_result(output_path, [], status="completed")
        return []

    # 中文说明：复用已有逐 Chunk 管线，它内部会调用 extract_from_text 并写入检查点。
    client = llm_client if llm_client is not None else LLMClient()
    return extract_text_chunks_to_file(
        chunks,
        output_path,
        llm_client=client,
        max_workers=max_workers,
        show_progress=show_progress,
    )


def main() -> None:
    """命令行入口：从第三阶段总结文件启动文本窗口抽取。"""

    parser = argparse.ArgumentParser(description="从第三阶段章节总结 JSON 执行文本窗口抽取")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="第三阶段章节总结 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="文本抽取结果 JSON")
    parser.add_argument("--workers", type=int, default=TEXT_EXTRACTION_WORKERS, help="LLM 并发线程数")
    parser.add_argument("--quiet", action="store_true", help="不显示逐 Chunk 进度条")
    args = parser.parse_args()

    # 中文说明：输出简明统计，便于确认递归章节读取和 JSON 保存均已完成。
    graphs = extract_text_from_summary_file(
        args.input,
        args.output,
        max_workers=args.workers,
        show_progress=not args.quiet,
    )
    print(f"文本抽取完成：Graph={len(graphs)}，输出文件：{args.output.resolve()}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "TEXT_EXTRACTION_WORKERS",
    "extract_text_from_summary_file",
    "main",
]
