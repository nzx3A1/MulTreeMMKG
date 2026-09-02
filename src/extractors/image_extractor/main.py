"""第三阶段章节总结 JSON 到图片知识图谱的独立执行入口。"""
from __future__ import annotations

import argparse
import sys
from time import perf_counter
from pathlib import Path
from typing import Any, Mapping

# 中文说明：支持直接执行本文件，并从仓库根目录导入统一模型与工具。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Graph
from src.extractors.extractor_init import collect_chunks
from src.extractors.image_extractor.classification import (
    ImageClassificationProvider,
    RecordImageClassificationProvider,
    VLMImageClassificationProvider,
)
from src.extractors.image_extractor.pipeline import extract_from_images, write_image_extraction_result
from src.extractors.image_extractor.schema_models import as_string_tuple
from src.utils.json_io import read_json
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger
from src.utils.vlm_client import VLMClient


logger = get_logger("extractors.image_extractor.main")


DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "stage_03_document_summary.json"
"""第三阶段章节总结 JSON 的默认输入路径。"""

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "stage_04_image_extraction.json"
"""图片模态 Graph 抽取结果的默认输出路径。"""


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """校验 JSON 顶层对象并给出清晰的输入错误。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 JSON 对象，实际为 {type(value).__name__}")
    return value


def _is_image_chunk(chunk: Mapping[str, Any]) -> bool:
    """兼容不同模态字段写法，只保留 ImageChunk。"""

    value = chunk.get("modality") or chunk.get("chunk_type") or chunk.get("type") or ""
    modality = str(getattr(value, "value", value)).strip().lower()
    aliases = {"imagechunk": "image"}
    return aliases.get(modality.replace("_", ""), modality) == "image"


def _chunk_id(chunk: Mapping[str, Any]) -> str:
    """返回输入 Chunk 的稳定标识。"""

    return str(chunk.get("id") or chunk.get("chunk_id") or "").strip()


def _load_reusable_graphs(output_path: str | Path) -> dict[str, Graph]:
    """读取已有结果，只复用同时含实体和关系的图片 Chunk Graph。"""

    path = Path(output_path)
    if not path.is_file():
        return {}

    try:
        payload = _as_mapping(read_json(path), "已有图片抽取结果")
    except Exception as exc:
        logger.warning(f"[图片抽取/续跑] 已有结果无法读取，将重新抽取：path={path.resolve()}，error={exc}")
        return {}

    reusable: dict[str, Graph] = {}
    raw_graphs = payload.get("graphs")
    if not isinstance(raw_graphs, list):
        logger.warning(f"[图片抽取/续跑] 已有结果缺少 graphs 数组，将重新抽取：path={path.resolve()}")
        return reusable

    for raw_graph in raw_graphs:
        if not isinstance(raw_graph, Mapping):
            continue
        metadata = raw_graph.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        chunk_id = str(metadata.get("chunk_id") or "").strip()
        entities = raw_graph.get("entities")
        relations = raw_graph.get("relations")
        if not chunk_id or not isinstance(entities, list) or not entities:
            continue
        if not isinstance(relations, list) or not relations:
            continue
        try:
            reusable[chunk_id] = Graph(**dict(raw_graph))
        except Exception as exc:
            logger.warning(
                f"[图片抽取/续跑] 已有 Chunk Graph 校验失败，将重新抽取：chunk={chunk_id}，error={exc}"
            )
    return reusable


def extract_images_from_summary_file(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    llm_client: Any | None = None,
    vlm_client: Any | None = None,
    classification_provider: ImageClassificationProvider | None = None,
    show_progress: bool = True,
) -> list[Graph]:
    """递归读取第三阶段图片 Chunk，调用现有图片管线并保存结构化 Graph。"""

    resolved_input = Path(input_path).resolve()
    resolved_output = Path(output_path).resolve()
    started_at = perf_counter()
    logger.info(f"[图片抽取/读取] 开始读取文章 Chunk：input={resolved_input}")
    payload = _as_mapping(read_json(input_path), "第三阶段章节总结输入")
    all_chunks = list(collect_chunks(payload))
    image_chunks = [chunk for chunk in all_chunks if _is_image_chunk(chunk)]
    image_count = sum(len(as_string_tuple(chunk.get("image_path"))) for chunk in image_chunks)
    reusable_graphs = _load_reusable_graphs(resolved_output)
    pending_chunks = [chunk for chunk in image_chunks if _chunk_id(chunk) not in reusable_graphs]
    logger.info(
        f"[图片抽取/扫描] Chunk 扫描完成：全部={len(all_chunks)}，图片Chunk={len(image_chunks)}，"
        f"图片文件={image_count}，output={resolved_output}"
    )
    logger.info(
        f"[图片抽取/续跑] 已有完整Chunk={len(reusable_graphs)}，"
        f"待抽取Chunk={len(pending_chunks)}"
    )

    extracted_graphs: list[Graph] = []
    if pending_chunks:
        # 中文说明：仅在确有待处理 Chunk 时初始化模型客户端，完整续跑不会产生模型调用。
        active_llm = llm_client if llm_client is not None else LLMClient()
        active_vlm = vlm_client if vlm_client is not None else VLMClient()
        provider = classification_provider or VLMImageClassificationProvider(active_vlm)
        logger.info(
            f"[图片抽取/初始化] 模型客户端与分类器就绪："
            f"llm={type(active_llm).__name__}，vlm={type(active_vlm).__name__}，"
            f"classification_provider={type(provider).__name__}"
        )
        extracted_graphs = extract_from_images(
            pending_chunks,
            active_llm,
            active_vlm,
            classification_provider=provider,
            output_path=None,
            show_progress=show_progress,
        )

    extracted_by_chunk_id = {
        str(graph.metadata.chunk_id): graph
        for graph in extracted_graphs
        if graph.metadata.chunk_id
    }
    graphs = [
        reusable_graphs.get(_chunk_id(chunk)) or extracted_by_chunk_id[_chunk_id(chunk)]
        for chunk in image_chunks
    ]
    write_image_extraction_result(resolved_output, graphs, status="completed")
    logger.info(
        f"[图片抽取/完成] 文章图片抽取结束：Graph={len(graphs)}，"
        f"实体={sum(len(graph.entities) for graph in graphs)}，"
        f"关系={sum(len(graph.relations) for graph in graphs)}，"
        f"事件={sum(len(graph.events) for graph in graphs)}，"
        f"耗时={perf_counter() - started_at:.2f}s，output={resolved_output}"
    )
    return graphs


def _build_cli() -> argparse.ArgumentParser:
    """构建图片抽取命令行参数。"""

    parser = argparse.ArgumentParser(description="从第三阶段章节总结 JSON 执行图片知识抽取")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="第三阶段章节总结 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="图片 Graph 结果 JSON")
    parser.add_argument("--image-classifications", type=Path, help="可选的逐图片分类 JSON；省略时调用 VLM 分类")
    parser.add_argument("--quiet", action="store_true", help="不显示逐 Chunk 进度")
    return parser


def main() -> int:
    """命令行入口：完成图片分类、路由、抽取和最终 JSON 保存。"""

    args = _build_cli().parse_args()
    logger.info(
        f"[图片抽取/启动] 命令行任务启动：input={args.input.resolve()}，"
        f"output={args.output.resolve()}，逐Chunk进度={'关闭' if args.quiet else '开启'}"
    )
    try:
        provider = (
            RecordImageClassificationProvider.from_json(args.image_classifications)
            if args.image_classifications
            else None
        )
        if args.image_classifications:
            logger.info(f"[图片抽取/分类] 使用已有逐图片分类文件：{args.image_classifications.resolve()}")
        else:
            logger.info("[图片抽取/分类] 未提供分类文件，将逐图调用 VLM 完成 A01-A20 分类")
        graphs = extract_images_from_summary_file(
            args.input,
            args.output,
            classification_provider=provider,
            show_progress=not args.quiet,
        )
    except Exception:
        logger.exception(
            f"[图片抽取/失败] 命令行任务异常终止：input={args.input.resolve()}，output={args.output.resolve()}"
        )
        raise
    logger.info(f"[图片抽取/退出] 命令行任务成功：Graph={len(graphs)}，输出文件={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "extract_images_from_summary_file",
    "main",
]
