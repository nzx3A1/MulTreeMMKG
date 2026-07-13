"""支持逐 Chunk 日志、断点续跑和最终 JSON 汇总的文本抽取持久化管线。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from model import Graph
from src.utils.json_io import write_json
from src.utils.logger import get_logger


logger = get_logger("extractors.text_extractor.pipeline")


def _dump_graph(graph: Graph) -> dict[str, Any]:
    """兼容 Pydantic v1/v2 地序列化 Graph。"""

    return graph.model_dump(mode="json") if hasattr(graph, "model_dump") else graph.dict()


def write_text_extraction_result(path: str | Path, graphs: Sequence[Graph], *, status: str) -> None:
    """写入可直接供后续阶段读取的文本 Graph 数组和统计信息。"""

    result = {
        "_status": status,
        "statistics": {
            "graph_count": len(graphs),
            "completed_text_chunk_count": len(graphs),
            "entity_count": sum(len(graph.entities) for graph in graphs),
            "relation_count": sum(len(graph.relations) for graph in graphs),
            "event_count": sum(len(graph.events) for graph in graphs),
        },
        "graphs": [_dump_graph(graph) for graph in graphs],
    }
    write_json(path, result)


def _load_journal(path: Path) -> list[Graph]:
    """读取已完成的 JSONL 记录，尾部损坏行会被忽略以便安全续跑。"""

    graphs: list[Graph] = []
    if not path.exists():
        return graphs
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            graphs.append(Graph(**json.loads(line)))
        except (ValueError, TypeError, json.JSONDecodeError):
            break
    return graphs


def extract_text_chunks_to_file(
    chunks: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    llm_client: Any,
    schema_selector: Any | None = None,
    show_progress: bool = True,
) -> list[Graph]:
    """逐段持久化文本 Graph，并跳过 JSONL 中已经成功完成的 Chunk。"""

    from .text_extractor import extract_from_text

    output = Path(output_path)
    journal = output.with_suffix(".jsonl")
    completed = _load_journal(journal)
    completed_ids = {str(graph.metadata.chunk_id) for graph in completed}
    pending = [chunk for chunk in chunks if str(chunk.get("id") or "") not in completed_ids]
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"文本抽取检查点载入：已完成={len(completed)}，待处理={len(pending)}，日志={journal}"
    )

    def persist(graph: Graph, index: int, total: int) -> None:
        """每完成一个 Chunk 立即追加日志并刷新阶段检查点。"""

        with journal.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_dump_graph(graph), ensure_ascii=False) + "\n")
        completed.append(graph)
        print(f"[{index}/{total}] Chunk {graph.metadata.chunk_id} 已写入 {journal}")
        logger.info(f"Chunk {graph.metadata.chunk_id} 检查点已写入：{journal}")
        write_text_extraction_result(output, completed, status="running" if index < total else "completed")

    if pending:
        extract_from_text(
            pending, llm_client, schema_selector,
            on_graph_completed=persist, show_progress=show_progress,
        )
    write_text_extraction_result(output, completed, status="completed")
    logger.info(f"文本抽取持久化完成：Graph={len(completed)}，输出={output}")
    return completed


__all__ = ["extract_text_chunks_to_file", "write_text_extraction_result"]
