"""表格 Chunk 的项目适配抽取器。"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from model import Graph

from ..schema_constrained import extract_chunk_graph
from ..schema_router import SchemaSelector


def _mapping(chunk: Any) -> Mapping[str, Any]:
    """把表格 Chunk 模型或字典统一转换为映射。"""

    if isinstance(chunk, Mapping):
        return chunk
    if hasattr(chunk, "to_dict"):
        return chunk.to_dict()
    if hasattr(chunk, "dict"):
        return chunk.dict()
    raise TypeError(f"不支持的表格 Chunk 类型：{type(chunk).__name__}")


def _table_text(item: Mapping[str, Any]) -> str:
    """组合标题、Markdown 表体、脚注与上下文，形成表格的完整文本证据。"""

    references = item.get("references") or []
    if isinstance(references, str):
        references = [references]
    parts = [
        str(item.get("caption") or ""),
        str(item.get("markdown") or item.get("table_body") or item.get("content") or ""),
        str(item.get("footnote") or ""),
        "\n".join(str(value) for value in references),
        str(item.get("context") or ""),
    ]
    return "\n".join(part for part in parts if part.strip())


def extract_from_tables(
    chunks: Sequence[Any],
    llm_client: Any,
    schema_selector: Any | None = None,
) -> list[Graph]:
    """逐个表格 Chunk 路由领域树，并用表体证据抽取统一 Graph。"""

    selector = schema_selector or SchemaSelector()
    graphs: list[Graph] = []
    for chunk in chunks:
        item = _mapping(chunk)
        source_text = _table_text(item)
        relevant = selector.select(
            source_text,
            modality="table",
            context={"section_title": item.get("section_title"), "caption": item.get("caption")},
        )
        graphs.append(
            extract_chunk_graph(
                item,
                "table",
                source_text,
                relevant,
                llm_client,
                extractor_name="table_schema_extractor",
            )
        )
    return graphs


__all__ = ["extract_from_tables"]
