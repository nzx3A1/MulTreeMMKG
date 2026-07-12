"""文本 Chunk 的项目适配抽取器。"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from model import Graph

from ..schema_constrained import extract_chunk_graph
from ..schema_router import SchemaSelector


def _mapping(chunk: Any) -> Mapping[str, Any]:
    """把文本 Chunk 模型或字典统一转换为映射。"""

    if isinstance(chunk, Mapping):
        return chunk
    if hasattr(chunk, "to_dict"):
        return chunk.to_dict()
    if hasattr(chunk, "dict"):
        return chunk.dict()
    raise TypeError(f"不支持的文本 Chunk 类型：{type(chunk).__name__}")


def extract_from_text(
    chunks: Sequence[Any],
    llm_client: Any,
    schema_selector: Any | None = None,
) -> list[Graph]:
    """逐个文本 Chunk 先选择 Schema 树，再抽取并返回统一 Graph 列表。"""

    selector = schema_selector or SchemaSelector()
    graphs: list[Graph] = []
    for chunk in chunks:
        item = _mapping(chunk)
        source_text = str(item.get("text") or item.get("content") or "").strip()
        context = {
            "section_title": item.get("section_title"),
            "document_title": item.get("document_title"),
        }
        relevant = selector.select(source_text, modality="text", context=context)
        graphs.append(
            extract_chunk_graph(
                item,
                "text",
                source_text,
                relevant,
                llm_client,
                extractor_name="text_schema_extractor",
            )
        )
    return graphs


__all__ = ["extract_from_text"]
