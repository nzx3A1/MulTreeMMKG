"""文本抽取阶段使用的轻量文档上下文。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TextDocumentContext:
    """保存文本抽取所需的顺序、章节和相邻片段，不包含 Schema 信息。"""

    document_id: str
    chunks: tuple[Mapping[str, Any], ...]
    chunk_indexes: Mapping[str, int]
    section_chunks: Mapping[str, tuple[str, ...]]
    topic_profile: str

    def local_context(self, chunk_id: str, *, neighbor_chars: int = 180) -> dict[str, str]:
        """返回当前章节信息和相邻正文，仅供模型消歧且不参与筛选。"""

        index = self.chunk_indexes[chunk_id]
        current = self.chunks[index]
        section_id = str(current.get("section_id") or "")
        section_ids = self.section_chunks.get(section_id, ())
        position = section_ids.index(chunk_id)
        previous_text = ""
        next_text = ""
        if position > 0:
            previous = self.chunks[self.chunk_indexes[section_ids[position - 1]]]
            previous_text = str(previous.get("text") or "")[-neighbor_chars:]
        if position + 1 < len(section_ids):
            following = self.chunks[self.chunk_indexes[section_ids[position + 1]]]
            next_text = str(following.get("text") or "")[:neighbor_chars]
        return {
            "section_title": str(current.get("section_title") or ""),
            "section_summary": str(current.get("section_summary") or ""),
            "previous_tail": previous_text,
            "next_head": next_text,
            "document_topic_profile": self.topic_profile,
        }


def _as_mapping(chunk: Any) -> dict[str, Any]:
    """把字典或 Pydantic Chunk 转成字典，统一文本抽取入口的输入格式。"""

    if isinstance(chunk, Mapping):
        return dict(chunk)
    if hasattr(chunk, "to_dict"):
        return dict(chunk.to_dict())
    if hasattr(chunk, "model_dump"):
        return dict(chunk.model_dump(mode="json"))
    if hasattr(chunk, "dict"):
        return dict(chunk.dict())
    raise TypeError(f"不支持的文本 Chunk 类型：{type(chunk).__name__}")


def build_text_document_context(chunks: Sequence[Any]) -> TextDocumentContext:
    """校验并排序文本 Chunk，构造不依赖 Schema 的文档上下文。"""

    if not chunks:
        return TextDocumentContext("", (), {}, {}, "")

    normalized: list[tuple[int, int, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    document_ids: set[str] = set()
    for input_index, raw_chunk in enumerate(chunks):
        chunk = _as_mapping(raw_chunk)
        chunk_id = str(chunk.get("id") or "").strip()
        if not chunk_id:
            raise ValueError(f"第 {input_index} 个文本 Chunk 缺少唯一 id")
        if chunk_id in seen_ids:
            raise ValueError(f"文本 Chunk id 重复：{chunk_id}")
        seen_ids.add(chunk_id)

        modality = str(getattr(chunk.get("modality"), "value", chunk.get("modality") or "text")).lower()
        if modality not in {"text", "textchunk"}:
            raise ValueError(f"Chunk {chunk_id} 不是文本模态：{modality}")
        chunk["text"] = str(chunk.get("text") or "").strip()
        document_id = str(chunk.get("document_id") or "").strip()
        if document_id:
            document_ids.add(document_id)
        try:
            order = int(chunk.get("order", input_index))
        except (TypeError, ValueError):
            order = input_index
        normalized.append((order, input_index, chunk))

    if len(document_ids) > 1:
        raise ValueError(f"一次文本抽取只能处理一篇论文，收到 document_id：{sorted(document_ids)}")

    normalized.sort(key=lambda item: (item[0], item[1]))
    ordered_chunks = tuple(item[2] for item in normalized)
    chunk_indexes = {str(chunk["id"]): index for index, chunk in enumerate(ordered_chunks)}
    section_map: dict[str, list[str]] = defaultdict(list)
    section_titles: list[str] = []
    section_summaries: list[str] = []
    for chunk in ordered_chunks:
        section_map[str(chunk.get("section_id") or "")].append(str(chunk["id"]))
        title = str(chunk.get("section_title") or "").strip()
        summary = str(chunk.get("section_summary") or "").strip()
        if title and title not in section_titles:
            section_titles.append(title)
        if summary and summary not in section_summaries:
            section_summaries.append(summary)

    first_chunk = ordered_chunks[0]
    document_id = next(iter(document_ids), "")
    # 中文说明：主题画像只传递上游摘要和章节文本，供文本语义消歧使用。
    topic_profile = "\n".join(
        (
            f"文档标题：{str(first_chunk.get('document_title') or document_id).strip()}",
            f"文档摘要：{str(first_chunk.get('document_abstract') or '').strip()}",
            f"文档总结：{str(first_chunk.get('document_summary') or '').strip()}",
            f"章节标题：{'；'.join(section_titles)}",
            f"章节总结：{'；'.join(section_summaries)}",
        )
    )
    return TextDocumentContext(
        document_id=document_id,
        chunks=ordered_chunks,
        chunk_indexes=chunk_indexes,
        section_chunks={key: tuple(value) for key, value in section_map.items()},
        topic_profile=topic_profile,
    )


__all__ = ["TextDocumentContext", "build_text_document_context"]
