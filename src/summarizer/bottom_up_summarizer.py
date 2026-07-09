"""自底向上摘要器（Bottom-up Summarizer）。

从最底层 chunk 摘要开始，逐层向上聚合到小节 → 节 → 章节 → 全文，
为下游 chunk 构建与图谱融合提供全局上下文。

输出：stage_04_section_summary.json
对应阶段：04 章节摘要
"""
from __future__ import annotations

import html
import re
from collections import defaultdict
from typing import Any


def _clean_text(value: Any) -> str:
    """清洗摘要候选文本，避免 HTML 标记进入阶段产物。"""

    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _chunk_text(chunk: dict[str, Any]) -> str:
    """按 chunk 模态提取可用于摘要的文本。"""

    modality = chunk.get("modality")
    if modality == "text":
        return _clean_text(chunk.get("text"))
    if modality == "table":
        return _clean_text(chunk.get("caption") or chunk.get("markdown"))
    if modality == "image":
        return _clean_text(chunk.get("caption") or chunk.get("ocr_text"))
    if modality == "formula":
        return _clean_text(chunk.get("caption") or chunk.get("latex"))
    return ""


def _shorten(text: str, limit: int = 260) -> str:
    """生成无外部 LLM 依赖的抽取式短摘要。"""

    text = _clean_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def summarize_bottom_up(chunks: list[dict], llm_client=None) -> dict:
    """对 chunk 列表自底向上生成各级摘要。"""

    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        section_id = str(chunk.get("section_id") or "")
        document_id = str(chunk.get("document_id") or "")
        if section_id:
            by_section[section_id].append(chunk)
        if document_id:
            by_document[document_id].append(chunk)

    summaries: list[dict[str, Any]] = []
    for section_id, section_chunks in by_section.items():
        text = " ".join(_chunk_text(chunk) for chunk in section_chunks if _chunk_text(chunk))
        summaries.append({
            "id": f"{section_id}_summary",
            "document_id": section_chunks[0].get("document_id"),
            "section_id": section_id,
            "section_title": section_chunks[0].get("section_title"),
            "level": "section",
            "summary": _shorten(text),
            "source_chunk_ids": [chunk.get("id") for chunk in section_chunks],
            "method": "extractive",
        })

    document_summaries: list[dict[str, Any]] = []
    for document_id, document_chunks in by_document.items():
        text = " ".join(_chunk_text(chunk) for chunk in document_chunks if _chunk_text(chunk))
        document_summaries.append({
            "id": f"{document_id}_summary",
            "document_id": document_id,
            "level": "document",
            "summary": _shorten(text, 500),
            "source_chunk_ids": [chunk.get("id") for chunk in document_chunks[:50]],
            "method": "extractive",
        })

    return {
        "_stage": 4,
        "_description": "自底向上生成的各级章节摘要。",
        "_produced_by": "src/summarizer/bottom_up_summarizer.py::summarize_bottom_up",
        "summaries": summaries,
        "document_summaries": document_summaries,
        "metadata": {
            "chunk_count": len(chunks),
            "section_summary_count": len(summaries),
            "document_summary_count": len(document_summaries),
            "llm_used": llm_client is not None,
        },
    }
