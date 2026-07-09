"""多模态 chunk 构建器。

依据文档骨架图与 MinerU 输出，把每一节内的文本块、表格、图像、公式
按模态类型组合为多模态 chunk，作为下游抽取的最小处理单元。

输出：stage_05_modal_chunks.json
对应阶段：05 多模态 chunk
"""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any


class _TableCellParser(HTMLParser):
    """解析 MinerU 输出的 HTML 表格，提取简化的二维单元格文本。"""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """记录当前进入的表格行或单元格。"""

        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"}:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        """收集当前单元格中的文本片段。"""

        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        """在行或单元格闭合时写入解析结果。"""

        if tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            self._current_row.append(_clean_text("".join(self._current_cell)))
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None


def _clean_text(value: Any) -> str:
    """清洗 chunk 文本，去掉 HTML 标记并压缩空白。"""

    text = "" if value is None else str(value)
    text = re.sub(r"</?(sub|sup)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_table_cells(markdown_or_html: str) -> list[list[str]]:
    """从 HTML 表格中解析单元格；非 HTML 表格返回空列表。"""

    if "<table" not in (markdown_or_html or "").lower():
        return []
    parser = _TableCellParser()
    parser.feed(markdown_or_html)
    return parser.rows


def _iter_sections(skeleton: dict) -> list[dict[str, Any]]:
    """读取骨架图中保留的章节展开列表。"""

    return skeleton.get("flat_sections") or skeleton.get("sections") or []


def _provenance(document_id: str, section_id: str, chunk_id: str, block: dict[str, Any]) -> dict[str, Any]:
    """为 chunk 生成统一溯源字段。"""

    return {
        "document_id": document_id,
        "section_id": section_id,
        "chunk_id": chunk_id,
        "source_file": block.get("resource_path"),
        "modality": block.get("type"),
        "extractor": "chunk_builder",
        "confidence": 1.0,
        "raw": {
            "block_id": block.get("id"),
            "content_id": block.get("content_id"),
            "page_num": block.get("page_num"),
        },
    }


def _make_chunk_id(document_id: str, order: int) -> str:
    """生成稳定的 chunk ID。"""

    return f"{document_id}_chunk_{order:06d}"


def build_modal_chunks(skeleton: dict, paper: dict, summary: dict | None = None) -> list[dict]:
    """构建多模态 chunk 列表。每个 chunk 包含章节定位 + 文本 + 表格 + 图像 + 公式引用。"""

    document_id = str(skeleton.get("document_id") or paper.get("document_id") or paper.get("paper_id") or "document")
    chunks: list[dict[str, Any]] = []
    chunk_order = 0
    max_text_chars = 1600

    def flush_text(section: dict[str, Any], buffer: list[dict[str, Any]]) -> None:
        """把当前章节内累积的文本块落成一个正文 chunk。"""

        nonlocal chunk_order
        if not buffer:
            return
        text = "\n".join(_clean_text(block.get("content") or block.get("text")) for block in buffer)
        text = text.strip()
        if not text:
            buffer.clear()
            return
        first_block = buffer[0]
        chunk_id = _make_chunk_id(document_id, chunk_order)
        chunks.append({
            "id": chunk_id,
            "document_id": document_id,
            "section_id": section.get("id"),
            "section_title": section.get("title"),
            "content_id": first_block.get("content_id"),
            "order": chunk_order,
            "modality": "text",
            "text": text,
            "token_count": len(text),
            "summary": (summary or {}).get(section.get("id")),
            "provenance": _provenance(document_id, section.get("id"), chunk_id, first_block),
            "metadata": {
                "source_block_ids": [block.get("id") for block in buffer],
                "page_nums": sorted({block.get("page_num") for block in buffer if block.get("page_num")}),
            },
        })
        chunk_order += 1
        buffer.clear()

    for section in _iter_sections(skeleton):
        text_buffer: list[dict[str, Any]] = []
        for block in section.get("blocks", []):
            block_type = block.get("type")
            role = block.get("role")
            if role in {"header", "footer", "page_number"}:
                continue

            if block_type == "text":
                content = _clean_text(block.get("content") or block.get("text"))
                if not content:
                    continue
                current_size = sum(len(_clean_text(item.get("content") or item.get("text"))) for item in text_buffer)
                if text_buffer and current_size + len(content) > max_text_chars:
                    flush_text(section, text_buffer)
                text_buffer.append(block)
                continue

            flush_text(section, text_buffer)
            chunk_id = _make_chunk_id(document_id, chunk_order)
            if block_type == "table":
                markdown = block.get("content") or ""
                chunks.append({
                    "id": chunk_id,
                    "document_id": document_id,
                    "section_id": section.get("id"),
                    "section_title": section.get("title"),
                    "content_id": block.get("content_id"),
                    "order": chunk_order,
                    "modality": "table",
                    "markdown": markdown,
                    "caption": block.get("caption") or block.get("footnote"),
                    "cells": _parse_table_cells(markdown),
                    "summary": None,
                    "provenance": _provenance(document_id, section.get("id"), chunk_id, block),
                    "metadata": {"page_num": block.get("page_num"), "resource_path": block.get("resource_path")},
                })
            elif block_type == "image":
                chunks.append({
                    "id": chunk_id,
                    "document_id": document_id,
                    "section_id": section.get("id"),
                    "section_title": section.get("title"),
                    "content_id": block.get("content_id"),
                    "order": chunk_order,
                    "modality": "image",
                    "image_path": block.get("resource_path"),
                    "image_url": None,
                    "caption": block.get("caption") or block.get("text"),
                    "ocr_text": block.get("text"),
                    "summary": None,
                    "provenance": _provenance(document_id, section.get("id"), chunk_id, block),
                    "metadata": {"page_num": block.get("page_num")},
                })
            elif block_type == "formula":
                chunks.append({
                    "id": chunk_id,
                    "document_id": document_id,
                    "section_id": section.get("id"),
                    "section_title": section.get("title"),
                    "content_id": block.get("content_id"),
                    "order": chunk_order,
                    "modality": "formula",
                    "latex": block.get("content") or block.get("text") or "",
                    "caption": block.get("caption"),
                    "context": section.get("title"),
                    "summary": None,
                    "provenance": _provenance(document_id, section.get("id"), chunk_id, block),
                    "metadata": {"page_num": block.get("page_num")},
                })
            chunk_order += 1

        flush_text(section, text_buffer)

    return chunks
