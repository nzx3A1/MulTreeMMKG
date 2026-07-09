"""章节切分器。

基于 MinerU content_list.json 中的标题层级与 Markdown 解析结果，
将一篇论文切分为多级章节树（章 → 节 → 小节），为骨架图与摘要提供节点。

输出：stage_02_section_tree.json
对应阶段：02 章节树
"""
from __future__ import annotations

import html
import re
from copy import deepcopy
from typing import Any


def _clean_text(value: Any) -> str:
    """清洗章节标题和正文中的 HTML 标记。"""

    text = "" if value is None else str(value)
    text = re.sub(r"</?(sub|sup)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _heading_level(block: dict[str, Any]) -> int | None:
    """读取 MinerU 或 Markdown block 中的标题层级。"""

    metadata = block.get("metadata") or {}
    level = metadata.get("text_level", block.get("level"))
    if level is None or block.get("role") != "heading":
        return None
    try:
        return int(level)
    except (TypeError, ValueError):
        return None


def _extract_number(title: str) -> str | None:
    """从章节标题前缀提取 1、1.2、0 等章节编号。"""

    match = re.match(r"^([0-9０-９]+(?:[.．、][0-9０-９]+)*)\s*", title)
    if not match:
        return None
    return match.group(1).replace("．", ".").replace("、", ".")


def _make_section(document_id: str, title: str, level: int, order: int, parent_id: str | None, block: dict[str, Any] | None = None) -> dict[str, Any]:
    """创建标准章节节点，供章节树与骨架图复用。"""

    clean_title = _clean_text(title) or "未命名章节"
    return {
        "id": f"{document_id}_sec_{order:04d}",
        "document_id": document_id,
        "title": clean_title,
        "level": level,
        "order": order,
        "parent_id": parent_id,
        "number": _extract_number(clean_title),
        "text": "",
        "blocks": [],
        "children": [],
        "summary": None,
        "metadata": {
            "raw_level": _heading_level(block or {}) if block else None,
            "page_num": (block or {}).get("page_num"),
            "source_block_id": (block or {}).get("id"),
        },
    }


def _block_ref(block: dict[str, Any], order: int) -> dict[str, Any]:
    """把原始 block 转为挂载到章节下的轻量引用。"""

    ref = deepcopy(block)
    ref["order"] = order
    return ref


def _flatten(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """深度优先展开章节树，便于后续模块顺序处理。"""

    flat: list[dict[str, Any]] = []
    for section in sections:
        flat.append(section)
        flat.extend(_flatten(section.get("children", [])))
    return flat


def _base_heading_level(blocks: list[dict[str, Any]]) -> int:
    """判断真正论文正文的起始标题级别，跳过论文题名使用的一级标题。"""

    levels = [level for block in blocks if (level := _heading_level(block)) is not None]
    body_levels = [level for level in levels if level >= 2]
    if body_levels:
        return min(body_levels)
    return min(levels) if levels else 1


def _append_block(section: dict[str, Any], block: dict[str, Any]) -> None:
    """把内容块加入章节，并同步维护章节合并正文。"""

    block = _block_ref(block, len(section["blocks"]))
    block["section_id"] = section["id"]
    section["blocks"].append(block)
    if block.get("type") == "text" and block.get("role") not in {"header", "footer", "page_number"}:
        text = _clean_text(block.get("content") or block.get("text"))
        if text:
            section["text"] = f"{section['text']}\n{text}".strip()


def split_sections(parsed_doc: dict) -> dict:
    """从归一化的论文结构中构建章节树。"""

    if isinstance(parsed_doc, list):
        document_id = "document"
        blocks = parsed_doc
        meta: dict[str, Any] = {}
    else:
        document_id = str(parsed_doc.get("document_id") or parsed_doc.get("paper_id") or "document")
        blocks = parsed_doc.get("blocks", [])
        meta = parsed_doc.get("meta", {})
    base_level = _base_heading_level(blocks)
    roots: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    section_order = 0

    front_matter = _make_section(document_id, "前置信息", 1, section_order, None)
    section_order += 1
    current_section: dict[str, Any] | None = None

    for block in blocks:
        raw_level = _heading_level(block)
        if raw_level is not None and raw_level >= base_level:
            level = raw_level - base_level + 1
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            parent = stack[-1] if stack else None
            section = _make_section(document_id, block.get("content") or block.get("text"), level, section_order, parent["id"] if parent else None, block)
            section_order += 1
            if parent:
                parent["children"].append(section)
            else:
                roots.append(section)
            stack.append(section)
            current_section = section
            continue

        if current_section is None:
            _append_block(front_matter, block)
        else:
            _append_block(current_section, block)

    if front_matter["blocks"]:
        roots.insert(0, front_matter)

    flat_sections = _flatten(roots)
    return {
        "_stage": 2,
        "_description": "文档章节树（章 → 节 → 小节）。",
        "_produced_by": "src/parser/section_splitter.py::split_sections",
        "paper_id": document_id,
        "document_id": document_id,
        "title": meta.get("title", ""),
        "sections": roots,
        "flat_sections": flat_sections,
        "statistics": {
            "section_count": len(flat_sections),
            "root_section_count": len(roots),
            "base_heading_level": base_level,
        },
    }
