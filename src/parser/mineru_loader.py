"""MinerU 解析结果加载器。

职责：
    - 读取 data/mineru_output/<paper_id>/ 目录下的 content_list.json、
      layout.json、full.md 以及 images/ 子目录。
    - 归一化为统一的 Paper 内部字典结构，供下游模块（章节切分、骨架构建）使用。
    - 写出 stage_01_mineru_parse.json 所需的数据。

对应阶段：01 MinerU 解析
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from config.app_config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Any:
    """读取 MinerU JSON 文件，集中处理 UTF-8 编码。"""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _project_relative(path: Path) -> str:
    """将项目内路径转为相对路径，便于 JSON 产物在不同机器上复用。"""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _clean_text(value: Any) -> str:
    """清洗 MinerU 文本中的 HTML 标记并压缩多余空白。"""

    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"</?(sub|sup)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _join_lines(value: Any) -> str:
    """把 MinerU caption/footnote 的列表结构归一为换行文本。"""

    if isinstance(value, list):
        return "\n".join(_clean_text(item) for item in value if _clean_text(item))
    return _clean_text(value)


def _load_index(mineru_root: Path) -> dict[str, dict[str, Any]]:
    """读取 data/mineru_output/index.json，建立 paper_id 到源 PDF 的索引。"""

    index_path = mineru_root / "index.json"
    if not index_path.exists():
        return {}
    rows = _read_json(index_path)
    if not isinstance(rows, list):
        return {}
    return {str(row.get("paper_id")): row for row in rows if row.get("paper_id")}


def _find_source_pdf(paper_id: str, mineru_root: Path) -> str | None:
    """根据 MinerU 输出索引反查原始 PDF 路径。"""

    row = _load_index(mineru_root).get(paper_id)
    if not row:
        return None
    file_name = row.get("source_pdf")
    if not file_name:
        return None
    pdf_path = settings.raw_pdf_dir / str(file_name)
    return _project_relative(pdf_path) if pdf_path.exists() else str(file_name)


def _item_text(item: dict[str, Any]) -> str:
    """提取不同 MinerU block 中可供下游使用的主体文本。"""

    raw_type = item.get("type")
    if raw_type == "table":
        return str(item.get("table_body") or item.get("html") or item.get("text") or "")
    if raw_type == "image":
        return _join_lines(item.get("image_caption") or item.get("caption") or item.get("content"))
    if raw_type in {"equation", "interline_equation", "inline_equation", "formula"}:
        return str(item.get("latex") or item.get("text") or item.get("content") or "")
    return str(item.get("text") or item.get("content") or "")


def _block_type(item: dict[str, Any]) -> str:
    """把 MinerU 原始类型映射为项目内部的四类主模态。"""

    raw_type = str(item.get("type") or "text")
    if raw_type == "table":
        return "table"
    if raw_type == "image":
        return "image"
    if raw_type in {"equation", "interline_equation", "inline_equation", "formula"}:
        return "formula"
    return "text"


def _block_role(item: dict[str, Any]) -> str:
    """根据 MinerU 类型和 text_level 判断内容块在论文中的结构角色。"""

    raw_type = str(item.get("type") or "text")
    if item.get("text_level") is not None:
        return "heading"
    if raw_type in {"header", "footer", "page_number", "page_footnote", "ref_text"}:
        return raw_type
    if raw_type == "table":
        return "table"
    if raw_type == "image":
        return "figure"
    return "body"


def _resource_path(paper_dir: Path, item: dict[str, Any]) -> str | None:
    """解析图片或表格截图资源路径，统一输出项目相对路径。"""

    raw_path = item.get("img_path") or item.get("image_path") or item.get("path")
    if not raw_path:
        return None
    resource = paper_dir / str(raw_path)
    return _project_relative(resource)


def _normalize_block(paper_id: str, paper_dir: Path, item: dict[str, Any], order: int) -> dict[str, Any]:
    """把单个 MinerU content_list 条目归一化为内部 block 字典。"""

    text = _item_text(item)
    page_idx = item.get("page_idx")
    caption = _join_lines(item.get("image_caption") or item.get("table_caption") or item.get("caption"))
    footnote = _join_lines(item.get("image_footnote") or item.get("table_footnote") or item.get("footnote"))
    block_id = f"{paper_id}_block_{order:06d}"
    content_id = f"{paper_id}_content_{order:06d}"
    return {
        "id": block_id,
        "content_id": content_id,
        "document_id": paper_id,
        "type": _block_type(item),
        "role": _block_role(item),
        "order": order,
        "content": text,
        "text": _clean_text(text),
        "caption": caption,
        "footnote": footnote,
        "page_idx": page_idx,
        "page_num": page_idx + 1 if isinstance(page_idx, int) else None,
        "bbox": item.get("bbox"),
        "resource_path": _resource_path(paper_dir, item),
        "metadata": {
            "raw_type": item.get("type"),
            "text_level": item.get("text_level"),
            "img_path": item.get("img_path"),
            "has_table_body": bool(item.get("table_body")),
        },
    }


def _extract_meta(paper_id: str, blocks: list[dict[str, Any]], markdown_path: Path, source_pdf: str | None) -> dict[str, Any]:
    """从 MinerU 块中抽取论文标题、摘要、关键词等基础元信息。"""

    title = ""
    authors: list[str] = []
    abstract = None
    keywords: list[str] = []
    doi = None
    for index, block in enumerate(blocks):
        text = block.get("text", "")
        if not title and block.get("role") == "heading" and block.get("metadata", {}).get("text_level") == 1:
            title = text
            next_text = blocks[index + 1].get("text", "") if index + 1 < len(blocks) else ""
            if next_text and not re.search(r"摘要|Abstract|关键词|Keywords", next_text, re.IGNORECASE):
                authors = [part.strip() for part in re.split(r"[，,;；]\s*", next_text) if part.strip()]
        if abstract is None and re.match(r"^(摘要|摘\s*要)\s*[:：]", text):
            abstract = re.sub(r"^(摘要|摘\s*要)\s*[:：]\s*", "", text).strip()
        if not keywords and re.match(r"^(关键词|关\s*键\s*词)\s*[:：]", text):
            keyword_text = re.sub(r"^(关键词|关\s*键\s*词)\s*[:：]\s*", "", text).strip()
            keywords = [part.strip() for part in re.split(r"[;；,，]\s*", keyword_text) if part.strip()]
        if doi is None:
            match = re.search(r"\b10\.\s*\d{4,9}\s*/\s*[-._;()/:A-Z0-9]+", text, flags=re.IGNORECASE)
            if match:
                doi = re.sub(r"\s+", "", match.group(0))
    if not title:
        markdown_title = re.search(r"^#\s+(.+)$", markdown_path.read_text(encoding="utf-8"), flags=re.MULTILINE)
        title = _clean_text(markdown_title.group(1)) if markdown_title else paper_id
    return {
        "title": title,
        "authors": authors,
        "organizations": [],
        "year": None,
        "doi": doi,
        "keywords": keywords,
        "abstract": abstract,
        "source_path": source_pdf,
        "metadata": {},
    }


def load_paper(paper_id: str) -> dict:
    """加载单篇论文的 MinerU 输出，返回归一化后的字典。"""

    mineru_root = settings.mineru_output_dir
    paper_dir = mineru_root / paper_id
    if not paper_dir.exists():
        raise FileNotFoundError(f"MinerU 输出目录不存在: {paper_dir}")

    markdown_path = paper_dir / "full.md"
    content_list_path = paper_dir / "content_list.json"
    layout_path = paper_dir / "layout.json"
    missing = [path.name for path in (markdown_path, content_list_path, layout_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{paper_id} 缺少 MinerU 文件: {', '.join(missing)}")

    raw_markdown = markdown_path.read_text(encoding="utf-8")
    content_list = _read_json(content_list_path)
    layout = _read_json(layout_path)
    if not isinstance(content_list, list):
        raise ValueError(f"{content_list_path} 须为列表结构")

    blocks = [_normalize_block(paper_id, paper_dir, item, order) for order, item in enumerate(content_list)]
    contents = [
        {
            "id": block["content_id"],
            "document_id": paper_id,
            "section_id": None,
            "order": block["order"],
            "modality": block["type"],
            "role": block["role"],
            "content": block["content"],
            "caption": block["caption"],
            "resource_path": block["resource_path"],
            "provenance": {
                "document_id": paper_id,
                "source_file": _project_relative(content_list_path),
                "modality": block["type"],
                "raw": {"block_id": block["id"], "page_num": block["page_num"]},
            },
            "metadata": block["metadata"],
        }
        for block in blocks
    ]
    source_pdf = _find_source_pdf(paper_id, mineru_root)
    page_count = len(layout.get("pdf_info", [])) if isinstance(layout, dict) else None
    type_counter = Counter(block["type"] for block in blocks)
    return {
        "_stage": 1,
        "_description": "MinerU 解析结果归一化输出。",
        "_produced_by": "src/parser/mineru_loader.py::load_paper",
        "paper_id": paper_id,
        "document_id": paper_id,
        "source_pdf": source_pdf,
        "mineru_dir": _project_relative(paper_dir),
        "markdown_path": _project_relative(markdown_path),
        "content_list_path": _project_relative(content_list_path),
        "layout_path": _project_relative(layout_path),
        "image_dir": _project_relative(paper_dir / "images"),
        "meta": _extract_meta(paper_id, blocks, markdown_path, source_pdf),
        "blocks": blocks,
        "contents": contents,
        "raw_markdown": raw_markdown,
        "statistics": {
            "page_count": page_count,
            "block_count": len(blocks),
            "content_count": len(contents),
            "type_counts": dict(type_counter),
        },
    }
