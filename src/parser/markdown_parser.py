"""Markdown 解析器。

基于 markdown-it-py 把 full.md 解析为 token 流 / AST，
保留段落、标题、列表、代码块、公式、表格、图片引用等元素位置信息，
为后续章节切分提供结构化输入。
"""
from __future__ import annotations

from typing import Any

from markdown_it import MarkdownIt


def _inline_content(token: Any) -> str:
    """提取 inline token 的可读文本，兼容图片 alt 文本。"""

    if not getattr(token, "children", None):
        return token.content or ""
    parts: list[str] = []
    for child in token.children:
        if child.type == "image":
            parts.append(child.content or "")
        else:
            parts.append(child.content or "")
    return "".join(parts).strip()


def _inline_images(token: Any) -> list[dict[str, str]]:
    """从 inline token 中抽取 Markdown 图片引用。"""

    images: list[dict[str, str]] = []
    for child in getattr(token, "children", None) or []:
        if child.type != "image":
            continue
        attrs = dict(child.attrs or [])
        images.append({"src": attrs.get("src", ""), "alt": child.content or ""})
    return images


def _line_range(token: Any) -> dict[str, int | None]:
    """把 markdown-it 的 0 基 line map 转为便于阅读的 1 基行号。"""

    if not token.map:
        return {"line_start": None, "line_end": None}
    return {"line_start": token.map[0] + 1, "line_end": token.map[1]}


def parse_markdown(md_text: str) -> list[dict]:
    """将 Markdown 文本解析为带元信息的 block 列表。"""

    parser = MarkdownIt("commonmark", {"html": True})
    tokens = parser.parse(md_text or "")
    blocks: list[dict[str, Any]] = []
    index = 0
    order = 0

    while index < len(tokens):
        token = tokens[index]
        if token.type == "heading_open":
            inline = tokens[index + 1] if index + 1 < len(tokens) else None
            level = int(token.tag[1:]) if token.tag.startswith("h") and token.tag[1:].isdigit() else 1
            text = _inline_content(inline) if inline else ""
            blocks.append({
                "id": f"md_block_{order:06d}",
                "type": "heading",
                "role": "heading",
                "order": order,
                "level": level,
                "text": text,
                "content": text,
                **_line_range(token),
            })
            order += 1
            index += 3
            continue

        if token.type == "paragraph_open":
            inline = tokens[index + 1] if index + 1 < len(tokens) else None
            text = _inline_content(inline) if inline else ""
            images = _inline_images(inline) if inline else []
            block_type = "image" if images and not text else "paragraph"
            blocks.append({
                "id": f"md_block_{order:06d}",
                "type": block_type,
                "role": "body",
                "order": order,
                "text": text,
                "content": text,
                "images": images,
                **_line_range(token),
            })
            order += 1
            index += 3
            continue

        if token.type == "html_block":
            content = token.content.strip()
            block_type = "table" if content.lower().startswith("<table") else "html"
            blocks.append({
                "id": f"md_block_{order:06d}",
                "type": block_type,
                "role": "body",
                "order": order,
                "text": content,
                "content": content,
                **_line_range(token),
            })
            order += 1
            index += 1
            continue

        if token.type in {"fence", "code_block"}:
            blocks.append({
                "id": f"md_block_{order:06d}",
                "type": "code",
                "role": "body",
                "order": order,
                "text": token.content,
                "content": token.content,
                "info": token.info,
                **_line_range(token),
            })
            order += 1
            index += 1
            continue

        index += 1

    return blocks
