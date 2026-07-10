"""第二阶段文档目录树构建器。

将第一阶段 MinerU 解析 JSON 适配为 ``Document -> DocumentSection -> Chunk``
对象树。构建结束时会创建内存索引，供下一阶段按章节或 Chunk ID 快速定位来源。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

# 允许通过 ``python src/skeleton/document_tree_builder.py`` 直接运行本文件。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.chunk import FormulaChunk, ImageChunk, TableChunk, TextChunk
from model.document import Document, DocumentMetadata, DocumentSection


DEFAULT_TEXT_CHUNK_MAX_LENGTH = 500
"""正文文本 Chunk 的默认最大字符数；超长单段会保持完整，不会被强制截断。"""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """将 MinerU 的资源条目规范为映射，兼容字符串形式的表格或公式。"""

    if isinstance(value, Mapping):
        return value
    return {"content": str(value)}


def _as_items(value: Any) -> list[Any]:
    """将可空的单值或序列字段规范为列表，避免字符串被拆成字符。"""

    if value is None:
        return []
    if isinstance(value, (str, Mapping)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _as_string_list(value: Any) -> list[str]:
    """将可空的单值或序列字段规范为字符串列表。"""

    return [str(item) for item in _as_items(value) if item is not None]


def _document_id(stage01: Mapping[str, Any], source_json: Optional[Path]) -> str:
    """优先复用已有 ID，否则依据 MinerU Markdown 所在目录生成稳定文档 ID。"""

    existing_id = stage01.get("document_id") or stage01.get("paper_id")
    if existing_id:
        return str(existing_id)
    input_file = stage01.get("input_file")
    if input_file:
        return Path(str(input_file)).parent.name
    if source_json is not None:
        return source_json.stem
    return "document"


def _resolve_image_paths(paths: list[str], source_file: Optional[str]) -> list[str]:
    """将相对图片路径解析为相对原始 Markdown 文件的绝对路径。"""

    if not source_file:
        return paths
    base_dir = Path(source_file).parent
    return [str((base_dir / path).resolve()) if not Path(path).is_absolute() else path for path in paths]


def _split_text_by_paragraph(content: str, max_length: int) -> list[str]:
    """按换行段落合并文本，普通 Chunk 不超过上限，超长单段独立保留。"""

    if max_length <= 0:
        raise ValueError("文本 Chunk 长度上限必须大于 0")

    text_chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_length = 0
    for paragraph in (item.strip() for item in content.splitlines()):
        if not paragraph:
            continue
        separator_length = 1 if current_paragraphs else 0
        if current_paragraphs and current_length + separator_length + len(paragraph) > max_length:
            text_chunks.append("\n".join(current_paragraphs))
            current_paragraphs = []
            current_length = 0
        # 单个段落超过上限时独占一个 Chunk，保证段落语义不会被截断。
        if len(paragraph) > max_length:
            text_chunks.append(paragraph)
            continue
        current_paragraphs.append(paragraph)
        current_length += (1 if current_length else 0) + len(paragraph)
    if current_paragraphs:
        text_chunks.append("\n".join(current_paragraphs))
    return text_chunks


def _build_chunks(
    section_id: str,
    raw_section: Mapping[str, Any],
    source_file: Optional[str],
    max_text_chunk_length: int,
) -> list[Any]:
    """按正文、图片、表格、公式的顺序，将一个原始章节转换为多模态 Chunk 对象。"""

    chunks: list[Any] = []
    content = str(raw_section.get("content") or "").strip()
    for index, text in enumerate(_split_text_by_paragraph(content, max_text_chunk_length)):
        chunks.append(TextChunk(id=f"{section_id}:text:{index}", order=len(chunks), text=text))

    for index, raw_image in enumerate(_as_items(raw_section.get("images"))):
        image = _as_mapping(raw_image)
        paths = _resolve_image_paths(_as_string_list(image.get("path")), source_file)
        chunks.append(ImageChunk(
            id=f"{section_id}:image:{index}",
            order=len(chunks),
            image_path=paths,
            caption=image.get("caption"),
            references=_as_string_list(image.get("references")) or None,
        ))

    tables = raw_section.get("table") or raw_section.get("tables") or []
    for index, raw_table in enumerate(_as_items(tables)):
        table = _as_mapping(raw_table)
        markdown = str(table.get("markdown") or table.get("content") or table.get("table") or "")
        chunks.append(TableChunk(
            id=f"{section_id}:table:{index}",
            order=len(chunks),
            markdown=markdown,
            caption=table.get("caption"),
            references=_as_string_list(table.get("references")) or None,
        ))

    for index, raw_formula in enumerate(_as_items(raw_section.get("formulas"))):
        formula = _as_mapping(raw_formula)
        latex = str(formula.get("latex") or formula.get("content") or formula.get("formula") or "")
        chunks.append(FormulaChunk(
            id=f"{section_id}:formula:{index}",
            order=len(chunks),
            latex=latex,
            caption=formula.get("caption"),
        ))
    return chunks


def _build_section(
    raw_section: Mapping[str, Any],
    parent_id: Optional[str],
    order: int,
    source_file: Optional[str],
    max_text_chunk_length: int,
) -> DocumentSection:
    """递归构建一个章节对象，并把原始子章节挂载到 ``children`` 字段。"""

    section_id = str(raw_section.get("id") or f"section-{order}")
    section = DocumentSection(
        id=section_id,
        title=str(raw_section.get("title") or ""),
        level=max(1, int(raw_section.get("level") or 1)),
        order=order,
        parent_id=parent_id,
        chunks=_build_chunks(section_id, raw_section, source_file, max_text_chunk_length),
    )
    section.children = [
        _build_section(_as_mapping(child), section.id, child_order, source_file, max_text_chunk_length)
        for child_order, child in enumerate(_as_items(raw_section.get("children")))
    ]
    return section


def build_document_tree(
    stage01: Mapping[str, Any],
    source_json: Optional[Path] = None,
    max_text_chunk_length: int = DEFAULT_TEXT_CHUNK_MAX_LENGTH,
) -> Document:
    """将一篇第一阶段结果转换为带章节/Chunk 索引的 ``Document`` 内存对象。"""

    # 兼容未来批量输出的 ``papers`` 包装以及当前直接包含 ``toc`` 的阶段一 JSON。
    if not stage01.get("toc") and isinstance(stage01.get("papers"), Sequence):
        papers = stage01["papers"]
        if len(papers) != 1:
            raise ValueError("build_document_tree 一次只能构建一篇文档，请先从 papers 中选择目标文档。")
        stage01 = _as_mapping(papers[0])

    source_file = str(stage01.get("input_file")) if stage01.get("input_file") else None
    basic_info = _as_mapping(stage01.get("basicInformation"))
    document = Document(
        id=_document_id(stage01, source_json),
        title=str(stage01.get("title") or basic_info.get("title") or "未命名文档"),
        metadata=DocumentMetadata(
            abstract=basic_info.get("abstract"),
            authors=_as_string_list(basic_info.get("authors")),
            keywords=_as_string_list(basic_info.get("keywords")),
            publish_organization=basic_info.get("publish_organization"),
            references=basic_info.get("references"),
        ),
        source_file=source_file,
        sections=[
            _build_section(_as_mapping(raw_section), None, order, source_file, max_text_chunk_length)
            for order, raw_section in enumerate(_as_items(stage01.get("toc")))
        ],
    )
    document.rebuild_indexes()
    return document


def load_stage01_document(
    path: str | Path,
    max_text_chunk_length: int = DEFAULT_TEXT_CHUNK_MAX_LENGTH,
) -> Document:
    """读取第一阶段 JSON，并返回已完成依赖关系装载的文档对象。"""

    source_json = Path(path)
    with source_json.open("r", encoding="utf-8") as file:
        stage01 = json.load(file)
    if not isinstance(stage01, Mapping):
        raise ValueError(f"第一阶段 JSON 根节点必须是对象：{source_json}")
    return build_document_tree(stage01, source_json=source_json, max_text_chunk_length=max_text_chunk_length)


def document_tree_to_dict(document: Document) -> dict[str, Any]:
    """将内存目录树序列化为第二阶段 JSON，同时附带统计信息。"""

    return {
        "_stage": 2,
        "_description": "Document -> Section -> Chunk 的多模态文档目录树。",
        "_produced_by": "src/skeleton/document_tree_builder.py::build_document_tree",
        "document": document.to_dict(),
        "statistics": {
            "section_count": sum(1 for _ in document.iter_sections()),
            "chunk_count": sum(1 for _ in document.iter_chunks()),
        },
    }


def write_document_tree(document: Document, path: str | Path) -> None:
    """将第二阶段目录树结果写入 JSON 文件，便于检查或断点续跑。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(document_tree_to_dict(document), file, ensure_ascii=False, indent=2)


def main() -> None:
    """提供命令行入口，直接将阶段一 JSON 构建为阶段二文档目录树 JSON。"""

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="构建 Document -> Section -> Chunk 文档目录树")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "output" / "stage_01_mineru_parse.json",
        help="第一阶段 MinerU 解析 JSON 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "output" / "stage_02_document_tree.json",
        help="第二阶段文档目录树 JSON 输出路径",
    )
    parser.add_argument(
        "--max-text-chunk-length",
        type=int,
        default=DEFAULT_TEXT_CHUNK_MAX_LENGTH,
        help="文本 Chunk 最大字符数；超过上限的单个段落保持完整",
    )
    args = parser.parse_args()

    document = load_stage01_document(args.input, max_text_chunk_length=args.max_text_chunk_length)
    write_document_tree(document, args.output)
    statistics = document_tree_to_dict(document)["statistics"]
    print(f"阶段二构建完成：{statistics['section_count']} 个章节，{statistics['chunk_count']} 个 Chunk。")
    print(f"输出文件：{args.output.resolve()}")


if __name__ == "__main__":
    main()
