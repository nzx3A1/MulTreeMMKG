"""src/parser 与 src/skeleton 阶段二模块的单元测试。"""
from __future__ import annotations

from src.parser.markdown_parser import parse_markdown
from src.parser.mineru_loader import load_paper
from src.parser.section_splitter import split_sections
from src.skeleton.chunk_builder import build_modal_chunks
from src.skeleton.document_skeleton_builder import build_skeleton


def test_mineru_loader_returns_normalized_doc():
    """MinerU 输出能被归一化为内部统一结构。"""

    paper = load_paper("paper_001")

    assert paper["paper_id"] == "paper_001"
    assert paper["meta"]["title"]
    assert paper["blocks"]
    assert paper["statistics"]["type_counts"]["text"] > 0
    assert any(block["type"] == "image" for block in paper["blocks"])
    assert any(block["type"] == "table" for block in paper["blocks"])


def test_parse_markdown_keeps_core_blocks():
    """Markdown 解析器应保留标题、图片和 HTML 表格块。"""

    blocks = parse_markdown("# 标题\n\n正文\n\n![](images/a.jpg)\n\n<table><tr><td>A</td></tr></table>")

    assert any(block["type"] == "heading" and block["text"] == "标题" for block in blocks)
    assert any(block["type"] == "image" for block in blocks)
    assert any(block["type"] == "table" for block in blocks)


def test_stage_two_builds_sections_skeleton_and_chunks():
    """阶段二主链路应能生成章节树、骨架图和多模态 chunk。"""

    paper = load_paper("paper_001")
    section_tree = split_sections(paper)
    skeleton = build_skeleton(section_tree)
    chunks = build_modal_chunks(skeleton, paper)

    assert section_tree["sections"]
    assert section_tree["statistics"]["section_count"] > 0
    assert any(section["title"] for section in section_tree["flat_sections"])
    assert skeleton["nodes"]
    assert skeleton["edges"]
    assert chunks
    assert {chunk["modality"] for chunk in chunks} >= {"text", "image", "table"}
