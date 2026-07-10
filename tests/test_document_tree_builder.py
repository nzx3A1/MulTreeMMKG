"""第二阶段文档目录树构建器的集成测试。"""
from pathlib import Path

from model.chunk import ImageChunk, TextChunk
from src.skeleton.document_tree_builder import document_tree_to_dict, load_stage01_document


def test_load_stage01_document_builds_hierarchy_and_indexes() -> None:
    """第一阶段 JSON 应被装载为可按章节和 Chunk ID 查询的对象树。"""

    project_root = Path(__file__).resolve().parents[1]
    document = load_stage01_document(project_root / "output" / "stage_01_mineru_parse.json")

    section = document.get_section("2")
    subsection = document.get_section("2.1")

    assert document.title
    assert section is not None
    assert subsection is not None
    assert subsection.parent_id == section.id
    assert any(isinstance(chunk, TextChunk) for chunk in document.iter_chunks())
    assert any(isinstance(chunk, ImageChunk) for chunk in document.iter_chunks())
    assert document.get_chunk("2.1:text:0") is not None
    assert document.get_chunk_section("2.1:text:0") is subsection
    assert document_tree_to_dict(document)["statistics"]["section_count"] == sum(1 for _ in document.iter_sections())
