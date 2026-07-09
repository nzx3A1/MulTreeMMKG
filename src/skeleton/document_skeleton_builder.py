"""文档骨架图构建器。

基于章节树构建“文档骨架图”：节点包括 Paper / Section / SubSection /
Chunk / Figure / Table / Formula，边表示包含与阅读顺序关系。

输出：stage_03_document_skeleton_graph.json
对应阶段：03 文档骨架图
"""
from __future__ import annotations

from typing import Any


def _node_type_for_block(block: dict[str, Any]) -> str:
    """根据内容块模态选择骨架节点类型。"""

    block_type = block.get("type")
    if block_type == "image":
        return "Figure"
    if block_type == "table":
        return "Table"
    if block_type == "formula":
        return "Formula"
    return "Chunk"


def _edge(edge_type: str, source_id: str, target_id: str, order: int, **properties: Any) -> dict[str, Any]:
    """创建稳定的骨架边 ID 和属性。"""

    return {
        "id": f"{source_id}__{edge_type.lower()}__{target_id}",
        "type": edge_type,
        "source_id": source_id,
        "target_id": target_id,
        "properties": {"order": order, **properties},
    }


def _walk_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """深度优先展开章节，保持论文阅读顺序。"""

    rows: list[dict[str, Any]] = []
    for section in sections:
        rows.append(section)
        rows.extend(_walk_sections(section.get("children", [])))
    return rows


def build_skeleton(section_tree: dict) -> dict:
    """从章节树构建文档骨架图（节点 + 边列表）。"""

    document_id = str(section_tree.get("document_id") or section_tree.get("paper_id") or "document")
    nodes: list[dict[str, Any]] = [{
        "id": document_id,
        "type": "Paper",
        "title": section_tree.get("title") or document_id,
        "properties": {
            "document_id": document_id,
            "section_count": section_tree.get("statistics", {}).get("section_count", 0),
        },
    }]
    edges: list[dict[str, Any]] = []

    def add_section(section: dict[str, Any], parent_id: str, sibling_order: int) -> None:
        """递归加入章节节点、内容节点和结构关系。"""

        section_type = "Section" if section.get("level", 1) == 1 else "SubSection"
        nodes.append({
            "id": section["id"],
            "type": section_type,
            "title": section.get("title", ""),
            "properties": {
                "document_id": document_id,
                "level": section.get("level"),
                "order": section.get("order"),
                "number": section.get("number"),
                "parent_id": section.get("parent_id"),
                "page_num": section.get("metadata", {}).get("page_num"),
            },
        })
        edges.append(_edge("HAS_SECTION", parent_id, section["id"], sibling_order))

        previous_block_id: str | None = None
        for block_order, block in enumerate(section.get("blocks", [])):
            block_id = block.get("content_id") or block.get("id")
            if not block_id:
                continue
            nodes.append({
                "id": block_id,
                "type": _node_type_for_block(block),
                "title": block.get("caption") or block.get("text", "")[:40],
                "properties": {
                    "document_id": document_id,
                    "section_id": section["id"],
                    "block_id": block.get("id"),
                    "modality": block.get("type"),
                    "role": block.get("role"),
                    "order": block_order,
                    "page_num": block.get("page_num"),
                    "resource_path": block.get("resource_path"),
                    "content_preview": (block.get("text") or block.get("content") or "")[:200],
                },
            })
            edges.append(_edge("HAS_CHUNK", section["id"], block_id, block_order, block_id=block.get("id")))
            if previous_block_id:
                edges.append(_edge("NEXT", previous_block_id, block_id, block_order))
            previous_block_id = block_id

        previous_child_id: str | None = None
        for child_order, child in enumerate(section.get("children", [])):
            add_section(child, section["id"], child_order)
            if previous_child_id:
                edges.append(_edge("NEXT", previous_child_id, child["id"], child_order))
            previous_child_id = child["id"]

    previous_root_id: str | None = None
    for root_order, section in enumerate(section_tree.get("sections", [])):
        add_section(section, document_id, root_order)
        if previous_root_id:
            edges.append(_edge("NEXT", previous_root_id, section["id"], root_order))
        previous_root_id = section["id"]

    flat_sections = section_tree.get("flat_sections") or _walk_sections(section_tree.get("sections", []))
    return {
        "_stage": 3,
        "_description": "文档骨架图：Paper / Section / Subsection / Chunk / Figure / Table / Formula 节点及结构关系。",
        "_produced_by": "src/skeleton/document_skeleton_builder.py::build_skeleton",
        "paper_id": document_id,
        "document_id": document_id,
        "nodes": nodes,
        "edges": edges,
        "sections": section_tree.get("sections", []),
        "flat_sections": flat_sections,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }
