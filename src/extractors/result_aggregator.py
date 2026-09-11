"""归类汇总第四阶段的文本、图片和表格抽取结果。

本模块不合并或去重任何实体、关系和事件，只将各模态的 Graph 按第三阶段
文档摘要中的章节与 Chunk 顺序排列，并把原始 Chunk 内容补充到 Graph.metadata。
同时为每个实体补充 ``metadata.source_modality``，以保留文本、表格、图片或
公式的来源信息。
输出继续使用 ``stage_04_text_extraction.json`` 的顶层结构：
``_status``、``statistics`` 和 ``graphs``。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# 中文说明：支持直接执行本文件，将项目根目录加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.json_io import read_json, write_json


DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "output" / "stage_03_document_summary.json"
DEFAULT_TEXT_PATH = PROJECT_ROOT / "output" / "stage_04_text_extraction.json"
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "output" / "stage_04_image_extraction.json"
DEFAULT_TABLE_PATH = PROJECT_ROOT / "output" / "stage_04_table_extraction.json"
DEFAULT_FORMULA_PATH = PROJECT_ROOT / "output" / "stage_04_formula_extraction.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "stage_04_merged_extraction.json"

MODALITY_METADATA_FIELDS: dict[str, tuple[str, ...]] = {
    "text": ("text",),
    "image": ("image_path", "caption", "references"),
    "table": ("markdown", "caption", "table_path", "references"),
    "formula": ("latex", "caption"),
}


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """校验值为 JSON 对象，并在输入结构异常时给出明确位置。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 JSON 对象，实际为 {type(value).__name__}")
    return value


def _as_sequence(value: Any, label: str) -> Sequence[Any]:
    """校验值为 JSON 数组，排除字符串等伪序列。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} 必须是 JSON 数组，实际为 {type(value).__name__}")
    return value


def _order_key(item: Mapping[str, Any], original_position: int) -> tuple[int, float | str, int]:
    """生成稳定排序键；数字 order 优先，缺失或非数字值保持可重复排序。"""

    value = item.get("order")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value), original_position)
    if isinstance(value, str):
        try:
            return (0, float(value.strip()), original_position)
        except ValueError:
            return (1, value, original_position)
    return (2, "", original_position)


def collect_ordered_chunks(summary_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """按章节树和各章节 Chunk 的 ``order`` 递归收集全部 Chunk。

    同级章节先按章节 ``order`` 排序；进入章节后，先输出该章节的 Chunk，再按
    ``children`` 的 ``order`` 深度优先访问子章节。相同 ``order`` 时保留原数组顺序。
    """

    document = _as_mapping(summary_payload.get("document", summary_payload), "document")
    result: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    def visit_sections(raw_sections: Any, label: str) -> None:
        sections = _as_sequence(raw_sections or [], label)
        ordered_sections = sorted(
            enumerate(sections),
            key=lambda pair: _order_key(_as_mapping(pair[1], f"{label}[{pair[0]}]"), pair[0]),
        )
        for section_position, raw_section in ordered_sections:
            section_label = f"{label}[{section_position}]"
            section = _as_mapping(raw_section, section_label)
            raw_chunks = _as_sequence(section.get("chunks") or [], f"{section_label}.chunks")
            ordered_chunks = sorted(
                enumerate(raw_chunks),
                key=lambda pair: _order_key(
                    _as_mapping(pair[1], f"{section_label}.chunks[{pair[0]}]"), pair[0]
                ),
            )
            for chunk_position, raw_chunk in ordered_chunks:
                chunk = dict(
                    _as_mapping(raw_chunk, f"{section_label}.chunks[{chunk_position}]")
                )
                chunk_id = str(chunk.get("id") or "").strip()
                if not chunk_id:
                    raise ValueError(f"{section_label}.chunks[{chunk_position}] 缺少非空 id")
                if chunk_id in seen_chunk_ids:
                    raise ValueError(f"第三阶段摘要中存在重复 chunk id：{chunk_id}")
                seen_chunk_ids.add(chunk_id)
                result.append(chunk)
            visit_sections(section.get("children") or [], f"{section_label}.children")

    visit_sections(document.get("sections") or [], "document.sections")
    return result


def _read_extraction_payload(path: str | Path, modality: str) -> dict[str, Any]:
    """读取并校验单个模态抽取结果的必要顶层字段。"""

    payload = dict(_as_mapping(read_json(path), f"{modality} 抽取结果"))
    _as_sequence(payload.get("graphs"), f"{modality} 抽取结果.graphs")
    return payload


def _enrich_graph(graph: Mapping[str, Any], chunk: Mapping[str, Any]) -> dict[str, Any]:
    """复制 Graph，回填 Chunk 元数据，并标记每个实体的来源模态。"""

    result = deepcopy(dict(graph))
    metadata = dict(_as_mapping(result.get("metadata"), "graph.metadata"))
    chunk_id = str(chunk["id"])
    modality = str(chunk.get("modality") or metadata.get("modality") or "").strip().lower()
    if modality not in MODALITY_METADATA_FIELDS:
        raise ValueError(f"chunk {chunk_id} 的模态不受支持：{modality or '<empty>'}")

    metadata["chunk_id"] = chunk_id
    metadata["modality"] = modality
    for field in MODALITY_METADATA_FIELDS[modality]:
        # 字段即使为 null 或空数组也原样写入，确保输出元数据结构稳定。
        metadata[field] = deepcopy(chunk.get(field))
    result["metadata"] = metadata

    entities = _as_sequence(result.get("entities") or [], "graph.entities")
    enriched_entities: list[dict[str, Any]] = []
    for entity_index, raw_entity in enumerate(entities):
        entity = deepcopy(dict(_as_mapping(raw_entity, f"graph.entities[{entity_index}]")))
        raw_entity_metadata = entity.get("metadata")
        entity_metadata = (
            dict(_as_mapping(raw_entity_metadata, f"graph.entities[{entity_index}].metadata"))
            if raw_entity_metadata is not None
            else {}
        )
        # 合并阶段以所属 Chunk 模态为准，覆盖抽取阶段可能遗留的不一致来源标识。
        entity_metadata["source_modality"] = modality
        entity["metadata"] = entity_metadata
        enriched_entities.append(entity)
    result["entities"] = enriched_entities
    return result


def _build_statistics(graphs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """依据未合并的 Graph 列表重新计算汇总统计。"""

    modality_counts = {modality: 0 for modality in MODALITY_METADATA_FIELDS}
    entity_count = 0
    relation_count = 0
    event_count = 0
    for graph in graphs:
        metadata = _as_mapping(graph.get("metadata"), "graph.metadata")
        modality = str(metadata.get("modality") or "").strip().lower()
        if modality in modality_counts:
            modality_counts[modality] += 1
        entity_count += len(_as_sequence(graph.get("entities") or [], "graph.entities"))
        relation_count += len(_as_sequence(graph.get("relations") or [], "graph.relations"))
        event_count += len(_as_sequence(graph.get("events") or [], "graph.events"))

    return {
        "graph_count": len(graphs),
        "completed_text_chunk_count": modality_counts["text"],
        "completed_image_chunk_count": modality_counts["image"],
        "completed_table_chunk_count": modality_counts["table"],
        "completed_formula_chunk_count": modality_counts["formula"],
        "entity_count": entity_count,
        "relation_count": relation_count,
        "event_count": event_count,
    }


def aggregate_extraction_results(
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    text_path: str | Path = DEFAULT_TEXT_PATH,
    image_path: str | Path = DEFAULT_IMAGE_PATH,
    table_path: str | Path = DEFAULT_TABLE_PATH,
    formula_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """汇总各模态结果、补充 Chunk 元数据并写出排序后的 JSON。

    每个输入 Graph 都完整保留，不执行节点、关系或事件合并。若抽取结果包含第三阶段
    摘要中不存在的 ``chunk_id``，函数会中止，避免产生无法排序和溯源的数据。公式
    结果文件为可选参数，避免影响当前只生成文本、图片、表格结果的调用方。
    """

    summary_payload = _as_mapping(read_json(summary_path), "第三阶段文档摘要")
    ordered_chunks = collect_ordered_chunks(summary_payload)
    chunks_by_id = {str(chunk["id"]): chunk for chunk in ordered_chunks}
    chunk_positions = {chunk_id: position for position, chunk_id in enumerate(chunks_by_id)}

    text_payload = _read_extraction_payload(text_path, "text")
    extraction_payloads = (
        ("text", text_payload),
        ("image", _read_extraction_payload(image_path, "image")),
        ("table", _read_extraction_payload(table_path, "table")),
    )
    if formula_path is not None:
        extraction_payloads += (("formula", _read_extraction_payload(formula_path, "formula")),)

    enriched_graphs: list[tuple[int, int, dict[str, Any]]] = []
    input_position = 0
    for expected_modality, payload in extraction_payloads:
        for raw_graph in _as_sequence(payload["graphs"], f"{expected_modality} 抽取结果.graphs"):
            graph = _as_mapping(raw_graph, f"{expected_modality} 抽取结果.graphs[]")
            metadata = _as_mapping(graph.get("metadata"), "graph.metadata")
            chunk_id = str(metadata.get("chunk_id") or "").strip()
            if not chunk_id:
                raise ValueError(f"{expected_modality} 抽取结果中存在缺少 metadata.chunk_id 的 Graph")
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                raise ValueError(f"抽取结果 chunk_id 不存在于第三阶段摘要：{chunk_id}")
            chunk_modality = str(chunk.get("modality") or "").strip().lower()
            if chunk_modality != expected_modality:
                raise ValueError(
                    f"chunk {chunk_id} 模态不一致：摘要为 {chunk_modality!r}，"
                    f"Graph 来自 {expected_modality!r} 文件"
                )
            enriched_graphs.append(
                (chunk_positions[chunk_id], input_position, _enrich_graph(graph, chunk))
            )
            input_position += 1

    # 同一 Chunk 意外产生多个 Graph 时不合并，仍保持输入中的稳定先后顺序。
    enriched_graphs.sort(key=lambda item: (item[0], item[1]))
    graphs = [item[2] for item in enriched_graphs]

    # 复制文本结果的顶层对象，确保输出与文本抽取结果保持同一结构约定。
    result = deepcopy(text_payload)
    result["_status"] = "completed"
    result["statistics"] = _build_statistics(graphs)
    result["graphs"] = graphs
    write_json(output_path, result)
    return result


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="归类汇总第四阶段文本、图片、表格和公式 Graph")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH, help="第三阶段文档摘要 JSON")
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT_PATH, help="第四阶段文本抽取 JSON")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="第四阶段图片抽取 JSON")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE_PATH, help="第四阶段表格抽取 JSON")
    parser.add_argument("--formula", type=Path, help="可选的第四阶段公式抽取 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="汇总结果 JSON")
    args = parser.parse_args()

    result = aggregate_extraction_results(
        summary_path=args.summary,
        text_path=args.text,
        image_path=args.image,
        table_path=args.table,
        formula_path=args.formula,
        output_path=args.output,
    )
    statistics = result["statistics"]
    print(
        "抽取结果汇总完成："
        f"Graph={statistics['graph_count']}，"
        f"文本={statistics['completed_text_chunk_count']}，"
        f"图片={statistics['completed_image_chunk_count']}，"
        f"表格={statistics['completed_table_chunk_count']}，"
        f"公式={statistics['completed_formula_chunk_count']}，"
        f"输出文件：{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_IMAGE_PATH",
    "DEFAULT_FORMULA_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_SUMMARY_PATH",
    "DEFAULT_TABLE_PATH",
    "DEFAULT_TEXT_PATH",
    "MODALITY_METADATA_FIELDS",
    "aggregate_extraction_results",
    "collect_ordered_chunks",
    "main",
]
