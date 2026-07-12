"""第四阶段多模态抽取器统一调度入口。

读取 ``stage_03_document_summary.json`` 中嵌套章节里的 Chunk，按模态调用同级目录
中的文本、表格、图片和公式抽取器，分别保存结果后再合并为统一 Graph。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from model import Graph
from src.utils.json_io import read_json, write_json
from src.utils.llm_client import LLMClient
from src.utils.vlm_client import VLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "stage_03_document_summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
SUPPORTED_MODALITIES = ("text", "table", "image", "formula")


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """校验输入为映射类型，并提供包含数据位置的清晰错误信息。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 JSON 对象，实际为 {type(value).__name__}")
    return value


def _modality(chunk: Mapping[str, Any]) -> str:
    """兼容 modality、chunk_type 和 type 字段，规范化 Chunk 模态名称。"""

    value = chunk.get("modality") or chunk.get("chunk_type") or chunk.get("type") or ""
    modality = str(getattr(value, "value", value)).strip().lower()
    aliases = {"textchunk": "text", "tablechunk": "table", "imagechunk": "image", "formulachunk": "formula"}
    return aliases.get(modality.replace("_", ""), modality)


def collect_chunks(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """递归收集文档章节 Chunk，并补充文档和章节溯源字段。"""

    document = _as_mapping(payload.get("document", payload), "document")
    document_id = str(document.get("id") or document.get("document_id") or "")
    source_file = document.get("source_file")
    document_summary = str(document.get("summary") or "")
    document_schema_keys = list(document.get("schemaKeys") or [])
    result: list[dict[str, Any]] = []

    def visit(section: Mapping[str, Any], inherited_title: str = "") -> None:
        """深度优先遍历单个章节，将章节上下文复制到其 Chunk。"""

        section_id = str(section.get("id") or section.get("section_id") or "")
        section_title = str(section.get("title") or inherited_title or "")
        section_summary = str(section.get("summary") or "")
        section_schema_keys = list(section.get("schemaKeys") or [])
        for raw_chunk in section.get("chunks") or []:
            chunk = dict(_as_mapping(raw_chunk, f"section[{section_id}].chunks[]"))
            chunk.setdefault("document_id", document_id)
            chunk.setdefault("section_id", section_id)
            chunk.setdefault("section_title", section_title)
            chunk.setdefault("document_summary", document_summary)
            chunk.setdefault("document_schema_keys", document_schema_keys)
            chunk.setdefault("section_summary", section_summary)
            chunk.setdefault("schemaKeys", section_schema_keys)
            if source_file is not None:
                chunk.setdefault("source_file", source_file)
            result.append(chunk)
        for child in section.get("children") or []:
            visit(_as_mapping(child, f"section[{section_id}].children[]"), section_title)

    for section in document.get("sections") or []:
        visit(_as_mapping(section, "document.sections[]"))
    return result


def _graph_dict(graph: Graph) -> dict[str, Any]:
    """把 Graph 模型转换为可直接写入 JSON 的普通字典。"""

    return graph.to_dict() if hasattr(graph, "to_dict") else graph.dict()


def _modality_payload(modality: str, chunks: Sequence[Mapping[str, Any]], graphs: Sequence[Graph]) -> dict[str, Any]:
    """构造单一模态的第四阶段输出及基础统计信息。"""

    return {
        "_stage": 4,
        "_modality": modality,
        "statistics": {
            f"{modality}_chunk_count": len(chunks),
            "graph_count": len(graphs),
            "entity_count": sum(len(graph.entities) for graph in graphs),
            "relation_count": sum(len(graph.relations) for graph in graphs),
            "event_count": sum(len(graph.events) for graph in graphs),
        },
        "graphs": [_graph_dict(graph) for graph in graphs],
    }


class InitExtractor:
    """持有模型客户端与 Schema 选择器，并统一调度四种模态抽取器。"""

    def __init__(
        self,
        llm_client: Any | None = None,
        vlm_client: Any | None = None,
        show_progress: bool = True,
    ) -> None:
        """初始化可注入的客户端；默认客户端均为延迟连接，不会在此处发起请求。"""

        self.llm_client = llm_client or LLMClient()
        self.vlm_client = vlm_client or VLMClient()
        self.show_progress = show_progress
###########################################################核心####################################################
    def extract(self, chunks: Sequence[Mapping[str, Any]]) -> dict[str, list[Graph]]:
        """按文本、表格、公式、图片的固定顺序调用同级抽取器。"""

        grouped: MutableMapping[str, list[Mapping[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            modality = _modality(chunk)
            if modality in SUPPORTED_MODALITIES:
                grouped[modality].append(chunk)

        results: dict[str, list[Graph]] = {}

        # 第一阶段：优先抽取正文文本，为后续多模态结果提供文本语义基础。
        text_chunks = grouped["text"]
        if self.show_progress:
            print(f"正在抽取 text Chunk：{len(text_chunks)} 个")
        if text_chunks:
            from .text_extractor import extract_from_text

            results["text"] = extract_from_text(
                text_chunks,
                self.llm_client,
            )
        else:
            results["text"] = []

        # 第二阶段：抽取表格中的结构化实体、属性与关系。
        table_chunks = grouped["table"]
        if self.show_progress:
            print(f"正在抽取 table Chunk：{len(table_chunks)} 个")
        if table_chunks:
            from .table_extractor import extract_from_tables

            results["table"] = extract_from_tables(
                table_chunks,
                self.llm_client,
            )
        else:
            results["table"] = []

        # 第三阶段：抽取公式、变量、参数及其领域关系。
        formula_chunks = grouped["formula"]
        if self.show_progress:
            print(f"正在抽取 formula Chunk：{len(formula_chunks)} 个")
        if formula_chunks:
            from .formula_extractor import extract_from_formulas

            results["formula"] = extract_from_formulas(
                formula_chunks,
                self.llm_client,
            )
        else:
            results["formula"] = []

        # 第四阶段：最后调用视觉模型抽取图片内容，避免提前占用较重的视觉请求资源。
        image_chunks = grouped["image"]
        if self.show_progress:
            print(f"正在抽取 image Chunk：{len(image_chunks)} 个")
        if image_chunks:
            from .image_extractor import extract_from_images

            results["image"] = extract_from_images(
                image_chunks,
                self.llm_client,
                self.vlm_client,
            )
        else:
            results["image"] = []

        return results


def extract_document_multimodal(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    llm_client: Any | None = None,
    vlm_client: Any | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """执行完整第四阶段流程，写出四路结果与最终多模态合并 Graph。"""

    source = _as_mapping(read_json(input_path), "第三阶段输入")
    chunks = collect_chunks(source)
    document = _as_mapping(source.get("document", source), "document")
    document_id = str(document.get("id") or document.get("document_id") or "") or None
    runner = InitExtractor(llm_client, vlm_client, show_progress)
    grouped_chunks = {modality: [chunk for chunk in chunks if _modality(chunk) == modality] for modality in SUPPORTED_MODALITIES}

    results = runner.extract(chunks)

    destination = Path(output_dir)
    all_graphs: list[Graph] = []
    for modality in SUPPORTED_MODALITIES:
        graphs = results[modality]
        all_graphs.extend(graphs)
        write_json(
            destination / f"stage_04_{modality}_extraction.json",
            _modality_payload(modality, grouped_chunks[modality], graphs),
        )

    merged = Graph.merge(all_graphs, document_id=document_id, stage="stage_04_multimodal_extraction")
    reference_errors = merged.validate_references()
    output = {
        "_stage": 4,
        "statistics": {
            "input_chunk_count": len(chunks),
            "extracted_chunk_graph_count": len(all_graphs),
            "skipped_chunk_count": len(chunks) - sum(len(items) for items in grouped_chunks.values()),
            "entity_count": len(merged.entities),
            "relation_count": len(merged.relations),
            "event_count": len(merged.events),
        },
        "validation": {"ok": not reference_errors, "errors": reference_errors},
        "graph": _graph_dict(merged),
    }
    write_json(destination / "stage_04_extraction.json", output)
    return output


def extract_document_text(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_DIR / "stage_04_text_extraction.json",
    *,
    llm_client: Any | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """仅执行文本 Chunk 抽取，供单模态运行与旧调用方兼容。"""

    source = _as_mapping(read_json(input_path), "第三阶段输入")
    text_chunks = [chunk for chunk in collect_chunks(source) if _modality(chunk) == "text"]
    runner = InitExtractor(llm_client=llm_client, show_progress=show_progress)
    graphs = runner.extract(text_chunks)["text"]
    output = _modality_payload("text", text_chunks, graphs)
    output["statistics"]["text_chunk_count"] = len(text_chunks)
    write_json(output_path, output)
    return output


def main() -> None:
    """解析命令行参数并启动第四阶段多模态知识抽取。"""

    parser = argparse.ArgumentParser(description="执行第四阶段文本、表格、图片和公式知识抽取")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="第三阶段摘要 JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="第四阶段输出目录")
    parser.add_argument("--quiet", action="store_true", help="不显示模态处理进度")
    args = parser.parse_args()
    result = extract_document_multimodal(args.input, args.output_dir, show_progress=not args.quiet)
    print(f"第四阶段完成，共生成 {result['statistics']['extracted_chunk_graph_count']} 个 Chunk Graph")


__all__ = ["InitExtractor", "collect_chunks", "extract_document_multimodal", "extract_document_text"]


if __name__ == "__main__":
    main()
