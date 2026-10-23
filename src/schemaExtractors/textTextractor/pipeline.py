"""递归读取第三步文本 Chunk，执行并发抽取、原子检查点和验证统计汇总。"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.utils.json_io import read_json

from ..schema_loader import DEFAULT_SCHEMA_PATH, load_schema
from .extractor import SchemaTextExtractor
from .prompts import PROMPT_VERSION
from .validator import VALIDATOR_VERSION, validate_entities, validate_relations


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "stage_03_document_summary.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "stage_04_schema_text_extraction.json"
logger = logging.getLogger(__name__)


def collect_dataset_text_chunks(records: list[Any]) -> list[dict[str, Any]]:
    """读取开发集文本样本，以完整 sample_id 保留跨文档唯一的图谱关联键。"""
    chunks: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("开发集文本样本必须为对象")
        identifier = record.get("sample_id")
        if not isinstance(identifier, str) or "::" not in identifier or identifier in identifiers:
            raise ValueError(f"开发集 sample_id 缺失、重复或不含文档前缀：{identifier!r}")
        identifiers.add(identifier)
        content, section = record.get("content"), record.get("section", {})
        if not isinstance(content, dict) or not isinstance(content.get("text"), str) or not isinstance(section, dict):
            raise ValueError(f"开发集样本 {identifier} 缺少文本或有效章节信息")
        if not content["text"].strip():
            logger.warning("跳过空文本样本：%s", identifier)
            continue
        chunks.append({
            "id": identifier, "text": content["text"], "order": record.get("order"),
            "document_id": identifier.split("::", 1)[0],
            "document_title": str(record.get("document_title") or ""),
            "document_summary": str(record.get("document_summary") or ""),
            "source_file": record.get("source_file", ""),
            "section_id": str(section.get("section_id") or ""),
            "section_title": str(section.get("section_title") or ""),
            "section_summary": str(section.get("section_summary") or ""),
        })
    return chunks


def collect_text_chunks(payload: Any) -> list[dict[str, Any]]:
    """兼容开发集样本数组和第三步章节树，收集非空文本并保留原始来源上下文。"""
    if isinstance(payload, list):
        return collect_dataset_text_chunks(payload)
    if not isinstance(payload, dict) or not isinstance(payload.get("document", payload), dict):
        raise ValueError("第三步输入及 document 必须是 JSON 对象")
    document = payload.get("document", payload)
    if "sections" not in document:
        raise ValueError("第三步输入缺少 document.sections")
    chunks: list[dict[str, Any]] = []
    identifiers: set[str] = set()

    def visit(sections: Any, inherited_title: str = "") -> None:
        """递归收集一个章节列表，遇到损坏结构或重复文本 ID 时立即报错。"""
        if not isinstance(sections, list):
            raise ValueError("sections/children 必须为数组")
        for section in sections:
            if not isinstance(section, dict):
                raise ValueError("章节必须为对象")
            title = section.get("title") or inherited_title
            raw_chunks = section.get("chunks", [])
            if not isinstance(raw_chunks, list):
                raise ValueError("章节 chunks 必须为数组")
            for raw in raw_chunks:
                if not isinstance(raw, dict):
                    raise ValueError("Chunk 必须为对象")
                modality = str(raw.get("modality") or raw.get("chunk_type") or raw.get("type") or "").lower().replace("_", "")
                if modality not in {"text", "textchunk"}:
                    continue
                identifier = raw.get("id")
                if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
                    raise ValueError(f"文本 Chunk id 为空或重复：{identifier!r}")
                identifiers.add(identifier)
                text = raw.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"文本 Chunk {identifier} 缺少字符串 text")
                if not text.strip():
                    logger.warning("跳过空文本 Chunk：%s", identifier)
                    continue
                chunks.append({
                    "id": identifier, "text": text, "order": raw.get("order"),
                    "document_id": str(document.get("id") or document.get("document_id") or ""),
                    "document_title": str(document.get("title") or ""),
                    "document_summary": str(document.get("summary") or ""),
                    "source_file": document.get("source_file", ""),
                    "section_id": str(section.get("id") or ""), "section_title": str(title),
                    "section_summary": str(section.get("summary") or ""),
                })
            visit(section.get("children") or [], str(title))

    visit(document["sections"])
    return chunks


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """先写同目录临时文件再替换目标，避免中断时破坏已有检查点。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _summarize(
    graphs: list[dict[str, Any]], total: int, extractor: SchemaTextExtractor, input_path: Path,
    resumed: int, *, running: bool,
) -> dict[str, Any]:
    """按 Chunk 汇总实体、关系、类型分布与验证问题，不跨范围合并同名对象。"""
    statuses = Counter(graph["status"] for graph in graphs)
    issues = [issue for graph in graphs for issue in graph["validation"]["issues"]]
    entity_counts = Counter(entity["type"] for graph in graphs for entity in graph["entities"])
    relation_counts = Counter(relation["type"] for graph in graphs for relation in graph["relations"])
    failed = [graph["metadata"]["chunk_id"] for graph in graphs if graph["status"] == "failed"]
    review = [graph["metadata"]["chunk_id"] for graph in graphs if graph["status"] == "needs_review"]
    status = "running" if running else (
        "partial_failure" if failed else "needs_review" if review else
        "completed_with_warnings" if statuses["completed_with_warnings"] else "completed"
    )
    return {
        "_status": status, "_produced_by": "src.schemaExtractors.textTextractor.pipeline",
        "input_path": str(input_path.resolve()), "schema_path": str(extractor.schema.path),
        "schema_fingerprint": extractor.schema.fingerprint, "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "model": getattr(getattr(extractor.client, "config", None), "model", "injected"),
        "statistics": {
            "text_chunk_count": total, "processed_text_chunk_count": len(graphs), "resumed_text_chunk_count": resumed,
            "completed_text_chunk_count": statuses["completed"] + statuses["completed_with_warnings"],
            "failed_text_chunk_count": len(failed), "needs_review_text_chunk_count": len(review),
            "entity_count": sum(entity_counts.values()), "relation_count": sum(relation_counts.values()),
            "entity_type_counts": dict(entity_counts), "relation_type_counts": dict(relation_counts),
            "entity_attribute_count": sum(len(entity.get("attribute", {})) for graph in graphs for entity in graph["entities"]),
            "attribute_completed_text_chunk_count": sum(
                graph["validation"].get("attribute_stage_completed") is True for graph in graphs),
        },
        "validation_summary": {
            "error_count": sum(issue["severity"] == "error" for issue in issues),
            "warning_count": sum(issue["severity"] == "warning" for issue in issues),
            "issue_code_counts": dict(Counter(issue["code"] for issue in issues)),
            "failed_chunk_ids": failed, "needs_review_chunk_ids": review,
        },
        "graphs": graphs,
    }


def _load_completed(path: Path, chunks: list[dict[str, Any]], extractor: SchemaTextExtractor) -> dict[str, dict[str, Any]]:
    """只恢复输入指纹匹配且再次通过基本校验的成功 Chunk，合法空结果也可恢复。"""
    if not path.exists():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("graphs"), list):
        raise ValueError("检查点缺少 graphs 数组；确认输出路径或使用 --no-resume 重跑")
    chunk_map = {chunk["id"]: chunk for chunk in chunks}
    completed: dict[str, dict[str, Any]] = {}
    for graph in payload["graphs"]:
        if not isinstance(graph, dict) or graph.get("status") not in {"completed", "completed_with_warnings"}:
            continue
        metadata = graph.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        identifier = metadata.get("chunk_id")
        if not isinstance(identifier, str) or identifier not in chunk_map:
            continue
        chunk = chunk_map[identifier]
        if metadata.get("input_fingerprint") != extractor.fingerprint(chunk):
            continue
        entities = validate_entities(graph, extractor.schema, chunk["text"], preserve_ids=True)
        relations = validate_relations(graph, extractor.schema, chunk["text"], entities.records)
        validation = graph.get("validation")
        if (not entities.has_errors and not relations.has_errors
                and graph.get("entities") == entities.records and graph.get("relations") == relations.records
                and isinstance(validation, dict) and isinstance(validation.get("issues"), list)
                and validation.get("attribute_stage_completed") is True
                and all(isinstance(issue, dict) and issue.get("severity") == "warning"
                        and isinstance(issue.get("code"), str) for issue in validation["issues"])):
            completed[identifier] = graph
    return completed


def extract_text_from_summary_file(
    input_path: str | Path = DEFAULT_INPUT_PATH, output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *, schema_path: str | Path = DEFAULT_SCHEMA_PATH, llm_client: Any | None = None,
    max_workers: int = 4, validation_retries: int = 1, resume: bool = True,
    limit: int | None = None, chunk_ids: list[str] | None = None, max_tokens: int = 8192,
    force_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    """从第三步输入执行实体、关系和属性补充三阶段抽取，每完成一段即保存。"""
    if max_workers < 1 or (limit is not None and limit < 1):
        raise ValueError("max_workers 和 limit 必须为正整数")
    source, output = Path(input_path), Path(output_path)
    schema = load_schema(schema_path)
    if output.resolve() in {source.resolve(), schema.path}:
        raise ValueError("输出文件不能覆盖输入文件或 Schema")
    chunks = collect_text_chunks(read_json(source))
    if chunk_ids:
        unknown = set(chunk_ids) - {chunk["id"] for chunk in chunks}
        if unknown:
            raise ValueError(f"指定的文本 Chunk 不存在：{sorted(unknown)}")
        chunks = [chunk for chunk in chunks if chunk["id"] in chunk_ids]
    if limit is not None:
        chunks = chunks[:limit]
    unknown_forced = set(force_chunk_ids or []) - {chunk["id"] for chunk in chunks}
    if unknown_forced:
        raise ValueError(f"指定重抽的文本 Chunk 不在本次输入中：{sorted(unknown_forced)}")
    extractor = SchemaTextExtractor(schema, llm_client, validation_retries=validation_retries, max_tokens=max_tokens)
    completed = _load_completed(output, chunks, extractor) if resume else {}
    for identifier in force_chunk_ids or []:
        completed.pop(identifier, None)
    resumed = len(completed)
    pending = [chunk for chunk in chunks if chunk["id"] not in completed]

    def save(running: bool) -> dict[str, Any]:
        """依照输入顺序保存当前进度，保证并发完成顺序不影响最终输出。"""
        graphs = [completed[chunk["id"]] for chunk in chunks if chunk["id"] in completed]
        result = _summarize(graphs, len(chunks), extractor, source, resumed, running=running)
        _atomic_write(output, result)
        return result

    logger.info("Schema 文本抽取：总数=%s，恢复=%s，待处理=%s，并发=%s", len(chunks), resumed, len(pending), max_workers)
    save(bool(pending))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(extractor.extract_chunk, chunk): chunk for chunk in pending}
        for future in as_completed(futures):
            chunk = futures[future]
            graph = future.result()
            completed[chunk["id"]] = graph
            logger.info("[%s/%s] %s：%s，实体=%s，关系=%s", len(completed), len(chunks), chunk["id"],
                        graph["status"], len(graph["entities"]), len(graph["relations"]))
            save(True)
    return save(False)
