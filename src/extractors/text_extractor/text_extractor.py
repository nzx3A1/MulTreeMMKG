"""不依赖 Schema 的文本实体和关系抽取流程。"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from model import Graph, SourceModality
from prompts.text_extraction_prompts import build_entity_prompt, build_relation_prompt
from src.utils.llm_client import safe_json_loads
from src.utils.logger import get_logger
from tqdm import tqdm

from .text_context import TextDocumentContext, build_text_document_context
from .text_extractor_parser import parse_extraction_payload
from .text_extractor_validator import ensure_valid_graph


logger = get_logger("extractors.text_extractor")


def _messages(prompt: Mapping[str, Any]) -> list[dict[str, str]]:
    """把结构化提示词序列化为项目 LLM 客户端统一使用的消息格式。"""

    return [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]


def _call_structured_llm(llm_client: Any, prompt: Mapping[str, Any]) -> Mapping[str, Any]:
    """调用模型并把异常降级为空结果，避免单个 Chunk 中断整篇抽取。"""

    try:
        messages = _messages(prompt)
        result = llm_client.chat_json(messages) if callable(getattr(llm_client, "chat_json", None)) else safe_json_loads(llm_client.chat(messages))
        return result if isinstance(result, Mapping) else {}
    except Exception as exc:
        logger.warning(f"文本抽取单次 LLM 调用失败，当前阶段降级为空结果：{exc}")
        return {}


def _entity_prompt_view(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """筛除非对象响应项，为关系抽取提供本轮实体及其临时 ID。"""

    return [item for item in payload.get("entities", []) if isinstance(item, Mapping)]


def extract_text_chunk_graph(
    chunk: Mapping[str, Any],
    document: TextDocumentContext,
    llm_client: Any,
) -> Graph:
    """对单个 Chunk 直接进行实体、关系两次抽取，不执行 Schema 归类或筛选。"""

    text = str(chunk.get("text") or "").strip()
    chunk_id = str(chunk.get("id") or "")
    if not text:
        # 中文说明：空正文不发起模型请求，仍写出统一 Graph 与结构校验信息供下游追溯。
        logger.info(f"Chunk {chunk_id} 正文为空，跳过实体、关系两阶段 LLM 抽取")
        graph = Graph.from_chunk(
            str(chunk.get("document_id") or ""),
            chunk_id,
            SourceModality.TEXT,
            stage="stage_04_text_extraction_front",
        )
        graph.metadata.extra["empty_reason"] = "empty_text"
        graph.metadata.extra["validation"] = {"accepted_count": 0, "rejected_count": 0, "rejected": []}
        return graph

    context = document.local_context(chunk_id)
    logger.info(f"Chunk {chunk_id} 开始无 Schema 抽取：字符数={len(text)}")

    # 中文说明：实体类型由本次模型直接根据正文语义生成，不读取 Schema 概念或候选池。
    entity_payload = _call_structured_llm(llm_client, build_entity_prompt(text, context))
    entity_candidates = _entity_prompt_view(entity_payload)
    logger.info(
        f"Chunk {chunk_id} 实体抽取完成：候选数={len(entity_candidates)}，"
        f"候选名称={[str(item.get('name') or '') for item in entity_candidates]}"
    )

    # 中文说明：关系只引用本轮实体 ID；不检查 Schema 关系方向或关系类型白名单。
    relation_payload = _call_structured_llm(llm_client, build_relation_prompt(entity_candidates, text, context))
    relation_candidates = [item for item in relation_payload.get("relations", []) if isinstance(item, Mapping)]
    logger.info(
        f"Chunk {chunk_id} 关系抽取完成：候选数={len(relation_candidates)}，"
        f"候选类型={[str(item.get('type') or item.get('relation_name') or '') for item in relation_candidates]}"
    )

    combined = {
        "entities": list(entity_payload.get("entities") or []),
        "relations": list(relation_payload.get("relations") or []),
        "events": [],
    }
    graph = ensure_valid_graph(parse_extraction_payload(combined, chunk=chunk), text)
    validation = graph.metadata.extra.get("validation", {})
    logger.info(
        f"Chunk {chunk_id} 解析校验完成：实体={len(graph.entities)}，关系={len(graph.relations)}，"
        f"事件={len(graph.events)}，拒绝={validation.get('rejected_count', 0)}"
    )
    return graph


def extract_from_text(
    chunks: Sequence[Mapping[str, Any]],
    llm_client: Any,
    *,
    max_workers: int,
    output_path: str | Path | None = None,
    on_graph_completed: Callable[[Graph, int, int], None] | None = None,
    show_progress: bool = True,
) -> list[Graph]:
    """按 Chunk 直接抽取文本图谱，不执行 Schema 预筛选、归类或约束校验。"""

    document = build_text_document_context(chunks)
    logger.info(
        f"无 Schema 文本抽取流程启动：document_id={document.document_id or 'unknown'}，"
        f"Chunk数={len(document.chunks)}，章节数={len(document.section_chunks)}"
    )

    graphs: list[Graph] = []
    total = len(document.chunks)
    if max_workers < 1:
        raise ValueError("max_workers 必须大于或等于 1")
    logger.info(f"文本抽取并发线程数={max_workers}")
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="text-extractor",
    ) as executor:
        # map 会并行执行任务并按输入顺序返回结果，保证 Graph 和检查点顺序稳定。
        extracted = executor.map(
            lambda chunk: extract_text_chunk_graph(chunk, document, llm_client),
            document.chunks,
        )
        progress = tqdm(
            extracted,
            total=total,
            desc="文本 Chunk 抽取",
            unit="chunk",
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for index, graph in enumerate(progress, start=1):
            progress.set_postfix_str(f"chunk={graph.metadata.chunk_id}")
            graphs.append(graph)
            if on_graph_completed:
                on_graph_completed(graph, index, total)

    if output_path is not None:
        from .pipeline import write_text_extraction_result

        write_text_extraction_result(output_path, graphs, status="completed")
        logger.info(f"文本抽取结果已保存：{output_path}")
    logger.info(
        f"无 Schema 文本抽取流程完成：Graph={len(graphs)}，实体={sum(len(item.entities) for item in graphs)}，"
        f"关系={sum(len(item.relations) for item in graphs)}，事件={sum(len(item.events) for item in graphs)}"
    )
    return graphs


__all__ = ["extract_from_text", "extract_text_chunk_graph"]
