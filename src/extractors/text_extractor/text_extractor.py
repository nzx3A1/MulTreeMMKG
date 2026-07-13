"""文本抽取五阶段总入口与单 Chunk 两次 LLM 调用控制。

本模块是文本模态知识抽取的核心调度器，职责包括：
1. 将结构化 Prompt 转换为 LLM 消息格式并调用模型；
2. 对单个文本 Chunk 依次执行「实体→关系」两阶段抽取，事件抽取暂时停用；
3. 对整篇论文执行「文档上下文→两级 Schema→逐 Chunk 抽取→校验保存」完整流程。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from model import Graph, SourceModality
from prompts.extractor_text_prompts import ENTITY_PROMPT, RELATION_PROMPT, ENTITY_PROMPT_noSchema, \
    RELATION_PROMPT_noSchema, ENTITYFULL_PROMPT_Schema, RELATIONFULL_PROMPT_Schema
from src.utils.llm_client import safe_json_loads
from src.utils.logger import get_logger
from tqdm import tqdm

from .document_context import build_document_context
from .schema_models import DocumentContext, RelevantSchema
from .schema_selector import SchemaSelector
from .text_extractor_parser import parse_extraction_payload
from .text_extractor_validator import ensure_valid_graph


logger = get_logger("extractors.text_extractor")


def _messages(prompt: Mapping[str, Any]) -> list[dict[str, str]]:
    """把项目的结构化 Prompt 转为模型客户端统一使用的消息格式。

    LLM 客户端要求输入为 ``[{"role": "user", "content": ...}]`` 形式的消息列表，
    而项目内部 Prompt 以字典形式组织，因此需要序列化为 JSON 字符串。
    """
    return [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]


def _call_structured_llm(llm_client: Any, prompt: Mapping[str, Any]) -> Mapping[str, Any]:
    """优先调用结构化接口，失败时降级为空对象且不阻断后续抽取阶段。

    调用策略：
    - 若客户端提供 ``chat_json`` 方法（如 Gemini / OpenAI function-calling），
      优先使用以获取原生 JSON 输出；
    - 否则使用 ``chat`` 获取文本响应，再通过 ``safe_json_loads`` 解析为字典；
    - 任何异常均捕获并打印警告，返回空字典，确保单次失败不阻断后续阶段。
    """
    try:
        messages = _messages(prompt)
        if callable(getattr(llm_client, "chat_json", None)):
            result = llm_client.chat_json(messages)
        else:
            result = safe_json_loads(llm_client.chat(messages))
        return result if isinstance(result, Mapping) else {}
    except Exception as exc:
        logger.warning(f"文本抽取单次 LLM 调用失败，当前阶段降级为空结果：{exc}")
        return {}


def _entity_prompt_view(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """为关系和事件 Prompt 提供已抽取实体的紧凑临时 ID 视图。

    关系抽取和事件抽取的 Prompt 需要引用第一次 LLM 调用产生的实体候选，
    此函数从实体载荷中提取 ``entities`` 列表并过滤非对象成员，
    使后续阶段可以基于 ``temp_id`` 建立实体间的引用关系。
    """
    return [item for item in payload.get("entities", []) if isinstance(item, Mapping)]


def _call_entityfull_llm(
    llm_client: Any,
    entity_payload: Mapping[str, Any],
    prompt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """调用 Schema 归类模型，并把归类结果安全合并回无约束实体候选。

    第二次调用只负责补充官方名称及中英文类型；第一次调用产生的原文证据、
    属性和置信度必须继续保留。模型漏项、乱序或调用失败时，按实体名称和位置
    回填结果，并为所有候选生成连续且唯一的临时 ID。
    """
    original_entities = [
        dict(item) for item in entity_payload.get("entities", [])
        if isinstance(item, Mapping)
    ]
    if not original_entities:
        return {**dict(entity_payload), "entities": []}

    classified_payload = _call_structured_llm(llm_client, prompt)
    classified_entities = [
        item for item in classified_payload.get("entities", [])
        if isinstance(item, Mapping)
    ]

    # 同名实体可能重复出现，因此名称索引保存列表，并在匹配后立即消费对应结果。
    classified_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for item in classified_entities:
        name = str(item.get("name") or "").strip()
        if name:
            classified_by_name.setdefault(name, []).append(item)

    merged_entities: list[dict[str, Any]] = []
    for index, original in enumerate(original_entities, start=1):
        name = str(original.get("name") or "").strip()
        matches = classified_by_name.get(name) or []
        classified = matches.pop(0) if matches else (
            classified_entities[index - 1] if index <= len(classified_entities) else {}
        )

        merged = dict(original)
        official_name = str(classified.get("official_name") or "").strip()
        schema_type = str(classified.get("type") or "").strip()
        type_zh = str(classified.get("type_zh") or "").strip()
        merged.update({
            "id": f"entity_{index}",
            "temp_id": f"entity_{index}",
            "name": name,
            "official_name": official_name or name,
            "type": schema_type or "other",
            "type_zh": type_zh or str(original.get("type_zh") or "").strip(),
        })
        merged_entities.append(merged)

    return {**dict(entity_payload), "entities": merged_entities}



def _call_relationfull_llm(
    llm_client: Any,
    relation_payload: Mapping[str, Any],
    prompt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """调用 Schema 归类模型，并把规范化结果合并回无约束关系候选。

    归类阶段只覆盖关系名称及中英文类型，首次抽取得到的实体端点、原文证据、
    属性和置信度保持不变。模型漏项、乱序或调用失败时，优先按端点匹配，
    再按原顺序回填，并为所有关系生成连续且唯一的临时 ID。
    """
    original_relations = [
        dict(item) for item in relation_payload.get("relations", [])
        if isinstance(item, Mapping)
    ]
    if not original_relations:
        return {**dict(relation_payload), "relations": []}

    classified_payload = _call_structured_llm(llm_client, prompt)
    classified_relations = [
        item for item in classified_payload.get("relations", [])
        if isinstance(item, Mapping)
    ]

    # 同一对实体可能存在多种关系，端点索引使用列表以便逐条消费归类结果。
    classified_by_endpoints: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for item in classified_relations:
        endpoint_key = (
            str(item.get("source_id") or "").strip(),
            str(item.get("target_id") or "").strip(),
        )
        if all(endpoint_key):
            classified_by_endpoints.setdefault(endpoint_key, []).append(item)

    merged_relations: list[dict[str, Any]] = []
    for index, original in enumerate(original_relations, start=1):
        endpoint_key = (
            str(original.get("source_id") or "").strip(),
            str(original.get("target_id") or "").strip(),
        )
        matches = classified_by_endpoints.get(endpoint_key) or []
        classified = matches.pop(0) if matches else (
            classified_relations[index - 1] if index <= len(classified_relations) else {}
        )

        original_name = str(
            original.get("relation_name") or original.get("type_zh") or original.get("type") or ""
        ).strip()
        relation_name = str(classified.get("relation_name") or "").strip()
        relation_type = str(classified.get("type") or "").strip()
        type_zh = str(classified.get("type_zh") or "").strip()
        merged = dict(original)
        merged.update({
            "id": f"relation_{index}",
            "temp_id": f"relation_{index}",
            "relation_name": relation_name or original_name,
            "type": relation_type or "other",
            "type_zh": type_zh or original_name,
        })
        merged_relations.append(merged)

    return {**dict(relation_payload), "relations": merged_relations}

def extract_text_chunk_graph(
    chunk: Mapping[str, Any],
    document: DocumentContext,
    schema: RelevantSchema,
    llm_client: Any,
) -> Graph:
    """对单个正文 Chunk 依次执行实体、关系抽取并解析校验，事件抽取暂时停用。

    执行流程：
    1. 空文检查 —— 若 Chunk 正文为空，直接返回带 ``empty_reason`` 的空 Graph；
    2. 构造上下文 —— 从 DocumentContext 获取当前 Chunk 的章节标题、相邻文本等；
    3. 第一次 LLM 调用（ENTITY_PROMPT）—— 抽取实体候选，获得 temp_id；
    4. 第二次 LLM 调用（RELATION_PROMPT）—— 在已抽取实体间判断 Schema 允许的有向关系；
    5. 事件载荷固定为空，不发起事件抽取 LLM 调用；
    6. 合并两阶段结果，交由 parse_extraction_payload 解析为稳定 ID 的 Graph 子对象；
    7. ensure_valid_graph 执行最终引用完整性和 provenance 证据校验。

    Args:
        chunk: 当前文本 Chunk，需包含 id、document_id、text 等字段。
        document: 整篇论文的文档上下文，提供章节信息和相邻文本。
        schema: 当前 Chunk 的局部 Schema 白名单，约束抽取范围。
        llm_client: 可注入的 LLM 客户端实例。

    Returns:
        经过解析和校验的统一 Graph 对象，空 Chunk 返回带原因的空 Graph。
    """
    text = str(chunk.get("text") or "").strip()
    chunk_id = str(chunk.get("id") or "")
    # 正文为空时直接返回空 Graph，附带 empty_reason 便于下游追溯
    if not text:
        logger.info(f"Chunk {chunk_id} 正文为空，跳过实体、关系两阶段 LLM 抽取")
        graph = Graph.from_chunk(
            str(chunk.get("document_id") or ""), str(chunk.get("id") or ""),
            SourceModality.TEXT, stage="stage_04_text_extraction_front",
        )
        graph.metadata.extra["empty_reason"] = "empty_text"
        graph.metadata.extra["schema_selection"] = schema.to_dict()
        graph.metadata.extra["validation"] = {"accepted_count": 0, "rejected_count": 0, "rejected": []}
        return graph

    # 获取当前 Chunk 的局部上下文：章节标题、章节总结、相邻上下文等
    context = document.local_context(str(chunk["id"]))
    # 将 Schema 概念和关系约束序列化为字典，供 Prompt 模板使用
    concepts = [item.to_dict() for item in schema.concepts]
    relations = [item.to_dict() for item in schema.relations]
    # 将 concepts中的数据只保留核心属性，其他属性不参与抽取，只保留schema，zh_name，description，example
    concepts = [{"schema": item["schema"], "zh_name": item["zhName"], "description": item["description"], "example": item["examples"]} for item in concepts]
    #relations 只保留'source_schema',relationEn,relationZh,target_schema
    relations = [{"source_schema": item["source_schema"], "relationEn": item["relationEn"], "relationZh": item["relationZh"], "target_schema": item["target_schema"]} for item in relations]

    logger.info(
        f"Chunk {chunk_id} 开始抽取：字符数={len(text)}，"
        f"局部Schema概念={len(concepts)}，局部Schema关系={len(relations)}"
    )

    # ── 第一次 LLM 调用：实体抽取 ──
    # 输入正文和 Schema 概念白名单，输出带 temp_id 的实体候选列表 ,选抽取再筛选
    entity_payload = _call_structured_llm(llm_client, ENTITY_PROMPT_noSchema(text, context))
    #为抽取后的实体添加id属性还有合适的定义类型
    entity_payload = _call_entityfull_llm(llm_client, entity_payload, ENTITYFULL_PROMPT_Schema(entity_payload.get("entities") or [], concepts))

    entity_candidates = _entity_prompt_view(entity_payload)



    logger.info(
        f"Chunk {chunk_id} 实体抽取完成：候选数={len(entity_candidates)}，"
        f"候选名称={[str(item.get('name') or '') for item in entity_candidates]}"
    )

    # ── 第二次 LLM 调用：关系抽取 ──
    # 仅在已抽取实体之间判断局部 Schema 允许的有向关系
    relation_payload = _call_structured_llm(
        llm_client, RELATION_PROMPT_noSchema(entity_candidates, text, context),
    )
    relation_payload= _call_relationfull_llm(llm_client, relation_payload, RELATIONFULL_PROMPT_Schema(relation_payload.get("relations") or [], relations))
    relation_candidates = list(relation_payload.get("relations") or [])
    logger.info(
        f"Chunk {chunk_id} 关系抽取完成：候选数={len(relation_candidates)}，"
        f"候选类型={[str(item.get('type') or item.get('relationEn') or '') for item in relation_candidates if isinstance(item, Mapping)]}"
    )
    # 事件抽取暂时停用：保留空载荷以兼容统一 Graph 输出及下游序列化结构。
    # 如需恢复，可重新启用 EVENT_PROMPT 导入和下方第三次 LLM 调用。
    # event_payload = _call_structured_llm(
    #     llm_client, EVENT_PROMPT(text, entity_candidates, context),
    # )
    # event_candidates = list(event_payload.get("events") or [])
    # logger.info(
    #     f"Chunk {chunk_id} 事件抽取完成：候选数={len(event_candidates)}，"
    #     f"候选名称={[str(item.get('name') or '') for item in event_candidates if isinstance(item, Mapping)]}"
    # )
    event_payload: Mapping[str, Any] = {}


    # 合并实体、关系结果和空事件载荷，保留各阶段原始响应用于追溯
    combined = {
        "entities": list(entity_payload.get("entities") or []),
        "relations": list(relation_payload.get("relations") or []),
        "events": list(event_payload.get("events") or []),
        "stage_responses": {
            "entity": dict(entity_payload), "relation": dict(relation_payload), "event": dict(event_payload),
        },
    }
    # 解析为稳定 ID 的 Entity/Relation/Event 对象，并执行 Schema 白名单和证据校验
    graph = ensure_valid_graph(parse_extraction_payload(combined, chunk=chunk, schema=schema), text)
    validation = graph.metadata.extra.get("validation", {})
    logger.info(
        f"Chunk {chunk_id} 解析校验完成：实体={len(graph.entities)}，关系={len(graph.relations)}，"
        f"事件={len(graph.events)}，拒绝={validation.get('rejected_count', 0)}"
    )
    return graph


def extract_from_text(
    chunks: Sequence[Mapping[str, Any]],
    llm_client: Any,
    schema_selector: Any | None = None,
    *,
    output_path: str | Path | None = None,
    on_graph_completed: Callable[[Graph, int, int], None] | None = None,
    show_progress: bool = True,
) -> list[Graph]:
    """执行整篇论文文本流程：上下文、两级 Schema、逐 Chunk 抽取、校验与保存。

    完整流程：
    1. build_document_context —— 校验、排序 Chunk，构建章节索引和主题画像；
    2. SchemaSelector.prepare_document 或 select_document_schema_pool —— 选择文档级候选池；
    3. 逐 Chunk 循环：
       a. select_chunk_schema —— 在文档候选池基础上选择当前 Chunk 的局部 Schema 子图；
       b. extract_text_chunk_graph —— 实体、关系两次 LLM 调用 + 解析校验；
       c. on_graph_completed 回调 —— 通知调用方当前 Chunk 已完成（用于持久化和进度展示）；
    4. 若指定 output_path，调用 pipeline.write_text_extraction_result 写入最终结果。

    Args:
        chunks: 同一篇论文的全部文本 Chunk 序列。
        llm_client: 可注入的 LLM 客户端实例。
        schema_selector: 可注入的 Schema 选择器，默认使用 SchemaSelector()。
        output_path: 若非空，抽取完成后将结果写入该路径。
        on_graph_completed: 每个 Chunk 抽取完成后的回调，参数为 (graph, index, total)。

    Returns:
        按 Chunk 顺序排列的 Graph 列表。
    """
    # 阶段 1：构建文档上下文（Chunk 排序、章节索引、jieba 术语提取、主题画像）
    document = build_document_context(chunks)
    selector = schema_selector or SchemaSelector()
    logger.info(
        f"文本抽取流程启动：document_id={document.document_id or 'unknown'}，"
        f"Chunk数={len(document.chunks)}，章节数={len(document.section_chunks)}，"
        f"领域术语数={len(document.domain_terms)}"
    )

    # 阶段 2：选择文档级 Schema 候选池
    # 优先使用 prepare_document 一次性生成文档池和全部 Chunk Schema；
    # 若选择器未实现该方法，则退化为先选文档池、再逐 Chunk 选择局部 Schema
    prepared = selector.prepare_document(document.chunks) if callable(getattr(selector, "prepare_document", None)) else None
    if prepared is not None:
        document = prepared.document
        document_pool = prepared.document_schema_pool
        chunk_schemas = prepared.chunk_schemas
    else:
        document_pool = selector.select_document_schema_pool(document)
        chunk_schemas = None
    logger.info(
        f"文档级Schema候选池完成：概念={len(document_pool.concepts)}，"
        f"关系={len(document_pool.relations)}，置信度={document_pool.selection_confidence:.3f}，"
        f"降级={document_pool.fallback_used}"
    )

    # 阶段 3~5：逐 Chunk 选择局部 Schema → 实体、关系两阶段 LLM 抽取 → 解析校验
    graphs: list[Graph] = []
    total = len(document.chunks)
    progress = tqdm(
        document.chunks,
        total=total,
        desc="文本 Chunk 抽取",
        unit="chunk",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for index, chunk in enumerate(progress, start=1):
        chunk_id = str(chunk["id"])
        progress.set_postfix_str(f"chunk={chunk_id}")
        # 阶段 3：选择当前 Chunk 的局部 Schema 子图
        schema = chunk_schemas[chunk_id] if chunk_schemas is not None else selector.select_chunk_schema(
            chunk, document, document_pool,
        )
        # 阶段 4+5：实体、关系两次 LLM 调用 + 解析校验
        graph = extract_text_chunk_graph(chunk, document, schema, llm_client)
        graphs.append(graph)
        # 通知调用方当前 Chunk 已完成（用于持久化、进度展示等）
        if on_graph_completed:
            on_graph_completed(graph, index, total)

    # 若指定了输出路径，将全部 Graph 写入文件
    if output_path is not None:
        from .pipeline import write_text_extraction_result
        write_text_extraction_result(output_path, graphs, status="completed")
        logger.info(f"文本抽取结果已保存：{output_path}")
    logger.info(
        f"文本抽取流程完成：Graph={len(graphs)}，实体={sum(len(item.entities) for item in graphs)}，"
        f"关系={sum(len(item.relations) for item in graphs)}，事件={sum(len(item.events) for item in graphs)}"
    )
    return graphs


__all__ = ["extract_from_text", "extract_text_chunk_graph"]
