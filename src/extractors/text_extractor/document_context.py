"""整篇文本 Chunk 的校验、排序、索引与主题画像构建。"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .schema_models import DocumentContext


TERM_PATTERN = re.compile(
    r"[A-Za-z]+(?:[-_/][A-Za-z0-9]+)*|[\u4e00-\u9fffA-Za-z0-9δΔ]+?(?:盆地|坳陷|洼陷|隆起|斜坡|断裂带|断层|区块|研究区|组|段|层|界面|砂岩|泥岩|灰岩|白云岩|储层|烃源岩|盖层|圈闭|油田|气田|油藏|气藏|井|裂缝|孔隙|有机质|干酪根|同位素|样品|实验|测井|地震剖面)"
)

TERM_BOUNDARY_PATTERN = re.compile(
    r"(?:包含|含有|组成|属于|隶属|位于|分布于|发育于|发育在|控制|影响|指示|表明|生成|产生|覆盖|分析|测试|采自|取自|具有|主要由|以及|和|与|及)"
)


def _as_mapping(chunk: Any) -> dict[str, Any]:
    """把字典或 Pydantic Chunk 转换为普通字典并保留扩展溯源字段。"""

    if isinstance(chunk, Mapping):
        return dict(chunk)
    if hasattr(chunk, "to_dict"):
        return dict(chunk.to_dict())
    if hasattr(chunk, "model_dump"):
        return dict(chunk.model_dump(mode="json"))
    if hasattr(chunk, "dict"):
        return dict(chunk.dict())
    raise TypeError(f"不支持的文本 Chunk 类型：{type(chunk).__name__}")


def extract_domain_terms(text: str, *, limit: int = 80) -> tuple[str, ...]:
    """使用石油地质后缀和中英文模式提取可解释的专业候选词。"""

    separated = TERM_BOUNDARY_PATTERN.sub(" ", str(text or ""))
    counter = Counter(match.group(0).strip() for match in TERM_PATTERN.finditer(separated))
    ordered = sorted(counter, key=lambda term: (-counter[term], -len(term), term))
    return tuple(ordered[:limit])


def _representative_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    max_chunks: int,
) -> tuple[str, ...]:
    """按章节均匀选择专业术语密度较高的代表性 Chunk。"""

    by_section: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        text = str(chunk.get("text") or "")
        term_count = len(extract_domain_terms(text, limit=200))
        by_section[str(chunk.get("section_id") or "")].append(
            (term_count, -index, str(chunk["id"]))
        )

    chosen: list[str] = []
    queues = [sorted(items, reverse=True) for items in by_section.values()]
    while queues and len(chosen) < max_chunks:
        next_round = []
        for queue in queues:
            if queue and len(chosen) < max_chunks:
                chosen.append(queue.pop(0)[2])
            if queue:
                next_round.append(queue)
        queues = next_round
    return tuple(chosen)


def build_document_context(
    chunks: Sequence[Any],
    *,
    representative_limit: int = 12,
    term_limit: int = 80,
) -> DocumentContext:
    """校验同一篇论文的全部文本 Chunk，并构建顺序、章节和主题上下文。"""

    if not chunks:
        return DocumentContext("", (), {}, {}, (), (), {}, (), (), "")

    normalized: list[tuple[int, int, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    document_ids: set[str] = set()
    for input_index, raw_chunk in enumerate(chunks):
        chunk = _as_mapping(raw_chunk)
        chunk_id = str(chunk.get("id") or "").strip()
        if not chunk_id:
            raise ValueError(f"第 {input_index} 个文本 Chunk 缺少唯一 id")
        if chunk_id in seen_ids:
            raise ValueError(f"文本 Chunk id 重复：{chunk_id}")
        seen_ids.add(chunk_id)

        modality = str(getattr(chunk.get("modality"), "value", chunk.get("modality") or "text")).lower()
        if modality not in {"text", "textchunk"}:
            raise ValueError(f"Chunk {chunk_id} 不是文本模态：{modality}")
        text = str(chunk.get("text") or "").strip()
        if not text:
            chunk["text"] = ""
        else:
            chunk["text"] = text
        document_id = str(chunk.get("document_id") or "").strip()
        if document_id:
            document_ids.add(document_id)
        try:
            order = int(chunk.get("order", input_index))
        except (TypeError, ValueError):
            order = input_index
        normalized.append((order, input_index, chunk))

    if len(document_ids) > 1:
        raise ValueError(f"一次 Schema 选择只能处理一篇论文，收到 document_id：{sorted(document_ids)}")

    normalized.sort(key=lambda item: (item[0], item[1]))
    ordered_chunks = tuple(item[2] for item in normalized)
    chunk_indexes = {str(chunk["id"]): index for index, chunk in enumerate(ordered_chunks)}
    section_map: dict[str, list[str]] = defaultdict(list)
    section_schema_keys: dict[str, tuple[str, ...]] = {}
    section_titles: list[str] = []
    document_schema_keys: list[str] = []
    for chunk in ordered_chunks:
        section_id = str(chunk.get("section_id") or "")
        section_map[section_id].append(str(chunk["id"]))
        title = str(chunk.get("section_title") or "").strip()
        if title and title not in section_titles:
            section_titles.append(title)
        keys = tuple(dict.fromkeys(str(item).strip() for item in chunk.get("schemaKeys") or [] if str(item).strip()))
        if keys:
            section_schema_keys[section_id] = keys
        for item in chunk.get("document_schema_keys") or []:
            key = str(item).strip()
            if key and key not in document_schema_keys:
                document_schema_keys.append(key)

    all_text = "\n".join(str(chunk.get("text") or "") for chunk in ordered_chunks)
    domain_terms = extract_domain_terms(all_text, limit=term_limit)
    representative_ids = _representative_chunks(ordered_chunks, max_chunks=representative_limit)
    representative_lookup = set(representative_ids)
    representative_texts = [
        str(chunk.get("text") or "")[:500]
        for chunk in ordered_chunks
        if str(chunk["id"]) in representative_lookup
    ]
    topic_profile = "\n".join(
        (
            f"章节标题：{'；'.join(section_titles)}",
            f"文档Schema关键词：{'、'.join(document_schema_keys)}",
            f"章节Schema关键词：{'、'.join(dict.fromkeys(key for keys in section_schema_keys.values() for key in keys))}",
            f"高频专业词：{'、'.join(domain_terms)}",
            f"代表性正文：{' '.join(representative_texts)}",
        )
    )
    return DocumentContext(
        document_id=next(iter(document_ids), ""),
        chunks=ordered_chunks,
        chunk_indexes=chunk_indexes,
        section_chunks={key: tuple(value) for key, value in section_map.items()},
        section_titles=tuple(section_titles),
        document_schema_keys=tuple(document_schema_keys),
        section_schema_keys=section_schema_keys,
        domain_terms=domain_terms,
        representative_chunk_ids=representative_ids,
        topic_profile=topic_profile,
    )


__all__ = ["build_document_context", "extract_domain_terms"]
