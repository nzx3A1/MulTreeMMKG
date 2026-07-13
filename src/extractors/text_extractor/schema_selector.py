"""整篇论文两级 Schema 选择算法。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jieba

from src.utils.embedding_client import EmbeddingClient
from src.utils.llm_client import safe_json_loads

from .document_context import build_document_context, extract_domain_terms
from .schema_models import (
    DocumentContext,
    DocumentSchemaContext,
    RelevantSchema,
    SchemaConcept,
    SchemaRelation,
)
from .schema_repository import Neo4jSchemaRepository


# 统一常见地质同义词与简称，弥补正文表述和 Schema 中文名不完全一致的问题。
CONCEPT_ALIASES = {
    "储集层": ("储层", "储集体"),
    "储层": ("储集层", "储集体"),
    "烃源岩": ("生油岩", "源岩"),
    "盖层": ("封盖层",),
    "断层": ("断裂",),
    "断裂": ("断层",),
    "盆地": ("沉积盆地",),
    "地层组": ("组",),
}


@dataclass(frozen=True)
class SchemaSelectorConfig:
    """集中保存核心概念树、跨树先验和 Chunk 一跳扩展预算。"""

    document_top_k: int = 30
    core_tree_top_k: int = 3
    chunk_vector_top_k: int = 10
    chunk_candidate_top_k: int = 10
    max_one_hop_nodes: int = 15
    min_vector_score: float = 0.55
    seed_score: float = 0.40
    neighbor_min_score: float = 0.35
    max_schema_nodes: int = 0
    max_schema_edges: int = 35
    max_expansion_edges: int = 200
    medium_confidence: float = 0.55
    high_confidence: float = 0.75


def _normalized_text(text: str) -> str:
    """归一化空白和常见 Markdown 标记，同时保留专业编号与单位。"""

    value = re.sub(r"[`*_>#|]+", " ", str(text or ""))
    return re.sub(r"\s+", " ", value).strip()


def build_chunk_query_text(chunk: Mapping[str, Any], context: Mapping[str, str]) -> str:
    """按文档、章节、相邻上下文和当前正文构造可解释查询文本。"""

    return "\n".join(
        (
            f"文档主题：{context.get('document_topic_profile', '')}",
            f"章节标题：{context.get('section_title', '')}",
            f"章节总结：{context.get('section_summary', '')}",
            f"章节Schema关键词：{context.get('schema_keys', '')}",
            f"相邻上文：{context.get('previous_tail', '')}",
            f"相邻下文：{context.get('next_head', '')}",
            f"当前正文：{_normalized_text(str(chunk.get('text') or ''))}",
        )
    )


def _concept_search_text(concept: SchemaConcept) -> str:
    """拼接概念的中英文名称、分类、描述和示例用于词法匹配。"""

    return " ".join(
        (
            concept.schema,
            concept.zh_name,
            concept.category,
            concept.description,
            " ".join(concept.examples),
        )
    ).lower()


def _lexical_candidates(concept: SchemaConcept) -> tuple[str, ...]:
    """汇总概念中英文名、斜杠别名和领域同义词，作为模糊词法匹配候选。"""

    names: list[str] = [concept.schema, concept.zh_name]
    names.extend(part for part in re.split(r"[/／、]", concept.zh_name) if part)
    for name in tuple(names):
        normalized = re.sub(r"\s+", "", str(name)).lower()
        names.extend(CONCEPT_ALIASES.get(normalized, ()))
    return tuple(dict.fromkeys(name.strip().lower() for name in names if name and name.strip()))


def _token_set(value: str) -> set[str]:
    """使用 jieba 将名称或术语切分为稳定 token 集，过滤空白和纯标点。"""

    return {
        token.strip().lower()
        for token in jieba.lcut(str(value or ""), cut_all=False)
        if token.strip() and re.search(r"[\w\u4e00-\u9fff]", token)
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    """计算两个 token 集的 Jaccard 相似度，空集合不产生模糊命中。"""

    return len(left & right) / len(left | right) if left and right else 0.0


def _weighted_score(parts: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """按可用信号重新归一化权重，避免缺失 schemaKeys 时固定损失 25% 分数。"""

    active_weights = {
        name: weight
        for name, weight in weights.items()
        if name != "schema_key" or parts.get(name, 0.0) > 0.0
    }
    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        return 0.0
    return sum(parts.get(name, 0.0) * weight for name, weight in active_weights.items()) / total_weight


def _lexical_score(text: str, terms: Sequence[str], concept: SchemaConcept) -> tuple[float, bool]:
    """融合精确命中、别名命中和 token-level Jaccard 计算概念词法支持度。"""

    normalized = text.lower()
    exact = False
    scores: list[float] = []
    if concept.zh_name and concept.zh_name.lower() in normalized:
        exact = True
        scores.append(1.0)
    if concept.schema and concept.schema.lower() in normalized:
        exact = True
        scores.append(1.0)
    for example in concept.examples:
        if example and str(example).lower() in normalized:
            exact = True
            scores.append(0.90)
    candidates = _lexical_candidates(concept)
    for candidate in candidates:
        if candidate and candidate in normalized:
            alias_exact = candidate not in {concept.schema.lower(), concept.zh_name.lower()}
            exact = exact or not alias_exact
            scores.append(0.90 if alias_exact else 1.0)
    text_units = tuple(dict.fromkeys((*terms, *_token_set(normalized))))
    for candidate in candidates:
        candidate_tokens = _token_set(candidate)
        similarity = max(
            (_jaccard(candidate_tokens, _token_set(unit)) for unit in text_units),
            default=0.0,
        )
        if similarity >= 0.34:
            scores.append(min(0.85, 0.35 + 0.50 * similarity))
    search_text = _concept_search_text(concept)
    matched_terms = [term for term in terms if len(term) >= 2 and term.lower() in search_text]
    if matched_terms:
        scores.append(min(0.85, 0.45 + 0.10 * len(matched_terms)))
    return (max(scores, default=0.0), exact)


def _schema_key_score(schema_keys: Sequence[str], concept: SchemaConcept) -> tuple[float, bool]:
    """独立计算大模型白名单 schemaKeys 对概念节点的直接支持度。"""

    normalized_keys = {re.sub(r"\s+", "", str(item)).lower() for item in schema_keys if str(item).strip()}
    schema_name = re.sub(r"\s+", "", concept.schema).lower()
    zh_name = re.sub(r"\s+", "", concept.zh_name).lower()
    if schema_name in normalized_keys or zh_name in normalized_keys:
        return 1.0, True
    aliases = {part for part in re.split(r"[/／、]", zh_name) if part}
    if aliases & normalized_keys:
        return 0.90, True
    if any(key and (key in zh_name or zh_name in key) for key in normalized_keys if len(key) >= 2):
        return 0.75, False
    return 0.0, False


def _merge_concept(base: SchemaConcept, candidate: SchemaConcept) -> SchemaConcept:
    """合并同一概念的元数据与向量分数，保留信息更完整的一侧。"""

    return replace(
        base,
        zh_name=base.zh_name or candidate.zh_name,
        category=base.category or candidate.category,
        description=base.description or candidate.description,
        examples=base.examples or candidate.examples,
        embedding=base.embedding or candidate.embedding,
        vector_score=max(base.vector_score, candidate.vector_score),
        selection_reasons=tuple(dict.fromkeys(base.selection_reasons + candidate.selection_reasons)),
    )


class SchemaSelector:
    """按“核心概念树 + 跨树先验 + Chunk 精召回”流程只筛选概念节点。"""

    def __init__(
        self,
        *,
        repository: Neo4jSchemaRepository | None = None,
        embedding_client: EmbeddingClient | None = None,
        llm_client: Any | None = None,
        config: SchemaSelectorConfig | None = None,
    ) -> None:
        """注入只读 Schema 仓储、向量客户端和可选主题分析模型。"""

        self.embedding_client = embedding_client or EmbeddingClient()
        self.repository = repository or Neo4jSchemaRepository(embedding_client=self.embedding_client)
        self.llm_client = llm_client
        self.config = config or SchemaSelectorConfig()

    def prepare_document(self, chunks: Sequence[Any]) -> DocumentSchemaContext:
        """一次接收整篇论文 Chunk，先建立文档先验，再为每个原始 Chunk 选择概念。"""

        document = build_document_context(chunks)
        if not document.chunks:
            return DocumentSchemaContext(document, RelevantSchema(), {})
        document_pool = self.select_document_schema_pool(document)
        chunk_schemas = {
            str(chunk["id"]): self.select_chunk_schema(chunk, document, document_pool)
            for chunk in document.chunks
        }
        return DocumentSchemaContext(document, document_pool, chunk_schemas)

    def _analyze_document_topic(
        self,
        document: DocumentContext,
        categories: Sequence[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
        """让 LLM 从白名单概念树中选 Top3，并在不可用时回退到确定性评分。"""

        if self.llm_client is None:
            return (), document.domain_terms, False
        prompt = {
            "task": "document_schema_topic_analysis",
            "instruction": (
                "根据文档主题画像提取领域关键词，并只能从 available_categories 中选择最相关的"
                " Top3 主概念树。严格返回 JSON："
                '{"domain_keywords": ["关键词"], "top_categories": ["概念树"]}。'
            ),
            "available_categories": list(categories),
            "document_topic_profile": document.topic_profile,
        }
        messages = [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]
        try:
            if callable(getattr(self.llm_client, "chat_json", None)):
                payload = self.llm_client.chat_json(messages)
            else:
                payload = safe_json_loads(self.llm_client.chat(messages))
        except Exception:
            return (), document.domain_terms, True
        if not isinstance(payload, Mapping):
            return (), document.domain_terms, True
        allowed = set(categories)
        selected = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in payload.get("top_categories") or payload.get("top3_concept_trees") or []
                if str(item).strip() in allowed
            )
        )[: self.config.core_tree_top_k]
        keywords = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in payload.get("domain_keywords") or []
                if str(item).strip()
            )
        )
        return selected, keywords or document.domain_terms, not bool(selected)

    def _score_document_concepts(
        self,
        document: DocumentContext,
        concepts: Sequence[SchemaConcept],
        vector_candidates: Sequence[SchemaConcept],
        query_text: str,
        query_terms: Sequence[str],
    ) -> tuple[dict[str, SchemaConcept], set[str]]:
        """融合关键词、向量、示例词和 schemaKeys，仅计算概念节点的文档相关度。"""

        vector_map = {item.schema: item for item in vector_candidates}
        scored: dict[str, SchemaConcept] = {}
        exact_schemas: set[str] = set()
        for concept in concepts:
            current = _merge_concept(vector_map.get(concept.schema, concept), concept)
            lexical, exact = _lexical_score(query_text, query_terms, current)
            schema_key_score, schema_key_exact = _schema_key_score(document.document_schema_keys, current)
            section_hits = sum(
                1
                for title in document.section_titles
                if _lexical_score(title, extract_domain_terms(title), current)[0] > 0
            )
            coverage = section_hits / max(1, len(document.section_titles))
            final_score = _weighted_score(
                {
                    "vector": current.vector_score,
                    "lexical": lexical,
                    "coverage": coverage,
                    "schema_key": schema_key_score,
                },
                {"vector": 0.45, "lexical": 0.25, "coverage": 0.10, "schema_key": 0.20},
            )
            reasons = list(current.selection_reasons)
            if current.vector_score > 0:
                reasons.append("文档级向量召回")
            if lexical > 0:
                reasons.append("文档关键词、示例词或名称命中")
            if schema_key_score > 0:
                reasons.append("文档 schemaKeys 命中")
            if exact or schema_key_exact:
                exact_schemas.add(current.schema)
                reasons.append("文档精确命中")
            scored[current.schema] = replace(
                current,
                lexical_score=lexical,
                context_score=coverage,
                schema_key_score=schema_key_score,
                document_score=final_score,
                final_score=final_score,
                selection_reasons=tuple(dict.fromkeys(reasons)),
            )
        return scored, exact_schemas

    def select_document_schema_pool(self, document: DocumentContext) -> RelevantSchema:
        """生成 Top3 核心树全量节点和 20～35 个跨树文档先验概念。"""

        all_concepts = self.repository.all_concepts()
        categories = tuple(sorted({item.category for item in all_concepts if item.category}))
        model_categories, model_terms, model_fallback = self._analyze_document_topic(document, categories)
        query_terms = tuple(dict.fromkeys((*model_terms, *document.document_schema_keys, *document.domain_terms)))
        query_text = f"{document.topic_profile}\n领域关键词：{'、'.join(query_terms)}"
        query_vector = self.embedding_client.encode_one(query_text)
        vector_candidates, vector_fallback = self.repository.vector_search(
            query_vector,
            top_k=max(self.config.document_top_k, self.config.core_tree_top_k),
            min_score=self.config.min_vector_score,
        )
        scored, exact_schemas = self._score_document_concepts(
            document, all_concepts, vector_candidates, query_text, query_terms,
        )

        # 中文说明：若模型少选或未选概念树，按树内头部概念分数补足 Top3，避免流程中断。
        category_scores: dict[str, float] = {}
        for category in categories:
            values = sorted(
                (item.final_score for item in scored.values() if item.category == category),
                reverse=True,
            )
            category_scores[category] = sum(values[:3]) / max(1, min(3, len(values)))
        core_categories = list(model_categories)
        for category in sorted(categories, key=lambda item: (-category_scores[item], item)):
            if category not in core_categories:
                core_categories.append(category)
            if len(core_categories) >= self.config.core_tree_top_k:
                break
        core_category_set = set(core_categories)

        core_nodes = [
            replace(
                concept,
                selection_reasons=tuple(dict.fromkeys(concept.selection_reasons + ("Top3 核心概念树全量保留",))),
            )
            for concept in scored.values()
            if concept.category in core_category_set
        ]
        cross_tree = [concept for concept in scored.values() if concept.category not in core_category_set]
        cross_tree.sort(
            key=lambda item: (item.schema not in exact_schemas, -item.final_score, item.schema)
        )
        cross_limit = max(0, min(35, self.config.document_top_k))
        document_prior = [
            replace(
                concept,
                selection_reasons=tuple(dict.fromkeys(concept.selection_reasons + ("跨树 Document Schema Prior",))),
            )
            for concept in cross_tree[:cross_limit]
        ]
        selected = sorted(
            (*core_nodes, *document_prior),
            key=lambda item: (item.category not in core_category_set, -item.final_score, item.schema),
        )
        if self.config.max_schema_nodes > 0:
            selected = selected[: self.config.max_schema_nodes]
        relations = self.repository.induced_relations([item.schema for item in selected])
        confidence = self._selection_confidence(selected, relations, query_terms)
        return RelevantSchema(
            concepts=tuple(selected),
            relations=relations,
            core_categories=tuple(core_categories),
            query_terms=query_terms,
            selection_confidence=confidence,
            fallback_used=model_fallback or vector_fallback,
            selector_version="core_tree_prior_v2",
        )

    def select_chunk_schema(
        self,
        chunk: Mapping[str, Any],
        document: DocumentContext,
        document_pool: RelevantSchema,
    ) -> RelevantSchema:
        """合并核心树、Chunk 跨树 TopK 与受控一跳扩展，最终只按节点集合取关系。"""

        chunk_id = str(chunk.get("id") or "")
        local_context = document.local_context(chunk_id)
        body = _normalized_text(str(chunk.get("text") or ""))
        context_text = " ".join(
            (
                local_context["section_title"],
                local_context["section_summary"],
                local_context["previous_tail"],
                local_context["next_head"],
            )
        )
        query_text = build_chunk_query_text(chunk, local_context)
        terms = extract_domain_terms(f"{body} {context_text}")
        schema_keys = document.section_schema_keys.get(str(chunk.get("section_id") or ""), ())
        query_vector = self.embedding_client.encode_one(query_text)
        vector_candidates, vector_fallback = self.repository.vector_search(
            query_vector,
            top_k=min(10, self.config.chunk_vector_top_k),
            min_score=self.config.min_vector_score,
        )
        vector_map = {item.schema: item for item in vector_candidates}
        document_map = document_pool.concept_map
        core_categories = set(document_pool.core_categories)
        core_nodes: list[SchemaConcept] = []
        for item in document_pool.concepts:
            if item.category not in core_categories:
                continue
            lexical, exact = _lexical_score(body, terms, item)
            context_score, _ = _lexical_score(context_text, extract_domain_terms(context_text), item)
            schema_key_score, schema_key_exact = _schema_key_score(schema_keys, item)
            reasons = list(item.selection_reasons)
            if lexical > 0:
                reasons.append("Chunk 正文关键词、示例词或名称命中")
            if context_score > 0:
                reasons.append("章节摘要或前后文命中")
            if schema_key_score > 0:
                reasons.append("章节 schemaKeys 命中")
            if schema_key_exact:
                reasons.append("章节 schemaKeys 精确命中")
            if exact or schema_key_exact:
                reasons.append("Chunk 精确命中")
            core_nodes.append(
                replace(
                    item,
                    lexical_score=max(item.lexical_score, lexical),
                    context_score=max(item.context_score, context_score),
                    schema_key_score=max(item.schema_key_score, schema_key_score),
                    selection_reasons=tuple(dict.fromkeys(reasons)),
                )
            )
        candidates: list[SchemaConcept] = []
        exact_schemas: set[str] = set()
        for concept in self.repository.all_concepts():
            if concept.category in core_categories:
                continue
            current = _merge_concept(vector_map.get(concept.schema, concept), concept)
            lexical, exact = _lexical_score(body, terms, current)
            context_score, _ = _lexical_score(context_text, extract_domain_terms(context_text), current)
            schema_key_score, schema_key_exact = _schema_key_score(schema_keys, current)
            prior_score = document_map.get(concept.schema, SchemaConcept(concept.schema)).document_score
            if not any((current.vector_score, lexical, context_score, schema_key_score, prior_score)):
                continue
            final_score = _weighted_score(
                {
                    "vector": current.vector_score,
                    "lexical": lexical,
                    "context": context_score,
                    "document": prior_score,
                    "schema_key": schema_key_score,
                },
                {"vector": 0.35, "lexical": 0.20, "context": 0.10, "document": 0.15, "schema_key": 0.20},
            )
            reasons = list(current.selection_reasons)
            if current.vector_score > 0:
                reasons.append("Chunk 向量召回")
            if lexical > 0:
                reasons.append("Chunk 正文关键词、示例词或名称命中")
            if context_score > 0:
                reasons.append("章节摘要或前后文命中")
            if prior_score > 0:
                reasons.append("Document Schema Prior")
            if schema_key_score > 0:
                reasons.append("章节 schemaKeys 命中")
            if exact or schema_key_exact:
                exact_schemas.add(current.schema)
                reasons.append("Chunk 精确命中")
            candidates.append(
                replace(
                    current,
                    lexical_score=lexical,
                    context_score=context_score,
                    schema_key_score=schema_key_score,
                    document_score=prior_score,
                    final_score=final_score,
                    selection_reasons=tuple(dict.fromkeys(reasons)),
                )
            )
        candidates.sort(
            key=lambda item: (item.schema not in exact_schemas, -item.final_score, item.schema)
        )
        chunk_seeds = candidates[: min(10, self.config.chunk_candidate_top_k)]

        neighbor_concepts, _ = self.repository.neighborhood(
            [item.schema for item in chunk_seeds],
            limit=self.config.max_expansion_edges,
        )
        selected_names = {item.schema for item in (*core_nodes, *chunk_seeds)}
        expansion: list[SchemaConcept] = []
        for neighbor in neighbor_concepts:
            if neighbor.schema in selected_names:
                continue
            lexical, exact = _lexical_score(body, terms, neighbor)
            context_score, _ = _lexical_score(context_text, extract_domain_terms(context_text), neighbor)
            support = max(lexical, context_score)
            if support <= 0 and neighbor.category not in core_categories:
                support = 0.20
            if exact:
                exact_schemas.add(neighbor.schema)
            expansion.append(
                replace(
                    neighbor,
                    lexical_score=lexical,
                    context_score=context_score,
                    final_score=support,
                    selection_reasons=("受控一跳概念扩展",),
                )
            )
        expansion.sort(key=lambda item: (item.schema not in exact_schemas, -item.final_score, item.schema))
        expansion = expansion[: min(15, self.config.max_one_hop_nodes)]

        final_map: dict[str, SchemaConcept] = {}
        for concept in (*core_nodes, *chunk_seeds, *expansion):
            final_map[concept.schema] = _merge_concept(final_map.get(concept.schema, concept), concept)
        final_nodes = sorted(
            final_map.values(),
            key=lambda item: (item.category not in core_categories, -item.final_score, item.schema),
        )
        if self.config.max_schema_nodes > 0:
            final_nodes = final_nodes[: self.config.max_schema_nodes]

        # 中文说明：关系不参与召回、打分或裁剪，只返回最终节点之间 Schema 中已有的全部关系。
        relations = self.repository.induced_relations([item.schema for item in final_nodes])
        confidence = self._selection_confidence(final_nodes, relations, terms)
        return RelevantSchema(
            concepts=tuple(final_nodes),
            relations=relations,
            core_categories=document_pool.core_categories,
            query_terms=tuple(terms),
            selection_confidence=confidence,
            fallback_used=vector_fallback or confidence < self.config.medium_confidence,
            selector_version="core_tree_prior_v2",
        )

    @staticmethod
    def _selection_confidence(
        concepts: Sequence[SchemaConcept],
        relations: Sequence[SchemaRelation],
        terms: Sequence[str],
    ) -> float:
        """根据节点头部得分、术语覆盖率和最终节点图连通度估计选择可信度。"""

        if not concepts:
            return 0.0
        ranked = sorted((item.final_score for item in concepts), reverse=True)
        top_average = sum(ranked[:3]) / min(3, len(ranked))
        matched_terms = sum(
            1
            for term in terms
            if any(term.lower() in _concept_search_text(concept) for concept in concepts)
        )
        coverage = matched_terms / max(1, len(terms))
        involved = {name for edge in relations for name in (edge.source_schema, edge.target_schema)}
        connectivity = len(involved) / max(1, len(concepts))
        confidence = 0.65 * top_average + 0.20 * coverage + 0.15 * connectivity
        return max(0.0, min(1.0, confidence))


__all__ = [
    "RelevantSchema",
    "SchemaConcept",
    "SchemaRelation",
    "SchemaSelector",
    "SchemaSelectorConfig",
    "build_chunk_query_text",
]
