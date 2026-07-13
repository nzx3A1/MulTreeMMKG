"""整篇论文两级 Schema 选择算法。"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jieba

from src.utils.embedding_client import EmbeddingClient

from .document_context import build_document_context, extract_domain_terms
from .schema_models import (
    DocumentContext,
    DocumentSchemaContext,
    RelevantSchema,
    SchemaConcept,
    SchemaRelation,
)
from .schema_repository import Neo4jSchemaRepository


RELATION_TRIGGERS = {
    "CONTAINS": ("包含", "含有", "组成"),
    "PART_OF": ("属于", "隶属", "组成部分"),
    "LOCATED_IN": ("位于", "分布于", "发育在"),
    "DEVELOPED_IN": ("发育于", "发育在", "赋存于"),
    "CONTROLS": ("控制", "制约"),
    "AFFECTS": ("影响", "作用于"),
    "INDICATES": ("指示", "表明", "反映"),
    "GENERATES": ("生成", "产生", "生烃"),
    "COVERS": ("覆盖", "上覆"),
    "OVERLIES": ("上覆", "位于其上"),
    "UNDERLIES": ("下伏", "位于其下"),
    "INTERPRETS": ("解释", "识别"),
    "ANALYZES": ("分析", "测试"),
    "COLLECTED_FROM": ("采自", "取自"),
}

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
    """集中保存文档级和 Chunk 级选择预算及冷启动阈值。"""

    document_top_k: int = 30
    chunk_vector_top_k: int = 12
    min_vector_score: float = 0.55
    seed_score: float = 0.40
    neighbor_min_score: float = 0.35
    max_schema_nodes: int = 15
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
    """先选择整篇论文候选池，再为每个 Chunk 选择受预算约束的局部子图。"""

    def __init__(
        self,
        *,
        repository: Neo4jSchemaRepository | None = None,
        embedding_client: EmbeddingClient | None = None,
        config: SchemaSelectorConfig | None = None,
    ) -> None:
        """注入 Schema 仓储和向量客户端，使算法可离线测试与替换模型。"""

        self.embedding_client = embedding_client or EmbeddingClient()
        self.repository = repository or Neo4jSchemaRepository(embedding_client=self.embedding_client)
        self.config = config or SchemaSelectorConfig()

    def prepare_document(self, chunks: Sequence[Any]) -> DocumentSchemaContext:
        """一次接收整篇论文 Chunk，生成文档候选池和全部 Chunk 的局部 Schema。"""

        document = build_document_context(chunks)
        if not document.chunks:
            return DocumentSchemaContext(document, RelevantSchema(), {})
        document_pool = self.select_document_schema_pool(document)
        chunk_schemas = {
            str(chunk["id"]): self.select_chunk_schema(chunk, document, document_pool)
            for chunk in document.chunks
        }
        return DocumentSchemaContext(document, document_pool, chunk_schemas)

    def select_document_schema_pool(self, document: DocumentContext) -> RelevantSchema:
        """依据整篇论文主题画像选择 20～35 个概念构成文档级软先验池。"""

        query_text = document.topic_profile
        query_vector = self.embedding_client.encode_one(query_text)
        vector_candidates, fallback = self.repository.vector_search(
            query_vector,
            top_k=self.config.document_top_k,
            min_score=self.config.min_vector_score,
        )
        all_concepts = self.repository.all_concepts()
        merged = {concept.schema: concept for concept in vector_candidates}
        schema_key_schemas: set[str] = set()
        for concept in all_concepts:
            lexical, exact = _lexical_score(query_text, document.domain_terms, concept)
            schema_key_score, schema_key_exact = _schema_key_score(document.document_schema_keys, concept)
            if lexical <= 0.0 and schema_key_score <= 0.0 and concept.schema not in merged:
                continue
            current = merged.get(concept.schema, concept)
            current = _merge_concept(current, concept)
            section_hits = sum(
                1
                for title in document.section_titles
                if _lexical_score(title, extract_domain_terms(title), concept)[0] > 0
            )
            coverage = section_hits / max(1, len(document.section_titles))
            document_score = _weighted_score(
                {
                    "vector": current.vector_score,
                    "lexical": lexical,
                    "coverage": coverage,
                    "schema_key": schema_key_score,
                },
                {"vector": 0.50, "lexical": 0.15, "coverage": 0.10, "schema_key": 0.25},
            )
            reasons = list(current.selection_reasons)
            if current.vector_score > 0:
                reasons.append("Neo4j 向量 Top-K 召回")
            if lexical > 0:
                reasons.append("文档词法命中")
            if exact:
                reasons.append("文档精确命中")
            if schema_key_score > 0:
                reasons.append("文档 schemaKeys 命中")
            if schema_key_exact:
                schema_key_schemas.add(concept.schema)
                reasons.append("文档 schemaKeys 精确命中")
            merged[concept.schema] = replace(
                current,
                lexical_score=lexical,
                context_score=coverage,
                schema_key_score=schema_key_score,
                document_score=document_score,
                final_score=document_score,
                selection_reasons=tuple(dict.fromkeys(reasons)),
            )

        for schema, concept in list(merged.items()):
            if concept.final_score == 0.0:
                score = 0.50 * concept.vector_score
                reasons = concept.selection_reasons
                if concept.vector_score > 0:
                    reasons = tuple(dict.fromkeys(reasons + ("Neo4j 向量 Top-K 召回",)))
                merged[schema] = replace(
                    concept,
                    document_score=score,
                    final_score=score,
                    selection_reasons=reasons,
                )
        ranked = sorted(
            merged.values(),
            key=lambda item: (item.schema not in schema_key_schemas, -item.final_score, item.schema),
        )
        selected = tuple(ranked[: self.config.document_top_k])
        relations = self.repository.induced_relations([item.schema for item in selected])
        confidence = self._selection_confidence(selected, relations, document.domain_terms)
        return RelevantSchema(
            concepts=selected,
            relations=relations,
            query_terms=document.domain_terms,
            selection_confidence=confidence,
            fallback_used=fallback,
        )

    def select_chunk_schema(
        self,
        chunk: Mapping[str, Any],
        document: DocumentContext,
        document_pool: RelevantSchema,
    ) -> RelevantSchema:
        """融合当前正文、局部上下文和文档先验，选择结构闭合的 Chunk Schema。"""

        chunk_id = str(chunk.get("id") or "")
        local_context = document.local_context(chunk_id)
        query_text = build_chunk_query_text(chunk, local_context)
        body = _normalized_text(str(chunk.get("text") or ""))
        context_text = " ".join(
            (
                local_context["section_title"],
                local_context["previous_tail"],
                local_context["next_head"],
            )
        )
        terms = extract_domain_terms(body)
        schema_keys = document.section_schema_keys.get(str(chunk.get("section_id") or ""), ())
        query_vector = self.embedding_client.encode_one(query_text)
        vector_candidates, fallback = self.repository.vector_search(
            query_vector,
            top_k=self.config.chunk_vector_top_k,
            min_score=self.config.min_vector_score,
        )

        all_concepts = self.repository.all_concepts()
        document_scores = {
            concept.schema: concept.document_score or concept.final_score
            for concept in document_pool.concepts
        }
        merged = {concept.schema: concept for concept in vector_candidates}
        exact_schemas: set[str] = set()
        for concept in all_concepts:
            lexical, exact = _lexical_score(body, terms, concept)
            context_score, _ = _lexical_score(context_text, extract_domain_terms(context_text), concept)
            schema_key_score, schema_key_exact = _schema_key_score(schema_keys, concept)
            if lexical <= 0 and context_score <= 0 and schema_key_score <= 0 and concept.schema not in merged:
                continue
            current = _merge_concept(merged.get(concept.schema, concept), concept)
            document_score = document_scores.get(concept.schema, 0.0)
            final_score = _weighted_score(
                {
                    "vector": current.vector_score,
                    "lexical": lexical,
                    "context": context_score,
                    "document": document_score,
                    "schema_key": schema_key_score,
                },
                {"vector": 0.35, "lexical": 0.15, "context": 0.10, "document": 0.15, "schema_key": 0.25},
            )
            reasons = list(current.selection_reasons)
            if current.vector_score > 0:
                reasons.append("Neo4j 向量 Top-K 召回")
            if lexical > 0:
                reasons.append("正文词法命中")
            if context_score > 0:
                reasons.append("章节或相邻上下文命中")
            if document_score > 0:
                reasons.append("文档候选池先验")
            if schema_key_score > 0:
                reasons.append("章节 schemaKeys 命中")
            if schema_key_exact:
                exact_schemas.add(concept.schema)
                reasons.append("章节 schemaKeys 精确命中")
            if exact:
                exact_schemas.add(concept.schema)
                reasons.append("正文精确命中")
            merged[concept.schema] = replace(
                current,
                lexical_score=lexical,
                context_score=context_score,
                schema_key_score=schema_key_score,
                document_score=document_score,
                final_score=final_score,
                selection_reasons=tuple(dict.fromkeys(reasons)),
            )

        seeds = [
            concept
            for concept in merged.values()
            if concept.final_score >= self.config.seed_score or concept.schema in exact_schemas
        ]
        if not seeds:
            seeds = sorted(merged.values(), key=lambda item: (-item.final_score, item.schema))[:3]
        return self._expand_and_prune(
            seeds,
            merged,
            body,
            terms,
            exact_schemas,
            fallback,
        )

    def _expand_and_prune(
        self,
        seeds: Sequence[SchemaConcept],
        candidates: Mapping[str, SchemaConcept],
        body: str,
        terms: Sequence[str],
        exact_schemas: set[str],
        fallback: bool,
    ) -> RelevantSchema:
        """对种子执行一跳扩展，并在节点和边预算下选择高收益闭合子图。"""

        neighbor_concepts, relations = self.repository.neighborhood(
            [seed.schema for seed in seeds],
            limit=self.config.max_expansion_edges,
        )
        pool = dict(candidates)
        for neighbor in neighbor_concepts:
            lexical, exact = _lexical_score(body, terms, neighbor)
            base = _merge_concept(pool.get(neighbor.schema, neighbor), neighbor)
            if base.final_score == 0.0:
                base = replace(
                    base,
                    lexical_score=lexical,
                    final_score=0.25 * lexical,
                    selection_reasons=("Schema 一跳扩展",),
                )
            if exact:
                exact_schemas.add(neighbor.schema)
            pool[neighbor.schema] = base

        seed_names = {item.schema for item in seeds}
        scored_relations: list[SchemaRelation] = []
        for relation in relations:
            source = pool.get(relation.source_schema, SchemaConcept(relation.source_schema))
            target = pool.get(relation.target_schema, SchemaConcept(relation.target_schema))
            trigger_score = self._relation_trigger_score(relation, body)
            connectivity = 1.0 if relation.source_schema in seed_names and relation.target_schema in seed_names else 0.4
            edge_score = (
                0.40 * source.final_score
                + 0.35 * target.final_score
                + 0.15 * trigger_score
                + 0.10 * connectivity
            )
            both_supported = source.final_score >= self.config.neighbor_min_score and target.final_score >= self.config.neighbor_min_score
            one_seed_supported = (
                (relation.source_schema in seed_names or relation.target_schema in seed_names)
                and max(source.lexical_score, target.lexical_score, trigger_score) > 0
            )
            if both_supported or one_seed_supported:
                scored_relations.append(replace(relation, edge_score=edge_score))

        hard_protected = [pool[name] for name in exact_schemas if name in pool]
        ranked_nodes = sorted(
            pool.values(),
            key=lambda item: (
                item.schema not in exact_schemas,
                -item.final_score,
                item.schema,
            ),
        )
        selected_nodes: dict[str, SchemaConcept] = {}
        for concept in hard_protected + ranked_nodes:
            if concept.schema in selected_nodes:
                continue
            if len(selected_nodes) >= self.config.max_schema_nodes:
                break
            selected_nodes[concept.schema] = concept

        eligible_edges = [
            relation
            for relation in scored_relations
            if relation.source_schema in selected_nodes and relation.target_schema in selected_nodes
        ]
        eligible_edges.sort(key=lambda item: (-item.edge_score, item.key))
        selected_relations = tuple(eligible_edges[: self.config.max_schema_edges])
        involved = {name for relation in selected_relations for name in (relation.source_schema, relation.target_schema)}
        final_nodes = tuple(
            concept
            for concept in selected_nodes.values()
            if concept.schema in involved or concept.schema in seed_names or concept.schema in exact_schemas
        )
        confidence = self._selection_confidence(final_nodes, selected_relations, terms)
        return RelevantSchema(
            concepts=tuple(sorted(final_nodes, key=lambda item: (-item.final_score, item.schema))),
            relations=selected_relations,
            query_terms=tuple(terms),
            selection_confidence=confidence,
            fallback_used=fallback or confidence < self.config.medium_confidence,
        )

    @staticmethod
    def _relation_trigger_score(relation: SchemaRelation, text: str) -> float:
        """判断正文是否出现关系中英文名称或常见中文触发词。"""

        triggers = list(RELATION_TRIGGERS.get(relation.relation_en.upper(), ()))
        if relation.relation_zh:
            triggers.append(relation.relation_zh)
        if relation.relation_en:
            triggers.append(relation.relation_en)
        lowered = text.lower()
        return 1.0 if any(trigger and trigger.lower() in lowered for trigger in triggers) else 0.0

    @staticmethod
    def _selection_confidence(
        concepts: Sequence[SchemaConcept],
        relations: Sequence[SchemaRelation],
        terms: Sequence[str],
    ) -> float:
        """根据头部得分、术语覆盖、图连通度和边界分差估计选择可信度。"""

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
        margin = ranked[-1] if len(ranked) == 1 else max(0.0, ranked[min(len(ranked), 3) - 1] - ranked[-1])
        confidence = 0.55 * top_average + 0.15 * coverage + 0.20 * connectivity + 0.10 * margin
        return max(0.0, min(1.0, confidence))


__all__ = [
    "RelevantSchema",
    "SchemaConcept",
    "SchemaRelation",
    "SchemaSelector",
    "SchemaSelectorConfig",
    "build_chunk_query_text",
]
