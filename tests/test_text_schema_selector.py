"""整篇文本 Chunk 上下文与两级 Schema 选择算法的离线测试。"""
from __future__ import annotations

import json
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest

from src.extractors import collect_chunks
from src.extractors.text_extractor import (
    Neo4jSchemaRepository,
    SchemaSelector,
    SchemaSelectorConfig,
    build_document_context,
)


CONCEPT_ROWS = [
    {
        "schema": "Basin",
        "zh_name": "盆地",
        "category": "地理与构造单元",
        "description": "大型地质空间单元",
        "examples": ["鄂尔多斯盆地"],
        "embedding": [1.0, 0.0],
    },
    {
        "schema": "Formation",
        "zh_name": "组",
        "category": "地层与地质时代",
        "description": "岩石地层基本单位",
        "examples": ["延长组"],
        "embedding": [0.9, 0.1],
    },
    {
        "schema": "SourceRock",
        "zh_name": "烃源岩",
        "category": "油气成藏要素",
        "description": "能够生成油气的岩石",
        "examples": ["长7烃源岩"],
        "embedding": [0.8, 0.2],
    },
]


class _FakeEmbeddingClient:
    """返回固定二维向量，避免测试访问外部 Embedding 服务。"""

    def encode_one(self, text: str) -> list[float]:
        """对任何非空查询返回统一测试向量。"""

        return [1.0, 0.0] if text else []

    @staticmethod
    def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
        """计算测试所需的二维余弦相似度。"""

        left_values = list(left)
        right_values = list(right)
        dot = sum(a * b for a, b in zip(left_values, right_values))
        left_norm = sum(a * a for a in left_values) ** 0.5
        right_norm = sum(a * a for a in right_values) ** 0.5
        return dot / (left_norm * right_norm)


def _query_runner(query: str, parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
    """根据 Cypher 结构返回概念、向量、邻域或诱导关系模拟记录。"""

    if "db.index.vector.queryNodes" in query:
        return [
            {**row, "vector_score": 0.92 - index * 0.04}
            for index, row in enumerate(CONCEPT_ROWS)
        ][: int(parameters["top_k"])]
    if "source.schema IN $seed_schemas OR" in query:
        return [
            {
                "source_schema": "Basin",
                "source_zh_name": "盆地",
                "source_category": "地理与构造单元",
                "source_description": "大型地质空间单元",
                "source_examples": ["鄂尔多斯盆地"],
                "target_schema": "Formation",
                "target_zh_name": "组",
                "target_category": "地层与地质时代",
                "target_description": "岩石地层基本单位",
                "target_examples": ["延长组"],
                "relation_en": "CONTAINS",
                "relation_zh": "包含",
            }
        ]
    if "source.schema IN $schemas AND" in query:
        return [
            {
                "source_schema": "Basin",
                "relation_en": "CONTAINS",
                "relation_zh": "包含",
                "target_schema": "Formation",
            }
        ]
    return [dict(row) for row in CONCEPT_ROWS]


def _chunks() -> list[dict[str, Any]]:
    """构造包含两个章节、顺序乱序输入的整篇论文文本 Chunk。"""

    return [
        {
            "id": "c2",
            "order": 2,
            "document_id": "doc-1",
            "section_id": "s2",
            "section_title": "烃源岩特征",
            "modality": "text",
            "text": "长7烃源岩具有较高生烃潜力。",
        },
        {
            "id": "c1",
            "order": 1,
            "document_id": "doc-1",
            "section_id": "s1",
            "section_title": "区域地质概况",
            "schemaKeys": ["烃源岩"],
            "modality": "text",
            "text": "鄂尔多斯盆地包含延长组。",
        },
    ]


def test_document_context_sorts_and_indexes_all_chunks() -> None:
    """整篇预处理应按 order 排序并保留章节、术语和代表性 Chunk 信息。"""

    context = build_document_context(_chunks())

    assert [chunk["id"] for chunk in context.chunks] == ["c1", "c2"]
    assert context.chunk_indexes == {"c1": 0, "c2": 1}
    assert context.section_chunks == {"s1": ("c1",), "s2": ("c2",)}
    assert "鄂尔多斯盆地" in context.domain_terms
    assert set(context.representative_chunk_ids) == {"c1", "c2"}


def test_document_context_rejects_mixed_documents() -> None:
    """一次调用混入多篇论文时应明确报错，避免上下文和 Schema 先验污染。"""

    chunks = _chunks()
    chunks[1]["document_id"] = "doc-2"

    with pytest.raises(ValueError, match="只能处理一篇论文"):
        build_document_context(chunks)


def test_collect_chunks_propagates_section_summary_and_schema_keys() -> None:
    """第三阶段章节总结和 schemaKeys 应随所属章节传递给每个文本 Chunk。"""

    payload = {
        "document": {
            "id": "doc-keys",
            "summary": "论文研究盆地构造。",
            "schemaKeys": ["盆地", "构造单元"],
            "sections": [
                {
                    "id": "s1",
                    "title": "断裂特征",
                    "summary": "本章研究断层与裂缝。",
                    "schemaKeys": ["断层", "裂缝"],
                    "chunks": [{"id": "c1", "modality": "text", "text": "正文未直接出现关键词。"}],
                    "children": [],
                }
            ],
        }
    }

    chunk = collect_chunks(payload)[0]

    assert chunk["document_summary"] == "论文研究盆地构造。"
    assert chunk["document_schema_keys"] == ["盆地", "构造单元"]
    assert chunk["section_summary"] == "本章研究断层与裂缝。"
    assert chunk["schemaKeys"] == ["断层", "裂缝"]


def test_two_level_selector_builds_document_pool_and_chunk_subgraphs() -> None:
    """选择器应先生成文档池，再为每个 Chunk 返回受节点和关系预算约束的子图。"""

    embedding = _FakeEmbeddingClient()
    repository = Neo4jSchemaRepository(query_runner=_query_runner, embedding_client=embedding)
    selector = SchemaSelector(
        repository=repository,
        embedding_client=embedding,
        config=SchemaSelectorConfig(
            document_top_k=3,
            max_schema_nodes=3,
            max_schema_edges=2,
            min_vector_score=0.1,
        ),
    )

    result = selector.prepare_document(_chunks())

    assert {item.schema for item in result.document_schema_pool.concepts} == {
        "Basin",
        "Formation",
        "SourceRock",
    }
    assert set(result.chunk_schemas) == {"c1", "c2"}
    first = result.chunk_schemas["c1"]
    assert {item.schema for item in first.concepts} >= {"Basin", "Formation"}
    assert any(relation.key == ("Basin", "CONTAINS", "Formation") for relation in first.relations)
    source_rock = next(item for item in first.concepts if item.schema == "SourceRock")
    assert source_rock.schema_key_score == 1.0
    assert "章节 schemaKeys 精确命中" in source_rock.selection_reasons
    assert len(first.concepts) <= 3
    assert len(first.relations) <= 2


def test_repository_falls_back_to_full_cosine_when_vector_index_is_missing() -> None:
    """向量索引查询失败时应自动读取全部概念并执行内存余弦召回。"""

    def failing_vector_runner(query: str, parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
        """模拟向量索引不存在，但允许全概念读取成功。"""

        if "db.index.vector.queryNodes" in query:
            raise RuntimeError("index not found")
        return [dict(row) for row in CONCEPT_ROWS]

    embedding = _FakeEmbeddingClient()
    repository = Neo4jSchemaRepository(query_runner=failing_vector_runner, embedding_client=embedding)

    concepts, fallback = repository.vector_search([1.0, 0.0], top_k=2, min_score=0.1)

    assert fallback is True
    assert [item.schema for item in concepts] == ["Basin", "Formation"]


def _north_china_craton_chunks() -> list[dict[str, Any]]:
    """返回用户提供的华北克拉通论文 Chunk，供真实两级 Schema 选择演示。"""

    return [
        {
            "id": "0:text:0",
            "order": 0,
            "modality": "text",
            "text": "作为形成于3．8 Ga 的古老陆核，华北克拉通经历了多阶段多期次的构造演化( Wu and Zhong，1998; 翟明国等，2014) 前人研究认为，华北克拉通基底于约2．5 Ga 由多个微陆块拼合形成，可划分为胶辽 前淮 许昌 阜平 济宁和阿拉善地块 6 个地块 ( Zhai et al． ，2005; Zhai and Santosh，2011 )2．25～2． 0 Ga 期间，其西北部 中部及东北－东部发育3 条主裂谷带，裂谷盆地进一步演化为洋盆并启动俯冲过程( Santosh，2010; Zhai and Santosh，2011;翟明国等，2021) 历经长期俯冲－增生作用后，克拉通于 1． 97～1． 82 Ga 通过陆块碰撞完成最终克拉通化，形成孔兹岩带( 北部) 中央造山带( 中部) 及胶－辽－吉带( 东部) 3 条主缝合带( Zhai and Peng，2007; Santosh et al． ，2011; 刘 建 辉 等，2025 )1．80 Ga 时期华北克拉通进入伸展构造环境，伴随区域性伸展作用，克拉通内发育多组陆内裂谷系统基性岩墙群及非造山岩浆活动产物( 李江海等，2001; 翟明国，2004; 陈杰等，2024)",
        },
        {
            "id": "0:text:1",
            "order": 1,
            "modality": "text",
            "text": "1．80～1． 75 Ga 是华北克拉通伸展作用的起始年代，这个阶段与南缘熊耳群火山岩及五台 恒山地区的基性岩墙群密切相关，期间岩浆活动记录了克拉通早期减薄和陆内伸展的地质过程。前人对这些岩浆事件进行了大量研究，认为这一时期的构造活动是 Columbia 超大陆裂解的初始( 陆松年等，2002;翟明 国 等，2014; 相 振 群，2014; Huang et al． ，2020) ，但对华北克拉通西南缘的研究仍有些薄弱( 赵太平等，2001，2004; 彭澎等，2004; 徐欢等，2020; 相振群等，2020; 李秋根等，2024) ，对地质事件的性质和构造背景的认识尚存在争议，如 Zhao 等( 2009) 根据火山岩地球化学特点，认为南缘的熊耳群火山岩具有钙碱性特点，形成于大陆边缘弧环境，而翟明国( 2004) 赵太平等( 2004) 姜盛洪( 2015)等认为其形成于与地幔柱相关的大陆裂解环境，Zhao 等( 2018) 相振群等( 2020) 认为南缘登封地区石秤花岗闪长岩是碰撞后形成的 因此，查明华北克拉通西南缘1．8 ～ 1．6 Ga 的构造事件性质及其动力学机制，对于探究华北克拉通的构造环境转变及其与哥伦比亚超大陆裂解的关系有着重要的意义。",
        },
        {
            "id": "0:text:2",
            "order": 2,
            "modality": "text",
            "text": "在1．80～1．75 Ga 期间，华北克拉通形成了广泛发育的双峰式火成岩组合，如熊耳群的双峰式火山－侵入岩。 最近，作者在位于鄂尔多斯盆地西南缘的陇县骑安沟地区，识别出一套新的由玄武安山岩与花岗斑岩组成的典型双峰式火成岩组合，对该套火成岩的岩石学特征 地球化学组成进行了研究，并进行了锆石 U-Pb 定年分析，厘定其成因机制及形成时代，进而为华北克拉通西部－鄂尔多斯盆地在中元古代的构造演化过程提供了重要证据，同时探究了其与南缘熊耳群岩浆事件及哥伦比亚超大陆裂解的关系。",
        },
    ]


def _print_schema_selection_result(result: Any) -> None:
    """按文档预处理、文档候选池和逐 Chunk 子图三个阶段打印过程信息。"""

    document = result.document
    print("\n" + "=" * 88)
    print("[阶段 1] 整篇 Chunk 批次预处理")
    print("=" * 88)
    print(f"document_id       : {document.document_id or '(输入未提供)'}")
    print(f"chunk_count       : {len(document.chunks)}")
    print(f"ordered_chunk_ids : {[chunk['id'] for chunk in document.chunks]}")
    print(f"section_titles    : {list(document.section_titles)}")
    print(f"document_schemaKeys: {list(document.document_schema_keys)}")
    print(f"section_schemaKeys : {dict(document.section_schema_keys)}")
    print(f"domain_terms      : {list(document.domain_terms)}")
    print(f"representatives   : {list(document.representative_chunk_ids)}")
    print("topic_profile:")
    print(document.topic_profile)

    pool = result.document_schema_pool
    print("\n" + "=" * 88)
    print("[阶段 2] 文档级 Schema 候选池")
    print("=" * 88)
    print(f"selection_confidence : {pool.selection_confidence:.4f}")
    print(f"fallback_used        : {pool.fallback_used}")
    print(f"concept_count        : {len(pool.concepts)}")
    print(f"relation_count       : {len(pool.relations)}")
    for rank, concept in enumerate(pool.concepts, start=1):
        print(
            f"  [{rank:02d}] {concept.schema:<28} {concept.zh_name:<14} "
            f"final={concept.final_score:.4f} vector={concept.vector_score:.4f} "
            f"lexical={concept.lexical_score:.4f} schemaKeys={concept.schema_key_score:.4f} "
            f"context={concept.context_score:.4f} "
            f"reasons={list(concept.selection_reasons)}"
        )
    print("文档池内部关系：")
    for relation in pool.relations:
        print(
            f"  {relation.source_schema} -[{relation.relation_en}/{relation.relation_zh}]-> "
            f"{relation.target_schema}"
        )

    print("\n" + "=" * 88)
    print("[阶段 3] 每个 Chunk 的局部 Schema 子图")
    print("=" * 88)
    for chunk in document.chunks:
        chunk_id = str(chunk["id"])
        selected = result.chunk_schemas[chunk_id]
        local = document.local_context(chunk_id)
        print(f"\n--- Chunk {chunk_id} / order={chunk.get('order')} ---")
        print(f"正文预览：{str(chunk.get('text') or '')[:160]}...")
        print(f"相邻上文：{local['previous_tail'][:100] or '(无)'}")
        print(f"相邻下文：{local['next_head'][:100] or '(无)'}")
        print(f"query_terms          : {list(selected.query_terms)}")
        print(f"selection_confidence : {selected.selection_confidence:.4f}")
        print(f"fallback_used        : {selected.fallback_used}")
        print("局部概念：")
        for concept in selected.concepts:
            print(
                f"  {concept.schema:<28} {concept.zh_name:<14} "
                f"final={concept.final_score:.4f} vector={concept.vector_score:.4f} "
                f"lexical={concept.lexical_score:.4f} schemaKeys={concept.schema_key_score:.4f} "
                f"document={concept.document_score:.4f} "
                f"reasons={list(concept.selection_reasons)}"
            )
        print("局部关系：")
        for relation in selected.relations:
            print(
                f"  {relation.source_schema} -[{relation.relation_en}/{relation.relation_zh}]-> "
                f"{relation.target_schema}  edge={relation.edge_score:.4f}"
            )

    print("\n" + "=" * 88)
    print("[完整 JSON 返回结果]")
    print("=" * 88)
    payload = {
        "document_schema_pool": pool.to_dict(),
        "chunk_schemas": {
            chunk_id: selected.to_dict()
            for chunk_id, selected in result.chunk_schemas.items()
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_and_save_schema_selection_result(result: Any) -> Path:
    """捕获完整过程日志，在终端打印后同步保存到项目 logs 目录。"""

    buffer = StringIO()
    with redirect_stdout(buffer):
        _print_schema_selection_result(result)
    output = buffer.getvalue()
    print(output, end="")

    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "text_schema_selection_north_china_craton.log"
    log_path.write_text(output, encoding="utf-8")
    print(f"\n日志已保存：{log_path}")
    return log_path



def test_live_north_china_craton_chunks_print_schema_selection_process() -> None:
    """使用真实 Neo4j 和 Embedding 打印华北克拉通 Chunk 的两级选择过程及结果。"""

    selector = SchemaSelector()
    result = selector.prepare_document(_north_china_craton_chunks())

    log_path = _print_and_save_schema_selection_result(result)

    assert len(result.document.chunks) == 3
    assert set(result.chunk_schemas) == {"0:text:0", "0:text:1", "0:text:2"}
    assert result.document_schema_pool.concepts
    assert log_path.exists()
    assert "[阶段 1] 整篇 Chunk 批次预处理" in log_path.read_text(encoding="utf-8")
