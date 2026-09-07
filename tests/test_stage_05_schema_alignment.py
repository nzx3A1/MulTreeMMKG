"""Stage 05 Schema 对齐的规则、语义判定、关系约束和输出测试。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from config.model_config import EmbeddingConfig
from src.schemaProcess.category_router import SchemaCategoryRouter
from src.schemaProcess.entity_aligner import EntityAligner
from src.schemaProcess.entity_cache import EntityMappingCache, build_entity_mapping_key
from src.schemaProcess.entity_rules import DEFAULT_RULES_PATH, EntityRuleMapper
from src.schemaProcess.models import (
    SchemaConcept,
    SchemaRelation,
    SchemaSnapshot,
    SelectionDecision,
    SemanticCandidate,
)
from src.schemaProcess.pipeline import DEFAULT_INPUT_PATH, SchemaAlignmentPipeline
from src.schemaProcess.relation_aligner import RelationAligner
from src.schemaProcess.schema_repository import InMemorySchemaRepository
from src.schemaProcess.semantic_retriever import (
    VectorSemanticRetriever,
    build_entity_embedding_text,
    build_schema_embedding_text,
)
from src.utils.embedding_client import EmbeddingClient
from src.utils.json_io import read_json


def _snapshot(*schemas: str, relations: Sequence[SchemaRelation] = ()) -> SchemaSnapshot:
    return SchemaSnapshot(
        concepts=[
            SchemaConcept(
                schema=schema,
                zh_name=schema,
                category="测试类别",
                description=f"{schema} 的测试定义",
                examples=(f"{schema}示例",),
            )
            for schema in schemas
        ],
        relations=list(relations),
        source="unit-test",
    )


class _NeverSelector:
    def select_entities(self, requests: Any) -> Any:
        raise AssertionError("高精度规则命中时不应调用 LLM")

    def select_relations(self, requests: Any) -> Any:
        raise AssertionError("单候选或精确关系命中时不应调用 LLM")


class _FixedSelector:
    def __init__(
        self,
        entity_decision: SelectionDecision | None = None,
        relation_decision: SelectionDecision | None = None,
    ) -> None:
        self.entity_decision = entity_decision or SelectionDecision(None, 0.9, "没有合适概念")
        self.relation_decision = relation_decision or SelectionDecision(None, 0.9, "没有合适关系")

    def select_entities(self, requests: Sequence[Any]) -> list[SelectionDecision]:
        return [self.entity_decision for _ in requests]

    def select_relations(self, requests: Sequence[Any]) -> list[SelectionDecision]:
        return [self.relation_decision for _ in requests]


class _FixedRetriever:
    def __init__(self, candidates: Sequence[SemanticCandidate]) -> None:
        self.candidates = list(candidates)

    def retrieve_many(self, entities: Sequence[Mapping[str, Any]]) -> list[list[SemanticCandidate]]:
        return [list(self.candidates) for _ in entities]


class _CountingRetriever(_FixedRetriever):
    """记录向量召回次数，用于验证实体去重和缓存是否生效。"""

    def __init__(self, candidates: Sequence[SemanticCandidate]) -> None:
        super().__init__(candidates)
        self.calls: list[int] = []

    def retrieve_many(self, entities: Sequence[Mapping[str, Any]]) -> list[list[SemanticCandidate]]:
        self.calls.append(len(entities))
        return super().retrieve_many(entities)


class _ConstantEmbedding:
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    @staticmethod
    def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        return 1.0


def test_entity_rules_honor_professional_priority() -> None:
    snapshot = _snapshot(
        "Basin",
        "Formation",
        "StratigraphicMember",
        "SubMember",
        "OilGasField",
        "OilGasReservoir",
        "Reservoir",
        "ReservoirInterval",
        "SourceRock",
        "Lithology",
    )
    mapper = EntityRuleMapper(snapshot)
    cases = {
        "鄂尔多斯盆地": "Basin",
        "马家沟组": "Formation",
        "马五段": "StratigraphicMember",
        "马五亚段": "SubMember",
        "靖边气田": "OilGasField",
        "马五气藏": "OilGasReservoir",
        "白云岩储层": "Reservoir",
        "马五储层段": "ReservoirInterval",
        "长7烃源岩": "SourceRock",
        "花岗岩": "Lithology",
        "白云岩": "Lithology",
    }
    for name, expected in cases.items():
        matched = mapper.map_entity({"name": name, "type": "open_type"})
        assert matched is not None
        assert matched.schema_type == expected

    # 专业特异规则优先，不能先被通用“岩石”语义吞掉。
    assert mapper.map_entity({"name": "长7烃源岩", "type": "rock_type"}).schema_type == "SourceRock"


def test_embedding_text_contains_all_required_fields() -> None:
    concept = SchemaConcept(
        schema="Reservoir",
        zh_name="储层",
        category="油气成藏要素",
        description="储集油气的岩层",
        examples=("白云岩储层",),
    )
    concept_text = build_schema_embedding_text(concept)
    for expected in ("schema: Reservoir", "zhName: 储层", "category:", "description:", "examples:"):
        assert expected in concept_text

    entity_text = build_entity_embedding_text(
        {
            "name": "未知对象",
            "type": "open_type",
            "type_zh": "开放类型",
            "attributes": {"depth": "3500m"},
            "provenance": "原文证据",
        }
    )
    for expected in ("name: 未知对象", "raw_type: open_type", "type_zh:", "attributes:", "provenance:"):
        assert expected in entity_text


def test_entity_semantic_none_preserves_raw_type_and_marks_unmapped() -> None:
    snapshot = _snapshot("Reservoir")
    concept = snapshot.concepts_by_schema["Reservoir"]
    aligner = EntityAligner(
        rule_mapper=EntityRuleMapper(snapshot),
        retriever=_FixedRetriever([SemanticCandidate(concept, 0.81)]),
        selector=_FixedSelector(entity_decision=SelectionDecision(None, 0.93, "该对象是表格单元格")),
    )
    source = {"id": "e1", "name": "总面积", "type": "open_type", "custom": {"keep": True}}
    result = aligner.align_many([source])[0]
    assert result["type"] == "open_type"
    assert result["raw_type"] == "open_type"
    assert result["schema_type"] is None
    assert result["schema_type_zh"] is None
    assert result["schema_alignment"]["status"] == "UNMAPPED"
    assert result["custom"] == {"keep": True}


def test_predefined_table_and_chart_types_skip_all_schema_mapping() -> None:
    snapshot = _snapshot("Reservoir", "TableCell")
    aligner = EntityAligner(
        rule_mapper=EntityRuleMapper(snapshot),
        retriever=_FixedRetriever([]),
        selector=_NeverSelector(),
    )
    entities = [
        {"id": "cell", "name": "单元格", "type": "TableCell"},
        {"id": "axis", "name": "X轴：孔隙度", "type": "chart_axis"},
        {"id": "trend", "name": "整体上升", "type": "chart_trend"},
        {"id": "theme", "name": "沉积相分布", "type": "MapTheme"},
        {"id": "reservoir", "name": "目标储层", "type": "Reservoir"},
    ]

    results = aligner.align_many(entities, ["table", "image", "image", "image", "image"])

    assert [item["schema_alignment"]["status"] for item in results] == [
        "SKIPPED",
        "SKIPPED",
        "SKIPPED",
        "SKIPPED",
        "RULE_MAPPED",
    ]
    assert all(item["schema_type"] is None for item in results[:4])
    assert all(
        item["schema_alignment"]["method"] == "predefined_type_filter"
        for item in results[:4]
    )
    assert results[4]["schema_type"] == "Reservoir"


def test_pipeline_skips_relations_connected_to_predefined_types() -> None:
    snapshot = _snapshot("Reservoir")
    payload = {
        "graphs": [
            {
                "entities": [
                    {"id": "axis", "name": "X轴：孔隙度", "type": "chart_axis"},
                    {"id": "trend", "name": "整体上升", "type": "chart_trend"},
                ],
                "relations": [
                    {
                        "id": "rel",
                        "type": "varies_along",
                        "source_id": "trend",
                        "target_id": "axis",
                    }
                ],
                "metadata": {"chunk_id": "image:1", "modality": "image"},
            }
        ]
    }
    pipeline = SchemaAlignmentPipeline(
        repository=InMemorySchemaRepository(snapshot),
        embedding_client=_ConstantEmbedding(),
        selector=_NeverSelector(),
        entity_mapping_cache_path=None,
        schema_embedding_cache_path=None,
    )

    aligned, entity_report, relation_report = pipeline.align_payload(payload)

    graph = aligned["graphs"][0]
    assert graph["relations"][0]["schema_alignment"]["status"] == "SKIPPED"
    assert graph["metadata"]["schema_alignment"] == {
        "entity_status_counts": {"SKIPPED": 2},
        "relation_status_counts": {"SKIPPED": 1},
    }
    assert entity_report["statistics"]["unmapped_entity_count"] == 0
    assert relation_report["statistics"]["unmapped_relation_count"] == 0


def test_entity_semantic_mapping_adds_schema_type_zh() -> None:
    concept = SchemaConcept(
        schema="Reservoir",
        zh_name="储层",
        category="油气成藏要素",
        description="储集油气的岩层",
    )
    snapshot = SchemaSnapshot([concept], [], source="unit-test")
    aligner = EntityAligner(
        rule_mapper=EntityRuleMapper(snapshot),
        retriever=_FixedRetriever([SemanticCandidate(concept, 0.91)]),
        selector=_FixedSelector(
            entity_decision=SelectionDecision("Reservoir", 0.95, "语义与储层一致")
        ),
    )

    result = aligner.align_many(
        [{"id": "e1", "name": "目标对象", "type": "open_type"}]
    )[0]

    assert result["schema_type"] == "Reservoir"
    assert result["schema_type_zh"] == "储层"
    assert result["schema_alignment"]["status"] == "SEMANTIC_MAPPED"


def test_relation_uses_original_semantics_for_multiple_endpoint_candidates() -> None:
    relations = [
        SchemaRelation("SourceRock", "GENERATES", "生成", "Hydrocarbon"),
        SchemaRelation("SourceRock", "EXPELS", "排出", "Hydrocarbon"),
    ]
    snapshot = _snapshot("SourceRock", "Hydrocarbon", relations=relations)
    aligner = RelationAligner(
        snapshot,
        selector=_FixedSelector(
            relation_decision=SelectionDecision("EXPELS", 0.96, "原文明确描述排烃")
        ),
    )
    source = {
        "id": "r1",
        "type": "RELEASES_HYDROCARBON",
        "relation_name": "排烃",
        "type_zh": "释放",
        "provenance": "烃源岩向储层排出油气",
    }
    result = aligner.align_many([(source, "SourceRock", "Hydrocarbon")])[0]
    assert result["type"] == "RELEASES_HYDROCARBON"
    assert result["raw_relation"] == "RELEASES_HYDROCARBON"
    assert result["schema_relation"] == "EXPELS"
    assert result["schema_alignment"]["status"] == "SEMANTIC_MAPPED"


def test_related_to_is_rejected_for_specific_original_relation() -> None:
    relation = SchemaRelation("Trap", "RELATED_TO", "相关于", "Lithology")
    snapshot = _snapshot("Trap", "Lithology", relations=[relation])
    aligner = RelationAligner(snapshot, selector=_NeverSelector())

    specific = aligner.align_many(
        [({"type": "CONTROLS", "type_zh": "控制"}, "Trap", "Lithology")]
    )[0]
    assert specific["schema_relation"] is None
    assert specific["schema_alignment"]["status"] == "UNMAPPED"

    weak = aligner.align_many(
        [({"type": "RELATED_TO", "type_zh": "相关"}, "Trap", "Lithology")]
    )[0]
    assert weak["schema_relation"] == "RELATED_TO"
    assert weak["schema_alignment"]["status"] == "DIRECT_MAPPED"


def test_pipeline_preserves_graph_and_builds_three_output_payloads() -> None:
    relation = SchemaRelation("OilGasField", "LOCATED_IN", "位于", "Basin")
    snapshot = _snapshot("OilGasField", "Basin", relations=[relation])
    payload = {
        "_status": "completed",
        "statistics": {"entity_count": 2, "relation_count": 1},
        "custom_root": {"must": "remain"},
        "graphs": [
            {
                "entities": [
                    {"id": "field", "name": "靖边气田", "type": "oil_gas_field"},
                    {"id": "basin", "name": "鄂尔多斯盆地", "type": "basin"},
                ],
                "relations": [
                    {
                        "id": "rel",
                        "type": "LOCATED_IN",
                        "relation_name": "located_in",
                        "source_id": "field",
                        "target_id": "basin",
                        "provenance": "靖边气田位于鄂尔多斯盆地",
                    }
                ],
                "events": [{"id": "event-kept"}],
                "metadata": {"chunk_id": "0:text:0", "modality": "text"},
            }
        ],
    }
    pipeline = SchemaAlignmentPipeline(
        repository=InMemorySchemaRepository(snapshot),
        embedding_client=_ConstantEmbedding(),
        selector=_NeverSelector(),
        entity_mapping_cache_path=None,
        schema_embedding_cache_path=None,
    )
    aligned, entity_report, relation_report = pipeline.align_payload(payload, "fixture.json")
    graph = aligned["graphs"][0]
    assert aligned["custom_root"] == {"must": "remain"}
    assert graph["events"] == [{"id": "event-kept"}]
    assert graph["entities"][0]["type"] == "oil_gas_field"
    assert graph["entities"][0]["raw_type"] == "oil_gas_field"
    assert graph["entities"][0]["schema_type"] == "OilGasField"
    assert graph["entities"][0]["schema_type_zh"] == "OilGasField"
    assert graph["relations"][0]["type"] == "LOCATED_IN"
    assert graph["relations"][0]["schema_relation"] == "LOCATED_IN"
    assert entity_report["statistics"]["unmapped_entity_count"] == 0
    assert relation_report["statistics"]["unmapped_relation_count"] == 0


def test_rules_against_existing_stage04_extraction_examples() -> None:
    """直接使用仓库当前抽取 JSON 中的案例验证规则，不触发外部模型。"""

    if not DEFAULT_INPUT_PATH.exists():
        return
    payload = read_json(DEFAULT_INPUT_PATH)
    entities = [entity for graph in payload["graphs"] for entity in graph.get("entities", [])]
    by_name = {entity.get("name"): entity for entity in entities}
    snapshot = _snapshot(
        "Basin",
        "Formation",
        "SubMember",
        "OilGasField",
        "Reservoir",
        "SourceRock",
        "Lithology",
    )
    mapper = EntityRuleMapper(snapshot)
    expected = {
        "鄂尔多斯盆地": "Basin",
        "马家沟组": "Formation",
        "马五 亚段": "SubMember",
        "靖边气田": "OilGasField",
        "白云岩储层": "Reservoir",
        "白云岩": "Lithology",
    }
    for name, schema_type in expected.items():
        assert name in by_name, f"现有 Stage 04 文件缺少测试案例：{name}"
        assert mapper.map_entity(by_name[name]).schema_type == schema_type


def test_vector_high_confidence_accepts_without_llm() -> None:
    """验证高相似度且有明显间隔的向量候选会跳过 LLM。"""

    snapshot = _snapshot("Reservoir", "Basin")
    candidates = [
        SemanticCandidate(snapshot.concepts_by_schema["Reservoir"], 0.95),
        SemanticCandidate(snapshot.concepts_by_schema["Basin"], 0.62),
    ]
    aligner = EntityAligner(
        rule_mapper=EntityRuleMapper(snapshot),
        retriever=_FixedRetriever(candidates),
        selector=_NeverSelector(),
    )

    result = aligner.align_many([{"name": "未知储集对象", "type": "open_type"}])[0]

    assert result["schema_type"] == "Reservoir"
    assert result["schema_alignment"]["method"] == "vector_auto_accept"


def test_entity_mapping_deduplicates_and_persists_cache(tmp_path: Path) -> None:
    """验证重复实体只召回一次，第二次运行直接命中持久化映射缓存。"""

    snapshot = _snapshot("Reservoir", "Basin")
    concept = snapshot.concepts_by_schema["Reservoir"]
    cache = EntityMappingCache(tmp_path / "entity-cache.json", "test-namespace")
    first_retriever = _CountingRetriever([SemanticCandidate(concept, 0.95)])
    first_aligner = EntityAligner(
        rule_mapper=EntityRuleMapper(snapshot),
        retriever=first_retriever,
        selector=_NeverSelector(),
        mapping_cache=cache,
    )
    entities = [
        {"id": "e1", "name": "未知储集对象", "type": "open_type"},
        {"id": "e2", "name": "未知储集对象", "type": "open_type"},
    ]

    first = first_aligner.align_many(entities)
    assert first_retriever.calls == [1]
    assert [item["schema_type"] for item in first] == ["Reservoir", "Reservoir"]

    second_cache = EntityMappingCache(tmp_path / "entity-cache.json", "test-namespace")
    second_retriever = _CountingRetriever([SemanticCandidate(concept, 0.1)])
    second_aligner = EntityAligner(
        rule_mapper=EntityRuleMapper(snapshot),
        retriever=second_retriever,
        selector=_NeverSelector(),
        mapping_cache=second_cache,
    )
    second = second_aligner.align_many(entities)

    assert second_retriever.calls == []
    assert second_aligner.cache_hits == 2
    assert all(item["schema_alignment"]["cache_hit"] for item in second)
    assert build_entity_mapping_key(entities[0]) in second_cache.entries


def test_schema_embeddings_are_loaded_from_persistent_cache(tmp_path: Path) -> None:
    """验证同一 Schema 哈希再次运行时不重复调用 embedding 服务。"""

    class _CountingEmbedding:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            self.calls += 1
            return [[1.0, 0.0] for _ in texts]

        @staticmethod
        def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
            return 1.0

    snapshot = _snapshot("Reservoir", "Basin")
    cache_path = tmp_path / "schema-embeddings.json"
    first_client = _CountingEmbedding()
    VectorSemanticRetriever(
        snapshot,
        embedding_client=first_client,
        batch_size=1,
        schema_embedding_cache_path=cache_path,
    ).prepare()
    second_client = _CountingEmbedding()
    VectorSemanticRetriever(
        snapshot,
        embedding_client=second_client,
        batch_size=1,
        schema_embedding_cache_path=cache_path,
    ).prepare()

    assert first_client.calls == 2
    assert second_client.calls == 0


def test_llm_selector_sends_requests_in_batches() -> None:
    """验证 LLM 选择器按 batch_size 分批，而不是逐实体请求。"""

    class _BatchLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat_json(self, messages: Any) -> dict[str, Any]:
            self.calls += 1
            user_content = messages[-1]["content"]
            count = user_content.count('"index":') - 1
            return {
                "decisions": [
                    {"index": index, "selected": "Reservoir", "confidence": 0.9, "reason": "批量测试"}
                    for index in range(count)
                ]
            }

    from src.schemaProcess.llm_selector import SchemaLLMSelector

    concept = SchemaConcept("Reservoir", "储层", "测试", "定义")
    requests = [
        ({"name": f"对象{index}", "type": "open_type"}, [SemanticCandidate(concept, 0.4)])
        for index in range(5)
    ]
    client = _BatchLLM()
    decisions = SchemaLLMSelector(client, batch_size=2).select_entities(requests)

    assert len(decisions) == 5
    assert client.calls == 3


def test_embedding_client_splits_requests_by_batch_size() -> None:
    """验证 EmbeddingClient 的公共批量接口按批大小拆分请求。"""

    client = EmbeddingClient(EmbeddingConfig(batch_size=2, dimensions=None))
    calls: list[list[str]] = []

    def fake_get_embeddings(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    client._get_embeddings = fake_get_embeddings  # type: ignore[method-assign]
    vectors = client.embed_batch(["a", "b", "c", "d", "e"])

    assert calls == [["a", "b"], ["c", "d"], ["e"]]
    assert len(vectors) == 5


def test_category_router_limits_vector_retrieval_and_falls_back() -> None:
    """验证可靠一级分类只召回类别内概念，未知实体则回退全 Schema。"""

    snapshot = SchemaSnapshot(
        concepts=[
            SchemaConcept("Basin", "盆地", "地理与构造单元", "构造地理单元"),
            SchemaConcept("Lithology", "岩性", "岩石、矿物与沉积体系", "岩石类型"),
            SchemaConcept("Reservoir", "储层", "油气成藏要素", "储集层"),
            SchemaConcept("PoreStructure", "孔隙结构", "储层结构与评价实体", "孔隙结构"),
        ],
        relations=[],
        source="unit-test",
    )
    router = SchemaCategoryRouter(snapshot, DEFAULT_RULES_PATH)
    entity = {"name": "页岩沉积特征", "type": "open_type"}
    route = router.route(entity)
    retriever = VectorSemanticRetriever(
        snapshot,
        embedding_client=_ConstantEmbedding(),
        top_k=5,
        schema_embedding_cache_path=None,
    )

    candidates = retriever.retrieve_many([entity], [route.category])[0]

    assert route.category == "岩石、矿物与沉积体系"
    assert [candidate.concept.schema for candidate in candidates] == ["Lithology"]
    assert router.route({"name": "X-17", "type": "unknown"}).category is None
    assert router.route({"name": "储层孔隙特征", "type": "unknown"}).category is None


def test_pipeline_uses_category_local_retrieval_before_vector_gate() -> None:
    """验证主流程在向量自动接受前已经应用一级类别候选过滤。"""

    snapshot = SchemaSnapshot(
        concepts=[
            SchemaConcept("Basin", "盆地", "地理与构造单元", "构造地理单元"),
            SchemaConcept("Lithology", "岩性", "岩石、矿物与沉积体系", "岩石类型"),
            SchemaConcept("Reservoir", "储层", "油气成藏要素", "储集层"),
        ],
        relations=[],
        source="unit-test",
    )
    payload = {
        "graphs": [
            {
                "entities": [{"id": "e1", "name": "页岩沉积特征", "type": "open_type"}],
                "relations": [],
                "metadata": {"modality": "text"},
            }
        ]
    }
    pipeline = SchemaAlignmentPipeline(
        repository=InMemorySchemaRepository(snapshot),
        embedding_client=_ConstantEmbedding(),
        selector=_NeverSelector(),
        entity_mapping_cache_path=None,
        schema_embedding_cache_path=None,
    )

    aligned, _, _ = pipeline.align_payload(payload)
    entity = aligned["graphs"][0]["entities"][0]

    assert entity["schema_type"] == "Lithology"
    assert entity["schema_alignment"]["category_route"] == "岩石、矿物与沉积体系"
    assert aligned["_schema_alignment"]["category_route_counts"] == {
        "岩石、矿物与沉积体系": 1
    }
