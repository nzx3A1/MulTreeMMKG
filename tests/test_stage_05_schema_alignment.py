"""Stage 05 Schema 对齐的规则、语义判定、关系约束和输出测试。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.schemaProcess.entity_aligner import EntityAligner
from src.schemaProcess.entity_rules import EntityRuleMapper
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
    build_entity_embedding_text,
    build_schema_embedding_text,
)
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
    source = {"id": "e1", "name": "总面积", "type": "TableCell", "custom": {"keep": True}}
    result = aligner.align_many([source])[0]
    assert result["type"] == "TableCell"
    assert result["raw_type"] == "TableCell"
    assert result["schema_type"] is None
    assert result["schema_alignment"]["status"] == "UNMAPPED"
    assert result["custom"] == {"keep": True}


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
    )
    aligned, entity_report, relation_report = pipeline.align_payload(payload, "fixture.json")
    graph = aligned["graphs"][0]
    assert aligned["custom_root"] == {"must": "remain"}
    assert graph["events"] == [{"id": "event-kept"}]
    assert graph["entities"][0]["type"] == "oil_gas_field"
    assert graph["entities"][0]["raw_type"] == "oil_gas_field"
    assert graph["entities"][0]["schema_type"] == "OilGasField"
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
