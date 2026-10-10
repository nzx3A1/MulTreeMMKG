"""验证 Stage 06 实体融合、关系迁移、证据汇集和来源守恒。"""
from __future__ import annotations

import json
from typing import Any

from tests._bootstrap import PROJECT_ROOT  # noqa: F401

from src.alignment.pipeline import MultimodalAlignmentPipeline


def _entity(
    entity_id: str,
    name: str,
    schema_type: str,
    schema_type_zh: str,
    *,
    raw_type: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造带 Stage 05 Schema 字段的测试实体。"""

    return {
        "id": entity_id,
        "name": name,
        "official_name": name,
        "type": schema_type,
        "type_zh": schema_type_zh,
        "raw_type": raw_type or schema_type,
        "schema_type": schema_type,
        "schema_type_zh": schema_type_zh,
        "aliases": [],
        "attributes": attributes or {},
        "provenance": f"{name} 的证据",
        "schema_alignment": {
            "status": "RULE_MAPPED" if schema_type_zh else "SKIPPED",
            "confidence": 0.99,
        },
    }


def _relation(
    relation_id: str,
    relation_type: str,
    source_id: str,
    target_id: str,
    *,
    status: str = "DIRECT_MAPPED",
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造可用于端点重写和事实融合的测试关系。"""

    return {
        "id": relation_id,
        "type": relation_type,
        "type_zh": relation_type,
        "source_id": source_id,
        "target_id": target_id,
        "attributes": attributes or {},
        "provenance": f"{relation_id} 的关系证据",
        "raw_relation": relation_type,
        "schema_relation": relation_type if status != "UNMAPPED" else None,
        "schema_alignment": {"status": status, "confidence": 0.9},
    }


def _graph(
    chunk_id: str,
    modality: str,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造一个具有完整来源元数据的测试子图。"""

    return {
        "entities": entities,
        "relations": relations,
        "events": events or [],
        "metadata": {
            "document_id": "doc-1",
            "chunk_id": chunk_id,
            "modality": modality,
            "text": "白云岩 马家沟组 鄂尔多斯盆地 Mg2+",
        },
    }


def _pipeline() -> MultimodalAlignmentPipeline:
    """创建不访问外部模型的确定性测试流水线。"""

    return MultimodalAlignmentPipeline(enable_embedding=False, enable_llm=False)


def _global_entity_by_name(aligned: dict[str, Any], name: str) -> dict[str, Any]:
    """按名称从单文档融合图中查找全局实体。"""

    return next(entity for entity in aligned["graphs"][0]["entities"] if entity["name"] == name)


class _SameDecisionLLM:
    """把所有灰区候选判为 SAME 的可控测试 LLM。"""

    def chat_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """读取 Prompt 中的 pair_id，并返回稳定的高置信 SAME 判定。"""

        _ = kwargs
        cases = json.loads(messages[1]["content"].split("\n", 1)[1])
        return {
            "decisions": [
                {
                    "pair_id": case["pair_id"],
                    "decision": "SAME",
                    "confidence": 0.95,
                    "reason": "测试中确认两种表达指向同一对象",
                }
                for case in cases
            ]
        }


def test_cross_modal_entity_merge_rewrites_all_relation_endpoints() -> None:
    """同名同类型实体应融合，原先连接两者的边均应迁移到新节点。"""

    payload = {
        "graphs": [
            _graph(
                "1:text:0",
                "text",
                [
                    _entity("dolomite-text", "白云岩", "Lithology", "岩性"),
                    _entity("formation", "马家沟组", "Formation", "组"),
                ],
                [_relation("r-text", "HAS_LITHOLOGY", "formation", "dolomite-text")],
            ),
            _graph(
                "1:image:0",
                "image",
                [
                    _entity("dolomite-image", "白云岩", "Lithology", "岩性"),
                    _entity("basin", "鄂尔多斯盆地", "Basin", "盆地"),
                ],
                [_relation("r-image", "DISTRIBUTED_IN", "dolomite-image", "basin")],
            ),
        ]
    }

    aligned, _, _, _ = _pipeline().align_payload(payload, "fixture.json")

    assert aligned["statistics"]["source_entity_count"] == 4
    assert aligned["statistics"]["global_entity_count"] == 3
    dolomite = _global_entity_by_name(aligned, "白云岩")
    assert dolomite["metadata"]["mention_count"] == 2
    assert dolomite["metadata"]["source_modalities"] == ["image", "text"]
    relations = aligned["graphs"][0]["relations"]
    assert any(relation["target_id"] == dolomite["id"] for relation in relations)
    assert any(relation["source_id"] == dolomite["id"] for relation in relations)
    assert aligned["validation"]["entity_source_conserved"] is True
    assert aligned["validation"]["relation_source_conserved"] is True


def test_duplicate_cross_modal_relation_fuses_evidence() -> None:
    """端点和限定条件一致的跨模态关系应合并并累计两条证据。"""

    graphs = []
    for suffix, modality in (("text", "text"), ("image", "image")):
        graphs.append(
            _graph(
                f"2:{modality}:0",
                modality,
                [
                    _entity(f"mg-{suffix}", "Mg2+", "ChemicalIon", "化学离子"),
                    _entity(f"rock-{suffix}", "白云岩", "Lithology", "岩性"),
                ],
                [
                    _relation(
                        f"r-{suffix}",
                        "AFFECTS",
                        f"mg-{suffix}",
                        f"rock-{suffix}",
                        attributes={"condition": "成岩期", "confidence": 0.9},
                    )
                ],
            )
        )

    aligned, _, _, _ = _pipeline().align_payload({"graphs": graphs}, "fixture.json")

    assert aligned["statistics"]["global_entity_count"] == 2
    assert aligned["statistics"]["global_relation_count"] == 1
    relation = aligned["graphs"][0]["relations"][0]
    assert relation["metadata"]["evidence_count"] == 2
    assert relation["metadata"]["cross_modal"] is True
    assert relation["qualifiers"] == {"condition": "成岩期"}


def test_incompatible_same_name_entities_do_not_merge() -> None:
    """同名但中文 Schema 类型显式互斥的实体必须保持分离。"""

    payload = {
        "graphs": [
            _graph(
                "3:text:0",
                "text",
                [_entity("well", "靖边", "Well", "井")],
                [],
            ),
            _graph(
                "3:image:0",
                "image",
                [_entity("location", "靖边", "Location", "地理位置")],
                [],
            ),
        ]
    }

    aligned, _, unresolved, _ = _pipeline().align_payload(payload, "fixture.json")

    assert aligned["statistics"]["global_entity_count"] == 2
    assert aligned["statistics"]["candidate_decision_counts"]["DIFFERENT"] == 1
    assert unresolved["statistics"]["unresolved_candidate_count"] == 0


def test_chunk_local_entities_never_merge_across_chunks() -> None:
    """图表数据系列等局部结构节点不能跨 Chunk 融合。"""

    payload = {
        "graphs": [
            _graph(
                "4:image:0",
                "image",
                [_entity("series-1", "白云岩", "data_series", "", raw_type="data_series")],
                [],
            ),
            _graph(
                "4:image:1",
                "image",
                [_entity("series-2", "白云岩", "data_series", "", raw_type="data_series")],
                [],
            ),
        ]
    }

    aligned, _, _, _ = _pipeline().align_payload(payload, "fixture.json")

    assert aligned["statistics"]["global_entity_count"] == 2
    assert all(
        entity["metadata"]["scope"] == "chunk_local"
        for entity in aligned["graphs"][0]["entities"]
    )


def test_unmapped_relation_is_preserved_and_can_fuse() -> None:
    """UNMAPPED 关系应保留原始语义，并在事实相同时正常汇集证据。"""

    graphs = []
    for suffix, modality in (("text", "text"), ("image", "image")):
        graphs.append(
            _graph(
                f"5:{modality}:0",
                modality,
                [
                    _entity(f"env-{suffix}", "萨布哈", "DepositionalEnvironment", "沉积环境"),
                    _entity(f"rock-{suffix}", "白云岩", "Lithology", "岩性"),
                ],
                [
                    _relation(
                        f"r-{suffix}",
                        "CONTROLS_CUSTOM",
                        f"env-{suffix}",
                        f"rock-{suffix}",
                        status="UNMAPPED",
                    )
                ],
            )
        )

    aligned, _, _, _ = _pipeline().align_payload({"graphs": graphs}, "fixture.json")

    relation = aligned["graphs"][0]["relations"][0]
    assert relation["schema_relation"] is None
    assert relation["type"] == "CONTROLS_CUSTOM"
    assert relation["schema_alignment_statuses"] == ["UNMAPPED"]
    assert relation["metadata"]["evidence_count"] == 2


def test_absorbed_coreference_self_loop_and_event_rewrite_remain_conserved() -> None:
    """共指边变成自环后应进入审计，事件参与者则应改写为全局实体。"""

    payload = {
        "graphs": [
            _graph(
                "6:image:0",
                "image",
                [
                    _entity("rock-a", "白云岩", "Lithology", "岩性"),
                    _entity("rock-b", "白云岩", "Lithology", "岩性"),
                ],
                [_relation("same", "SAME_AS", "rock-a", "rock-b", status="UNMAPPED")],
                events=[
                    {
                        "id": "event-1",
                        "type": "diagenesis",
                        "name": "白云石化",
                        "participants": ["rock-a", "rock-b"],
                        "attributes": {},
                        "provenance": "图中成岩过程",
                    }
                ],
            )
        ]
    }

    aligned, _, _, conflicts = _pipeline().align_payload(payload, "fixture.json")

    assert aligned["statistics"]["global_entity_count"] == 1
    assert aligned["statistics"]["global_relation_count"] == 0
    assert aligned["validation"]["relation_source_conserved"] is True
    assert conflicts["statistics"]["absorbed_self_loops_count"] == 1
    event = aligned["graphs"][0]["events"][0]
    assert len(event["participants"]) == 1
    assert event["original_participants"] == ["rock-a", "rock-b"]


def test_stratigraphic_abbreviation_merges_with_full_name_deterministically() -> None:
    """同类型地层简称和全称应融合，重复运行得到相同全局 ID。"""

    payload = {
        "graphs": [
            _graph(
                "7:text:0",
                "text",
                [_entity("member-a", "马五", "StratigraphicMember", "段")],
                [],
            ),
            _graph(
                "7:image:0",
                "image",
                [_entity("member-b", "马五段", "StratigraphicMember", "段")],
                [],
            ),
        ]
    }

    first, _, _, _ = _pipeline().align_payload(payload, "fixture.json")
    second, _, _, _ = _pipeline().align_payload(payload, "fixture.json")

    assert first["statistics"]["global_entity_count"] == 1
    assert first["graphs"][0]["entities"][0]["id"] == second["graphs"][0]["entities"][0]["id"]


def test_different_well_numbers_do_not_merge_through_shared_neighbor() -> None:
    """不同井号即使共享盆地邻居，也不能通过图结构传播错误融合。"""

    payload = {
        "graphs": [
            _graph(
                "8:text:0",
                "text",
                [
                    _entity("well-134", "陕134", "Well", "井"),
                    _entity("basin-text", "鄂尔多斯盆地", "Basin", "盆地"),
                ],
                [_relation("located-134", "LOCATED_IN", "well-134", "basin-text")],
            ),
            _graph(
                "8:image:0",
                "image",
                [
                    _entity("well-157", "陕157", "Well", "井"),
                    _entity("basin-image", "鄂尔多斯盆地", "Basin", "盆地"),
                ],
                [_relation("located-157", "LOCATED_IN", "well-157", "basin-image")],
            ),
        ]
    }

    aligned, _, _, _ = _pipeline().align_payload(payload, "fixture.json")

    assert aligned["statistics"]["global_entity_count"] == 3
    well_entities = [
        entity
        for entity in aligned["graphs"][0]["entities"]
        if entity["schema_type_zh"] == "井"
    ]
    assert sorted(entity["name"] for entity in well_entities) == ["陕134", "陕157"]


def test_lithology_modifier_is_not_treated_as_an_abbreviation() -> None:
    """岩性修饰语表达不同概念，不能因包含基础岩性名称而自动融合。"""

    payload = {
        "graphs": [
            _graph(
                "9:text:0",
                "text",
                [_entity("rock-a", "白云岩", "Lithology", "岩性")],
                [],
            ),
            _graph(
                "9:image:0",
                "image",
                [_entity("rock-b", "泥晶白云岩", "Lithology", "岩性")],
                [],
            ),
        ]
    }

    aligned, _, unresolved, _ = _pipeline().align_payload(payload, "fixture.json")

    assert aligned["statistics"]["global_entity_count"] == 2
    assert unresolved["statistics"]["unresolved_candidate_count"] == 1


def test_direct_domain_relation_blocks_same_name_entity_merge() -> None:
    """两个实体间已有非共指领域关系时，即使同名也不能被融合成自环。"""

    payload = {
        "graphs": [
            _graph(
                "10:text:0",
                "text",
                [
                    _entity("member", "马五段", "StratigraphicMember", "段"),
                    _entity("submember", "马五段", "SubMember", "亚段/小层"),
                ],
                [_relation("part", "PART_OF", "submember", "member")],
            )
        ]
    }

    aligned, _, _, _ = _pipeline().align_payload(payload, "fixture.json")

    assert aligned["statistics"]["global_entity_count"] == 2
    assert aligned["statistics"]["global_relation_count"] == 1
    relation = aligned["graphs"][0]["relations"][0]
    assert relation["source_id"] != relation["target_id"]


def test_llm_can_merge_rule_uncertain_candidate_without_changing_source() -> None:
    """LLM 应能补充融合灰区候选，同时原实体仍完整保存在 mentions 中。"""

    payload = {
        "graphs": [
            _graph(
                "11:text:0",
                "text",
                [_entity("rock-a", "白云岩", "Lithology", "岩性")],
                [],
            ),
            _graph(
                "11:image:0",
                "image",
                [_entity("rock-b", "泥晶白云岩", "Lithology", "岩性")],
                [],
            ),
        ]
    }
    pipeline = MultimodalAlignmentPipeline(
        enable_embedding=False,
        enable_llm=True,
        llm_client=_SameDecisionLLM(),
        max_llm_candidates=-1,
    )

    aligned, report, unresolved, _ = pipeline.align_payload(payload, "fixture.json")

    assert aligned["statistics"]["global_entity_count"] == 1
    entity = aligned["graphs"][0]["entities"][0]
    assert sorted(item["original_entity"]["name"] for item in entity["mentions"]) == [
        "泥晶白云岩",
        "白云岩",
    ]
    assert report["statistics"]["same_decision_method_counts"] == {
        "llm_pair_resolution": 1
    }
    assert unresolved["statistics"]["unresolved_candidate_count"] == 0
