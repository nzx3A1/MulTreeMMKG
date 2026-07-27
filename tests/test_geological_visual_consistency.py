"""地质模式图视觉一致性校正规则的离线测试。"""
from __future__ import annotations

from src.extractors.image_extractor.geological_process import (
    apply_geological_review_overlay,
    build_geological_process_graph,
    normalize_geological_visual_result,
)
from src.extractors.image_extractor.schema_models import ImageExtractionTask


def _task() -> ImageExtractionTask:
    """中文说明：构造无需模型客户端的 A08 图片任务。"""

    return ImageExtractionTask(
        document_id="doc",
        chunk_id="chunk",
        image_id="image",
        image_index=0,
        image_path="figure.jpg",
        caption="走滑断裂成藏模式",
        references=(),
        classification_code="A08",
        classification_type="油气藏、成藏与富集模式图",
    )


def test_consistency_rules_prefer_labels_and_disambiguate_wells_from_faults() -> None:
    """明确岩性文字应覆盖图例猜测，井号复制出的断裂实体应合并为分段主断裂。"""

    visual = {
        "legend_lithologies": [{"id": "lith_old", "name": "泥质白云岩"}],
        "stratigraphic_units": [
            {
                "id": "unit_source",
                "name": "Є1y",
                "evidence": "底部标签Є1y烃源岩",
                "lithologies": [{"lithology_id": "lith_old", "name": "泥质白云岩", "role": "主要", "confidence": 0.9}],
            },
            {
                "id": "unit_seal",
                "name": "O3S",
                "evidence": "顶部标签O3S泥质岩",
                "lithologies": [{"lithology_id": "lith_old", "name": "泥质白云岩", "role": "主要", "confidence": 0.9}],
            },
        ],
        "entities": [
            {"id": "fault_sb1", "name": "SB1断裂", "type": "fault"},
            {"id": "fault_sb1_10", "name": "SB1-10断裂", "type": "fault"},
            {"id": "reservoir_1", "name": "储集体1", "type": "reservoir_body"},
            {"id": "reservoir_2", "name": "储集体2", "type": "reservoir_body"},
            {"id": "zone_1", "name": "聚集区1", "type": "location"},
            {"id": "zone_2", "name": "聚集区2", "type": "location"},
        ],
        "structural_segments": [
            {
                "id": "segment_pull",
                "name": "走滑拉分",
                "wells": [{"id": "well_sb1", "name": "SB1井"}, {"id": "well_sb1_10", "name": "SB1-10井"}],
                "structures": ["fault_sb1", "fault_sb1_10", "reservoir_1", "reservoir_2"],
            }
        ],
        "petroleum_system": {
            "reservoirs": [{"id": "reservoir_1"}, {"id": "reservoir_2"}],
            "accumulation_zones": [{"id": "zone_1"}, {"id": "zone_2"}],
        },
        "process_events": [
            {"id": "migration", "type": "migration", "target": "reservoir_1", "explicit": True, "evidence": "正文描述"},
            {"id": "accumulation", "type": "accumulation", "target": "zone_1", "explicit": True, "evidence": "正文描述"},
        ],
        "causal_relations": [
            {"source": "fault_sb1", "type": "controls", "target": "reservoir_1", "explicit": True, "evidence": "正文说明断裂控储"}
        ],
        "spatial_relations": [{"source": "fault_sb1_10", "type": "cuts_through", "target": "unit_seal", "explicit": True, "evidence": "图中断裂"}],
        "uncertainties": [],
    }

    corrections = normalize_geological_visual_result(_task(), visual)

    source_lithology = visual["stratigraphic_units"][0]["lithologies"][0]
    seal_lithology = visual["stratigraphic_units"][1]["lithologies"][0]
    assert source_lithology["role"] == "不确定"
    assert source_lithology["confidence"] == 0.55
    assert seal_lithology["name"] == "泥质岩"
    assert seal_lithology["validation_source"] == "explicit_unit_label"

    fault_entities = [item for item in visual["entities"] if item.get("type") == "fault"]
    assert [(item["id"], item["name"]) for item in fault_entities] == [("fault_sb1", "走滑拉分主走滑断裂")]
    assert visual["spatial_relations"][0]["source"] == "fault_sb1"
    assert visual["process_events"][0]["targets"] == ["reservoir_1", "reservoir_2"]
    assert visual["process_events"][0]["explicit"] is False
    assert visual["process_events"][1]["targets"] == ["zone_1", "zone_2"]
    assert visual["causal_relations"][0]["explicit"] is False
    assert any(item["type"] == "well_fault_disambiguation" for item in corrections)
    assert any(item["type"] == "explicit_lithology_override" for item in corrections)


def test_review_overlay_updates_units_and_preserves_parent_child_hierarchy() -> None:
    """外部证据复核应能纠正层段并把统级父单元排除在直接垂向相邻推导之外。"""

    visual = {
        "coordinate_system": {"normal_stratigraphic_order": True},
        "legend_lithologies": [{"id": "lith_limestone", "name": "石灰岩"}],
        "stratigraphic_units": [
            {"id": "unit_lower", "name": "误读下层", "order_bottom_to_top": 0, "lithologies": [], "confidence": 0.9},
            {"id": "unit_child", "name": "误读上层", "order_bottom_to_top": 1, "lithologies": [], "confidence": 0.9},
        ],
        "entities": [],
        "process_events": [],
        "uncertainties": ["误读待确认"],
    }
    overlay = {
        "review_basis": "裁剪OCR",
        "unit_updates": [
            {"id": "unit_lower", "name": "O2yj", "order_bottom_to_top": 0},
            {"id": "unit_child", "name": "O3S", "order_bottom_to_top": 1, "parent_unit_id": "unit_parent"},
        ],
        "add_units": [
            {"id": "unit_parent", "name": "O3", "order_bottom_to_top": 1, "sequence_role": "container", "confidence": 0.95}
        ],
        "remove_uncertainties_containing": ["误读"],
    }

    apply_geological_review_overlay(visual, overlay)
    graph = build_geological_process_graph(_task(), visual, {})

    triples = {(relation.source_name, relation.type, relation.target_name) for relation in graph.relations}
    assert ("O3S", "within", "O3") in triples
    assert ("O3", "contains", "O3S") in triples
    assert ("O3S", "directly_overlies", "O2yj") in triples
    assert ("O3", "directly_overlies", "O3S") not in triples
    assert visual["uncertainties"] == []
