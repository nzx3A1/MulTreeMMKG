"""二维平面地层—测井图专用流水线、Graph 和目标清单测试。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src.extractors.image_extractor.schema_models import ImageExtractionTask
from src.extractors.image_extractor.stratigraphic_profile.two_dimensional_stratigraphic_log import (
    TwoDimensionalStratigraphicLogPipeline,
    build_two_dimensional_stratigraphic_log_graph,
    load_two_dimensional_target_chunks,
    validate_two_dimensional_graph,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "src"
    / "extractors"
    / "image_extractor"
    / "stratigraphic_profile"
    / "testImage"
    / "stratigraphic_profile_subtype_mock.json"
)


def _task() -> ImageExtractionTask:
    """中文说明：建立不依赖网络的二维剖面单图任务。"""

    return ImageExtractionTask(
        document_id="doc-2d",
        chunk_id="chunk-2d",
        image_id="chunk-2d:image:0",
        image_index=0,
        image_path="C:/fixtures/two-dimensional.jpg",
        caption="测试二维连井剖面",
        classification_code="A03",
        classification_type="地层—剖面—测井",
    )


def _payload() -> dict:
    """中文说明：构造同时含上下层序、井位横序和断层切层关系的最小专用响应。"""

    return {
        "schema_version": "two_dimensional_stratigraphic_log.v1",
        "diagram_type": "multiwell_correlation",
        "diagram_name": "测试二维连井剖面",
        "coordinate_system": {
            "horizontal_meaning": "井间位置",
            "horizontal_unit": "",
            "vertical_meaning": "深度",
            "vertical_unit": "m",
            "vertical_increases": "downward",
            "evidence": "左侧深度轴",
        },
        "entities": [
            {
                "id": "unit_upper",
                "name": "上部地层",
                "type": "stratigraphic_unit",
                "parent_id": "",
                "position": {"bbox": [0, 100, 1000, 300], "x_order": None, "y_order": 0},
                "attributes": {},
                "evidence": "上部连续层带",
                "confidence": 0.95,
            },
            {
                "id": "unit_lower",
                "name": "下部地层",
                "type": "stratigraphic_unit",
                "parent_id": "",
                "position": {"bbox": [0, 300, 1000, 600], "x_order": None, "y_order": 1},
                "attributes": {},
                "evidence": "下部连续层带",
                "confidence": 0.94,
            },
            {
                "id": "well_left",
                "name": "左井",
                "type": "well",
                "parent_id": "",
                "position": {"bbox": [100, 0, 180, 900], "x_order": 0, "y_order": None},
                "attributes": {},
                "evidence": "左侧井柱",
                "confidence": 0.96,
            },
            {
                "id": "well_right",
                "name": "右井",
                "type": "well",
                "parent_id": "",
                "position": {"bbox": [800, 0, 880, 900], "x_order": 1, "y_order": None},
                "attributes": {},
                "evidence": "右侧井柱",
                "confidence": 0.96,
            },
            {
                "id": "fault_1",
                "name": "断层F1",
                "type": "fault",
                "parent_id": "",
                "position": {"bbox": [450, 100, 520, 800], "x_order": None, "y_order": None},
                "attributes": {},
                "evidence": "红色断层线",
                "confidence": 0.92,
            },
        ],
        "stratigraphic_sequences": [
            {
                "id": "regional_sequence",
                "context_id": "regional",
                "ordered_unit_ids_top_to_bottom": ["unit_upper", "unit_lower"],
                "evidence": "上部地层位于下部地层之上",
                "confidence": 0.95,
            }
        ],
        "spatial_groups": [
            {
                "id": "well_order",
                "kind": "wells",
                "ordered_member_ids_left_to_right": ["well_left", "well_right"],
                "evidence": "两口井从左到右排列",
                "confidence": 0.96,
            }
        ],
        "relations": [
            {
                "source_id": "fault_1",
                "type": "cuts_through",
                "target_id": "unit_upper",
                "dimension": "structural",
                "explicit": True,
                "basis": "断层线穿过层带",
                "evidence": "红色断层线切穿上部地层",
                "confidence": 0.92,
            },
            {
                "source_id": "well_left",
                "type": "intersects",
                "target_id": "unit_lower",
                "dimension": "spatial",
                "explicit": True,
                "basis": "井柱穿过层带",
                "evidence": "左井贯穿下部地层",
                "confidence": 0.94,
            },
        ],
        "uncertainties": [],
    }


def test_pipeline_keeps_only_adjacent_vertical_inverse_relations() -> None:
    """层序只生成相邻直接上覆/下伏，横向组不生成位置边且孤立节点被删除。"""

    intermediate = TwoDimensionalStratigraphicLogPipeline().run(_task(), _payload())
    triples = {
        (item["source_id"], item["type"], item["target_id"])
        for item in intermediate["relations"]
    }
    assert ("unit_upper", "directly_overlies", "unit_lower") in triples
    assert ("unit_lower", "directly_underlies", "unit_upper") in triples
    assert ("fault_1", "cuts_through", "unit_upper") in triples
    assert not ({"above", "below", "left_of", "right_of", "adjacent_to"} & {item[1] for item in triples})
    assert "well_right" not in {item["id"] for item in intermediate["entities"]}
    assert intermediate["quality"]["pruned_isolated_entity_count"] == 1
    assert intermediate["quality"]["dropped_relations"] == []


def test_sequence_never_connects_nonadjacent_nodes() -> None:
    """三节点层序只连接首中和中末，不生成跨越中间节点的首末关系。"""

    payload = deepcopy(_payload())
    payload["entities"] = payload["entities"][:2]
    middle = deepcopy(payload["entities"][1])
    middle["id"] = "unit_middle"
    middle["name"] = "中部地层"
    payload["entities"].insert(1, middle)
    payload["stratigraphic_sequences"][0]["ordered_unit_ids_top_to_bottom"] = [
        "unit_upper",
        "unit_middle",
        "unit_lower",
    ]
    payload["spatial_groups"] = []
    payload["relations"] = []
    intermediate = TwoDimensionalStratigraphicLogPipeline().run(_task(), payload)
    triples = {
        (item["source_id"], item["type"], item["target_id"])
        for item in intermediate["relations"]
    }
    assert ("unit_upper", "directly_overlies", "unit_middle") in triples
    assert ("unit_middle", "directly_underlies", "unit_upper") in triples
    assert ("unit_middle", "directly_overlies", "unit_lower") in triples
    assert ("unit_lower", "directly_underlies", "unit_middle") in triples
    assert ("unit_upper", "directly_overlies", "unit_lower") not in triples
    assert ("unit_lower", "directly_underlies", "unit_upper") not in triples


def test_graph_keeps_source_provenance_and_no_events() -> None:
    """Graph 中每个实体和关系必须精确指向当前来源图片且不生成事件。"""

    task = _task()
    intermediate = TwoDimensionalStratigraphicLogPipeline().run(task, _payload())
    graph = build_two_dimensional_stratigraphic_log_graph(task, intermediate)
    validation = validate_two_dimensional_graph(graph, task)
    assert graph.metadata.extra["status"] == "completed"
    assert graph.events == []
    assert validation["reference_errors"] == []
    assert validation["provenance_errors"] == []
    assert validation["quality_errors"] == []
    assert validation["isolated_entity_ids"] == []
    triples = {(item.source_name, item.type, item.target_name) for item in graph.relations}
    assert ("上部地层", "directly_overlies", "下部地层") in triples
    assert ("断层F1", "cuts_through", "上部地层") in triples
    assert "右井" not in {item.name for item in graph.entities}


def test_single_named_horizon_uses_trajectory_relation_without_inventing_layer() -> None:
    """单层位地震图可用轨迹—层位关系验收，不能为满足上下门控虚构第二层。"""

    payload = deepcopy(_payload())
    payload["entities"] = [
        {
            "id": "mudstone_interval",
            "name": "长7₂泥岩段",
            "type": "shale_interval",
            "parent_id": "",
            "position": {"bbox": [200, 450, 900, 560], "x_order": None, "y_order": 0},
            "attributes": {},
            "evidence": "图中长7₂泥岩段标注",
            "confidence": 0.95,
        },
        {
            "id": "adjusted_trajectory",
            "name": "调整后轨迹",
            "type": "well_trajectory",
            "parent_id": "",
            "position": {"bbox": [200, 200, 900, 550], "x_order": None, "y_order": None},
            "attributes": {},
            "evidence": "红色调整后井轨迹",
            "confidence": 0.95,
        },
    ]
    payload["stratigraphic_sequences"] = []
    payload["spatial_groups"] = []
    payload["relations"] = [
        {
            "source_id": "adjusted_trajectory",
            "type": "tracks_along",
            "target_id": "mudstone_interval",
            "dimension": "spatial",
            "explicit": True,
            "basis": "轨迹沿标注层位延伸",
            "evidence": "红色调整后轨迹贴近长7₂泥岩段",
            "confidence": 0.94,
        }
    ]
    task = _task()
    intermediate = TwoDimensionalStratigraphicLogPipeline().run(task, payload)
    graph = build_two_dimensional_stratigraphic_log_graph(task, intermediate)
    validation = validate_two_dimensional_graph(graph, task)
    assert validation["vertical_relation_count"] == 0
    assert validation["semantic_position_relation_count"] == 1
    assert validation["quality_errors"] == []
    assert validation["quality_warnings"]


def test_user_manifest_has_exactly_seven_existing_two_dimensional_targets() -> None:
    """用户指定清单必须稳定筛出 7 个二维目标，且每张来源图片都真实存在。"""

    source_count, targets = load_two_dimensional_target_chunks(MANIFEST_PATH)
    assert source_count == 23
    assert len(targets) == 7
    assert all(Path(item["image_path"][0]).is_file() for item in targets)
    assert [item["target_index"] for item in targets] == list(range(7))
