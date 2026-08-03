"""地层—剖面—测井图片三分类与分支策略的离线测试。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.extractors.image_extractor.schema_models import ImageExtractionContext, ImageExtractionTask
from src.extractors.image_extractor.stratigraphic_profile import (
    StratigraphicProfileExtractor,
    StratigraphicProfileSubtype,
    StratigraphicProfileSubtypeClassifier,
    VLMStratigraphicProfileSubtypeClassifier,
    build_stratigraphic_subclassification_prompt,
    get_stratigraphic_subtype_strategy,
)
from src.utils.json_io import read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_DIR = PROJECT_ROOT / "src" / "extractors" / "image_extractor" / "stratigraphic_profile" / "testImage"
CHUNKS_PATH = TEST_IMAGE_DIR / "image_chunks_stratigraphic_profile.json"
MOCK_PATH = TEST_IMAGE_DIR / "stratigraphic_profile_subtype_mock.json"


def _chunks() -> list[dict[str, Any]]:
    """中文说明：读取用户指定的 23 个真实图片 Chunk，保证测试源与实际批处理一致。"""

    chunks = read_json(CHUNKS_PATH)
    assert isinstance(chunks, list)
    return chunks


def test_manual_visual_mock_covers_every_image_without_api() -> None:
    """23 张图必须全部命中人工视觉 mock，分类过程不接收也不调用任何模型客户端。"""

    classifier = StratigraphicProfileSubtypeClassifier()
    results = classifier.classify_chunks(_chunks())

    assert len(results) == 23
    assert all(Path(item["image_path"]).is_file() for item in results)
    assert {item["source"] for item in results} == {"manual_visual_mock"}
    assert Counter(item["subtype"] for item in results) == {
        StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID.value: 14,
        StratigraphicProfileSubtype.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL.value: 2,
        StratigraphicProfileSubtype.TWO_DIMENSIONAL_STRATIGRAPHIC_LOG.value: 7,
    }


def test_mock_file_uses_image_chunk_shape_with_one_added_field() -> None:
    """分类文件必须保持原始图片 Chunk 字段，并只用 stratigraphic_subtype 承载子分类信息。"""

    records = read_json(MOCK_PATH)
    assert isinstance(records, list)
    assert len(records) == 23
    required_fields = {"id", "order", "modality", "image_path", "caption", "references", "stratigraphic_subtype"}
    for record in records:
        assert required_fields <= set(record)
        assert record["modality"] == "image"
        assert isinstance(record["image_path"], list) and len(record["image_path"]) == 1
        classification = record["stratigraphic_subtype"]
        assert classification["subtype"] in {subtype.value for subtype in StratigraphicProfileSubtype}
        assert classification["subtype_name"]
        assert classification["evidence"]


def test_representative_visual_decisions_are_stable() -> None:
    """中文说明：固定三类代表图，防止后续仅凭图题关键词改坏人工视觉结论。"""

    results = StratigraphicProfileSubtypeClassifier().classify_chunks(_chunks())
    by_filename = {Path(item["image_path"]).name: item for item in results}

    assert by_filename["af4dce79db28e729dc664ae367262cb0023fd38e9b86eab7316d7a5efaa4d168.jpg"]["subtype"] == "table_embedded_hybrid"
    assert by_filename["f0ac23ffa801dccdccad61dda0ee06d1417576276e440abdd8db77b5a4919ff1.jpg"]["subtype"] == "three_dimensional_stratigraphic_model"
    assert by_filename["0875b0d3913ae8bd912712c6b9edc98998eaffa85966ef21dc2d29de51324264.jpg"]["subtype"] == "two_dimensional_stratigraphic_log"
    assert "实际是二维时间剖面切片" in by_filename["0875b0d3913ae8bd912712c6b9edc98998eaffa85966ef21dc2d29de51324264.jpg"]["evidence"]


def test_each_subtype_uses_a_distinct_extraction_algorithm() -> None:
    """三个子类型必须分别路由到网格轨道、三维拓扑和二维层界算法。"""

    algorithms = {
        get_stratigraphic_subtype_strategy(subtype).algorithm_name
        for subtype in StratigraphicProfileSubtype
    }
    assert algorithms == {
        "ppstructurev3_geometry_semantic_vlm_depth_alignment_graph_assembly",
        "surface_topology_spatial_extraction",
        "axis_layer_correlation_extraction",
    }


class RecordingOfflineVLM:
    """依次模拟 VLM 子分类和视觉抽取并记录 Prompt 的本地桩。"""

    def __init__(self) -> None:
        """中文说明：初始化 Prompt 调用记录。"""

        self.prompts: list[str] = []

    def ppstructure_geometry_extractor(self, image_path: str) -> dict[str, Any]:
        """中文说明：离线测试注入确定性的 PP 几何契约，避免下载或运行真实 OCR 模型。"""

        with Image.open(image_path) as image:
            width, height = image.size
        bbox = [0, 0, width, height]
        return {
            "schema_version": "ppstructurev3.table_geometry.v1",
            "engine": "offline_contract_fixture",
            "coordinate_space": "original_pixels",
            "source_image_path": str(Path(image_path).resolve()),
            "image_size": {"width": width, "height": height},
            "content_bbox": bbox,
            "tracks": [
                {
                    "id": "pp_track_000",
                    "order": 0,
                    "bbox": bbox,
                    "header_text": "测试地层",
                }
            ],
            "cells": [
                {
                    "id": "pp_cell_0000",
                    "bbox": bbox,
                    "ocr_ids": ["pp_ocr_0000"],
                    "text": "测试层",
                    "track_ids": ["pp_track_000"],
                }
            ],
            "ocr_lines": [
                {
                    "id": "pp_ocr_0000",
                    "text": "测试层",
                    "confidence": 1.0,
                    "bbox": bbox,
                    "cell_id": "pp_cell_0000",
                    "track_id": "pp_track_000",
                }
            ],
            "rule_lines": {
                "available": True,
                "method": "offline_contract_fixture",
                "vertical_lines": [0, width],
                "horizontal_lines": [0, height],
            },
            "quality": {"ocr_line_count": 1, "cell_count": 1, "track_count": 1},
        }

    def describe_image(self, image_path: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """中文说明：按 task_name 区分子分类与抽取响应，且不创建网络或模型客户端。"""

        self.prompts.append(prompt)
        if str(kwargs.get("task_name") or "").startswith("地层图片子分类"):
            subtype = {
                "af4dce79db28e729dc664ae367262cb0023fd38e9b86eab7316d7a5efaa4d168.jpg": "table_embedded_hybrid",
                "f0ac23ffa801dccdccad61dda0ee06d1417576276e440abdd8db77b5a4919ff1.jpg": "three_dimensional_stratigraphic_model",
                "0875b0d3913ae8bd912712c6b9edc98998eaffa85966ef21dc2d29de51324264.jpg": "two_dimensional_stratigraphic_log",
            }[Path(image_path).name]
            return {
                "subtype": subtype,
                "confidence": 0.96,
                "evidence": ["测试可见版式证据"],
            }
        if "表格嵌入混合节点官方名规范化" in str(kwargs.get("task_name") or ""):
            # 中文说明：第四次调用只覆盖完整抽取已经产生的两个节点，不得新增节点或关系。
            return {
                "schema_version": "table_embedded_hybrid.node_enrichment.v1",
                "nodes": [
                    {
                        "id": "test_table",
                        "official_name": "测试混合地层表",
                        "basis": "standardized_domain_term",
                        "confidence": 0.95,
                    },
                    {
                        "id": "unit_1",
                        "official_name": "测试层",
                        "basis": "already_official",
                        "confidence": 0.99,
                    },
                ],
            }
        if "table_embedded_hybrid.v1" in prompt:
            # 中文说明：表格混合分支必须返回专用几何协议，不能再用原通用地层 JSON。
            return {
                "schema_version": "table_embedded_hybrid.v1",
                "diagram_id": "test_table",
                "diagram_name": "测试混合表格",
                "image_size": {"width": 606, "height": 807},
                "coordinate_system": {
                    "content_bbox": [0, 0, 606, 807],
                    "vertical_axis": {
                        "kind": "relative_sequence",
                        "unit": "无量纲",
                        "increases": "downward",
                        "calibration_points": [
                            {"pixel_y": 0, "value": 0},
                            {"pixel_y": 807, "value": 807},
                        ],
                    },
                },
                "tracks": [
                    {
                        "id": "test_track",
                        "order": 0,
                        "role": "stratigraphy",
                        "header": "测试地层",
                        "bbox": [0, 0, 606, 807],
                        "parser": "hierarchical_merged_cell_parser",
                    }
                ],
                "primitives": {
                    "stratigraphic_intervals": [
                        {
                            "id": "unit_1",
                            "name": "测试层",
                            "track_id": "test_track",
                            "top_y": 0,
                            "bottom_y": 807,
                            "evidence": "测试图内标签",
                            "confidence": 0.9,
                        }
                    ],
                    "lithology_intervals": [],
                    "facies_intervals": [],
                    "curve_tracks": [],
                    "curve_observations": [],
                    "reservoir_intervals": [],
                    "point_markers": [],
                },
                "uncertainties": [],
            }
        if "three_dimensional_stratigraphic_model.v1" in prompt:
            # 中文说明：三维分支返回专用层序与上下文协议，验证外层确实分发到新子目录。
            return {
                "schema_version": "three_dimensional_stratigraphic_model.v1",
                "model": {
                    "id": "model_1",
                    "name": "测试三维地层模型",
                    "visible_surfaces": ["top", "front", "side"],
                    "topology_evidence": "测试块体可见三个面",
                    "evidence": "测试块体可见三个面",
                    "confidence": 0.95,
                },
                "lithologies": [],
                "stratigraphic_units": [
                    {
                        "id": "unit_lower",
                        "name": "下层",
                        "column_id": "main_section",
                        "order_bottom_to_top": 0,
                        "evidence": "侧壁下部",
                        "confidence": 0.9,
                    },
                    {
                        "id": "unit_upper",
                        "name": "上层",
                        "column_id": "main_section",
                        "order_bottom_to_top": 1,
                        "evidence": "侧壁上部",
                        "confidence": 0.9,
                    },
                ],
                "wells": [],
                "faults": [],
                "zones": [],
                "fluids": [],
                "objects": [],
                "relations": [],
                "context_relations": [],
                "uncertainties": [],
            }
        if "two_dimensional_stratigraphic_log.v1" in prompt:
            # 中文说明：二维专用目录已实现，返回最小合法实体协议验证调度而不触发真实模型。
            return {
                "schema_version": "two_dimensional_stratigraphic_log.v1",
                "diagram_type": "geological_section",
                "diagram_name": "测试二维剖面",
                "coordinate_system": {},
                "entities": [
                    {
                        "id": "unit_2d_upper",
                        "name": "测试二维上层",
                        "type": "stratigraphic_unit",
                        "bbox": [100, 100, 900, 300],
                        "evidence": "测试二维上层标签",
                        "confidence": 0.9,
                    },
                    {
                        "id": "unit_2d_lower",
                        "name": "测试二维下层",
                        "type": "stratigraphic_unit",
                        "bbox": [100, 300, 900, 600],
                        "evidence": "测试二维下层标签",
                        "confidence": 0.9,
                    }
                ],
                "relations": [],
                "stratigraphic_sequences": [
                    {
                        "id": "test_sequence",
                        "context_id": "regional",
                        "ordered_unit_ids_top_to_bottom": ["unit_2d_upper", "unit_2d_lower"],
                        "evidence": "测试二维上下层序",
                        "confidence": 0.9,
                    }
                ],
                "spatial_groups": [],
                "uncertainties": [],
            }
        return {
            "diagram_type": "offline_test",
            "stratigraphic_units": [
                {
                    "id": "unit_1",
                    "name": "测试层",
                    "order_bottom_to_top": 0,
                    "evidence": "测试图内标签",
                    "confidence": 0.9,
                }
            ],
            "relations": [],
            "uncertainties": [],
        }


@pytest.mark.parametrize(
    ("chunk_index", "expected_subtype", "expected_algorithm", "expected_status"),
    [
        (
            0,
            "table_embedded_hybrid",
            "ppstructurev3_geometry_semantic_vlm_depth_alignment_graph_assembly",
            "completed",
        ),
        (4, "three_dimensional_stratigraphic_model", "surface_topology_spatial_extraction", "completed"),
        (18, "two_dimensional_stratigraphic_log", "axis_layer_correlation_extraction", "completed"),
    ],
)
def test_extractor_dispatches_the_selected_strategy(
    chunk_index: int,
    expected_subtype: str,
    expected_algorithm: str,
    expected_status: str,
) -> None:
    """中文说明：验证三个已实现子类型分别进入专用目录，不再回退到旧通用算法。"""

    chunk = _chunks()[chunk_index]
    image_path = str(chunk["image_path"][0])
    task = ImageExtractionTask(
        document_id=str(chunk["id"]).split(":section:", 1)[0],
        chunk_id=str(chunk["id"]),
        image_id=f"{chunk['id']}:image:0",
        image_index=0,
        image_path=image_path,
        caption=str(chunk.get("caption") or ""),
        classification_code="A03",
        classification_type="地层—剖面—测井",
        raw_chunk=chunk,
    )
    vlm = RecordingOfflineVLM()
    graph = StratigraphicProfileExtractor().extract(
        task,
        ImageExtractionContext(
            vlm_client=vlm,
            llm_client=None,
            options={"enable_relation_audit": False},
        ),
    )

    assert graph.metadata.extra["status"] == expected_status
    assert graph.metadata.extra["stratigraphic_subtype"] == expected_subtype
    assert graph.metadata.extra["subtype_extraction_algorithm"] == expected_algorithm
    assert graph.metadata.extra["subtype_classification_vlm_called"] is True
    expected_call_count = 3 if expected_subtype == "table_embedded_hybrid" else 2
    assert graph.metadata.extra["vlm_call_count"] == expected_call_count
    assert len(vlm.prompts) == expected_call_count
    assert "只能从以下三个 subtype 中选择一个" in vlm.prompts[0]
    assert expected_algorithm in vlm.prompts[1]


def test_vlm_subclassification_prompt_and_json_parser() -> None:
    """提示词必须定义三类边界，VLM 分类器必须把证据数组规范为稳定结果。"""

    chunk = _chunks()[0]
    task = ImageExtractionTask(
        document_id=str(chunk["id"]).split(":section:", 1)[0],
        chunk_id=str(chunk["id"]),
        image_id=f"{chunk['id']}:image:0",
        image_index=0,
        image_path=str(chunk["image_path"][0]),
        caption=str(chunk.get("caption") or ""),
        raw_chunk=chunk,
    )
    prompt = build_stratigraphic_subclassification_prompt(task)
    assert "table_embedded_hybrid" in prompt
    assert "three_dimensional_stratigraphic_model" in prompt
    assert "two_dimensional_stratigraphic_log" in prompt
    assert "三维地震数据的一张平面剖切图" in prompt

    vlm = RecordingOfflineVLM()
    result = VLMStratigraphicProfileSubtypeClassifier().classify(task, vlm)
    assert result.subtype is StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID
    assert result.confidence == 0.96
    assert result.evidence == "测试可见版式证据"
    assert result.source == "vlm_visual_classification"
