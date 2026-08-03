"""表格嵌入混合抽取器的 PP-StructureV3 几何回归测试。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import pytest

from src.extractors.image_extractor.schema_models import ImageExtractionTask
from src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid.pipeline import (
    TableEmbeddedHybridPipeline,
)
from src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid.graph import (
    build_table_embedded_hybrid_graph,
)
from src.extractors.image_extractor.stratigraphic_profile.subclassifier import (
    StratigraphicProfileSubtype,
    StratigraphicSubtypeClassification,
)
from src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid.segmented_vlm import (
    apply_node_enrichment,
    extract_segmented_table_visual,
)
from src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid.ppstructure_geometry import (
    PPStructureV3GeometryExtractor,
    apply_ppstructure_geometry,
    normalize_ppstructure_result,
)


def _image(tmp_path: Path) -> Path:
    """中文说明：创建固定尺寸白底图，仅用于验证原图坐标契约。"""

    path = tmp_path / "table.png"
    Image.new("RGB", (400, 300), "white").save(path)
    return path


def _raw_pp_result() -> dict[str, Any]:
    """中文说明：模拟 PP-StructureV3 的真实字段形状，包含 OCR 框和主表单元格框。"""

    return {
        "width": 400,
        "height": 300,
        "overall_ocr_res": {
            "rec_texts": ["地层", "深度/m", "层A", "层B", "100", "200"],
            "rec_scores": [0.99, 0.99, 0.96, 0.96, 0.98, 0.98],
            "rec_boxes": [
                [10, 10, 70, 40],
                [90, 10, 150, 40],
                [10, 80, 70, 170],
                [10, 180, 70, 280],
                [90, 95, 145, 115],
                [90, 195, 145, 215],
            ],
        },
        "table_res_list": [
            {
                "cell_box_list": [
                    [0, 0, 80, 60],
                    [80, 0, 160, 60],
                    [0, 60, 80, 175],
                    [80, 60, 160, 175],
                    [0, 175, 80, 300],
                    [80, 175, 160, 300],
                ],
                "pred_html": "<table></table>",
                "table_ocr_pred": {},
            }
        ],
    }


def _geometry() -> dict[str, Any]:
    """中文说明：构造两个轨道、两个地层单元格和两个深度刻度的最小 PP 几何目录。"""

    return {
        "schema_version": "ppstructurev3.table_geometry.v1",
        "engine": "test_fixture",
        "coordinate_space": "original_pixels",
        "source_image_path": "table.png",
        "image_size": {"width": 400, "height": 300},
        "content_bbox": [0, 0, 160, 300],
        "tracks": [
            {"id": "pp_track_000", "order": 0, "bbox": [0, 0, 80, 300], "header_text": "地层"},
            {"id": "pp_track_001", "order": 1, "bbox": [80, 0, 160, 300], "header_text": "深度/m"},
        ],
        "cells": [
            {"id": "pp_cell_a", "bbox": [0, 60, 80, 175], "ocr_ids": ["pp_ocr_a"], "text": "层A", "track_ids": ["pp_track_000"]},
            {"id": "pp_cell_b", "bbox": [0, 175, 80, 300], "ocr_ids": ["pp_ocr_b"], "text": "层B", "track_ids": ["pp_track_000"]},
        ],
        "ocr_lines": [
            {"id": "pp_ocr_a", "text": "层A", "confidence": 0.96, "bbox": [10, 80, 70, 170], "cell_id": "pp_cell_a", "track_id": "pp_track_000"},
            {"id": "pp_ocr_b", "text": "层B", "confidence": 0.96, "bbox": [10, 185, 70, 275], "cell_id": "pp_cell_b", "track_id": "pp_track_000"},
            {"id": "pp_depth_100", "text": "100", "confidence": 0.98, "bbox": [90, 95, 145, 115], "track_id": "pp_track_001"},
            {"id": "pp_depth_200", "text": "200", "confidence": 0.98, "bbox": [90, 195, 145, 215], "track_id": "pp_track_001"},
        ],
        "rule_lines": {"available": True, "method": "test_fixture", "vertical_lines": [0, 80, 160], "horizontal_lines": [0, 60, 175, 300]},
        "quality": {"ocr_line_count": 4, "cell_count": 2, "track_count": 2},
    }


def _semantic_payload(axis_kind: str = "depth") -> dict[str, Any]:
    """中文说明：故意写入错误 VLM 坐标，验证最终结果只能采用 geometry_refs。"""

    return {
        "schema_version": "table_embedded_hybrid.v1",
        "diagram_id": "test_table",
        "diagram_name": "测试地层表",
        "layout_family": "stratigraphic_column_table",
        "image_size": {"width": 1000, "height": 1000},
        "coordinate_system": {
            "content_bbox": [900, 900, 999, 999],
            "vertical_axis": {
                "kind": axis_kind,
                "unit": "m",
                "increases": "downward",
                "track_id": "pp_track_001",
                "calibration_ocr_ids": ["pp_depth_100", "pp_depth_200"],
                "calibration_points": [
                    {"pixel_y": 999, "value": -999},
                    {"pixel_y": 1000, "value": 99999},
                ],
            },
        },
        "tracks": [
            {"id": "pp_track_000", "role": "stratigraphy", "header": "地层", "bbox": [900, 0, 999, 999]},
            {"id": "pp_track_001", "role": "depth", "header": "深度/m", "bbox": [0, 0, 1, 1]},
        ],
        "primitives": {
            "stratigraphic_intervals": [
                {"id": "unit_a", "name": "层A", "track_id": "pp_track_000", "geometry_refs": ["pp_cell_a"], "top_y": 900, "bottom_y": 950, "evidence": "层A 单元格", "confidence": 0.95},
                {"id": "unit_b", "name": "层B", "track_id": "pp_track_000", "geometry_refs": ["pp_cell_b"], "top_y": 950, "bottom_y": 999, "evidence": "层B 单元格", "confidence": 0.95},
            ],
            "reference_intervals": [],
            "lithology_intervals": [],
            "facies_intervals": [],
            "reservoir_intervals": [],
            "oil_layer_intervals": [],
            "geological_feature_intervals": [],
            "curve_observations": [],
            "curve_tracks": [],
            "point_markers": [],
            "objects": [],
            "explicit_relations": [],
        },
        "uncertainties": [],
    }


def test_normalize_real_ppstructure_fields_and_cache_once(tmp_path: Path) -> None:
    """PP 原始 OCR/单元格字段应生成稳定轨道，且同图第二次读取不重复预测。"""

    image_path = _image(tmp_path)
    geometry = normalize_ppstructure_result(_raw_pp_result(), image_path=image_path)
    assert geometry["content_bbox"] == [0, 0, 160, 300]
    assert geometry["quality"]["ocr_line_count"] == 6
    assert geometry["quality"]["cell_count"] == 6
    assert len(geometry["tracks"]) == 2
    assert {cell["text"] for cell in geometry["cells"]} >= {"层A", "层B"}

    class FakePipeline:
        """中文说明：记录 predict 次数，模拟 PaddleX Result.json 接口。"""

        def __init__(self) -> None:
            self.count = 0

        def predict(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            """中文说明：返回单页 PP-StructureV3 测试结果。"""

            self.count += 1
            return [{"res": _raw_pp_result()}]

    predictor = FakePipeline()
    extractor = PPStructureV3GeometryExtractor(pipeline=predictor, cache_dir=tmp_path / "cache")
    assert extractor.extract(image_path)["quality"]["cell_count"] == 6
    assert extractor.extract(image_path)["quality"]["cell_count"] == 6
    assert predictor.count == 1


def test_rule_fallback_recovers_merged_cell_and_promotes_ocr_refs(tmp_path: Path) -> None:
    """PP 无单元格时应从表线恢复合并格，并把“马”“五”“段”字框解析为整个“马五段”单元格。"""

    image_path = tmp_path / "merged-table.png"
    image = Image.new("RGB", (240, 320), "white")
    draw = ImageDraw.Draw(image)
    # 中文说明：第一列正文不画中间横线，用于模拟跨整段深度的“马五段”合并单元格。
    for x in (10, 60, 120, 230):
        draw.line((x, 10, x, 310), fill="black", width=2)
    for y in (10, 50, 310):
        draw.line((10, y, 230, y), fill="black", width=2)
    draw.line((60, 180, 230, 180), fill="black", width=2)
    image.save(image_path)

    raw = {
        "width": 240,
        "height": 320,
        "overall_ocr_res": {
            "rec_texts": ["地层", "马", "五", "段", "马五2", "马五A"],
            "rec_scores": [0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
            "rec_boxes": [
                [22, 18, 45, 40],
                [22, 80, 45, 105],
                [22, 150, 45, 175],
                [22, 240, 48, 285],
                [72, 90, 108, 120],
                [72, 230, 108, 260],
            ],
        },
        "table_res_list": [],
    }
    geometry = normalize_ppstructure_result(raw, image_path=image_path)
    assert geometry["quality"]["cell_geometry_source"] == "image_rule_morphology_fallback"
    merged_cell = next(cell for cell in geometry["cells"] if cell["text"] == "马 五 段")
    assert merged_cell["bbox"][1] <= 51
    assert merged_cell["bbox"][3] >= 309

    payload = _semantic_payload()
    payload["primitives"]["stratigraphic_intervals"] = [
        {
            "id": "ma_wu",
            "name": "马五段",
            "track_id": "pp_track_000",
            "geometry_refs": ["pp_ocr_0001", "pp_ocr_0002", "pp_ocr_0003"],
            "evidence": "合并单元格",
            "confidence": 0.95,
        },
        {
            "id": "ma_wu_2",
            "name": "马五2",
            "track_id": "pp_track_000",
            "geometry_refs": ["pp_ocr_0001", "pp_ocr_0004"],
            "evidence": "正确子层单元格与错误父层文字混合引用",
            "confidence": 0.95,
        },
        {
            "id": "ma_wu_4",
            "name": "马五4",
            "track_id": "pp_track_000",
            "geometry_refs": ["pp_ocr_0002", "pp_ocr_0005"],
            "evidence": "下标 4 被 OCR 成 A",
            "confidence": 0.95,
        },
    ]
    payload["coordinate_system"]["vertical_axis"]["kind"] = "relative_sequence"
    payload["coordinate_system"]["vertical_axis"]["calibration_ocr_ids"] = []
    enriched = apply_ppstructure_geometry(payload, geometry)
    interval = enriched["primitives"]["stratigraphic_intervals"][0]
    assert interval["geometry_refs"] == [merged_cell["id"]]
    assert interval["geometry_bbox"] == merged_cell["bbox"]
    unit_2, unit_4 = enriched["primitives"]["stratigraphic_intervals"][1:]
    assert len(unit_2["geometry_refs"]) == 1
    assert unit_2["geometry_bbox"][0] >= 60
    assert len(unit_4["geometry_refs"]) == 1
    assert unit_4["geometry_bbox"][0] >= 60


def test_ppstructure_retries_without_table_recognition_when_paddlex_ocr_is_missing(tmp_path: Path) -> None:
    """PaddleX 表格 OCR 未初始化时应保留布局与通用 OCR，而不是让整张图失败。"""

    image_path = _image(tmp_path)

    class FallbackPipeline:
        """中文说明：首轮模拟 PaddleX 的 text_rec_model 异常，第二轮返回可用几何。"""

        def __init__(self) -> None:
            self.options: list[dict[str, Any]] = []

        def predict(self, *_args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            """中文说明：记录表格识别开关，并验证退化调用关闭了问题分支。"""

            self.options.append(dict(kwargs))
            if kwargs.get("use_table_recognition"):
                raise AttributeError("'NoneType' object has no attribute 'text_rec_model'")
            return [{"res": _raw_pp_result()}]

    predictor = FallbackPipeline()
    extractor = PPStructureV3GeometryExtractor(pipeline=predictor, cache_dir=tmp_path / "fallback-cache")
    geometry = extractor.extract(image_path)
    assert [item["use_table_recognition"] for item in predictor.options] == [True, False]
    assert geometry["runtime"]["table_recognition_fallback"] is True
    assert any("ppstructure_table_fallback" in item for item in geometry["uncertainties"])


def test_duplicate_semantic_track_ids_are_merged_without_duplicate_geometry() -> None:
    """同一 PP 整表轨道的多种 VLM 角色应合并，最终轨道 ID 必须保持唯一。"""

    payload = _semantic_payload()
    payload["tracks"] = [
        {"id": "pp_track_000", "role": "depth", "header": "深度", "evidence": "深度表头"},
        {"id": "pp_track_000", "role": "curve", "header": "伽马", "evidence": "伽马表头"},
    ]
    payload["coordinate_system"]["vertical_axis"]["track_id"] = "pp_track_000"
    enriched = apply_ppstructure_geometry(payload, _geometry())
    assert [track["id"] for track in enriched["tracks"]] == ["pp_track_000"]
    assert enriched["tracks"][0]["semantic_roles"] == ["depth", "curve"]
    assert enriched["tracks"][0]["semantic_headers"] == ["深度", "伽马"]


def test_node_enrichment_rejects_missing_existing_nodes() -> None:
    """第四次调用必须覆盖前三段的每个节点，缺少任一局部 ID 时不得静默回退。"""

    payload = apply_ppstructure_geometry(_semantic_payload(), _geometry())
    with pytest.raises(ValueError, match="缺少节点"):
        apply_node_enrichment(
            payload,
            {
                "schema_version": "table_embedded_hybrid.node_enrichment.v1",
                "nodes": [
                    {
                        "id": "test_table",
                        "official_name": "测试地层表",
                        "basis": "already_official",
                        "confidence": 0.99,
                    }
                ],
            },
        )


def test_ppstructure_replaces_vlm_pixels_before_depth_calculation(tmp_path: Path) -> None:
    """区间和深度刻度必须来自 PP 框，故意注入的错误 VLM 坐标不能进入拟合。"""

    image_path = _image(tmp_path)
    enriched = apply_ppstructure_geometry(_semantic_payload(), _geometry())
    assert enriched["coordinate_system"]["content_bbox"] == [0, 0, 160, 300]
    assert enriched["tracks"][0]["bbox"] == [0, 0, 80, 300]
    assert enriched["primitives"]["stratigraphic_intervals"][0]["top_y"] == 60.0
    assert enriched["primitives"]["stratigraphic_intervals"][0]["bottom_y"] == 175.0
    assert [point["pixel_y"] for point in enriched["coordinate_system"]["vertical_axis"]["calibration_points"]] == [105.0, 205.0]
    assert enriched["geometry_policy"]["vlm_pixel_coordinates_used"] is False

    task = ImageExtractionTask(
        document_id="doc",
        chunk_id="chunk",
        image_id="image",
        image_index=0,
        image_path=str(image_path),
    )
    intermediate = TableEmbeddedHybridPipeline().run(task, enriched)
    assert intermediate["coordinate_system"]["coordinate_source"] == "PP-StructureV3"
    assert intermediate["coordinate_system"]["vlm_pixel_coordinates_used"] is False
    assert intermediate["quality"]["axis_rmse"] == pytest.approx(0.0, abs=1e-10)
    assert intermediate["parsed"]["stratigraphic_intervals"][0]["top_value"] == pytest.approx(55.0)


def test_thickness_cells_do_not_create_fake_absolute_depth() -> None:
    """逐层厚度不是连续深度轴，程序必须退化到相对层序并记录原因。"""

    enriched = apply_ppstructure_geometry(_semantic_payload(axis_kind="thickness"), _geometry())
    axis = enriched["coordinate_system"]["vertical_axis"]
    assert axis["kind"] == "relative_sequence"
    assert axis["unit"] == "relative"
    assert [point["value"] for point in axis["calibration_points"]] == [0.0, 1.0]
    assert any("禁止输出伪绝对深度" in item for item in enriched["uncertainties"])


def test_segmented_vlm_selects_pp_ids_without_returning_pixels(tmp_path: Path) -> None:
    """三段 VLM 选择 PP ID，第四次调用为全部现有节点补官方名和轨道表头。"""

    image_path = _image(tmp_path)

    class SegmentedVLM:
        """中文说明：按分段返回最小语义 JSON，并检查 Prompt 已携带 PP 几何目录。"""

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def describe_image(self, _image_path: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
            """中文说明：模拟模型只选择轨道、单元格和 OCR ID，不提供任何像素数字。"""

            self.prompts.append(prompt)
            task_name = str(kwargs.get("task_name") or "")
            if "官方名规范化" in task_name:
                assert '"track_header":"地层"' in prompt
                return {
                    "schema_version": "table_embedded_hybrid.node_enrichment.v1",
                    "nodes": [
                        {
                            "id": "test",
                            "official_name": "测试地层综合表",
                            "basis": "standardized_domain_term",
                            "confidence": 0.96,
                        },
                        {
                            "id": "unit_a",
                            "official_name": "层A",
                            "basis": "already_official",
                            "confidence": 0.99,
                        },
                    ],
                }
            assert '"id":"pp_cell_a"' in prompt
            if ":layout" in task_name:
                return {
                    "schema_version": "table_embedded_hybrid.segment.v1",
                    "segment": "layout",
                    "layout_family": "stratigraphic_column_table",
                    "diagram_id": "test",
                    "diagram_name": "测试表",
                    "coordinate_system": {
                        "vertical_axis": {
                            "kind": "depth",
                            "unit": "m",
                            "increases": "downward",
                            "track_id": "pp_track_001",
                            "calibration_ocr_ids": ["pp_depth_100", "pp_depth_200"],
                        }
                    },
                    "tracks": [
                        {"id": "pp_track_000", "role": "stratigraphy", "header": "地层", "parser": "cell", "evidence": "地层表头"},
                        {"id": "pp_track_001", "role": "depth", "header": "深度/m", "parser": "axis", "evidence": "深度表头"},
                    ],
                    "uncertainties": [],
                }
            if ":stratigraphy_lithology" in task_name:
                return {
                    "schema_version": "table_embedded_hybrid.segment.v1",
                    "segment": "stratigraphy_lithology",
                    "primitives": {
                        "stratigraphic_intervals": [
                            {"id": "unit_a", "name": "层A", "track_id": "pp_track_000", "geometry_refs": ["pp_cell_a"], "evidence": "层A 单元格", "confidence": 0.95}
                        ],
                        "reference_intervals": [],
                        "lithology_intervals": [],
                        "geological_feature_intervals": [],
                    },
                    "uncertainties": [],
                }
            return {
                "schema_version": "table_embedded_hybrid.segment.v1",
                "segment": "facies_reservoir_wells",
                "primitives": {
                    "facies_intervals": [],
                    "curve_tracks": [],
                    "curve_observations": [],
                    "reservoir_intervals": [],
                    "oil_layer_intervals": [],
                    "point_markers": [],
                    "objects": [],
                    "explicit_relations": [],
                },
                "uncertainties": [],
            }

    task = ImageExtractionTask(
        document_id="doc",
        chunk_id="chunk",
        image_id="image",
        image_index=0,
        image_path=str(image_path),
    )
    classification = StratigraphicSubtypeClassification(
        subtype=StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID,
        confidence=0.99,
        evidence="测试表格",
        source="test",
    )
    vlm = SegmentedVLM()
    payload = extract_segmented_table_visual(
        task,
        vlm,
        classification,
        geometry=_geometry(),
    )
    interval = payload["primitives"]["stratigraphic_intervals"][0]
    assert (interval["top_y"], interval["bottom_y"]) == (60.0, 175.0)
    assert payload["coordinate_system"]["vertical_axis"]["calibration_points"][0]["pixel_y"] == 105.0
    assert payload["geometry_policy"]["vlm_pixel_coordinates_used"] is False
    assert interval["official_name"] == "层A"
    assert interval["track_header"] == "地层"
    intermediate = TableEmbeddedHybridPipeline().run(task, payload)
    graph = build_table_embedded_hybrid_graph(task, intermediate)
    unit = next(entity for entity in graph.entities if entity.name == "层A")
    assert unit.official_name == "层A"
    assert unit.attributes["track_header"] == "地层"
    assert all(entity.official_name for entity in graph.entities)
    assert all("track_header" in entity.attributes for entity in graph.entities)
    assert len(vlm.prompts) == 4
