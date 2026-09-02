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
from src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid.submember_refinement import (
    build_submember_coverage,
    recover_submember_intervals,
    refine_small_cell_geometry,
)
from src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid.visual_track_extraction import (
    apply_visual_track_slice_responses,
    build_adjacent_visual_track_slices,
    build_semantic_tracks,
    build_visual_track_slice_prompt,
    cache_visual_track_slices,
    describe_adjacent_visual_track_slices,
    enrich_visual_track_primitives,
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
            {"id": "pp_track_000", "track_type": "table_text", "role": "stratigraphy", "header": "地层", "bbox": [900, 0, 999, 999]},
            {"id": "pp_track_001", "track_type": "table_text", "role": "depth", "header": "深度/m", "bbox": [0, 0, 1, 1]},
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
        {"id": "pp_track_000", "track_type": "table_text", "role": "depth", "header": "深度", "evidence": "深度表头"},
        {"id": "pp_track_000", "track_type": "curve", "role": "curve", "header": "伽马", "evidence": "伽马表头"},
    ]
    payload["coordinate_system"]["vertical_axis"]["track_id"] = "pp_track_000"
    enriched = apply_ppstructure_geometry(payload, _geometry())
    assert [track["id"] for track in enriched["tracks"]] == ["pp_track_000"]
    assert enriched["tracks"][0]["semantic_roles"] == ["depth", "curve"]
    assert enriched["tracks"][0]["semantic_track_types"] == ["table_text", "curve"]
    assert enriched["tracks"][0]["semantic_headers"] == ["深度", "伽马"]


def test_node_enrichment_rejects_missing_existing_nodes() -> None:
    """最终规范化调用必须覆盖既有节点，缺少任一局部 ID 时不得静默回退。"""

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


def test_segmented_vlm_selects_pp_ids_without_returning_pixels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """三段 VLM 选择 PP ID，最终规范化调用为全部现有节点补官方名和轨道表头。"""

    image_path = _image(tmp_path)

    class SegmentedVLM:
        """中文说明：按分段返回最小语义 JSON，并检查 Prompt 已携带 PP 几何目录。"""

        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.request_options: list[dict[str, Any]] = []

        def describe_image(self, _image_path: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
            """中文说明：模拟模型只选择轨道、单元格和 OCR ID，不提供任何像素数字。"""

            self.prompts.append(prompt)
            self.request_options.append(dict(kwargs))
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
                        {"id": "pp_track_000", "track_type": "table_text", "role": "stratigraphy", "header": "地层", "parser": "cell", "evidence": "地层表头"},
                        {"id": "pp_track_001", "track_type": "table_text", "role": "depth", "header": "深度/m", "parser": "axis", "evidence": "深度表头"},
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
    monkeypatch.delenv("STRATIGRAPHIC_TABLE_VLM_TIMEOUT_SECS", raising=False)
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
    assert [options["timeout"] for options in vlm.request_options] == [600.0] * 4


def test_small_low_confidence_cell_uses_generic_local_ocr(tmp_path: Path) -> None:
    """任意小尺寸低置信度单元格都应触发局部 OCR，不得限定为亚段或代号列。"""

    image_path = _image(tmp_path)
    geometry = _geometry()
    geometry["cells"][0].update(
        {
            "bbox": [0, 60, 80, 88],
            "text": "细粉砂",
            "ocr_ids": ["pp_ocr_a"],
        }
    )
    geometry["ocr_lines"][0].update(
        {
            "text": "细粉砂",
            "confidence": 0.55,
            "bbox": [5, 64, 72, 82],
        }
    )
    seen: list[str] = []

    def fake_recognizer(_image_path: Path, cells: Any) -> dict[str, list[dict[str, Any]]]:
        """中文说明：模拟局部放大后补全普通岩性单元格文字，并记录实际触发的单元格。"""

        seen.extend(str(cell["id"]) for cell in cells)
        return {
            "pp_cell_a": [
                {"text": "细粉砂岩", "confidence": 0.96, "variant": "autocontrast_gray"}
            ]
        }

    refined = refine_small_cell_geometry(image_path, geometry, recognizer=fake_recognizer)
    cell = next(item for item in refined["cells"] if item["id"] == "pp_cell_a")
    assert "pp_cell_a" in seen
    assert cell["refined_text"] == "细粉砂岩"
    assert cell["small_cell_ocr"]["decision"] == "local_ocr_higher_confidence"
    assert refined["quality"]["small_cell_refined_count"] == 1


def test_submember_cross_column_recovery_and_coverage_gate(tmp_path: Path) -> None:
    """亚段局部 OCR 应与代号列锚点交叉校验，并让三个行单元格全部被独立实体消费。"""

    image_path = _image(tmp_path)
    geometry = {
        "schema_version": "ppstructurev3.table_geometry.v1",
        "source_image_path": str(image_path),
        "image_size": {"width": 400, "height": 300},
        "tracks": [
            {"id": "segment", "header_text": "段", "bbox": [0, 0, 90, 300]},
            {"id": "submember", "header_text": "亚段", "bbox": [90, 0, 180, 300]},
            {"id": "code", "header_text": "代号", "bbox": [180, 0, 260, 300]},
        ],
        "cells": [
            {"id": "parent_ma5", "bbox": [0, 40, 90, 190], "text": "马五段", "ocr_ids": [], "track_ids": ["segment"]},
            {"id": "sub_1", "bbox": [90, 40, 180, 90], "text": "马五", "ocr_ids": ["ocr_sub_1"], "track_ids": ["submember"]},
            {"id": "sub_2", "bbox": [90, 90, 180, 140], "text": "马五", "ocr_ids": ["ocr_sub_2"], "track_ids": ["submember"]},
            {"id": "sub_3", "bbox": [90, 140, 180, 190], "text": "马五", "ocr_ids": ["ocr_sub_3"], "track_ids": ["submember"]},
            {"id": "code_1", "bbox": [180, 40, 260, 90], "text": "Om1", "ocr_ids": ["ocr_code_1"], "track_ids": ["code"]},
            {"id": "code_2", "bbox": [180, 90, 260, 140], "text": "Om8", "ocr_ids": ["ocr_code_2"], "track_ids": ["code"]},
            {"id": "code_3", "bbox": [180, 140, 260, 190], "text": "Om3", "ocr_ids": ["ocr_code_3"], "track_ids": ["code"]},
        ],
        "ocr_lines": [
            {"id": name, "text": "马五" if "sub" in name else "Om", "confidence": 0.6, "bbox": box}
            for name, box in (
                ("ocr_sub_1", [100, 50, 160, 75]),
                ("ocr_sub_2", [100, 100, 160, 125]),
                ("ocr_sub_3", [100, 150, 160, 175]),
                ("ocr_code_1", [190, 50, 245, 75]),
                ("ocr_code_2", [190, 100, 245, 125]),
                ("ocr_code_3", [190, 150, 245, 175]),
            )
        ],
        "quality": {},
    }

    def fake_recognizer(_image_path: Path, _cells: Any) -> dict[str, list[dict[str, Any]]]:
        """中文说明：首尾行提供双列数字锚点，中间行保留冲突以验证结构序列审计。"""

        return {
            "sub_1": [{"text": "马五1", "confidence": 0.98}],
            "code_1": [{"text": "Om1", "confidence": 0.97}],
            "sub_3": [{"text": "马五3", "confidence": 0.98}],
            "code_3": [{"text": "Om3", "confidence": 0.97}],
        }

    refined_geometry = refine_small_cell_geometry(
        image_path,
        geometry,
        recognizer=fake_recognizer,
    )
    payload = {
        "primitives": {
            "stratigraphic_intervals": [
                {
                    "id": "ma5_parent",
                    "name": "马五段",
                    "geometry_refs": ["parent_ma5"],
                }
            ]
        },
        "ppstructure_geometry": refined_geometry,
    }
    recovered = recover_submember_intervals(payload)
    children = recovered["primitives"]["stratigraphic_intervals"][1:]
    assert [item["name"] for item in children] == ["马五1", "马五2", "马五3"]
    assert all(item["parent_id"] == "ma5_parent" for item in children)
    assert children[1]["recognition_source"] == "cross_column_anchored_row_sequence"
    coverage = build_submember_coverage(
        recovered["primitives"],
        refined_geometry,
    )
    assert coverage["ok"] is True
    assert coverage["detected_cell_count"] == 3
    assert coverage["entity_count"] == 3
    assert coverage["unconsumed_cell_ids"] == []


def test_submember_coverage_gate_reports_unconsumed_cells(tmp_path: Path) -> None:
    """存在亚段行却没有实体引用时，质量门必须明确列出未消费单元格。"""

    image_path = _image(tmp_path)
    geometry = {
        "quality": {"small_cell_refinement_version": "small-cell-ocr-with-submember-crosscheck.v1"},
        "submember_groups": [
            {
                "id": "group",
                "rows": [
                    {"cell_id": "sub_1"},
                    {"cell_id": "sub_2"},
                ],
            }
        ],
    }
    coverage = build_submember_coverage({"stratigraphic_intervals": []}, geometry)
    assert coverage["ok"] is False
    assert coverage["detected_cell_count"] == 2
    assert coverage["unconsumed_cell_ids"] == ["sub_1", "sub_2"]
    assert "未生成实体" in coverage["errors"][0]


def test_semantic_track_group_merges_internal_curve_grid_columns() -> None:
    """曲线网格的空白/刻度窄列应归入中间曲线轨道，不能各自成为实体列。"""

    physical = [
        {"id": "p0", "order": 0, "bbox": [0, 0, 80, 300], "header_text": "地层"},
        {"id": "p1", "order": 1, "bbox": [80, 0, 105, 300], "header_text": ""},
        {"id": "p2", "order": 2, "bbox": [105, 0, 145, 300], "header_text": "RS / RD"},
        {"id": "p3", "order": 3, "bbox": [145, 0, 180, 300], "header_text": "2 / 20000"},
        {"id": "p4", "order": 4, "bbox": [180, 0, 280, 300], "header_text": "岩性特征"},
    ]
    mapped = [
        {**physical[0], "track_type": "table_text", "role": "stratigraphy", "header": "地层"},
        {**physical[2], "track_type": "curve", "role": "curve", "header": "RS / RD"},
        {**physical[4], "track_type": "table_text", "role": "text", "header": "岩性特征"},
    ]
    tracks, mapping = build_semantic_tracks(mapped, physical)
    curve = next(item for item in tracks if item["role"] == "curve")
    assert curve["bbox"] == [80, 0, 180, 300]
    assert curve["member_track_ids"] == ["p1", "p2", "p3"]
    assert {mapping[item] for item in ("p1", "p2", "p3")} == {"p2"}


def test_semantic_track_group_merges_adjacent_profile_columns_with_shared_cells() -> None:
    """中文说明：相邻岩性图例列共享 PP 单元格时，应合并为一条完整剖面轨道并保留物理列来源。"""

    physical = [
        {"id": "pp_track_005", "order": 5, "bbox": [363, 49, 380, 1066], "header_text": ""},
        {"id": "pp_track_006", "order": 6, "bbox": [380, 49, 405, 1066], "header_text": ""},
        {"id": "pp_track_007", "order": 7, "bbox": [405, 49, 450, 1066], "header_text": "剖面"},
        {"id": "pp_track_008", "order": 8, "bbox": [450, 49, 622, 1066], "header_text": "CNL / AC / DEN"},
    ]
    mapped = [
        {**physical[0], "track_type": "table_text", "role": "other", "header": ""},
        {**physical[1], "track_type": "legend", "role": "lithology", "header": ""},
        {**physical[2], "track_type": "legend", "role": "lithology", "header": "剖面"},
        {**physical[3], "track_type": "curve", "role": "curve", "header": "CNL / AC / DEN"},
    ]
    cells = [
        {
            "id": "pp_cell_0003",
            "bbox": [363, 49, 449, 187],
            "track_ids": ["pp_track_006", "pp_track_007"],
        },
        {
            "id": "pp_cell_0024",
            "bbox": [363, 188, 449, 214],
            "track_ids": ["pp_track_006", "pp_track_007"],
        },
    ]

    tracks, mapping = build_semantic_tracks(mapped, physical, cells)

    profiles = [item for item in tracks if item.get("role") == "lithology"]
    assert len(profiles) == 1
    assert profiles[0]["id"] == "pp_track_007"
    assert profiles[0]["bbox"] == [363, 49, 450, 1066]
    assert profiles[0]["member_track_ids"] == ["pp_track_006", "pp_track_007"]
    assert profiles[0]["merge_evidence_cell_ids"] == ["pp_cell_0003", "pp_cell_0024"]
    assert mapping["pp_track_006"] == "pp_track_007"
    assert mapping["pp_track_007"] == "pp_track_007"


def test_every_track_cell_becomes_entity_and_only_aligns_with_left_neighbor(tmp_path: Path) -> None:
    """三条文字轨道都应建点，第三轨只能连第二轨，不能越过第二轨直接连接第一轨。"""

    image_path = _image(tmp_path)
    geometry = _geometry()
    geometry["source_image_path"] = str(image_path)
    geometry["content_bbox"] = [0, 0, 260, 300]
    geometry["tracks"][1]["header_text"] = "沉积微相"
    geometry["tracks"].append(
        {"id": "pp_track_002", "order": 2, "bbox": [160, 0, 260, 300], "header_text": "岩性特征"}
    )
    geometry["cells"].extend(
        [
            {"id": "facies_a", "bbox": [80, 60, 160, 175], "ocr_ids": ["facies_ocr_a"], "text": "含膏结核云坪", "track_ids": ["pp_track_001"]},
            {"id": "facies_b", "bbox": [80, 175, 160, 300], "ocr_ids": ["facies_ocr_b"], "text": "灰云坪", "track_ids": ["pp_track_001"]},
            {"id": "feature_a", "bbox": [160, 60, 260, 175], "ocr_ids": ["feature_ocr_a"], "text": "泥质白云岩", "track_ids": ["pp_track_002"]},
            {"id": "feature_b", "bbox": [160, 175, 260, 300], "ocr_ids": ["feature_ocr_b"], "text": "灰质云岩", "track_ids": ["pp_track_002"]},
        ]
    )
    geometry["ocr_lines"].extend(
        [
            {"id": "facies_ocr_a", "text": "含膏结核云坪", "confidence": 0.98, "bbox": [90, 90, 150, 120], "cell_id": "facies_a", "track_id": "pp_track_001"},
            {"id": "facies_ocr_b", "text": "灰云坪", "confidence": 0.98, "bbox": [90, 205, 150, 235], "cell_id": "facies_b", "track_id": "pp_track_001"},
            {"id": "feature_ocr_a", "text": "泥质白云岩", "confidence": 0.97, "bbox": [170, 90, 245, 120], "cell_id": "feature_a", "track_id": "pp_track_002"},
            {"id": "feature_ocr_b", "text": "灰质云岩", "confidence": 0.96, "bbox": [170, 205, 245, 235], "cell_id": "feature_b", "track_id": "pp_track_002"},
        ]
    )
    geometry["rule_lines"]["vertical_lines"].append(260)
    payload = _semantic_payload()
    payload["coordinate_system"]["vertical_axis"].update(
        {"kind": "relative_sequence", "unit": "relative", "track_id": "pp_track_000", "calibration_ocr_ids": []}
    )
    payload["tracks"][1].update(
        {"track_type": "table_text", "role": "facies", "header": "沉积微相", "parser": "text"}
    )
    payload["tracks"].append(
        {"id": "pp_track_002", "track_type": "table_text", "role": "text", "header": "岩性特征", "parser": "text", "evidence": "表头"}
    )
    enriched = apply_ppstructure_geometry(payload, geometry)
    task = ImageExtractionTask(
        document_id="doc",
        chunk_id="chunk",
        image_id="image",
        image_index=0,
        image_path=str(image_path),
        caption="测试综合柱状图",
    )
    intermediate = TableEmbeddedHybridPipeline().run(task, enriched)
    track_nodes = intermediate["parsed"]["track_intervals"]
    assert {item["name"] for item in track_nodes} == {
        "含膏结核云坪", "灰云坪", "泥质白云岩", "灰质云岩"
    }
    middle_nodes = [item for item in track_nodes if item["track_id"] == "pp_track_001"]
    right_nodes = [item for item in track_nodes if item["track_id"] == "pp_track_002"]
    cross_edges = [
        item
        for item in intermediate["alignment_relations"]
        if item.get("basis") == "adjacent_track_shared_vertical_axis_overlap"
    ]
    middle_ids = {node["id"] for node in middle_nodes}
    right_ids = {node["id"] for node in right_nodes}
    assert {
        item["target_id"] for item in cross_edges if item["source_id"] in middle_ids
    } == {"unit_a", "unit_b"}
    assert {
        item["target_id"] for item in cross_edges if item["source_id"] in right_ids
    } == middle_ids
    assert not any(
        item["source_id"] in right_ids and item["target_id"] in {"unit_a", "unit_b"}
        for item in cross_edges
    )
    graph = build_table_embedded_hybrid_graph(task, intermediate)
    assert sum(entity.type == "diagram_track" for entity in graph.entities) == 3
    assert sum(relation.type == "has_track" for relation in graph.relations) == 3
    assert sum(relation.type == "located_in_track" for relation in graph.relations) >= 6


def test_measurement_track_ocr_noise_is_not_promoted_to_graph_entities() -> None:
    """中文说明：孔隙度、渗透率图形中的伪文字不得被 PP 文本兜底升级为图谱实体，地质分类文字仍需保留。"""

    tracks = [
        {"id": "porosity", "order": 0, "bbox": [0, 0, 100, 300], "track_type": "table_text", "role": "porosity", "header": "测井孔隙度%"},
        {"id": "permeability", "order": 1, "bbox": [100, 0, 200, 300], "track_type": "table_text", "role": "permeability", "header": "测井渗透率M3"},
        {"id": "facies", "order": 2, "bbox": [200, 0, 300, 300], "track_type": "table_text", "role": "facies", "header": "沉积相"},
    ]
    geometry = {
        "content_bbox": [0, 0, 300, 300],
        "tracks": [{"id": item["id"], "order": item["order"], "bbox": item["bbox"], "header_text": item["header"]} for item in tracks],
        "cells": [
            {"id": "porosity_header", "bbox": [0, 0, 100, 60], "ocr_ids": [], "text": "测井孔隙度%", "track_ids": ["porosity"]},
            {"id": "porosity_noise", "bbox": [0, 60, 100, 160], "ocr_ids": ["ocr_h"], "refined_text": "H", "track_ids": ["porosity"]},
            {"id": "permeability_header", "bbox": [100, 0, 200, 60], "ocr_ids": [], "text": "测井渗透率M3", "track_ids": ["permeability"]},
            {"id": "permeability_noise", "bbox": [100, 60, 200, 160], "ocr_ids": ["ocr_1"], "refined_text": "1", "track_ids": ["permeability"]},
            {"id": "facies_header", "bbox": [200, 0, 300, 60], "ocr_ids": [], "text": "沉积相", "track_ids": ["facies"]},
            {"id": "facies_label", "bbox": [200, 60, 300, 160], "ocr_ids": ["ocr_facies"], "refined_text": "灰云坪", "track_ids": ["facies"]},
        ],
        "ocr_lines": [
            {"id": "ocr_h", "cell_id": "porosity_noise", "text": "H", "confidence": 0.45},
            {"id": "ocr_1", "cell_id": "permeability_noise", "text": "1", "confidence": 0.45},
            {"id": "ocr_facies", "cell_id": "facies_label", "text": "灰云坪", "confidence": 0.98},
        ],
        "rule_lines": {"horizontal_lines": [0, 60, 160, 300], "vertical_lines": [0, 100, 200, 300]},
    }
    payload = {
        "tracks": tracks,
        "primitives": {
            "track_intervals": [
                {
                    "id": "stale_noise",
                    "name": "ANV",
                    "track_id": "porosity",
                    "geometry_refs": ["porosity_noise"],
                    "recognition_source": "PP-StructureV3.refined_cell_text",
                }
            ]
        },
    }

    enriched = enrich_visual_track_primitives(payload, geometry)
    assert [item["name"] for item in enriched["primitives"]["track_intervals"]] == ["灰云坪"]
    audit = {
        item["track_id"]: item
        for item in enriched["visual_track_extraction"]["track_entity_coverage"]["tracks"]
    }
    assert audit["porosity"]["fallback_skipped"] is True
    assert audit["permeability"]["fallback_skipped"] is True
    assert audit["facies"]["fallback_entity_cell_count"] == 1


def legacy_curve_track_is_sliced_by_richer_adjacent_entities_and_described_by_vlm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中文说明：保留旧曲线 VLM 切片回归场景作为历史参考，当前默认流程不再收集该测试。"""

    image_path = tmp_path / "microfacies-curve.png"
    Image.new("RGB", (300, 300), "white").save(image_path)
    cache_dir = tmp_path / "track-slice-cache"
    monkeypatch.setenv("TABLE_VISUAL_TRACK_SLICE_CACHE_DIR", str(cache_dir))
    geometry = {
        "schema_version": "ppstructurev3.table_geometry.v1",
        "engine": "test_fixture",
        "coordinate_space": "original_pixels",
        "source_image_path": str(image_path),
        "image_size": {"width": 300, "height": 300},
        "content_bbox": [0, 0, 300, 300],
        "tracks": [
            {"id": "left", "order": 0, "bbox": [0, 0, 80, 300], "header_text": "地层"},
            {"id": "microfacies", "order": 1, "bbox": [80, 0, 180, 300], "header_text": "沉积微相"},
            {"id": "sea_level", "order": 2, "bbox": [180, 0, 300, 300], "header_text": "相对海平面变化"},
        ],
        "cells": [
            {"id": "left_body", "bbox": [0, 60, 80, 300], "ocr_ids": ["ocr_left"], "text": "马五段", "track_ids": ["left"]},
            {"id": "micro_1", "bbox": [80, 60, 180, 180], "ocr_ids": ["ocr_micro_1"], "text": "含膏结核云坪", "track_ids": ["microfacies"]},
            {"id": "micro_2", "bbox": [80, 180, 180, 220], "ocr_ids": ["ocr_micro_2"], "text": "灰云坪", "track_ids": ["microfacies"]},
            {"id": "micro_3", "bbox": [80, 220, 180, 300], "ocr_ids": ["ocr_micro_3"], "text": "膏云坪", "track_ids": ["microfacies"]},
            {"id": "curve_body", "bbox": [180, 60, 300, 300], "ocr_ids": [], "text": "", "track_ids": ["sea_level"]},
        ],
        "ocr_lines": [
            {"id": "ocr_left", "text": "马五段", "confidence": 0.99, "bbox": [10, 100, 60, 130], "cell_id": "left_body", "track_id": "left"},
            {"id": "ocr_micro_1", "text": "含膏结核云坪", "confidence": 0.99, "bbox": [90, 90, 170, 120], "cell_id": "micro_1", "track_id": "microfacies"},
            {"id": "ocr_micro_2", "text": "灰云坪", "confidence": 0.99, "bbox": [95, 188, 165, 210], "cell_id": "micro_2", "track_id": "microfacies"},
            {"id": "ocr_micro_3", "text": "膏云坪", "confidence": 0.99, "bbox": [95, 240, 165, 270], "cell_id": "micro_3", "track_id": "microfacies"},
        ],
        "rule_lines": {
            "available": True,
            "vertical_lines": [0, 80, 180, 300],
            "horizontal_lines": [0, 60, 180, 220, 300],
        },
        "quality": {"ocr_line_count": 4, "cell_count": 5, "track_count": 3},
    }
    payload = {
        "schema_version": "table_embedded_hybrid.v1",
        "diagram_id": "microfacies_curve",
        "diagram_name": "沉积微相与相对海平面变化",
        "layout_family": "stratigraphic_column_table",
        "coordinate_system": {
            "vertical_axis": {
                "kind": "relative_sequence",
                "unit": "relative",
                "increases": "downward",
                "track_id": "left",
                "calibration_ocr_ids": [],
            }
        },
        # 中文说明：故意打乱 VLM 返回顺序，验证最终轨道仍按 PP 的 x 坐标从左到右重排。
        "tracks": [
            {"id": "sea_level", "track_type": "curve", "role": "curve", "header": "相对海平面变化", "parser": "curve"},
            {"id": "left", "track_type": "table_text", "role": "stratigraphy", "header": "地层", "parser": "text"},
            {"id": "microfacies", "track_type": "table_text", "role": "facies", "header": "沉积微相", "parser": "text"},
        ],
        "primitives": {
            **{field: [] for field in (
                "reference_intervals", "lithology_intervals", "reservoir_intervals",
                "oil_layer_intervals", "geological_feature_intervals", "curve_observations",
                "track_intervals",
            )},
            "stratigraphic_intervals": [
                {"id": "ma5", "name": "马五段", "track_id": "left", "geometry_refs": ["left_body"], "evidence": "马五段", "confidence": 0.99}
            ],
            "facies_intervals": [
                {"id": "facies_1", "name": "含膏结核云坪", "track_id": "microfacies", "geometry_refs": ["micro_1"], "evidence": "含膏结核云坪", "confidence": 0.99},
                {"id": "facies_2", "name": "灰云坪", "track_id": "microfacies", "geometry_refs": ["micro_2"], "evidence": "灰云坪", "confidence": 0.99},
                {"id": "facies_3", "name": "膏云坪", "track_id": "microfacies", "geometry_refs": ["micro_3"], "evidence": "膏云坪", "confidence": 0.99},
            ],
            "curve_tracks": [
                {"id": "curve_sea_level", "name": "相对海平面变化", "track_id": "sea_level", "color": "black", "visual_form": "continuous_curve", "scale_transform": "linear", "evidence": "曲线表头"}
            ],
            "point_markers": [],
            "objects": [],
            "legend_entries": [],
            "explicit_relations": [],
        },
        "uncertainties": [],
    }
    enriched = apply_ppstructure_geometry(payload, geometry)
    assert [item["id"] for item in enriched["tracks"]] == ["left", "microfacies", "sea_level"]
    assert [item["order"] for item in enriched["tracks"]] == [0, 1, 2]
    assert enriched["tracks"][2]["previous_track_id"] == "microfacies"
    assert enriched["tracks"][2]["track_type_source"] == "VLM.layout_track_classification"
    slices = build_adjacent_visual_track_slices(enriched, geometry)
    assert [item["bbox"] for item in slices] == [
        [180, 60, 300, 180],
        [180, 180, 300, 220],
        [180, 220, 300, 300],
    ]
    assert {item["source_track_side"] for item in slices} == {"left"}
    assert {item["source_track_entity_count"] for item in slices} == {3}
    assert all(
        item["pixel_range"]
        == {
            "left_x": item["bbox"][0],
            "top_y": item["top_y"],
            "right_x": item["bbox"][2],
            "bottom_y": item["bottom_y"],
            "coordinate_space": "original_pixels",
        }
        for item in slices
    )

    class SliceVLM:
        """中文说明：记录每个真实裁剪图尺寸，并按切片 ID 返回受约束曲线变化描述。"""

        def __init__(self) -> None:
            self.crop_sizes: list[tuple[int, int]] = []
            self.crop_paths: list[Path] = []

        def describe_image(self, crop_path: str, _prompt: str, **kwargs: Any) -> dict[str, Any]:
            """中文说明：验证传入 VLM 的是目标轨道裁剪图，而不是整张原图。"""

            with Image.open(crop_path) as crop:
                self.crop_sizes.append(crop.size)
            self.crop_paths.append(Path(crop_path))
            task_name = str(kwargs.get("task_name") or "")
            expected = next(item for item in slices if item["slice_id"] in task_name)
            changes = {"micro_1": "falling", "micro_2": "peak", "micro_3": "rising"}
            return {
                "schema_version": "table_embedded_hybrid.track_slice_description.v2",
                "slice_id": expected["slice_id"],
                "track_id": expected["track_id"],
                "track_type": expected["track_type"],
                "source_cell_id": expected["source_cell_id"],
                "description": f"{expected['source_label']}区间内曲线发生可见变化",
                "curve_change": changes[expected["source_cell_id"]],
                "curve_readings": [
                    {
                        "curve_name": "相对海平面",
                        "unit": "relative",
                        "left_scale_value": 0,
                        "right_scale_value": 1,
                        "scale_transform": "linear",
                        "top_value": 0.2,
                        "middle_value": 0.5,
                        "bottom_value": 0.8,
                        "minimum_value": 0.2,
                        "maximum_value": 0.8,
                        "change": changes[expected["source_cell_id"]],
                        "value_basis": "表头刻度与曲线位置",
                        "confidence": 0.9,
                        "uncertainty": "",
                    }
                ],
                "legend_interpretations": [],
                "visual_features": ["黑色连续曲线"],
                "confidence": 0.95,
                "uncertainty": "",
            }

    task = ImageExtractionTask(
        document_id="doc",
        chunk_id="chunk",
        image_id="microfacies-curve",
        image_index=0,
        image_path=str(image_path),
    )
    vlm = SliceVLM()
    described = describe_adjacent_visual_track_slices(task, vlm, enriched)
    assert vlm.crop_sizes == [(120, 120), (120, 40), (120, 80)]
    assert all(path.is_absolute() and path.is_file() for path in vlm.crop_paths)
    assert all(cache_dir.resolve() in path.parents for path in vlm.crop_paths)
    observations = [
        item
        for item in described["primitives"]["curve_observations"]
        if item.get("recognition_source") == "VLM.adjacent_track_slice_description"
    ]
    assert [(item["top_y"], item["bottom_y"]) for item in observations] == [
        (60.0, 180.0), (180.0, 220.0), (220.0, 300.0)
    ]
    assert {item["aligned_from_track_id"] for item in observations} == {"microfacies"}
    assert all(item["pixel_range"]["coordinate_space"] == "original_pixels" for item in observations)
    assert all(item["curve_readings"][0]["curve_name"] == "相对海平面" for item in observations)
    assert all(item["curve_value_source"] == "VLM.track_header_and_cropped_slice" for item in observations)
    assert {
        item["cropped_image_cache_path"] for item in observations
    } == {str(path.resolve()) for path in vlm.crop_paths}
    intermediate = TableEmbeddedHybridPipeline().run(task, described)
    observation_ids = {item["id"] for item in intermediate["parsed"]["curve_observations"]}
    adjacent_edges = [
        item
        for item in intermediate["alignment_relations"]
        if item.get("relation_type") == "aligned_with"
        and item.get("source_id") in observation_ids
    ]
    assert {item["target_id"] for item in adjacent_edges} == {"facies_1", "facies_2", "facies_3"}
    assert all(item["target_track_id"] == "microfacies" for item in adjacent_edges)
    visual_order_edges = [
        item
        for item in intermediate["alignment_relations"]
        if item.get("relation_type") == "directly_overlies"
        and item.get("basis") == "same_visual_track_adjacent_pixel_order"
    ]
    ordered_observation_ids = [item["id"] for item in intermediate["parsed"]["curve_observations"]]
    assert [
        (item["source_id"], item["target_id"])
        for item in visual_order_edges
    ] == list(zip(ordered_observation_ids, ordered_observation_ids[1:]))
    assert intermediate["quality"]["visual_track_directly_overlies_count"] == 2
    assert intermediate["quality"]["track_type_and_slice_quality"]["ok"] is True
    graph = build_table_embedded_hybrid_graph(task, intermediate)
    description_nodes = [
        entity
        for entity in graph.entities
        if entity.attributes.get("recognition_source")
        == "VLM.adjacent_track_slice_description"
    ]
    assert len(description_nodes) == 3
    assert all(entity.attributes["pixel_range"]["coordinate_space"] == "original_pixels" for entity in description_nodes)
    description_node_ids = {entity.id for entity in description_nodes}
    graph_visual_order_edges = [
        relation
        for relation in graph.relations
        if relation.type == "directly_overlies"
        and relation.source_id in description_node_ids
        and relation.target_id in description_node_ids
    ]
    assert len(graph_visual_order_edges) == 2
    assert all(
        relation.attributes["inference_basis"]
        == "same_visual_track_adjacent_pixel_order"
        for relation in graph_visual_order_edges
    )
    assert all(
        Path(entity.attributes["cropped_image_cache_path"]).is_absolute()
        and Path(entity.attributes["cropped_image_cache_path"]).is_file()
        for entity in description_nodes
    )


def test_curve_track_does_not_generate_or_request_vlm_slices(tmp_path: Path) -> None:
    """中文说明：curve 轨道仍保留语义和 PP 几何，但不生成切片，也不允许调用 VLM。"""

    image_path = _image(tmp_path)
    payload = {
        "tracks": [
            {"id": "text", "bbox": [0, 0, 100, 300], "track_type": "table_text", "header": "岩性特征"},
            {"id": "curve", "bbox": [100, 0, 200, 300], "track_type": "curve", "role": "curve", "header": "GR 0-350"},
        ],
        "ppstructure_geometry": {
            "source_image_path": str(image_path),
            "content_bbox": [0, 0, 200, 300],
            "cells": [],
        },
        "primitives": {
            "stratigraphic_intervals": [],
            "curve_tracks": [{"id": "curve_gr", "track_id": "curve", "name": "GR"}],
            "curve_observations": [],
            "track_intervals": [],
        },
        "visual_track_extraction": {},
        "uncertainties": [],
    }

    class RejectCurveVLM:
        """中文说明：任何切片模型调用都立即失败，用于证明 curve 路径被彻底跳过。"""

        def describe_image(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            """中文说明：测试替身不应被调用。"""

            raise AssertionError("curve 切片不应调用 VLM")

    assert build_adjacent_visual_track_slices(
        payload,
        payload["ppstructure_geometry"],
    ) == []
    task = ImageExtractionTask(
        document_id="doc",
        chunk_id="chunk",
        image_id="curve-vlm-disabled",
        image_index=0,
        image_path=str(image_path),
    )
    described = describe_adjacent_visual_track_slices(task, RejectCurveVLM(), payload)
    quality = described["visual_track_extraction"]["adjacent_slice_description"]
    assert quality["slice_count"] == 0
    assert quality["completed_count"] == 0
    assert quality["vlm_slice_track_types"] == ["legend"]
    assert quality["curve_slice_vlm_enabled"] is False
    assert described["primitives"]["curve_observations"] == []


def test_visual_track_slice_failure_does_not_stop_following_slices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中文说明：即使旧上游混入 curve 切片，描述入口也必须在调用 VLM 前丢弃。"""

    image_path = _image(tmp_path)

    def make_slice(slice_id: str, source_cell_id: str, top_y: float, bottom_y: float) -> dict[str, Any]:
        """中文说明：构造带完整文字轨道基准审计字段的曲线切片。"""

        return {
            "slice_id": slice_id,
            "track_id": "curve",
            "track_order": 1,
            "track_type": "curve",
            "recognition_stage": 3,
            "track_role": "curve",
            "is_lithology_profile": False,
            "track_header": "GR 0-350",
            "source_track_id": "text",
            "source_track_order": 0,
            "source_track_side": "left",
            "source_track_distance": 1,
            "source_track_entity_count": 2,
            "adjacent_track_entity_counts": {
                "left": {"track_id": "text", "entity_count": 2, "track_distance": 1, "track_type": "table_text"}
            },
            "candidate_table_text_track_entity_counts": {
                "left": {"track_id": "text", "entity_count": 2, "track_distance": 1, "track_type": "table_text"}
            },
            "source_track_type": "table_text",
            "source_track_header": "岩性特征",
            "source_entity_id": source_cell_id,
            "source_cell_id": source_cell_id,
            "source_geometry_refs": [source_cell_id],
            "source_label": source_cell_id,
            "bbox": [100, top_y, 200, bottom_y],
            "top_y": top_y,
            "bottom_y": bottom_y,
            "pixel_range": {
                "left_x": 100,
                "top_y": top_y,
                "right_x": 200,
                "bottom_y": bottom_y,
                "coordinate_space": "original_pixels",
            },
            "coordinate_source": "PP-StructureV3.table_text_entity_y_projection",
            "cropped_image_cache_path": str(image_path.resolve()),
        }

    slices = [
        make_slice("bad_slice", "row_1", 0.0, 100.0),
        make_slice("unreadable_slice", "row_2", 100.0, 200.0),
    ]
    monkeypatch.setattr(
        "src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid.visual_track_extraction.build_adjacent_visual_track_slices",
        lambda _payload, _geometry: slices,
    )
    monkeypatch.setattr(
        "src.extractors.image_extractor.stratigraphic_profile.table_embedded_hybrid.visual_track_extraction.cache_visual_track_slices",
        lambda _image_path, _slices: slices,
    )

    class ContinueVLM:
        """中文说明：让首片连续失败、次片返回不可读结果，以验证循环不会提前终止。"""

        def __init__(self) -> None:
            self.call_count = 0

        def describe_image(self, _crop_path: str, _prompt: str, **kwargs: Any) -> dict[str, Any]:
            """中文说明：根据任务名返回坏数组或结构完整的不可读曲线响应。"""

            self.call_count += 1
            expected = slices[0] if "bad_slice" in str(kwargs.get("task_name") or "") else slices[1]
            response = {
                "schema_version": "table_embedded_hybrid.track_slice_description.v2",
                "slice_id": expected["slice_id"],
                "track_id": "wrong_track" if expected["slice_id"] == "bad_slice" else "curve",
                "track_type": "curve",
                "source_cell_id": expected["source_cell_id"],
                "description": "切片过窄，无法可靠读取曲线",
                "curve_change": "none" if expected["slice_id"] == "unreadable_slice" else "not_applicable",
                "curve_readings": [None],
                "legend_interpretations": [],
                "visual_features": ["窄切片"],
                "confidence": 0.0,
                "uncertainty": "无法读取",
            }
            return response

    payload = {
        "ppstructure_geometry": {"source_image_path": str(image_path)},
        "primitives": {"curve_tracks": [], "curve_observations": [], "track_intervals": []},
        "visual_track_extraction": {},
        "uncertainties": [],
    }
    task = ImageExtractionTask(
        document_id="doc",
        chunk_id="chunk",
        image_id="continue-after-error",
        image_index=0,
        image_path=str(image_path),
    )
    vlm = ContinueVLM()

    described = describe_adjacent_visual_track_slices(task, vlm, payload)

    quality = described["visual_track_extraction"]["adjacent_slice_description"]
    assert vlm.call_count == 0
    assert quality["slice_count"] == 0
    assert quality["completed_count"] == 0
    assert quality["missing_count"] == 0
    assert quality["curve_slice_vlm_enabled"] is False
    assert described["primitives"]["curve_observations"] == []


def test_visual_track_uses_right_neighbor_when_it_has_more_entities() -> None:
    """右邻轨道已识别实体数更多时，图像轨道必须改用右侧实体的 y 像素范围切割。"""

    geometry = {
        "content_bbox": [0, 0, 300, 300],
        "cells": [],
        "rule_lines": {"horizontal_lines": [0, 50, 160, 300]},
    }
    payload = {
        "tracks": [
            {"id": "left", "bbox": [0, 0, 100, 300], "track_type": "table_text", "header": "左表"},
            {"id": "image", "bbox": [100, 0, 200, 300], "track_type": "legend", "header": "图像"},
            {"id": "right", "bbox": [200, 0, 300, 300], "track_type": "table_text", "header": "右表"},
        ],
        "primitives": {
            "stratigraphic_intervals": [
                {
                    "id": "left_one",
                    "name": "左侧单元",
                    "track_id": "left",
                    "geometry_refs": ["left_one_cell"],
                    "geometry_bbox": [0, 50, 100, 300],
                    "top_y": 50,
                    "bottom_y": 300,
                }
            ],
            "lithology_intervals": [
                {
                    "id": "right_upper",
                    "name": "右侧上部",
                    "track_id": "right",
                    "geometry_refs": ["right_upper_cell"],
                    "geometry_bbox": [200, 50, 300, 160],
                    "top_y": 50,
                    "bottom_y": 160,
                },
                {
                    "id": "right_lower",
                    "name": "右侧下部",
                    "track_id": "right",
                    "geometry_refs": ["right_lower_cell"],
                    "geometry_bbox": [200, 160, 300, 300],
                    "top_y": 160,
                    "bottom_y": 300,
                },
            ],
        },
    }

    slices = build_adjacent_visual_track_slices(payload, geometry)

    assert [item["bbox"] for item in slices] == [
        [100, 50, 200, 160],
        [100, 160, 200, 300],
    ]
    assert {item["source_track_id"] for item in slices} == {"right"}
    assert {item["source_track_side"] for item in slices} == {"right"}
    assert {
        item["source_entity_id"] for item in slices
    } == {"right_upper", "right_lower"}
    assert all(
        item["adjacent_track_entity_counts"]
        == {
            "left": {"track_id": "left", "entity_count": 1, "track_distance": 1, "track_type": "table_text"},
            "right": {"track_id": "right", "entity_count": 2, "track_distance": 1, "track_type": "table_text"},
        }
        for item in slices
    )


def test_visual_tracks_skip_visual_neighbors_and_only_slice_legend() -> None:
    """中文说明：legend 必须跳过中间视觉列寻找文字基准，curve 不再输出 VLM 切片。"""

    geometry = {
        "content_bbox": [0, 0, 400, 300],
        "cells": [],
        "rule_lines": {"horizontal_lines": [0, 50, 160, 300]},
    }
    payload = {
        "tracks": [
            {"id": "left_text", "bbox": [0, 0, 100, 300], "track_type": "table_text", "header": "地层"},
            {"id": "profile", "bbox": [100, 0, 200, 300], "track_type": "legend", "role": "lithology", "header": "岩性剖面"},
            {"id": "curve", "bbox": [200, 0, 300, 300], "track_type": "curve", "header": "GR 0—150"},
            {"id": "right_text", "bbox": [300, 0, 400, 300], "track_type": "table_text", "header": "岩性特征"},
        ],
        "primitives": {
            "stratigraphic_intervals": [
                {
                    "id": "left_all",
                    "name": "马五段",
                    "track_id": "left_text",
                    "geometry_refs": ["left_cell"],
                    "geometry_bbox": [0, 50, 100, 300],
                }
            ],
            "lithology_intervals": [
                {
                    "id": "right_upper",
                    "name": "白云岩",
                    "track_id": "right_text",
                    "geometry_refs": ["right_upper_cell"],
                    "geometry_bbox": [300, 50, 400, 160],
                },
                {
                    "id": "right_lower",
                    "name": "石灰岩",
                    "track_id": "right_text",
                    "geometry_refs": ["right_lower_cell"],
                    "geometry_bbox": [300, 160, 400, 300],
                },
            ],
        },
    }

    slices = build_adjacent_visual_track_slices(payload, geometry)

    assert [item["track_type"] for item in slices] == ["legend", "legend"]
    assert {item["source_track_id"] for item in slices} == {"right_text"}
    assert {item["source_track_type"] for item in slices} == {"table_text"}
    assert {item["source_track_distance"] for item in slices if item["track_type"] == "legend"} == {2}
    assert [item["recognition_stage"] for item in slices] == [2, 2]


def test_lithology_profile_prompt_and_legend_type_are_constrained(tmp_path: Path) -> None:
    """岩性剖面切片必须带用户给定纹理参考，并把候选内类型写回 legend 节点。"""

    image_path = tmp_path / "profile.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    slice_record = {
        "slice_id": "track_slice_profile_row_1",
        "track_id": "profile",
        "track_order": 1,
        "track_type": "legend",
        "recognition_stage": 2,
        "track_role": "lithology",
        "is_lithology_profile": True,
        "track_header": "岩性剖面",
        "source_track_id": "description",
        "source_track_order": 2,
        "source_track_side": "right",
        "source_track_distance": 1,
        "source_track_entity_count": 1,
        "adjacent_track_entity_counts": {
            "right": {"track_id": "description", "entity_count": 1, "track_distance": 1, "track_type": "table_text"}
        },
        "candidate_table_text_track_entity_counts": {
            "right": {"track_id": "description", "entity_count": 1, "track_distance": 1, "track_type": "table_text"}
        },
        "source_track_type": "table_text",
        "source_track_header": "岩性特征",
        "source_entity_id": "row_1",
        "source_cell_id": "row_1_cell",
        "source_geometry_refs": ["row_1_cell"],
        "source_label": "白云岩",
        "bbox": [0, 0, 100, 100],
        "top_y": 0.0,
        "bottom_y": 100.0,
        "pixel_range": {"left_x": 0, "top_y": 0.0, "right_x": 100, "bottom_y": 100.0, "coordinate_space": "original_pixels"},
        "coordinate_source": "PP-StructureV3.table_text_entity_y_projection",
        "cropped_image_cache_path": str(image_path.resolve()),
    }
    prompt = build_visual_track_slice_prompt(slice_record)
    assert "泥岩 Mudstone" in prompt
    assert "白云岩 Dolomite" in prompt
    assert "legend_interpretations" in prompt
    payload = {
        "ppstructure_geometry": {"source_image_path": str(image_path)},
        "primitives": {"curve_tracks": [], "curve_observations": [], "track_intervals": []},
        "visual_track_extraction": {},
        "uncertainties": [],
    }
    response = {
        "schema_version": "table_embedded_hybrid.track_slice_description.v2",
        "slice_id": slice_record["slice_id"],
        "track_id": "profile",
        "track_type": "legend",
        "source_cell_id": "row_1_cell",
        "description": "同向斜线组成的白云岩剖面纹理",
        "curve_change": "not_applicable",
        "curve_readings": [],
        "legend_interpretations": [
            {
                "legend_type": "白云岩 Dolomite",
                "category": "lithology",
                "visual_basis": "大量同方向斜线",
                "confidence": 0.92,
                "uncertainty": "",
            }
        ],
        "visual_features": ["同向斜线"],
        "confidence": 0.92,
        "uncertainty": "",
    }

    enriched = apply_visual_track_slice_responses(payload, [response], slices=[slice_record])
    interval = enriched["primitives"]["track_intervals"][0]
    assert interval["legend_type"] == "白云岩"
    assert interval["lithology_types"] == ["白云岩"]
    assert interval["legend_interpretations"][0]["visual_basis"] == "大量同方向斜线"


def test_curve_vlm_input_combines_track_header_and_body_slice(tmp_path: Path) -> None:
    """曲线请求图必须把同轨表头置于正文切片上方，使模型可读取刻度并估算数值。"""

    image_path = tmp_path / "curve-header-and-body.png"
    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 0, 109, 39), fill=(220, 40, 40))
    draw.rectangle((10, 60, 109, 99), fill=(40, 80, 220))
    image.save(image_path)
    record = {
        "slice_id": "curve_header_slice",
        "track_type": "curve",
        "header_bbox": [10, 0, 110, 40],
        "bbox": [10, 60, 110, 100],
    }

    cached = cache_visual_track_slices(image_path, [record])[0]

    assert cached["vlm_input_layout"] == "track_header_above_slice"
    assert Path(cached["track_header_image_cache_path"]).is_file()
    assert Path(cached["vlm_input_image_cache_path"]).is_file()
    with Image.open(cached["vlm_input_image_cache_path"]) as combined:
        assert combined.size == (100, 84)
        assert combined.getpixel((50, 20)) == (220, 40, 40)
        assert combined.getpixel((50, 70)) == (40, 80, 220)


def test_local_colored_curve_vectorization_is_disabled(tmp_path: Path) -> None:
    """视觉轨道描述改由 VLM 负责后，本地颜色追踪不得再生成曲线样点或响应节点。"""

    image_path = tmp_path / "curve.png"
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((160, 0, 320, 299), outline="black", width=1)
    draw.line((160, 60, 320, 60), fill="black", width=1)
    points = [(205 + round((y - 60) * 0.2), y) for y in range(62, 298)]
    draw.line(points, fill=(230, 20, 20), width=2)
    image.save(image_path)
    geometry = _geometry()
    geometry["source_image_path"] = str(image_path)
    geometry["image_size"] = {"width": 400, "height": 300}
    geometry["content_bbox"] = [0, 0, 320, 300]
    geometry["tracks"].append(
        {"id": "pp_track_002", "order": 2, "bbox": [160, 0, 320, 300], "header_text": "GR 0 100"}
    )
    geometry["cells"].extend(
        [
            {"id": "curve_header", "bbox": [160, 0, 320, 60], "ocr_ids": ["curve_header_ocr"], "text": "GR 0 100", "track_ids": ["pp_track_002"]},
            {"id": "curve_body", "bbox": [160, 60, 320, 300], "ocr_ids": [], "text": "", "track_ids": ["pp_track_002"]},
        ]
    )
    geometry["ocr_lines"].append(
        {"id": "curve_header_ocr", "text": "GR", "confidence": 0.99, "bbox": [215, 15, 255, 40], "cell_id": "curve_header", "track_id": "pp_track_002"}
    )
    geometry["rule_lines"]["vertical_lines"].extend([160, 320])
    geometry["rule_lines"]["horizontal_lines"].append(60)
    payload = _semantic_payload()
    payload["tracks"].append(
        {"id": "pp_track_002", "track_type": "curve", "role": "curve", "header": "GR", "parser": "curve", "evidence": "GR 表头"}
    )
    payload["primitives"]["curve_tracks"] = [
        {
            "id": "curve_gr",
            "name": "GR",
            "track_id": "pp_track_002",
            "left_value": 0,
            "right_value": 100,
            "unit": "API",
            "color": "red",
            "visual_form": "continuous_curve",
            "scale_transform": "linear",
            "evidence": "GR 0—100",
        }
    ]
    enriched = apply_ppstructure_geometry(payload, geometry)
    assert enriched["primitives"]["curve_traces"] == []
    assert (
        enriched["visual_track_extraction"]["recognition_mode"]
        == "vlm_legend_slice_interpretation_curve_slice_disabled"
    )
    assert enriched["visual_track_extraction"]["vlm_slice_track_types"] == ["legend"]
    assert enriched["visual_track_extraction"]["curve_slice_vlm_enabled"] is False
    assert enriched["visual_track_extraction"]["curve"]["available"] is False
    task = ImageExtractionTask(
        document_id="doc",
        chunk_id="chunk",
        image_id="curve-image",
        image_index=0,
        image_path=str(image_path),
    )
    intermediate = TableEmbeddedHybridPipeline().run(task, enriched)
    local_trace_observations = [
        item
        for item in intermediate["parsed"]["curve_observations"]
        if item.get("curve_trace_id") == "trace_curve_gr"
    ]
    assert local_trace_observations == []


def test_vlm_curve_track_is_not_rewritten_by_local_color_rules(tmp_path: Path) -> None:
    """VLM 已输出的曲线节点应原样保留，本地颜色规则不得改名、改色或补造轨迹。"""

    image_path = tmp_path / "porosity.png"
    image = Image.new("RGB", (260, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(180, y) for y in range(62, 298)], fill=(230, 20, 20), width=2)
    image.save(image_path)
    geometry = _geometry()
    geometry["source_image_path"] = str(image_path)
    geometry["content_bbox"] = [0, 0, 260, 300]
    geometry["tracks"].append(
        {"id": "pp_track_002", "order": 2, "bbox": [160, 0, 260, 300], "header_text": "测井孔隙度% 0 10"}
    )
    geometry["cells"].append(
        {
            "id": "porosity_header",
            "bbox": [160, 0, 260, 60],
            "ocr_ids": [],
            "text": "测井孔隙度% 0 10",
            "track_ids": ["pp_track_002"],
        }
    )
    geometry["rule_lines"]["vertical_lines"].append(260)
    geometry["rule_lines"]["horizontal_lines"].append(60)
    payload = _semantic_payload()
    payload["tracks"].append(
        {"id": "pp_track_002", "track_type": "curve", "role": "porosity", "header": "测井孔隙度", "parser": "curve"}
    )
    payload["primitives"]["curve_tracks"] = [
        {
            "id": "log_phi",
            "name": "LOG_POROSITY",
            "track_id": "pp_track_002",
            "left_value": 0,
            "right_value": 10,
            "color": "unknown",
            "visual_form": "continuous_curve",
            "scale_transform": "linear",
        }
    ]
    enriched = apply_ppstructure_geometry(payload, geometry)
    assert [item["id"] for item in enriched["primitives"]["curve_tracks"]] == ["log_phi"]
    assert enriched["primitives"]["curve_tracks"][0]["color"] == "unknown"
    assert enriched["primitives"]["curve_traces"] == []


def test_empty_visual_track_fails_entity_coverage_gate(tmp_path: Path) -> None:
    """没有文字但包含图像内容的岩性剖面轨道若未生成实体，质量门必须明确报错。"""

    image_path = _image(tmp_path)
    geometry = _geometry()
    geometry["source_image_path"] = str(image_path)
    geometry["content_bbox"] = [0, 0, 260, 300]
    geometry["tracks"].append(
        {"id": "pp_track_002", "order": 2, "bbox": [160, 0, 260, 300], "header_text": "岩性剖面"}
    )
    geometry["cells"].extend(
        [
            {
                "id": "profile_header",
                "bbox": [160, 0, 260, 60],
                "ocr_ids": [],
                "text": "岩性剖面",
                "track_ids": ["pp_track_002"],
            },
            {
                "id": "profile_body",
                "bbox": [160, 60, 260, 300],
                "ocr_ids": [],
                "text": "",
                "track_ids": ["pp_track_002"],
            },
        ]
    )
    geometry["rule_lines"]["vertical_lines"].append(260)
    payload = _semantic_payload()
    payload["tracks"].append(
        {"id": "pp_track_002", "track_type": "legend", "role": "lithology", "header": "岩性剖面", "parser": "pattern_legend"}
    )
    enriched = apply_ppstructure_geometry(payload, geometry)
    task = ImageExtractionTask(
        document_id="doc",
        chunk_id="chunk",
        image_id="empty-profile",
        image_index=0,
        image_path=str(image_path),
    )
    intermediate = TableEmbeddedHybridPipeline().run(task, enriched)
    coverage = intermediate["quality"]["track_entity_coverage"]
    profile = next(item for item in coverage["tracks"] if item["track_id"] == "pp_track_002")
    assert profile["covered"] is False
    assert intermediate["quality"]["quality_gate_ok"] is False
    assert any("pp_track_002" in error for error in intermediate["quality"]["quality_gate_errors"])


def test_local_legend_pattern_matching_is_disabled(tmp_path: Path) -> None:
    """图例纹理即使相似也不得由本地算法命名岩性，后续只接受 VLM 切片描述。"""

    image_path = tmp_path / "legend.png"
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    for box in ((100, 40, 200, 120), (100, 120, 200, 200), (10, 220, 70, 270)):
        draw.rectangle(box, outline="black", width=2)
        for offset in range(-60, 140, 12):
            draw.line((box[0] + offset, box[3], box[0] + offset + box[3] - box[1], box[1]), fill="black", width=1)
    image.save(image_path)
    geometry = {
        "schema_version": "ppstructurev3.table_geometry.v1",
        "engine": "test_fixture",
        "coordinate_space": "original_pixels",
        "source_image_path": str(image_path),
        "image_size": {"width": 400, "height": 300},
        "content_bbox": [0, 0, 200, 200],
        "tracks": [
            {"id": "depth", "order": 0, "bbox": [0, 0, 100, 200], "header_text": "深度"},
            {"id": "pattern", "order": 1, "bbox": [100, 0, 200, 200], "header_text": "岩性剖面"},
        ],
        "cells": [
            {"id": "pattern_header", "bbox": [100, 0, 200, 40], "ocr_ids": [], "text": "岩性剖面", "track_ids": ["pattern"]},
            {"id": "pattern_a", "bbox": [100, 40, 200, 120], "ocr_ids": [], "text": "", "track_ids": ["pattern"]},
            {"id": "pattern_b", "bbox": [100, 120, 200, 200], "ocr_ids": [], "text": "", "track_ids": ["pattern"]},
        ],
        "ocr_lines": [
            {"id": "depth_0", "text": "0", "confidence": 0.99, "bbox": [20, 45, 50, 65], "track_id": "depth"},
            {"id": "depth_100", "text": "100", "confidence": 0.99, "bbox": [20, 165, 60, 185], "track_id": "depth"},
            {"id": "legend_label", "text": "泥质云岩", "confidence": 0.99, "bbox": [80, 225, 170, 265], "track_id": ""},
        ],
        "rule_lines": {"available": True, "vertical_lines": [0, 100, 200], "horizontal_lines": [0, 40, 120, 200]},
        "quality": {"ocr_line_count": 3, "cell_count": 3, "track_count": 2},
    }
    payload = {
        "schema_version": "table_embedded_hybrid.v1",
        "diagram_id": "legend_test",
        "diagram_name": "岩性图例测试",
        "layout_family": "stratigraphic_column_table",
        "coordinate_system": {
            "vertical_axis": {
                "kind": "depth",
                "unit": "m",
                "increases": "downward",
                "track_id": "depth",
                "calibration_ocr_ids": ["depth_0", "depth_100"],
            }
        },
        "tracks": [
            {"id": "depth", "track_type": "table_text", "role": "depth", "header": "深度", "parser": "axis"},
            {"id": "pattern", "track_type": "legend", "role": "lithology", "header": "岩性剖面", "parser": "pattern_legend"},
        ],
        "primitives": {
            **{field: [] for field in (
                "stratigraphic_intervals", "reference_intervals", "lithology_intervals",
                "facies_intervals", "reservoir_intervals", "oil_layer_intervals",
                "geological_feature_intervals", "curve_observations", "track_intervals",
            )},
            "curve_tracks": [],
            "point_markers": [],
            "objects": [],
            "legend_entries": [
                {"id": "legend_muddy_dolomite", "name": "泥质云岩", "geometry_refs": ["legend_label"]}
            ],
            "explicit_relations": [],
        },
        "uncertainties": [],
    }
    enriched = apply_ppstructure_geometry(payload, geometry)
    matched = [
        item
        for item in enriched["primitives"]["lithology_intervals"]
        if item.get("legend_entry_id") == "legend_muddy_dolomite"
    ]
    assert matched == []
    assert enriched["primitives"]["legend_entries"] == []
    assert enriched["visual_track_extraction"]["legend"]["available"] is False
