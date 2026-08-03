"""三维地层建模图专用抽取、层序建图和批处理测试。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.extractors.image_extractor.schema_models import ImageExtractionContext, ImageExtractionTask
from src.extractors.image_extractor.stratigraphic_profile import (
    StratigraphicProfileExtractor,
    StratigraphicProfileSubtypeClassifier,
    build_three_dimensional_stratigraphic_model_prompt,
)
from src.extractors.image_extractor.stratigraphic_profile.subclassifier import (
    StratigraphicSubtypeClassification,
    StratigraphicProfileSubtype,
)
from src.extractors.image_extractor.stratigraphic_profile.three_dimensional_stratigraphic_model.batch import (
    DEFAULT_SOURCE,
    _load_target_chunks,
    run_batch,
)
from src.extractors.image_extractor.stratigraphic_profile.three_dimensional_stratigraphic_model.graph import (
    build_three_dimensional_stratigraphic_model_graph,
)
from src.extractors.image_extractor.stratigraphic_profile.three_dimensional_stratigraphic_model.pipeline import (
    ThreeDimensionalStratigraphicModelPipeline,
)
from src.extractors.image_extractor.stratigraphic_profile.three_dimensional_stratigraphic_model.multipass import (
    _normalize_inventory,
    _normalize_unit_color_batch,
    apply_multipass_quality_gates,
)
from src.extractors.image_extractor.stratigraphic_profile.three_dimensional_stratigraphic_model.review import (
    apply_manual_review_corrections,
)


def _payload() -> dict[str, Any]:
    """中文说明：构造同时覆盖上下层序、构造关系和正文上下文三元组的专用离线响应。"""

    return {
        "schema_version": "three_dimensional_stratigraphic_model.v1",
        "model": {
            "id": "model_1",
            "name": "测试三维地层块体",
            "view_direction": "透视",
            "visible_surfaces": ["top", "front", "side"],
            "topology_evidence": "可见顶面、正面和侧面",
            "evidence": "整张三维块体",
            "image_region": "整图",
            "confidence": 0.98,
        },
        "lithologies": [
            {
                "id": "lith_dolomite",
                "name": "白云岩",
                "source_kind": "visual",
                "evidence": "图例标注白云岩",
                "image_region": "右下图例",
                "confidence": 0.95,
            }
        ],
        "stratigraphic_units": [
            {
                "id": "unit_group",
                "name": "马家沟组",
                "sequence_role": "container",
                "column_id": "main_section",
                "order_bottom_to_top": 0,
                "source_kind": "visual_context",
                "context_evidence": "正文讨论马家沟组层序",
                "evidence": "侧壁地层层状体",
                "image_region": "侧壁",
                "confidence": 0.9,
            },
            {
                "id": "unit_lower",
                "name": "马五5",
                "parent_unit_id": "unit_group",
                "column_id": "main_section",
                "order_bottom_to_top": 0,
                "lithology_ids": ["lith_dolomite"],
                "source_kind": "visual",
                "evidence": "侧壁下部标注马五5",
                "image_region": "左下侧壁",
                "confidence": 0.94,
            },
            {
                "id": "unit_upper",
                "name": "马五1-4",
                "parent_unit_id": "unit_group",
                "column_id": "main_section",
                "order_bottom_to_top": 1,
                "source_kind": "visual",
                "evidence": "侧壁上部标注马五1-4",
                "image_region": "左侧壁上部",
                "confidence": 0.93,
            },
        ],
        "wells": [],
        "faults": [
            {
                "id": "fault_1",
                "name": "走滑断裂",
                "source_kind": "visual",
                "evidence": "红色断裂带切穿块体",
                "image_region": "块体中部",
                "confidence": 0.96,
            }
        ],
        "zones": [
            {
                "id": "reservoir_1",
                "name": "奥陶系储层",
                "zone_type": "reservoir",
                "source_kind": "visual_context",
                "context_evidence": "正文明确称奥陶系储层受走滑断裂控制",
                "evidence": "断裂带周缘层状储集体",
                "image_region": "断裂带两侧",
                "confidence": 0.88,
            }
        ],
        "fluids": [],
        "objects": [
            {
                "id": "oil_gas",
                "name": "油气",
                "entity_type": "hydrocarbon",
                "source_kind": "visual_context",
                "context_evidence": "正文提及油气沿走滑断裂运移",
                "evidence": "断裂带中的红色运移通道",
                "image_region": "断裂带",
                "confidence": 0.9,
            }
        ],
        "relations": [
            {
                "source": "fault_1",
                "type": "cuts_through",
                "target": "unit_upper",
                "explicit": True,
                "basis": "visible_intersection",
                "evidence": "红色断裂带穿过上部地层",
                "confidence": 0.95,
            },
            {
                "source": "unit_upper",
                "type": "directly_overlies",
                "target": "unit_lower",
                "explicit": True,
                "basis": "model_duplicate",
                "evidence": "模型重复提交的上下边",
                "confidence": 0.95,
            },
        ],
        "context_relations": [
            {
                "source": "fault_1",
                "type": "controls",
                "target": "reservoir_1",
                "explicit": True,
                "basis": "reference_context",
                "context_evidence": "奥陶系储层主要受走滑断裂带控制",
                "visual_anchor_evidence": "图中红色断裂带邻接奥陶系层状体",
                "confidence": 0.91,
            },
            {
                "source": "unit_lower",
                "type": "acts_as_source_rock",
                "target": "oil_gas",
                "context_evidence": "烃源岩生成油气",
                "visual_anchor_evidence": "下部地层和断裂运移通道",
                "confidence": 0.9,
            },
            {
                "source": "missing",
                "type": "controls",
                "target": "reservoir_1",
                "context_evidence": "无效端点测试",
                "confidence": 0.9,
            },
            {
                "source": "fault_1",
                "type": "transports",
                "target": "unit_lower",
                "context_evidence": "错误地把地层当作被输导对象",
                "visual_anchor_evidence": "断裂带贯穿地层",
                "confidence": 0.9,
            },
        ],
        "uncertainties": [],
    }


class ThreeDimensionalOfflineVLM:
    """为全部三维目标返回专用 JSON 的离线视觉模型桩。"""

    def __init__(self) -> None:
        """中文说明：初始化调用记录，便于断言每张目标只调用一次内容抽取。"""

        self.prompts: list[str] = []

    def describe_image(self, image_path: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """中文说明：记录真实任务 Prompt，并返回可由确定性流水线装配的三维响应。"""

        _ = image_path, kwargs
        self.prompts.append(prompt)
        return _payload()


class ThreeDimensionalMultiPassOfflineVLM:
    """模拟整图、层界、图例、逐层识别和全图审查的多轮视觉模型。"""

    supports_three_dimensional_lithology_multipass = True

    def __init__(self, *, review_segment: str = "") -> None:
        """中文说明：保存调用记录，并可指定一个低置信层段来验证自动盲审分支。"""

        self.calls: list[tuple[str, str]] = []
        self.review_segment = review_segment

    def describe_image(
        self,
        image_path: str,
        prompt: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """中文说明：按任务阶段返回严格协议，并让三个冻结层段都高置信匹配白云岩图例。"""

        _ = prompt
        task_name = str(kwargs.get("task_name") or "")
        self.calls.append((task_name, image_path))
        if task_name.startswith("三维地层建模图抽取:"):
            return _payload()
        if task_name.startswith("三维地层层界冻结:") or task_name.startswith("三维地层层界独立审查:"):
            stage = (
                "layer_inventory_review"
                if task_name.startswith("三维地层层界独立审查:")
                else "layer_inventory"
            )
            return {
                "schema_version": "three_dimensional_lithology.segment.v1",
                "stage": stage,
                "units": [
                    {
                        "unit_id": unit_id,
                        "visible_regions": [
                            {
                                "surface": "side",
                                "bbox": [20, 20 + index * 90, 220, 90 + index * 90],
                                "boundary_evidence": "可见颜色边界",
                                "confidence": 0.96,
                            }
                        ],
                    }
                    for index, unit_id in enumerate(("unit_group", "unit_lower", "unit_upper"))
                ],
                "visual_layer_segments": [
                    {
                        "segment_id": f"segment_{index + 1}",
                        "parent_unit_id": unit_id,
                        "order_within_parent": 0,
                        "visible_regions": [
                            {
                                "surface": "side",
                                "bbox": [20, 20 + index * 90, 220, 90 + index * 90],
                                "boundary_evidence": "可见颜色边界",
                                "confidence": 0.96,
                            }
                        ],
                        "boundary_evidence": "上下颜色边界清楚",
                        "confidence": 0.96,
                    }
                    for index, unit_id in enumerate(("unit_group", "unit_lower", "unit_upper"))
                ],
                "uncertainties": [],
            }
        if task_name.startswith("三维地层图例目录:"):
            return {
                "schema_version": "three_dimensional_lithology.segment.v1",
                "stage": "legend_catalog",
                "legend_bbox": [240, 20, 430, 180],
                "legend_items": [
                    {
                        "lithology_id": "lith_dolomite",
                        "raw_text": "白云岩",
                        "normalized_name": "白云岩",
                        "bbox": [250, 40, 420, 80],
                        "color_features": ["浅灰色"],
                        "pattern_features": ["砖纹"],
                        "text_explicit": True,
                        "evidence": "图例明确标注白云岩",
                        "confidence": 0.98,
                    },
                    {
                        "lithology_id": "lith_limestone",
                        "raw_text": "灰岩",
                        "normalized_name": "灰岩",
                        "bbox": [250, 90, 420, 130],
                        "color_features": ["深灰色"],
                        "pattern_features": ["块状纹理"],
                        "text_explicit": True,
                        "evidence": "图例明确标注灰岩",
                        "confidence": 0.97,
                    },
                ],
                "uncertainties": [],
            }
        if task_name.startswith("三维地层同色岩性批次:"):
            # 中文说明：同色批次从第二个地层单元补出逐层调用漏掉的灰岩，用于验证全量多岩性合并。
            return {
                "schema_version": "three_dimensional_lithology.segment.v1",
                "stage": "unit_color_lithology",
                "batch_index": 1,
                "unit_results": [
                    {
                        "unit_id": "unit_group",
                        "observed_color_signature": ["#E0E0E0"],
                        "same_color_region_evidence": "浅灰色范围已覆盖目标地层可见侧面",
                        "checked_surfaces": ["side"],
                        "composition": [],
                        "coverage_complete": True,
                        "decision_confidence": 0.94,
                        "unresolved_reason": "",
                        "uncertainties": [],
                    },
                    {
                        "unit_id": "unit_lower",
                        "observed_color_signature": ["#B0B0B0"],
                        "same_color_region_evidence": "同一深灰底色范围出现独立块状纹理",
                        "checked_surfaces": ["side"],
                        "composition": [
                            {
                                "lithology_id": "lith_limestone",
                                "role": "secondary",
                                "matched_patterns": ["深灰色块状纹理"],
                                "evidence": "同色地层范围内的块状纹理与灰岩图例一致",
                                "confidence": 0.93,
                            }
                        ],
                        "coverage_complete": True,
                        "decision_confidence": 0.94,
                        "unresolved_reason": "",
                        "uncertainties": [],
                    },
                    {
                        "unit_id": "unit_upper",
                        "observed_color_signature": ["#D0D0D0"],
                        "same_color_region_evidence": "浅灰色范围已覆盖目标地层可见侧面",
                        "checked_surfaces": ["side"],
                        "composition": [],
                        "coverage_complete": True,
                        "decision_confidence": 0.94,
                        "unresolved_reason": "",
                        "uncertainties": [],
                    },
                ],
                "uncertainties": [],
            }
        if task_name.startswith("三维逐层岩性:"):
            segment_id = task_name.rsplit(":", 1)[-1]
            needs_review = segment_id == self.review_segment
            composition = [
                {
                    "lithology_id": "lith_dolomite",
                    "role": "primary",
                    "visible_fraction": 1.0,
                    "evidence_source": "legend_pattern",
                    "evidence": "目标层段浅灰砖纹与白云岩图例一致",
                    "confidence": 0.7 if needs_review else 0.96,
                }
            ]
            return {
                "schema_version": "three_dimensional_lithology.segment.v1",
                "stage": "layer_lithology",
                "segment_id": segment_id,
                "surface_observations": [
                    {
                        "surface": "side",
                        "observed_color": "浅灰色",
                        "observed_pattern": "砖纹",
                        "occlusion": "无",
                        "candidates": [
                            {
                                "lithology_id": "lith_dolomite",
                                "matched_features": ["颜色", "砖纹"],
                                "conflicting_features": [],
                                "confidence": 0.7 if needs_review else 0.96,
                            }
                        ],
                    }
                ],
                "composition": composition,
                "unknown_patterns": [],
                "decision_confidence": 0.7 if needs_review else 0.96,
                "needs_independent_review": needs_review,
                "uncertainties": [],
            }
        if task_name.startswith("三维逐层岩性盲审:"):
            segment_id = task_name.rsplit(":", 1)[-1]
            review_composition = [
                {
                    "lithology_id": "lith_dolomite",
                    "role": "primary",
                    "visible_fraction": 1.0,
                    "evidence_source": "legend_pattern",
                    "evidence": "独立盲审再次确认浅灰砖纹与白云岩图例一致",
                    "confidence": 0.95,
                }
            ]
            return {
                "schema_version": "three_dimensional_lithology.segment.v1",
                "stage": "independent_review",
                "segment_id": segment_id,
                "surface_observations": [],
                "composition": review_composition,
                "unknown_patterns": [],
                "decision_confidence": 0.95,
                "needs_independent_review": False,
                "uncertainties": [],
            }
        if task_name.startswith("三维逐层岩性全图审查:"):
            return {
                "schema_version": "three_dimensional_lithology.segment.v1",
                "stage": "global_audit",
                "corrections": [],
                "unit_corrections": [],
                "conflicts": [],
                "unresolved_segments": [],
                "unresolved_units": [],
            }
        raise AssertionError(f"未覆盖的多轮任务：{task_name}")


def _task() -> ImageExtractionTask:
    """中文说明：使用清单中的第一张真实三维图片构造带正文参考的规范任务。"""

    chunk = _load_target_chunks(DEFAULT_SOURCE)[0]
    chunk_id = str(chunk["id"])
    return ImageExtractionTask(
        document_id=chunk_id.split(":section:", 1)[0],
        chunk_id=chunk_id,
        image_id=f"{chunk_id}:image:0",
        image_index=0,
        image_path=str(chunk["image_path"][0]),
        caption=str(chunk.get("caption") or ""),
        references=tuple(chunk.get("references") or []),
        classification_code="A04",
        classification_type="三维地层建模图",
        raw_chunk=chunk,
    )


def test_target_manifest_contains_two_existing_three_dimensional_images() -> None:
    """用户指定 JSON 必须恰好筛出两张真实存在的三维地层建模图。"""

    targets = _load_target_chunks(DEFAULT_SOURCE)
    assert len(targets) == 2
    assert all(Path(chunk["image_path"][0]).is_file() for chunk in targets)
    assert {
        Path(chunk["image_path"][0]).name for chunk in targets
    } == {
        "f0ac23ffa801dccdccad61dda0ee06d1417576276e440abdd8db77b5a4919ff1.jpg",
        "c05e5f21288b34d21112005f87dc0a9547e2df64064ad7002b0b41abc27d6d0b.jpg",
    }


def test_prompt_prioritizes_vertical_order_and_separates_context_evidence() -> None:
    """三维 Prompt 必须携带正文，并明确同柱同父层级和 context explicit=false 约束。"""

    task = _task()
    prompt = build_three_dimensional_stratigraphic_model_prompt(
        task,
        StratigraphicSubtypeClassification(
            subtype=StratigraphicProfileSubtype.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL,
            confidence=0.99,
            evidence="测试三维块体",
            source="manual_visual_mock",
        ),
    )
    assert task.references[0] in prompt
    assert "同一 column_id 且具有同一 parent_unit_id" in prompt
    assert "context_relations" in prompt
    assert "explicit 必须为 false" in prompt
    assert "order_bottom_to_top" in prompt


def test_multipass_normalizes_zero_to_one_thousand_bboxes_for_short_images() -> None:
    """多轮层界响应使用零到一千坐标时，必须按真实短图高度缩放而不是直接截断。"""

    raw = {
        "schema_version": "three_dimensional_lithology.segment.v1",
        "stage": "layer_inventory",
        "units": [
            {
                "unit_id": "unit_1",
                "visible_regions": [
                    {"surface": "front", "bbox": [0, 530, 1000, 680], "confidence": 0.9}
                ],
            }
        ],
        "visual_layer_segments": [
            {
                "segment_id": "segment_1",
                "parent_unit_id": "unit_1",
                "segment_role": "vertical_layer",
                "visible_regions": [
                    {"surface": "front", "bbox": [0, 530, 1000, 680], "confidence": 0.9}
                ],
                "confidence": 0.9,
            }
        ],
        "uncertainties": [],
    }
    normalized = _normalize_inventory(
        raw,
        [{"id": "unit_1", "name": "马五5", "order_bottom_to_top": 0}],
        width=1000,
        height=442,
    )
    assert normalized["input_coordinate_space"] == "normalized_0_1000"
    assert normalized["visual_layer_segments"][0]["visible_regions"][0]["bbox"] == [0, 234, 1000, 301]


def test_unit_color_batch_requires_pattern_evidence_and_keeps_all_supported_lithologies() -> None:
    """同色范围只能限定地层归属，必须有图例纹理且达到阈值才能补充一种或多种岩性。"""

    unit_batch = [
        {
            "unit_id": "unit_lower",
            "unit_name": "马五5",
            "visible_regions": [{"surface": "side", "bbox": [10, 20, 200, 100]}],
            "color_signature": {"rgb": [[176, 176, 176]], "hex": ["#B0B0B0"]},
        }
    ]
    raw = {
        "unit_results": [
            {
                "unit_id": "unit_lower",
                "same_color_region_evidence": "同色范围覆盖整个侧壁",
                "checked_surfaces": ["side"],
                "composition": [
                    {
                        "lithology_id": "lith_dolomite",
                        "role": "primary",
                        "matched_patterns": ["斜线砖纹"],
                        "evidence": "斜线砖纹与白云岩图例一致",
                        "confidence": 0.94,
                    },
                    {
                        "lithology_id": "lith_limestone",
                        "role": "secondary",
                        "matched_patterns": ["矩形砖纹"],
                        "evidence": "矩形砖纹与灰岩图例一致",
                        "confidence": 0.91,
                    },
                    {
                        "lithology_id": "lith_mud",
                        "role": "secondary",
                        "matched_patterns": [],
                        "evidence": "只有颜色相近",
                        "confidence": 0.99,
                    },
                ],
                "coverage_complete": True,
                "decision_confidence": 0.95,
                "unresolved_reason": "",
                "uncertainties": [],
            }
        ],
        "uncertainties": [],
    }
    normalized, uncertainties = _normalize_unit_color_batch(
        raw,
        unit_batch,
        {"lith_dolomite", "lith_limestone", "lith_mud"},
    )
    assert uncertainties == []
    assert [item["lithology_id"] for item in normalized[0]["composition"]] == [
        "lith_dolomite",
        "lith_limestone",
    ]
    assert normalized[0]["rejected_candidates"][0]["gate_reason"] == "missing_pattern_evidence"


def test_multipass_quality_gates_reject_unresolved_and_misapplied_explicit_lithology() -> None:
    """未解决候选和只属于其他层名的显式岩性必须留作审计，不得生成正式岩性关系。"""

    payload = {
        "stratigraphic_units": [
            {"id": "unit_source", "name": "Є1y烃源岩"},
            {"id": "unit_yj", "name": "O2yj"},
            {"id": "unit_mud", "name": "O3S泥质岩"},
        ],
        "visual_layer_segments": [
            {
                "id": "segment_source",
                "parent_unit_id": "unit_source",
                "lithology_ids": ["lith_mud"],
                "lithology_assignments": [{"lithology_id": "lith_mud", "confidence": 0.95}],
                "confidence": 0.95,
            },
            {
                "id": "segment_yj",
                "parent_unit_id": "unit_yj",
                "lithology_ids": ["lith_salt"],
                "lithology_assignments": [{"lithology_id": "lith_salt", "confidence": 0.8}],
                "confidence": 0.8,
            },
            {
                "id": "segment_mud",
                "parent_unit_id": "unit_mud",
                "lithology_ids": ["lith_mud"],
                "lithology_assignments": [{"lithology_id": "lith_mud", "confidence": 0.95}],
                "confidence": 0.95,
            },
        ],
        "uncertainties": [],
        "multipass_lithology": {
            "legend_catalog": {
                "legend_items": [
                    {
                        "lithology_id": "lith_mud",
                        "normalized_name": "泥质岩",
                        "catalog_source": "explicit_layer_label",
                    },
                    {
                        "lithology_id": "lith_salt",
                        "normalized_name": "膏盐岩",
                        "catalog_source": "legend",
                    },
                ]
            },
            "global_audit": {"unresolved_segments": [{"segment_id": "segment_yj"}]},
            "summary": {},
        },
    }
    corrected = apply_multipass_quality_gates(payload)
    by_id = {item["id"]: item for item in corrected["visual_layer_segments"]}
    assert by_id["segment_source"]["lithology_ids"] == []
    assert by_id["segment_yj"]["lithology_ids"] == []
    assert by_id["segment_mud"]["lithology_ids"] == ["lith_mud"]
    assert corrected["multipass_lithology"]["summary"]["unknown_segment_count"] == 2
    assert corrected["multipass_lithology"]["quality_gates"]["rejected_candidate_count"] == 2


def test_multipass_quality_gate_rejects_vertical_bbox_contained_by_top_surface() -> None:
    """所谓垂向层段完全落在同父顶面区时，必须视为表面角色冲突而不输出岩性边。"""

    assignment = {
        "lithology_id": "lith_limestone",
        "role": "primary",
        "evidence_source": "legend_pattern",
        "confidence": 0.9,
    }
    payload = {
        "stratigraphic_units": [{"id": "unit_2", "name": "马五1-4"}],
        "visual_layer_segments": [
            {
                "id": "vertical_1",
                "parent_unit_id": "unit_2",
                "segment_role": "vertical_layer",
                "visible_regions": [{"surface": "front", "bbox": [0, 155, 1000, 234]}],
                "lithology_ids": ["lith_limestone"],
                "lithology_assignments": [assignment],
                "confidence": 0.9,
            },
            {
                "id": "facies_1",
                "parent_unit_id": "unit_2",
                "segment_role": "lateral_facies_zone",
                "visible_regions": [{"surface": "top", "bbox": [0, 0, 1000, 234]}],
                "lithology_ids": ["lith_limestone"],
                "lithology_assignments": [assignment],
                "confidence": 0.9,
            },
        ],
        "multipass_lithology": {
            "legend_catalog": {
                "legend_items": [
                    {
                        "lithology_id": "lith_limestone",
                        "normalized_name": "灰岩",
                        "catalog_source": "legend",
                    }
                ]
            },
            "global_audit": {"unresolved_segments": []},
            "summary": {},
        },
    }
    corrected = apply_multipass_quality_gates(payload)
    by_id = {item["id"]: item for item in corrected["visual_layer_segments"]}
    assert by_id["vertical_1"]["lithology_ids"] == []
    assert by_id["vertical_1"]["review_status"] == "unresolved_surface_role_overlap"
    assert by_id["facies_1"]["lithology_ids"] == ["lith_limestone"]


def test_pipeline_and_graph_build_vertical_and_context_triples_with_provenance() -> None:
    """确定性装配必须生成双向相邻层序，强制上下文边为非显式并丢弃悬空端点。"""

    task = _task()
    intermediate = ThreeDimensionalStratigraphicModelPipeline().run(task, _payload())
    assert intermediate["quality"]["vertical_relation_count"] == 2
    assert intermediate["quality"]["context_relation_count"] == 2
    assert intermediate["quality"]["context_relation_repairs"][0]["repair"] == "acts_as_source_rock_to_generates"
    assert any(item["reason"] == "unresolved_endpoint" for item in intermediate["quality"]["dropped_relations"])
    assert any(
        item["reason"] == "vertical_relation_rebuilt_from_order"
        for item in intermediate["quality"]["dropped_relations"]
    )
    assert any(
        item["reason"] == "transports_target_must_be_fluid_ion_or_hydrocarbon"
        for item in intermediate["quality"]["dropped_relations"]
    )

    graph = build_three_dimensional_stratigraphic_model_graph(task, intermediate)
    triples = {(relation.source_name, relation.type, relation.target_name) for relation in graph.relations}
    assert ("马五1-4", "directly_overlies", "马五5") in triples
    assert ("马五5", "directly_underlies", "马五1-4") in triples
    assert ("走滑断裂", "controls", "奥陶系储层") in triples
    assert ("马五5", "generates", "油气") in triples
    vertical_relation = next(relation for relation in graph.relations if relation.type == "directly_overlies")
    assert vertical_relation.attributes["explicit"] is False
    assert vertical_relation.attributes["inference_basis"] == "order_bottom_to_top_same_parent_and_column"
    context_relation = next(relation for relation in graph.relations if relation.type == "controls")
    assert context_relation.attributes["explicit"] is False
    assert context_relation.attributes["evidence_scope"] == "context"
    assert context_relation.metadata["context_evidence"] == "奥陶系储层主要受走滑断裂带控制"
    assert graph.events == []
    assert graph.validate_references() == []
    for record in [*graph.entities, *graph.relations]:
        assert record.metadata["source_image_path"] == task.image_path
        assert record.metadata["source_image_id"] == task.image_id
        assert record.metadata["source_chunk_id"] == task.chunk_id
        assert record.metadata["visual_evidence"]


def test_targeted_ocr_review_merges_duplicate_blue_unit_and_corrects_yellow_unit() -> None:
    """专项 OCR 复核必须合并同一蓝色层的重复面识别，并把黄色薄层修正为 O2yj。"""

    payload = _payload()
    payload["stratigraphic_units"] = [
        {
            "id": "unit_5",
            "name": "O1-2v1",
            "order_bottom_to_top": 4,
            "evidence": "蓝色层正面",
            "confidence": 0.8,
        },
        {
            "id": "unit_6",
            "name": "O1-2y1",
            "order_bottom_to_top": 5,
            "evidence": "蓝色层侧面",
            "confidence": 0.8,
        },
        {
            "id": "unit_7",
            "name": "O3y1",
            "order_bottom_to_top": 6,
            "evidence": "黄色薄层",
            "confidence": 0.8,
        },
        {
            "id": "unit_8",
            "name": "O3",
            "order_bottom_to_top": 7,
            "evidence": "白色层",
            "confidence": 0.9,
        },
        {
            "id": "unit_9",
            "name": "O3S泥质岩",
            "order_bottom_to_top": 8,
            "evidence": "顶部紫色层",
            "confidence": 0.9,
        },
    ]
    payload["relations"] = [
        {
            "source": "unit_7",
            "type": "directly_overlies",
            "target": "unit_6",
            "evidence": "原始重复端点关系",
            "confidence": 0.8,
        }
    ]
    corrected, audit = apply_manual_review_corrections(_task(), payload)
    names = [item["name"] for item in corrected["stratigraphic_units"]]
    assert names == ["O1-2y", "O2yj", "O3", "O3S泥质岩"]
    assert [item["order_bottom_to_top"] for item in corrected["stratigraphic_units"]] == [4, 5, 6, 7]
    assert corrected["relations"][0]["target"] == "unit_5"
    assert audit["review_source"] == "targeted_live_vlm_ocr_review"
    assert len(audit["applied"]) == 4
    assert audit["skipped"] == []


def test_targeted_review_normalizes_current_single_pass_ocr_aliases() -> None:
    """专项复核必须兼容模型新出现的 O3p/O3v1 单节点别名，避免同图重跑后标签漂移。"""

    payload = _payload()
    payload["stratigraphic_units"] = [
        {"id": "unit_4", "name": "O3p", "order_bottom_to_top": 3},
        {"id": "unit_5", "name": "O3v1", "order_bottom_to_top": 4},
        {"id": "unit_6", "name": "O2yj", "order_bottom_to_top": 5},
        {"id": "unit_7", "name": "O3", "order_bottom_to_top": 6},
        {"id": "unit_8", "name": "O3S泥质岩", "order_bottom_to_top": 7},
    ]
    corrected, audit = apply_manual_review_corrections(_task(), payload)
    assert [item["name"] for item in corrected["stratigraphic_units"]] == [
        "O1p",
        "O1-2y",
        "O2yj",
        "O3",
        "O3S泥质岩",
    ]
    assert {item["new_name"] for item in audit["applied"]} >= {"O1p", "O1-2y", "O2yj"}


def test_dispatcher_completes_three_dimensional_subtype_with_one_content_call() -> None:
    """外层调度器必须把人工已分类目标直接交给三维子目录而不重复调用子分类 VLM。"""

    task = _task()
    vlm = ThreeDimensionalOfflineVLM()
    graph = StratigraphicProfileExtractor(
        subtype_classifier=StratigraphicProfileSubtypeClassifier(DEFAULT_SOURCE)
    ).extract(
        task,
        ImageExtractionContext(
            vlm_client=vlm,
            options={"allowed_stratigraphic_subtypes": ["three_dimensional_stratigraphic_model"]},
        ),
    )
    assert graph.metadata.extra["status"] == "completed"
    assert graph.metadata.extra["stratigraphic_subtype"] == "three_dimensional_stratigraphic_model"
    assert graph.metadata.extra["subtype_extraction_algorithm"] == "surface_topology_spatial_extraction"
    assert graph.metadata.extra["subtype_classification_vlm_called"] is False
    assert graph.metadata.extra["vlm_call_count"] == 1
    assert len(vlm.prompts) == 1


def test_dispatcher_promotes_multipass_lithologies_to_direct_multi_lithology_unit_triples() -> None:
    """多轮模式必须保留逐段识别证据，但最终图谱只将一种或多种岩性直接挂到地层单元。"""

    task = _task()
    vlm = ThreeDimensionalMultiPassOfflineVLM()
    graph = StratigraphicProfileExtractor(
        subtype_classifier=StratigraphicProfileSubtypeClassifier(DEFAULT_SOURCE)
    ).extract(
        task,
        ImageExtractionContext(
            vlm_client=vlm,
            options={"allowed_stratigraphic_subtypes": ["three_dimensional_stratigraphic_model"]},
        ),
    )
    assert graph.metadata.extra["status"] == "completed"
    assert graph.metadata.extra["vlm_call_count"] == 9
    segments = [entity for entity in graph.entities if entity.type == "visual_layer_segment"]
    assert segments == []
    lithology_relations = [
        relation for relation in graph.relations if relation.type == "has_lithology"
    ]
    assert len(lithology_relations) == 4
    assert all(relation.source_type == "stratigraphic_unit" for relation in lithology_relations)
    triples = {
        (relation.source_name, relation.type, relation.target_name)
        for relation in lithology_relations
    }
    assert ("马五5", "has_lithology", "白云岩") in triples
    assert ("马五5", "has_lithology", "灰岩") in triples
    basis_values = {
        relation.attributes["inference_basis"] for relation in lithology_relations
    }
    assert basis_values == {
        "multipass_layer_crop_promoted_to_stratigraphic_unit",
        "same_color_unit_pattern_scan",
    }
    assert all(Path(image_path).suffix == ".png" for _, image_path in vlm.calls[4:8])
    assert vlm.calls[4][0].startswith("三维地层同色岩性批次:")
    summary = graph.metadata.extra["quality"]["multipass_lithology_summary"]
    assert summary == {
        "vlm_call_count": 8,
        "segment_count": 3,
        "assigned_segment_count": 3,
        "reviewed_segment_count": 0,
        "unknown_segment_count": 0,
        "unit_color_batch_count": 1,
        "unit_color_scanned_count": 3,
        "unit_color_assigned_count": 1,
        "unit_color_rejected_candidate_count": 0,
    }
    assert graph.metadata.extra["quality"]["visual_layer_segment_count"] == 0
    assert graph.metadata.extra["quality"]["lithology_assignment_count"] == 4
    assert graph.validate_references() == []


def test_low_confidence_layer_triggers_independent_review_and_merges_agreement() -> None:
    """首次置信度不足时必须调用独立盲审，并在两次结论一致后保留复核证据。"""

    task = _task()
    vlm = ThreeDimensionalMultiPassOfflineVLM(review_segment="segment_2")
    graph = StratigraphicProfileExtractor(
        subtype_classifier=StratigraphicProfileSubtypeClassifier(DEFAULT_SOURCE)
    ).extract(
        task,
        ImageExtractionContext(
            vlm_client=vlm,
            options={"allowed_stratigraphic_subtypes": ["three_dimensional_stratigraphic_model"]},
        ),
    )
    task_names = [task_name for task_name, _ in vlm.calls]
    assert graph.metadata.extra["status"] == "completed"
    assert graph.metadata.extra["vlm_call_count"] == 10
    assert sum(name.startswith("三维逐层岩性盲审:") for name in task_names) == 1
    assert not any(name.startswith("三维逐层岩性裁决:") for name in task_names)
    unit = next(entity for entity in graph.entities if entity.name == "马五5")
    dolomite_assignment = next(
        item
        for item in unit.attributes["lithology_assignments"]
        if item["lithology_id"] == "lith_dolomite"
    )
    assert dolomite_assignment["evidence_source"] == "independent_review_agreement"
    summary = graph.metadata.extra["quality"]["multipass_lithology_summary"]
    assert summary["reviewed_segment_count"] == 1
    assert summary["unknown_segment_count"] == 0


def test_batch_processes_every_target_and_writes_unique_standalone_results(tmp_path: Path) -> None:
    """批处理必须覆盖两张目标、逐项完成，并写出两个不同文件名的独立结果。"""

    vlm = ThreeDimensionalOfflineVLM()
    output = tmp_path / "batch.json"
    payload = run_batch(DEFAULT_SOURCE, output, vlm_client=vlm)
    assert payload["status"] == "completed"
    assert payload["summary"]["target_chunk_count"] == 2
    assert payload["summary"]["completed_count"] == 2
    assert payload["summary"]["event_count"] == 0
    assert payload["summary"]["vertical_relation_count"] == 4
    assert payload["summary"]["context_relation_count"] == 4
    assert len(vlm.prompts) == 2
    standalone = sorted(tmp_path.glob("*_extraction.json"))
    assert len(standalone) == 2
    assert standalone[0].name != standalone[1].name
