"""验证多子图检测、像素隔离、复核门禁及图片流水线的分类和来源追踪。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import pytest

from model import Entity, Graph, Relation, SourceModality
from src.extractors.image_extractor import BaseImageExtractor, ImageExtractorKind, ImageExtractorRegistry
from src.extractors.image_extractor.classification import RecordImageClassificationProvider
from src.extractors.image_extractor.panel_splitter import VLMPanelSplitter
from src.extractors.image_extractor.pipeline import build_image_tasks, extract_from_images
from src.extractors.image_extractor import main as image_main


SINGLE = {"has_subfigures": False, "single_type": True, "reason": "单个完整面板", "panels": []}


class ScriptedVLM:
    """按顺序返回检测或分类结果，并保存模型实际收到的图片像素。"""

    def __init__(self, responses: list[Any]) -> None:
        """保存预定响应和调用记录，用于验证检测、复核、分类的先后顺序。"""

        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def describe_image(self, path: str, prompt: str, **kwargs: Any) -> Any:
        """读取输入图并返回响应，允许模拟接口异常和无效模型输出。"""

        with Image.open(path) as image:
            self.calls.append({"path": path, "image": image.copy(), "prompt": prompt, **kwargs})
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _panel(box: list[int], label: str = "a", masks: list[list[int]] | None = None) -> dict[str, Any]:
    """构造一条归一化子图记录。"""

    return {"label": label, "bbox": box, "exclude_bboxes": masks or []}


def _multi(*panels: dict[str, Any]) -> dict[str, Any]:
    """构造多子图检测结果。"""

    return {"has_subfigures": True, "single_type": False, "reason": "多个独立面板", "panels": list(panels)}


def _chunk(tmp_path: Path) -> dict[str, Any]:
    """用不同颜色模拟主图、嵌套图和右侧面板，支持检查裁剪后的真实像素。"""

    image = Image.new("RGB", (200, 100), "red")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 0, 199, 99), fill="blue")
    draw.rectangle((60, 0, 99, 39), fill="green")
    path = tmp_path / "原图.png"
    image.save(path)
    return {"id": "doc:section:1:image:0", "document_id": "doc", "modality": "image",
            "caption": "混合图", "image_path": [str(path)], "classification": {"primary_code": "A20"}}


def _task(tmp_path: Path):
    """取得尚未实际调用视觉模型的单图任务。"""

    return build_image_tasks(_chunk(tmp_path))[0]


def _classification(code: str, secondary: list[str] | None = None) -> dict[str, Any]:
    """构造符合现有 A01-A20 分类协议的视觉分类结果。"""

    return {"primary_code": code, "secondary_codes": secondary or [], "confidence": 0.98,
            "reason": "可见当前子图主体", "visual_evidence": ["独立面板"]}


class RecordingExtractor(BaseImageExtractor):
    """记录被放行的真实子图任务，返回带实体关系的可聚合图。"""

    def __init__(self, kind: ImageExtractorKind) -> None:
        """为指定专业路由创建离线抽取器。"""

        self.kind = kind
        self.display_name = "记录子图任务"
        self.tasks = []

    def extract(self, task: Any, context: Any) -> Graph:
        """仅保存任务身份，避免测试依赖地质专业模型与 OCR。"""

        self.tasks.append(task)
        entity_id = f"{task.image_id}:e"
        graph = Graph.from_chunk(task.document_id, task.chunk_id, SourceModality.IMAGE,
                                 entities=[Entity(id=entity_id, name=task.image_id, type="figure")],
                                 relations=[Relation(id=f"{task.image_id}:r", type="self", source_id=entity_id, target_id=entity_id)])
        graph.metadata.extra.update(status="completed", image_id=task.image_id,
                                    image_path=task.image_path, extractor_kind=self.kind.value,
                                    classification_code=task.classification_code)
        return graph


def test_inset_is_cropped_and_hidden_in_main_map(tmp_path: Path) -> None:
    """同类型嵌套图也要单独切出，主图对应区域自动涂白且不会损坏旁边像素。"""

    client = ScriptedVLM([_multi(_panel([0, 0, 500, 1000]), _panel([300, 0, 500, 400], "inset"),
                                _panel([500, 0, 1000, 1000], "b")), SINGLE, SINGLE, SINGLE])
    prepared = VLMPanelSplitter(client, tmp_path / "panels").prepare(_task(tmp_path))
    assert len(prepared.tasks) == 3
    assert prepared.call_count == 4
    main, inset, right = prepared.tasks
    with Image.open(main.image_path) as image:
        assert image.size == (100, 100)
        assert image.getpixel((70, 20)) == (255, 255, 255)
        assert image.getpixel((59, 20)) == (255, 0, 0)
        assert image.getpixel((99, 40)) == (255, 0, 0)
    with Image.open(inset.image_path) as image:
        assert image.size == (40, 40)
        assert image.getpixel((10, 10)) == (0, 128, 0)
    assert right.panel_metadata["bbox_pixels"] == [100, 0, 200, 100]
    assert main.panel_metadata["masked_bboxes_pixels"] == [[60, 0, 100, 40]]
    assert all(task.panel_metadata["status"] == "verified_single" for task in prepared.tasks)
    assert len(list((tmp_path / "panels").rglob("panels.json"))) == 1


def test_recursive_split_maps_coordinates_back_to_source(tmp_path: Path) -> None:
    """第一次裁剪后仍有子图时继续拆分，叶子坐标正确累积到原图。"""

    client = ScriptedVLM([
        _multi(_panel([0, 0, 500, 1000]), _panel([500, 0, 1000, 1000], "b")), SINGLE,
        _multi(_panel([0, 0, 1000, 500]), _panel([0, 500, 1000, 1000], "b2")), SINGLE, SINGLE,
    ])
    prepared = VLMPanelSplitter(client, tmp_path / "panels").prepare(_task(tmp_path))
    assert [task.panel_metadata["bbox_pixels"] for task in prepared.tasks] == [
        [0, 0, 100, 100], [100, 0, 200, 50], [100, 50, 200, 100],
    ]
    assert prepared.tasks[-1].image_id.endswith(":panel:1:panel:1")


@pytest.mark.parametrize("response", [
    "not json", RuntimeError("接口失败"),
    {**SINGLE, "single_type": False}, {**SINGLE, "has_subfigures": "false"},
    _multi(_panel([-10, 0, 500, 1000]), _panel([500, 0, 1000, 1000])),
    _multi(_panel([0, 0, 500, 1000]), _panel([0, 0, 500, 1000])),
    _multi(_panel([0, 0, 700, 1000]), _panel([500, 0, 1000, 1000])),
    _multi(_panel([0, 0, 500, 1000], masks=[[0, 0, 500, 1000]]), _panel([500, 0, 1000, 1000])),
])
def test_invalid_or_uncertain_detection_blocks_extraction(tmp_path: Path, response: Any) -> None:
    """错误、越界、重复、未隔离重叠与不确定输出均不能回退到原图抽取。"""

    prepared = VLMPanelSplitter(ScriptedVLM([response, response]), tmp_path / "panels").prepare(_task(tmp_path))
    assert len(prepared.tasks) == 1
    assert prepared.tasks[0].panel_metadata["status"] == "failed"
    assert prepared.tasks[0].panel_metadata["error"]


def test_depth_limit_blocks_mixed_leaf_but_keeps_verified_sibling(tmp_path: Path) -> None:
    """达到深度限制的混合图被阻断，同一原图的合格兄弟图仍可继续。"""

    multi = _multi(_panel([0, 0, 500, 1000]), _panel([500, 0, 1000, 1000]))
    prepared = VLMPanelSplitter(ScriptedVLM([multi, multi, SINGLE]), tmp_path / "panels", max_depth=1).prepare(_task(tmp_path))
    assert [task.panel_metadata["status"] for task in prepared.tasks] == ["failed", "verified_single"]


def test_complementary_masks_allow_shared_legend_layout(tmp_path: Path) -> None:
    """两张子图在不同位置保留图例时，按双方遮蔽框的并集校验重复可见区域。"""

    client = ScriptedVLM([_multi(
        _panel([0, 0, 700, 1000], masks=[[500, 0, 700, 500]]),
        _panel([500, 0, 1000, 1000], masks=[[500, 500, 700, 1000]]),
    ), SINGLE, SINGLE])
    result = VLMPanelSplitter(client, tmp_path / "panels").prepare(_task(tmp_path))
    assert [task.panel_metadata["status"] for task in result.tasks] == ["verified_single", "verified_single"]


def test_geometry_failure_can_be_corrected_once(tmp_path: Path) -> None:
    """首次模型边界重叠时，把几何错误回传模型，修正后仍须逐子图复核。"""

    client = ScriptedVLM([
        _multi(_panel([0, 0, 700, 1000]), _panel([500, 0, 1000, 1000])),
        _multi(_panel([0, 0, 500, 1000]), _panel([500, 0, 1000, 1000])), SINGLE, SINGLE,
    ])
    result = VLMPanelSplitter(client, tmp_path / "panels").prepare(_task(tmp_path))
    assert len(result.tasks) == 2
    assert result.call_count == 4
    assert "程序校验失败" in client.calls[1]["prompt"]


def test_explicit_shared_legend_can_remain_visible_in_both_panels(tmp_path: Path) -> None:
    """共享图例可在两个裁剪结果中保留，但不能将大面积主体标为共享图例。"""
    source = _task(tmp_path)
    multi = _multi(_panel([0, 0, 700, 1000], masks=[[500, 0, 700, 800]]),
                   _panel([500, 0, 1000, 1000]))
    multi["shared_bboxes"] = [[500, 800, 700, 1000]]
    result = VLMPanelSplitter(ScriptedVLM([multi, SINGLE, SINGLE]), tmp_path / "panels").prepare(source)
    assert [task.panel_metadata["status"] for task in result.tasks] == ["verified_single", "verified_single"]
    assert result.tasks[0].panel_metadata["shared_bboxes_pixels"] == [[100, 80, 140, 100]]
    multi["shared_bboxes"] = [[0, 0, 1000, 1000]]
    blocked = VLMPanelSplitter(ScriptedVLM([multi, multi]), tmp_path / "blocked").prepare(source)
    assert blocked.tasks[0].panel_metadata["status"] == "failed"


def test_source_summary_tracks_entity_ids_after_same_name_merge(tmp_path: Path) -> None:
    """原图汇总索引应随同名实体合并更新，不能引用已被去重的子图节点。"""

    from src.extractors.image_extractor.pipeline import _summarize_source_images
    from src.extractors.image_extractor.postprocess import merge_same_name_entities
    from src.extractors.image_extractor.pipeline import _merge_task_graphs

    graphs = []
    for index in range(2):
        graph = Graph.from_chunk("doc", "chunk", SourceModality.IMAGE,
                                  entities=[Entity(id=f"e{index}", name="延长组", type="stratum")])
        graph.metadata.extra.update(status="completed", image_id=f"source:panel:{index}", image_panel={
            "source_image_id": "source", "source_image_path": "source.png", "panel_path": [index],
        })
        graphs.append(graph)
    merged = merge_same_name_entities(_merge_task_graphs({"id": "chunk"}, graphs))
    _summarize_source_images(merged, graphs)
    assert len(merged.entities) == 1
    assert merged.metadata.extra["source_image_results"][0]["entity_ids"] == ["e0"]
    assert len(merged.entities[0].metadata["same_name_merge"]["source_entities"]) == 2


def test_pipeline_reclassifies_panels_and_preserves_provenance(tmp_path: Path) -> None:
    """切分后必须重新分类，不能沿用父图 A20；抽取器仅接收复核后的单图。"""

    chunk = _chunk(tmp_path)
    client = ScriptedVLM([
        _multi(_panel([0, 0, 500, 1000], "a"), _panel([500, 0, 1000, 1000], "b")),
        SINGLE, SINGLE, _classification("A01"), _classification("A03"),
    ])
    registry = ImageExtractorRegistry()
    map_extractor = RecordingExtractor(ImageExtractorKind.MAP_SPATIAL)
    strata_extractor = RecordingExtractor(ImageExtractorKind.STRATIGRAPHIC_PROFILE)
    registry.register(map_extractor)
    registry.register(strata_extractor)
    # 中文说明：故意传入整图 A20 记录，验证这些记录不会覆盖子图视觉分类。
    provider = RecordImageClassificationProvider([{"chunk_id": chunk["id"], "image_index": 0, "primary_code": "A20"}])
    graph = extract_from_images([chunk], vlm_client=client, registry=registry, classification_provider=provider,
                                context_options={"panel_output_dir": tmp_path / "panels"}, show_progress=False)[0]
    assert len(map_extractor.tasks) == len(strata_extractor.tasks) == 1
    assert [route["classification_code"] for route in graph.metadata.extra["routes"]] == ["A01", "A03"]
    assert graph.metadata.extra["status"] == "completed"
    assert graph.metadata.extra["vlm_call_count"] == 5
    assert graph.metadata.extra["panel_detection_vlm_call_count"] == 3
    assert graph.validate_references() == []
    assert all(entity.metadata["image_panel"]["source_image_path"] == chunk["image_path"][0] for entity in graph.entities)
    assert all("classification" not in task.raw_chunk for task in map_extractor.tasks + strata_extractor.tasks)
    assert all(call["image"].size == (100, 100) for call in client.calls[1:])
    assert all(":panel:" in call["task_name"] for call in client.calls[-2:])
    # 中文说明：两个子图的实体和关系归入同一原图，最终仍只输出一个 Chunk Graph。
    assert graph.metadata.extra["source_image_count"] == 1
    summary = graph.metadata.extra["source_image_results"][0]
    assert summary["panel_count"] == 2
    assert summary["classification_codes"] == ["A01", "A03"]
    assert set(summary["entity_ids"]) == {entity.id for entity in graph.entities}
    assert set(summary["relation_ids"]) == {relation.id for relation in graph.relations}


def test_failed_detection_never_calls_registered_extractor(tmp_path: Path) -> None:
    """检测失败的图片写入可追踪错误，不能误进入 A20 或其他专业抽取器。"""

    graph = extract_from_images([_chunk(tmp_path)], vlm_client=ScriptedVLM([RuntimeError("检测失败")]),
                                registry=ImageExtractorRegistry(),
                                context_options={"panel_output_dir": tmp_path / "panels"}, show_progress=False)[0]
    assert graph.entities == []
    assert graph.metadata.extra["status"] == "model_error"
    assert "检测失败" in graph.metadata.extra["routes"][0]["model_errors"][0]
    assert graph.metadata.extra["vlm_call_count"] == 1


def test_failed_classification_is_not_replaced_by_parent_type(tmp_path: Path) -> None:
    """视觉复核通过但子图分类失败时，不能使用父图类别继续抽取。"""

    client = ScriptedVLM([SINGLE, RuntimeError("分类失败")])
    graph = extract_from_images([_chunk(tmp_path)], vlm_client=client, registry=ImageExtractorRegistry(),
                                context_options={"panel_output_dir": tmp_path / "panels"}, show_progress=False)[0]
    assert graph.metadata.extra["status"] == "model_error"
    assert graph.metadata.extra["vlm_call_count"] == 2


def test_single_panel_preserves_original_path_classification_lookup(tmp_path: Path) -> None:
    """没有子图时，原图路径分类记录仍能用于朝向归一化后的图片。"""

    chunk = _chunk(tmp_path)
    provider = RecordImageClassificationProvider([{"image_path": chunk["image_path"][0], "primary_code": "A01"}])
    registry = ImageExtractorRegistry()
    extractor = RecordingExtractor(ImageExtractorKind.MAP_SPATIAL)
    registry.register(extractor)
    graph = extract_from_images([chunk], vlm_client=ScriptedVLM([SINGLE]), registry=registry,
                                classification_provider=provider,
                                context_options={"panel_output_dir": tmp_path / "panels"}, show_progress=False)[0]
    assert extractor.tasks[0].classification_code == "A01"
    assert graph.metadata.extra["vlm_call_count"] == 1


def test_legacy_or_partial_cache_cannot_bypass_panel_checks(tmp_path: Path) -> None:
    """旧版整图结果或部分失败结果即使有实体关系，也必须重新执行子图流程。"""

    cached = Graph.from_chunk("doc", "chunk", SourceModality.IMAGE,
                              entities=[Entity(id="e", name="e", type="figure")],
                              relations=[Relation(id="r", type="self", source_id="e", target_id="e")])
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"graphs": [cached.to_dict()]}), encoding="utf-8")
    assert image_main._load_reusable_graphs(path) == {}
    cached.metadata.extra.update(panel_preprocessing_version=1, status="partial")
    path.write_text(json.dumps({"graphs": [cached.to_dict()]}), encoding="utf-8")
    assert image_main._load_reusable_graphs(path) == {}


def test_multiple_source_images_have_separate_aggregated_results(tmp_path: Path) -> None:
    """同一 Chunk 的多张原图分别汇总，每张原图只引用自己的子图结果。"""

    chunk = _chunk(tmp_path)
    chunk["image_path"] *= 2
    client = ScriptedVLM([
        _multi(_panel([0, 0, 500, 1000]), _panel([500, 0, 1000, 1000])),
        SINGLE, SINGLE, _classification("A01"), _classification("A03"),
        SINGLE, _classification("A01"),
    ])
    registry = ImageExtractorRegistry()
    registry.register(RecordingExtractor(ImageExtractorKind.MAP_SPATIAL))
    registry.register(RecordingExtractor(ImageExtractorKind.STRATIGRAPHIC_PROFILE))
    graph = extract_from_images([chunk], vlm_client=client, registry=registry,
                                context_options={"panel_output_dir": tmp_path / "panels"}, show_progress=False)[0]
    summaries = graph.metadata.extra["source_image_results"]
    assert graph.metadata.extra["source_image_count"] == 2
    assert [item["panel_count"] for item in summaries] == [2, 1]
    assert [item["split_into_subfigures"] for item in summaries] == [True, False]
    assert [item["source_image_index"] for item in summaries] == [0, 1]
    assert set(summaries[0]["entity_ids"]).isdisjoint(summaries[1]["entity_ids"])
    assert len(graph.entities) == 3
