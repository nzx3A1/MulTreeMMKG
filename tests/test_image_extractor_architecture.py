"""图片抽取器路由、扩展注册和无模型调用骨架的离线测试。"""
from __future__ import annotations

import json

from model import Graph
from src.extractors.image_extractor import (
    BaseImageExtractor,
    ImageExtractorKind,
    ImageExtractorRegistry,
    ImageExtractorRouter,
    RecordImageClassificationProvider,
    build_default_registry,
    build_image_tasks,
    extract_from_images,
)
from src.extractors.extractor_init import InitExtractor


class ExplodingClient:
    """任何模型方法访问都会失败，用于证明架构阶段没有调用模型。"""

    def __getattr__(self, name):
        """中文说明：模型接口一旦被访问就立即抛错。"""

        raise AssertionError(f"架构测试不应调用模型方法：{name}")


class ReplacementMapExtractor(BaseImageExtractor):
    """用于验证注册表可替换能力的最小扩展抽取器。"""

    kind = ImageExtractorKind.MAP_SPATIAL
    display_name = "测试地图抽取器"


def _chunk(code: str = "A01", paths=None):
    """中文说明：构造一个与 extractor_init 输出边界一致的图片 Chunk。"""

    return {
        "id": "doc-1:section:1:image:0",
        "document_id": "doc-1",
        "section_id": "section-1",
        "section_title": "区域地质",
        "modality": "image",
        "image_path": paths or ["data/images/a.jpg"],
        "caption": "研究区构造位置图",
        "references": ["正文引用"],
        "classification": {"primary_code": code, "primary_type": "测试类型"},
    }


def test_default_registry_contains_six_independent_extractors() -> None:
    """默认注册表必须包含六类独立图片抽取器。"""

    registry = build_default_registry()
    assert set(registry.kinds()) == set(ImageExtractorKind)


def test_router_maps_all_twenty_codes_to_expected_families() -> None:
    """A01-A20 应稳定归并到六类抽取器，未知类型进入综合图版兜底。"""

    router = ImageExtractorRouter()
    expected = {
        "A01": ImageExtractorKind.MAP_SPATIAL,
        "A06": ImageExtractorKind.STRATIGRAPHIC_PROFILE,
        "A12": ImageExtractorKind.ROCK_MICRO,
        "A16": ImageExtractorKind.QUANTITATIVE_CHART,
        "A09": ImageExtractorKind.GEOLOGICAL_PROCESS,
        "A20": ImageExtractorKind.COMPOSITE_PANEL,
        "": ImageExtractorKind.COMPOSITE_PANEL,
    }
    for code, kind in expected.items():
        assert router.route(build_image_tasks(_chunk(code))[0]) == kind


def test_chunk_with_multiple_paths_expands_and_merges_without_model_calls(tmp_path) -> None:
    """一个 Chunk 的多张图片应分别路由，最终仍返回一个 Chunk Graph。"""

    output_path = tmp_path / "image-scaffold.json"
    results = extract_from_images(
        [_chunk("A01", ["data/images/a.jpg", "data/images/b.jpg"])],
        ExplodingClient(),
        ExplodingClient(),
        output_path=output_path,
        show_progress=False,
    )
    assert len(results) == 1
    assert isinstance(results[0], Graph)
    assert results[0].metadata.chunk_id == "doc-1:section:1:image:0"
    assert results[0].metadata.extra["image_task_count"] == 2
    assert all(route["extractor_kind"] == "map_spatial" for route in results[0].metadata.extra["routes"])
    assert all(route["model_called"] is False for route in results[0].metadata.extra["routes"])
    assert len(json.loads(output_path.read_text(encoding="utf-8"))) == 1


def test_images_in_same_chunk_can_use_individual_mock_classifications() -> None:
    """同一 Chunk 中的图片可使用 mock 单图分类分别进入不同抽取器。"""

    chunk = _chunk("A20", ["data/images/map.jpg", "data/images/sem.jpg"])
    chunk["image_classifications"] = [
        {"image_index": 0, "classification": {"primary_code": "A01", "primary_type": "构造图"}},
        {"image_index": 1, "classification": {"primary_code": "A12", "primary_type": "扫描电镜"}},
    ]
    result = extract_from_images([chunk], ExplodingClient(), ExplodingClient(), show_progress=False)[0]
    assert [route["extractor_kind"] for route in result.metadata.extra["routes"]] == ["map_spatial", "rock_micro"]


def test_mock_record_provider_matches_parent_chunk_and_image_index() -> None:
    """外部逐图分类记录应按父 Chunk 和图片序号覆盖 Chunk 级分类。"""

    chunk = _chunk("A20", ["data/images/map.jpg", "data/images/sem.jpg"])
    provider = RecordImageClassificationProvider(
        [
            {
                "parent_chunk_id": chunk["id"],
                "image_index_in_chunk": 0,
                "image_path": "data/images/map.jpg",
                "classification": {"primary_code": "A01", "primary_type": "构造图"},
            },
            {
                "parent_chunk_id": chunk["id"],
                "image_index_in_chunk": 1,
                "image_path": "data/images/sem.jpg",
                "classification": {"primary_code": "A12", "primary_type": "扫描电镜"},
            },
        ]
    )
    result = extract_from_images(
        [chunk],
        ExplodingClient(),
        ExplodingClient(),
        classification_provider=provider,
        show_progress=False,
    )[0]
    assert [route["extractor_kind"] for route in result.metadata.extra["routes"]] == ["map_spatial", "rock_micro"]


def test_init_extractor_is_the_image_chunk_integration_entry(tmp_path) -> None:
    """InitExtractor 应把分组后的图片 Chunk 原样交给图片架构，且不触发模型调用。"""

    extractor = InitExtractor(llm_client=ExplodingClient(), vlm_client=ExplodingClient(), show_progress=False)
    results = extractor.extract([_chunk("A01")], output_dir=tmp_path)
    assert len(results["image"]) == 1
    route = results["image"][0].metadata.extra["routes"][0]
    assert route["extractor_kind"] == "map_spatial"
    assert route["model_called"] is False
    assert (tmp_path / "stage_04_image_extraction.json").is_file()


def test_registry_allows_explicit_replacement_for_future_extension() -> None:
    """后续实现可以显式替换某一类抽取器而不修改路由和流水线。"""

    registry = ImageExtractorRegistry()
    original = ReplacementMapExtractor()
    registry.register(original)
    replacement = ReplacementMapExtractor()
    registry.register(replacement, replace=True)
    assert registry.get(ImageExtractorKind.MAP_SPATIAL) is replacement
