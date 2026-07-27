"""地层—剖面—测井抽取器针对 14 张真实图片的全 mock 离线测试。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

from src.extractors.image_extractor import extract_from_images
from src.utils.json_io import read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = PROJECT_ROOT / "src" / "extractors" / "image_extractor" / "stratigraphic_profile" / "test_data"
CHUNKS_PATH = TEST_DATA_DIR / "image_chunks.json"
MOCK_PATH = TEST_DATA_DIR / "mock_responses.json"

# 中文说明：该文件验证已移除的通用 14 图算法，历史 mock 目录也已不存在；新表格分支由专用测试接管。
pytestmark = pytest.mark.skip(reason="legacy generic stratigraphic extractor replaced by subtype-specific algorithms")


def _records() -> dict[str, dict[str, Any]]:
    """中文说明：加载人工逐图读取后形成的 14 份 VLM/LLM mock 响应。"""

    payload = read_json(MOCK_PATH)
    records = payload.get("responses", {}) if isinstance(payload, Mapping) else {}
    assert isinstance(records, dict)
    return records


def _classified_chunks(records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """中文说明：给原始测试 Chunk 注入人工确认的 A03—A06 分类，确保全部进入本抽取器。"""

    chunks = deepcopy(read_json(CHUNKS_PATH))
    assert isinstance(chunks, list)
    for chunk in chunks:
        image_path = Path(chunk["image_path"][0])
        chunk["classification"] = dict(records[image_path.name]["classification"])
        chunk.setdefault("document_id", str(chunk["id"]).split(":section:", 1)[0])
    return chunks


class MockStratigraphicVLM:
    """按来源图片文件名返回人工构造的视觉模型 JSON。"""

    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        """中文说明：保存 mock 索引和逐图调用记录。"""

        self.records = records
        self.calls: list[str] = []

    def describe_image(self, image_path: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """中文说明：验证 Prompt 包含来源约束后返回对应图片的人工视觉响应。"""

        filename = Path(image_path).name
        assert filename in self.records
        assert image_path in prompt
        assert "不得凭花纹猜岩性" in prompt
        assert kwargs["response_format"] == {"type": "json_object"}
        self.calls.append(filename)
        return deepcopy(self.records[filename]["visual"])


class MockStratigraphicLLM:
    """按审查 Prompt 中的图片文件名返回关系审查 JSON。"""

    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        """中文说明：保存关系审查 mock 索引和调用记录。"""

        self.records = records
        self.calls: list[str] = []

    def call_openai_json(self, prompt: str, task_name: str = "") -> dict[str, Any]:
        """中文说明：匹配当前来源图片并返回不创造新实体的人工审查结果。"""

        assert "不得创建视觉结果中不存在的实体" in prompt
        for filename, record in self.records.items():
            if filename in prompt:
                self.calls.append(filename)
                return deepcopy(record["audit"])
        raise AssertionError(f"关系审查 Prompt 未包含已知图片文件名：{task_name}")


def _run_all_images():
    """中文说明：通过统一图片流水线运行 14 张真实图片的 mock 模型响应。"""

    records = _records()
    vlm = MockStratigraphicVLM(records)
    llm = MockStratigraphicLLM(records)
    results = extract_from_images(_classified_chunks(records), llm, vlm, show_progress=False)
    return records, vlm, llm, results


def test_all_fourteen_images_complete_without_events() -> None:
    """14 张样例都必须完成实体关系抽取，并按当前要求保持事件列表为空。"""

    records, vlm, llm, results = _run_all_images()
    assert len(records) == len(results) == 14
    assert len(vlm.calls) == len(llm.calls) == 14
    assert set(vlm.calls) == set(llm.calls) == set(records)
    for graph in results:
        assert graph.metadata.extra["status"] == "completed"
        assert graph.metadata.extra["routes"][0]["extractor_kind"] == "stratigraphic_profile"
        assert graph.metadata.extra["routes"][0]["model_called"] is True
        assert graph.entities
        assert graph.relations
        assert graph.events == []
        assert graph.validate_references() == []


def test_every_entity_and_relation_has_exact_source_image() -> None:
    """每个实体和关系都必须标注对应的来源图片、图片任务、Chunk 和图内证据。"""

    _, _, _, results = _run_all_images()
    for graph in results:
        route = graph.metadata.extra["routes"][0]
        source_path = route["image_path"]
        source_image_id = route["image_id"]
        for item in [*graph.entities, *graph.relations]:
            assert item.provenance
            assert item.metadata["source_modality"] == "image"
            assert item.metadata["source_image_path"] == source_path
            assert item.metadata["source_image_id"] == source_image_id
            assert item.metadata["source_chunk_id"] == graph.metadata.chunk_id
            assert "visual_evidence" in item.metadata


def test_each_image_keeps_its_key_entities_and_relations() -> None:
    """逐图检查人工识别出的关键实体和方向敏感关系没有在装配时丢失。"""

    expected = {
        "af4dce79db28e729dc664ae367262cb0023fd38e9b86eab7316d7a5efaa4d168.jpg": ("芦草沟组", ("P₂l₂²", "directly_overlies", "P₂l₂²⁻¹")),
        "21856a21840417ea12c8ac5ab804d4e4e713abaea5dc3f4323435c25ddbc36b7.jpg": ("顺托果勒低隆起", ("顺北断裂群", "cuts_through", "O₁—₂y")),
        "25710ddea32114b634f99605ec251f03f196b779dad9ed49984857e947639138.jpg": ("放空段", ("SB1-4H井轨迹", "intersects", "放空段")),
        "f09f659e47b62516a44aa61839eb7a17020158f9fc8e507e46106c1800013690.jpg": ("高角度溶蚀缝", ("SBP1井8398—8402 m井段", "contains_reservoir", "高角度溶蚀缝")),
        "f0ac23ffa801dccdccad61dda0ee06d1417576276e440abdd8db77b5a4919ff1.jpg": ("走滑拉分段", ("SB1-10", "located_in", "走滑拉分段")),
        "08471e7940670b3c490c8861c63cd59641bb54cabe2d8de2e6dc520b9d0a26a4.jpg": ("伽马", ("伽马", "characterizes", "1520—1542 m黄色重点层段")),
        "f78edd05df62c17f2f9ef128081b5dc848da93c9471f07207ae8507c5530efad.jpg": ("龙一段", ("龙一段", "higher_response_than", "龙二段")),
        "feae59d54d90508387d221cd079b0d8d3988221362bd5ba6a6a9c9b02a421141.jpg": ("优质页岩", ("优质页岩", "bounded_by", "龙马溪组优质页岩顶板")),
        "08c6b2e072c2de158841113a7bd3a49c7741ee74312d7d678f7257705294647c.jpg": ("宁201", ("深水陆棚相", "lateral_transition_to", "半深水泥质陆棚相")),
        "3966a658606e612bbc12f83b3a53fc9f93476ce10ee670286999724988ea38c4.jpg": ("研究地层", ("研究地层", "located_in", "K₂qn¹ 青山口组一段")),
        "0875b0d3913ae8bd912712c6b9edc98998eaffa85966ef21dc2d29de51324264.jpg": ("NH6-2调整后轨迹", ("NH6-2调整后轨迹", "tracks_along", "长7泥岩段")),
        "787873a7863480cdb8cd2b32373fe10780afb8365cc9153ebd6aa6e0ebcea629.jpg": ("马五₆", ("马四段", "contains_reservoir", "层序界面短期暴露溶蚀储层")),
        "c05e5f21288b34d21112005f87dc0a9547e2df64064ad7002b0b41abc27d6d0b.jpg": ("萨布哈", ("Mg²⁺流体", "dolomitizes", "含硬石膏结核白云岩")),
        "2a03b69fde6aea8f72f7195ee9f92bf059f0dabd9656489f8d54e0a6753b86c3.jpg": ("YY12", ("YY12", "intersects", "长7₃")),
    }
    _, _, _, results = _run_all_images()
    by_filename = {Path(graph.metadata.extra["routes"][0]["image_path"]).name: graph for graph in results}
    assert set(by_filename) == set(expected)
    for filename, (entity_name, triple) in expected.items():
        graph = by_filename[filename]
        assert entity_name in {entity.name for entity in graph.entities}
        triples = {(relation.source_name, relation.type, relation.target_name) for relation in graph.relations}
        assert triple in triples

    # 中文说明：马五段十个亚段属于同一父层，仍应按子层内部顺序生成相邻层序边。
    majiagou = by_filename["787873a7863480cdb8cd2b32373fe10780afb8365cc9153ebd6aa6e0ebcea629.jpg"]
    majiagou_triples = {(relation.source_name, relation.type, relation.target_name) for relation in majiagou.relations}
    assert ("马五₉", "directly_overlies", "马五₁₀") in majiagou_triples
