"""验证提示词图片评测的选择性重跑、原样本汇总、备份和断点续跑。"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from src.extractors.image_extractor.panel_splitter import PanelPreparation


@pytest.fixture
def runner(monkeypatch):
    """按独立模块加载中文目录中的评测入口，避免与其他同名脚本冲突。"""
    path = Path(__file__).resolve().parents[1] / "eval/抽取评测/chunk抽取/LLM抽取/image_extract.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("llm_image_panel_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(tmp_path: Path, name: str = "a") -> dict:
    """生成可用图片路径及与真实评测一致的样本结构。"""
    path = tmp_path / f"{name}.png"
    path.write_bytes(b"offline image fixture")
    return {"sample_id": f"doc::{name}", "section": {"section_title": "测试章节"},
            "content": {"image_path": [str(path)], "caption": "父图标题"}}


def test_split_children_are_merged_to_original_sample(runner, monkeypatch, tmp_path) -> None:
    """每个子图独立提示词抽取，最终只产生原样本 Graph 并保留子图证据。"""
    sample = _sample(tmp_path)

    class Splitter:
        """返回两个已复核子图的离线切分器。"""

        def __init__(self, *args):
            """兼容正式切分器的构造参数。"""

        def prepare(self, task):
            """返回带同一原图身份的两个叶子任务。"""
            return PanelPreparation([replace(task, image_id=f"{task.image_id}:panel:{index}", panel_metadata={
                "status": "verified_single", "panel_path": [index], "source_image_id": task.image_id,
                "source_image_path": task.image_path, "source_image_index": 0,
            }) for index in range(2)], [])

    class Classifier:
        """返回单一类型，避免测试依赖真实分类模型。"""

        def classify(self, task, client):
            """为两个独立子图返回同一个合法类型。"""
            return SimpleNamespace(primary_code="A01", primary_type="地图", secondary_codes=[])

    def fake_call(sample, *args):
        """相同实体名用于验证跨子图合并不会丢失来源。"""
        return {"entities": [{"name": "延长组", "type": "stratum", "type_zh": "地层", "provenance": sample["sample_id"]}], "relations": []}

    monkeypatch.setattr(runner, "VLMPanelSplitter", Splitter)
    monkeypatch.setattr(runner, "VLMImageClassifier", Classifier)
    monkeypatch.setattr(runner, "call_model", fake_call)
    result = runner.extract_split_sample(sample, tmp_path / "input.json", tmp_path / "panels")
    assert result["report"]["status"] == "completed"
    graph = result["graph"]
    assert graph["metadata"]["chunk_id"] == sample["sample_id"]
    assert len(graph["entities"]) == 1
    assert len(graph["entities"][0]["metadata"]["same_name_merge"]["source_entities"]) == 2
    summary = graph["metadata"]["extra"]["source_image_results"][0]
    assert summary["panel_count"] == 2
    assert summary["entity_ids"] == [graph["entities"][0]["id"]]


def test_selective_rerun_preserves_unsplit_graph_and_resumes(runner, monkeypatch, tmp_path) -> None:
    """只替换需切分的样本，旧文件先备份，完成后再次运行不会重复请求模型。"""
    samples = [_sample(tmp_path, "a"), _sample(tmp_path, "b")]
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(samples), encoding="utf-8")
    old_graphs = [{"entities": [], "relations": [], "events": [], "metadata": {"chunk_id": sample["sample_id"], "extra": {"marker": "old"}}} for sample in samples]
    output_path = tmp_path / "result.json"
    output_path.write_text(json.dumps({"input": str(input_path.resolve()), "graphs": old_graphs, "statistics": {"model_request_count": 9}}), encoding="utf-8")
    old_bytes = output_path.read_bytes()
    calls = []

    def fake_extract(sample, *args):
        """一个样本无需切分，另一个返回已重新抽取的样本图。"""
        sample_id = sample["sample_id"]
        calls.append(sample_id)
        split = sample_id.endswith("b")
        return {"report": {"sample_id": sample_id, "status": "completed" if split else "preserved_unsplit", "model_request_count": 2},
                "graph": {**old_graphs[1], "metadata": {"chunk_id": sample_id, "extra": {"marker": "new"}}} if split else None}

    monkeypatch.setattr(runner, "extract_split_sample", fake_extract)
    result = runner.rerun_split_images(input_path, output_path, workers=2)
    assert result["_status"] == "completed"
    assert result["graphs"][0] == old_graphs[0]
    assert result["graphs"][1]["metadata"]["extra"]["marker"] == "new"
    assert Path(result["panel_rerun"]["backup"]).read_bytes() == old_bytes
    assert result["statistics"]["model_request_count"] == 13
    runner.rerun_split_images(input_path, output_path)
    assert len(calls) == 2
