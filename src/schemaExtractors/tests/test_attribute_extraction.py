"""验证中文开放属性的补充、冲突保护与断点恢复。"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from src.schemaExtractors.schema_loader import load_schema
from src.schemaExtractors.textTextractor.attribute_extraction import merge_attribute_updates, validate_attribute_updates
from src.schemaExtractors.textTextractor.pipeline import extract_text_from_summary_file
from src.schemaExtractors.textTextractor.validator import validate_entities


TEXT = "马家沟组储层发育。该储层孔隙度为8%，厚度100～900 m，发育晶间孔和溶蚀孔。"


def reservoir() -> dict[str, Any]:
    """构造仅带实体证据和初始中文属性的储层响应。"""
    return {"id": "temporary", "name": "马家沟组储层", "type": "储层",
            "attribute": {"储层类型": "白云岩"}, "provenance": "马家沟组储层发育。"}


class AttributeLLM:
    """从属性提示中读取正式实体 ID 并返回无逐项证据的增量。"""

    def __init__(self) -> None:
        """初始化调用记录。"""
        self.messages: list[Any] = []

    def chat_json(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        """按当前阶段返回固定实体或基于正式 ID 的属性补丁。"""
        self.messages.append(copy.deepcopy(messages))
        prompt = json.loads(messages[1]["content"])
        if "relations" not in prompt:
            return {"entities": [reservoir()]}
        identifier = prompt["entities"][0]["id"]
        return {"entity_attributes": [{"entity_id": identifier, "attribute": {
            "孔隙度": "8%", "厚度范围": "100～900 m", "孔隙类型": ["晶间孔", "溶蚀孔"]}}]}


def test_attribute_stage_and_checkpoint_resume(tmp_path: Path) -> None:
    """属性补充后只留下中文开放属性，成功检查点可直接恢复。"""
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps([{"sample_id": "doc::text:1", "content": {"text": TEXT}}], ensure_ascii=False), encoding="utf-8")
    client = AttributeLLM()
    result = extract_text_from_summary_file(source, output, llm_client=client)
    assert result["_status"] == "completed"
    assert len(client.messages) == 2
    record = result["graphs"][0]["entities"][0]
    assert re.fullmatch(r"entity_\d{16,}", record["id"])
    assert record["attribute"] == {"储层类型": "白云岩", "孔隙度": "8%",
                                   "厚度范围": "100～900 m", "孔隙类型": ["晶间孔", "溶蚀孔"]}
    assert "attributes" not in record and "attribute_provenance" not in record
    assert result["statistics"]["entity_attribute_count"] == 4
    assert result["statistics"]["attribute_completed_text_chunk_count"] == 1
    cached_client = AttributeLLM()
    cached = extract_text_from_summary_file(source, output, llm_client=cached_client)
    assert cached["statistics"]["resumed_text_chunk_count"] == 1
    assert cached["graphs"] == result["graphs"]
    assert cached_client.messages == []


def test_updates_are_open_and_conflicts_preserve_first_value() -> None:
    """补丁可增加任意中文属性，重复补丁不能覆盖已有值。"""
    schema = load_schema()
    entities = validate_entities({"entities": [reservoir()]}, schema, TEXT).records
    identifier = entities[0]["id"]
    result = validate_attribute_updates({"entity_attributes": [
        {"entity_id": identifier, "attribute": {"孔隙度": "8%", "储层类型": "砂岩"}},
        {"entity_id": identifier, "attribute": {"孔隙度": "9%", "分布": "局部"}},
    ]}, schema, TEXT, entities)
    merged = merge_attribute_updates(entities, result.records)
    assert merged[0]["attribute"] == {"储层类型": "白云岩", "孔隙度": "8%", "分布": "局部"}
    assert sum(issue["code"] == "attribute_conflict" for issue in result.issues) == 2


def test_invalid_updates_do_not_mutate_existing_entities() -> None:
    """未知实体、英文键和空值均被拒绝，原始实体不受影响。"""
    schema = load_schema()
    entities = validate_entities({"entities": [reservoir()]}, schema, TEXT).records
    original = copy.deepcopy(entities)
    result = validate_attribute_updates({"entity_attributes": [
        {"entity_id": "entity_0", "attribute": {"孔隙度": "8%"}},
        {"entity_id": entities[0]["id"], "attribute": {"porosity": 8, "厚度": None}},
    ]}, schema, TEXT, entities)
    assert result.has_errors
    assert not result.records
    assert {issue["code"] for issue in result.issues} >= {
        "invalid_attribute_entity", "invalid_attribute_name", "invalid_attribute_value"}
    assert entities == original


def test_legacy_attribute_patch_is_rejected() -> None:
    """旧 attributes 属性补丁不能进入新的开放属性输出。"""
    schema = load_schema()
    entities = validate_entities({"entities": [reservoir()]}, schema, TEXT).records
    result = validate_attribute_updates({"entity_attributes": [
        {"entity_id": entities[0]["id"], "attributes": {"porosity_pct": 8}}
    ]}, schema, TEXT, entities)
    assert result.has_errors and not result.records
