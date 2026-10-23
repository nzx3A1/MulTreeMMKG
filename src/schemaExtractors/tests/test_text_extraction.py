"""验证文本实体的中文开放属性、时间戳身份及关系端点。"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from src.schemaExtractors.schema_loader import DEFAULT_SCHEMA_PATH, load_schema
from src.schemaExtractors.textTextractor.extractor import SchemaTextExtractor
from src.schemaExtractors.textTextractor.pipeline import collect_text_chunks
from src.schemaExtractors.textTextractor.validator import validate_entities, validate_relations


TEXT = "靖边气田位于鄂尔多斯盆地，马家沟组白云岩储层孔隙度为8%。"


def entity(identifier: str, name: str, kind: str, **attribute: Any) -> dict[str, Any]:
    """构造含中文开放属性和实体原文证据的模型响应。"""
    return {"id": identifier, "name": name, "type": kind, "attribute": attribute, "provenance": TEXT}


class DynamicLLM:
    """按提示中的正式实体 ID 构造关系和属性响应。"""

    def __init__(self) -> None:
        """记录调用内容与阶段，供离线测试检查。"""
        self.messages: list[list[dict[str, str]]] = []

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """模拟模型读取上阶段生成的正式 ID，避免依赖临时编号。"""
        self.messages.append(copy.deepcopy(messages))
        prompt = json.loads(messages[1]["content"])
        if "candidate_relation_schema" in prompt:
            source, target = [item["id"] for item in prompt["entities"]]
            return {"relations": [{"type": "located_in", "source_id": source,
                                    "target_id": target, "attributes": {}, "provenance": TEXT}]}
        if "relations" in prompt:
            return {"entity_attributes": [{"entity_id": prompt["entities"][0]["id"],
                                            "attribute": {"开发状态": "已开发"}}]}
        return {"entities": [entity("temporary_a", "靖边气田", "OilGasField", 类型="气田"),
                             entity("temporary_b", "鄂尔多斯盆地", "Basin")]}


def test_schema_and_chunk_collection(tmp_path: Path) -> None:
    """使用实际本体和输入结构，确认类型及 Chunk 来源可追溯。"""
    schema = load_schema()
    assert len(schema.entities) == 20 and len(schema.relations) == 32
    assert schema.relations["located_in"].zh_name == "位于"
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"document": {"id": "doc", "sections": [
        {"id": "s", "chunks": [{"id": "s:text:1", "modality": "text", "text": TEXT}]}
    ]}}, ensure_ascii=False), encoding="utf-8")
    chunks = collect_text_chunks(json.loads(source.read_text(encoding="utf-8")))
    assert len(chunks) == 1 and chunks[0]["id"] == "s:text:1"


def test_broken_markdown_fails_closed(tmp_path: Path) -> None:
    """本体表格损坏时停止抽取，避免类型集合部分加载。"""
    path = tmp_path / "schema.md"
    path.write_text(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8").replace("##### `Basin`", "##### `Unknown`"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_schema(path)


def test_open_attributes_have_chinese_names_without_property_evidence() -> None:
    """开放属性不受固定字段限制，且不输出属性证据容器。"""
    payload = {"entities": [entity("a", "马家沟组白云岩储层", "Reservoir",
                                   孔隙度="8%", 厚度范围={"最小值": 100, "最大值": 900, "单位": "m"})]}
    result = validate_entities(payload, load_schema(), TEXT)
    assert not result.has_errors
    record = result.records[0]
    assert re.fullmatch(r"entity_\d{16,}", record["id"])
    assert record["attribute"]["厚度范围"]["最大值"] == 900
    assert "attributes" not in record and "attribute_provenance" not in record


def test_invalid_attribute_names_and_old_fields_are_reported() -> None:
    """英文属性名及旧固定字段被标为错误，合法中文属性仍可保留。"""
    raw = entity("a", "马家沟组白云岩储层", "Reservoir", 孔隙度="8%", porosity_pct=8)
    raw["attributes"] = {"porosity_pct": 8}
    raw["attribute_provenance"] = {"porosity_pct": TEXT}
    result = validate_entities({"entities": [raw]}, load_schema(), TEXT)
    assert result.has_errors
    assert result.records[0]["attribute"] == {"孔隙度": "8%"}
    assert {issue["code"] for issue in result.issues} >= {"invalid_attribute_name", "obsolete_entity_attribute_fields"}


def test_timestamp_ids_are_distinct_and_preserved_on_revalidation() -> None:
    """同一批实体获得不同时间戳 ID，检查点复核保持身份不变。"""
    schema = load_schema()
    payload = {"entities": [entity("a", "靖边气田", "OilGasField"),
                            entity("b", "鄂尔多斯盆地", "Basin")]}
    first = validate_entities(payload, schema, TEXT)
    ids = [item["id"] for item in first.records]
    assert len(set(ids)) == 2 and all(re.fullmatch(r"entity_\d{16,}", item) for item in ids)
    resumed = validate_entities({"entities": first.records}, schema, TEXT, preserve_ids=True)
    assert not resumed.has_errors and resumed.records == first.records


def test_relations_reference_timestamp_ids() -> None:
    """关系端点使用正式时间戳实体 ID，并保持原文证据。"""
    schema = load_schema()
    entities = validate_entities({"entities": [entity("a", "靖边气田", "OilGasField"),
                                               entity("b", "鄂尔多斯盆地", "Basin")]}, schema, TEXT).records
    result = validate_relations({"relations": [{"type": "located_in", "source_id": entities[0]["id"],
                                                "target_id": entities[1]["id"], "provenance": TEXT}]}, schema, TEXT, entities)
    assert not result.has_errors
    assert result.records[0]["source_id"] == entities[0]["id"]
    assert result.records[0]["target_id"] == entities[1]["id"]


def test_three_stage_extraction_uses_new_contract() -> None:
    """三阶段串行抽取的实体、关系和属性补充使用一致的正式 ID。"""
    client = DynamicLLM()
    graph = SchemaTextExtractor(llm_client=client).extract_chunk({"id": "s:text:1", "text": TEXT})
    assert graph["status"] == "completed"
    assert len(client.messages) == 3
    assert graph["entities"][0]["attribute"] == {"类型": "气田", "开发状态": "已开发"}
    assert graph["relations"][0]["source_id"] == graph["entities"][0]["id"]
    assert graph["validation"]["attribute_stage_completed"] is True
