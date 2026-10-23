"""验证并合并实体的中文开放属性补丁，不要求逐项属性证据。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..schema_loader import MarkdownSchema
from .validator import ValidationResult, _array, _open_attributes


def merge_attribute_updates(
    entities: list[dict[str, Any]], updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按实体 ID 合并已校验属性，保留实体身份与已有属性值。"""
    merged = deepcopy(entities)
    entity_map = {entity["id"]: entity for entity in merged}
    for update in updates:
        entity_map[update["entity_id"]].setdefault("attribute", {}).update(update["attribute"])
    return merged


def validate_attribute_updates(
    payload: Any, schema: MarkdownSchema, text: str, entities: list[dict[str, Any]],
) -> ValidationResult:
    """验证中文开放属性补丁，拒绝未知实体、无效值和覆盖冲突。"""
    result = ValidationResult()
    entity_map = {entity["id"]: deepcopy(entity.get("attribute", {})) for entity in entities}
    if isinstance(payload, dict) and set(payload) - {"entity_attributes"}:
        result.add("invalid_attribute_response", "entity_attributes", "属性阶段只能返回 entity_attributes")
    for index, raw in enumerate(_array(payload, "entity_attributes", result)):
        location = f"entity_attributes[{index}]"
        if not isinstance(raw, dict) or set(raw) - {"entity_id", "attribute"}:
            result.add("invalid_attribute_update", location, "属性补丁只允许 entity_id 和 attribute")
            result.rejected_count += 1
            continue
        identifier = raw.get("entity_id")
        if not isinstance(identifier, str) or identifier not in entity_map:
            result.add("invalid_attribute_entity", location, "entity_id 必须引用已有实体")
            result.rejected_count += 1
            continue
        clean = _open_attributes(raw.get("attribute", {}), result, location)
        update: dict[str, Any] = {}
        for key, value in clean.items():
            if key in entity_map[identifier]:
                if entity_map[identifier][key] != value:
                    result.add("attribute_conflict", f"{location}.attribute.{key}", "补充值与已有值冲突，保留已有值", "warning")
                    result.rejected_count += 1
            else:
                update[key] = value
        if update:
            result.records.append({"entity_id": identifier, "attribute": update})
            entity_map[identifier].update(update)
    return result
