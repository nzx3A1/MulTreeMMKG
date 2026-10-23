"""校验实体属性、原文证据、关系端点及部分可确定执行的本体业务约束。"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ..schema_loader import MarkdownSchema, RelationDefinition, resolve_type_name
from .evidence import resolve_provenance


VALIDATOR_VERSION = "schema_validator_v6_open_chinese_attributes"
RELATION_QUALIFIERS = {
    "spatial_scope_id", "stratigraphic_scope", "temporal_scope", "source_property", "target_property",
    "effect_direction", "polarity", "certainty", "attribution",
}
QUALIFIER_ENUMS = {
    "relation_basis": {"formal_hierarchy", "physical_inclusion"},
    "boundary_role": {"top", "bottom", "other"},
    "transition_axis": {"lateral", "vertical"},
    "polarity": {"positive", "negative"},
    "certainty": {"certain", "uncertain"},
}
HIERARCHIES = {"structural_part_of", "stratigraphic_part_of", "facies_part_of"}
_ENTITY_ID_LOCK = Lock()
_LAST_ENTITY_TIMESTAMP = 0


def _new_entity_id() -> str:
    """在并发 Chunk 中生成单调递增的纳秒时间戳实体 ID。"""
    global _LAST_ENTITY_TIMESTAMP
    with _ENTITY_ID_LOCK:
        _LAST_ENTITY_TIMESTAMP = max(time.time_ns(), _LAST_ENTITY_TIMESTAMP + 1)
        return f"entity_{_LAST_ENTITY_TIMESTAMP}"


@dataclass
class ValidationResult:
    """保存有效记录及被拒绝项的问题，便于重试和最终汇总。"""

    records: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    rejected_count: int = 0

    def add(self, code: str, location: str, message: str, severity: str = "error") -> None:
        """记录可定位的问题，避免只静默丢弃模型输出。"""
        self.issues.append({"severity": severity, "code": code, "location": location, "message": message})

    @property
    def has_errors(self) -> bool:
        """判断是否存在需要反馈给模型修复的错误。"""
        return any(issue["severity"] == "error" for issue in self.issues)


def _compact(text: str) -> str:
    """只归一化空白，不改写证据的标点、数字或否定词。"""
    return re.sub(r"\s+", "", text)


def _has_evidence(value: Any, text: str) -> bool:
    """确保证据是当前 Chunk 的非空连续原文，不能引用摘要。"""
    return isinstance(value, str) and bool(_compact(value)) and _compact(value) in _compact(text)


def _array(payload: Any, key: str, result: ValidationResult) -> list[Any]:
    """区分合法空抽取与无效 JSON、缺失键及错误容器类型。"""
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        result.add("invalid_response", key, f"响应必须是包含 {key} 数组的 JSON 对象")
        return []
    return payload[key]


def _valid_extension_value(value: Any) -> bool:
    """允许开放属性保留非空 JSON 值和嵌套区间，同时拒绝空值及非有限数。"""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return bool(value) and all(_valid_extension_value(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and all(isinstance(key, str) and bool(key.strip())
                                   and _valid_extension_value(item) for key, item in value.items())
    return False


def _open_attributes(raw: Any, result: ValidationResult, location: str) -> dict[str, Any]:
    """校验中文开放属性名称及非空 JSON 值，不要求逐项属性证据。"""
    if not isinstance(raw, dict):
        result.add("invalid_attribute", location, "attribute 必须为对象")
        return {}
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        item_location = f"{location}.attribute.{key}"
        if not isinstance(key, str) or key != key.strip() or not re.search(r"[\u3400-\u9fff]", key):
            result.add("invalid_attribute_name", item_location, "属性名称必须包含中文且不能有首尾空格")
        elif not _valid_extension_value(value):
            result.add("invalid_attribute_value", item_location, "属性值必须是非空、有限的 JSON 值")
        else:
            clean[key] = value
    return clean


def validate_entities(
    payload: Any, schema: MarkdownSchema, text: str, *, preserve_ids: bool = False,
) -> ValidationResult:
    """校验实体及中文开放属性，并为新实体生成唯一时间戳 ID。"""
    result = ValidationResult()
    identifiers: set[str] = set()
    signatures: set[str] = set()
    for index, raw in enumerate(_array(payload, "entities", result)):
        location = f"entities[{index}]"
        if not isinstance(raw, dict):
            result.add("invalid_entity", location, "实体必须为对象")
            result.rejected_count += 1
            continue
        identifier, name = raw.get("id"), raw.get("name")
        entity_type = resolve_type_name(raw.get("type"), schema.entities)
        if preserve_ids and (not isinstance(identifier, str) or not re.fullmatch(r"entity_\d+", identifier)):
            result.add("invalid_entity_id", location, "已有实体 id 必须采用 entity_时间戳 格式")
            result.rejected_count += 1
            continue
        if identifier is not None and (not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers):
            result.add("invalid_entity_id", location, "临时实体 id 不能为空或重复")
            result.rejected_count += 1
            continue
        if identifier is not None:
            identifiers.add(identifier)
        if not isinstance(name, str) or not name.strip() or not isinstance(entity_type, str) or entity_type not in schema.entities:
            result.add("invalid_entity_type_or_name", location, "实体名称为空或 type 不在核心实体白名单")
            result.rejected_count += 1
            continue
        provenance = resolve_provenance(raw, text)
        if not _has_evidence(provenance, text):
            result.add("invalid_provenance", location, "provenance 必须为当前 chunk 的连续原文")
            result.rejected_count += 1
            continue
        definition = schema.entities[entity_type]
        if "attributes" in raw or "attribute_provenance" in raw:
            result.add("obsolete_entity_attribute_fields", location, "实体属性只允许使用中文键的 attribute")
        attribute = _open_attributes(raw.get("attribute", {}), result, location)
        official = raw.get("official_name", raw.get("canonical_name", name))
        if not isinstance(official, str) or not official.strip():
            official = name
            result.add("missing_official_name", location, "规范名称为空，回退为原文名称", "warning")
        aliases = raw.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            result.add("invalid_aliases", location, "aliases 必须为字符串列表")
            aliases = []
        supported_aliases = list(dict.fromkeys(alias.strip() for alias in aliases if _has_evidence(alias, text)))
        if len(supported_aliases) != len(set(aliases)):
            result.add("unsupported_alias", location, "删除原文未出现的别名或空别名", "warning")
        entity = {
            "id": identifier, "name": name.strip(), "official_name": official.strip(),
            "type": definition.zh_name, "type_zh": definition.zh_name, "type_en": entity_type,
            "entity_kind": definition.kind,
            "aliases": supported_aliases, "attribute": attribute, "provenance": provenance.strip(),
        }
        signature = json.dumps({key: value for key, value in entity.items() if key not in {"id", "provenance"}}, sort_keys=True, ensure_ascii=False)
        if signature in signatures and not preserve_ids:
            result.add("duplicate_entity", location, "忽略属性及范围完全相同的重复实体", "warning")
            continue
        signatures.add(signature)
        if not preserve_ids:
            entity["id"] = _new_entity_id()
        result.records.append(entity)
    return result


def _qualifiers(
    raw: Any, relation: RelationDefinition, entity_map: dict[str, dict[str, Any]],
    text: str, result: ValidationResult, location: str,
) -> dict[str, str] | None:
    """校验关系必需限定词及范围引用，禁止虚构的范围实体 ID。"""
    if not isinstance(raw, dict):
        result.add("invalid_attributes", location, "关系 attributes 必须为对象")
        return None
    allowed = RELATION_QUALIFIERS | set(relation.required_qualifiers)
    if relation.name == "stratigraphic_part_of":
        allowed.add("spatial_scope_id")
    attributes: dict[str, str] = {}
    valid = True
    for key, value in raw.items():
        if key not in allowed or not isinstance(value, str) or not value.strip():
            result.add("invalid_qualifier", location, f"关系不支持限定词 {key} 或其值不是非空字符串")
            valid = False
        elif key in QUALIFIER_ENUMS and value not in QUALIFIER_ENUMS[key]:
            result.add("invalid_qualifier", location, f"限定词 {key} 不在允许枚举中")
            valid = False
        else:
            attributes[key] = value.strip()
    for key in relation.required_qualifiers:
        if key not in attributes:
            result.add("missing_relation_qualifier", location, f"关系 {relation.name} 缺少 {key}")
            valid = False
    scope = attributes.get("spatial_scope_id")
    if scope:
        scope_entity = entity_map.get(scope)
        if scope_entity is not None:
            scope_valid = scope_entity["type_en"] in {"Basin", "StructuralUnit", "StudyArea", "OilGasField", "Well"}
        else:
            scope_valid = not re.fullmatch(r"entity_\d+", scope) and _has_evidence(scope, text)
        if not scope_valid:
            result.add("invalid_spatial_scope", location, "范围须为已有空间实体 ID 或当前原文中的明确范围短语")
            valid = False
    return attributes if valid else None


def _business_error(
    relation: RelationDefinition, source: dict[str, Any], target: dict[str, Any],
    attributes: dict[str, str], records: list[dict[str, Any]],
) -> str:
    """检查层级方向、局部无环和单父约束；语义真伪仍需领域审核。"""
    name = relation.name
    source_attrs, target_attrs = source.get("attribute", {}), target.get("attribute", {})
    if name == "stratigraphic_part_of":
        source_scheme, target_scheme = source_attrs.get("地层划分体系"), target_attrs.get("地层划分体系")
        if source_scheme and target_scheme and source_scheme != target_scheme:
            if attributes.get("relation_basis") != "physical_inclusion" or not attributes.get("spatial_scope_id"):
                return "跨体系地层隶属须有 physical_inclusion 和 spatial_scope_id"
        elif attributes.get("relation_basis") == "formal_hierarchy":
            levels = {"formation": 0, "member": 1, "submember": 2, "bed": 3}
            left, right = source_attrs.get("地层级别"), target_attrs.get("地层级别")
            if left in levels and right in levels and levels[left] <= levels[right]:
                return "形式地层隶属须从较低层级指向较高层级"
    if name == "facies_part_of":
        levels = {"facies": 0, "subfacies": 1, "microfacies": 2}
        left, right = source_attrs.get("沉积相级别"), target_attrs.get("沉积相级别")
        if left in levels and right in levels and levels[left] <= levels[right]:
            return "相级隶属必须从微相/亚相指向更高层级"
    context_keys = ("spatial_scope_id", "stratigraphic_scope", "temporal_scope", "facies_hierarchy_scheme")
    existing = [record for record in records if record["type_en"] == name and all(
        record["attributes"].get(key) == attributes.get(key) for key in context_keys
    )]
    if name in {"sample_from_well", "facies_part_of"} or (
        name == "stratigraphic_part_of" and attributes.get("relation_basis") == "formal_hierarchy"
    ):
        if any(record["source_id"] == source["id"] and record["target_id"] != target["id"] for record in existing):
            return "同一范围或分类体系内违反单一来源/形式父级约束"
    if name in HIERARCHIES:
        adjacency: dict[str, list[str]] = {}
        for record in existing:
            adjacency.setdefault(record["source_id"], []).append(record["target_id"])
        pending, visited = [target["id"]], set()
        while pending:
            current = pending.pop()
            if current == source["id"]:
                return "层级关系不得形成有向环"
            if current not in visited:
                visited.add(current)
                pending.extend(adjacency.get(current, []))
    return ""


def validate_relations(
    payload: Any, schema: MarkdownSchema, text: str, entities: list[dict[str, Any]],
) -> ValidationResult:
    """只保留有合法实体端点、方向、原文证据和限定词的 Schema 关系。"""
    result = ValidationResult()
    entity_map = {entity["id"]: {**entity, "type_en": resolve_type_name(entity.get("type"), schema.entities)}
                  for entity in entities}
    signatures: set[str] = set()
    for index, raw in enumerate(_array(payload, "relations", result)):
        location = f"relations[{index}]"
        if not isinstance(raw, dict):
            result.add("invalid_relation", location, "关系必须为对象")
            result.rejected_count += 1
            continue
        relation_type = resolve_type_name(raw.get("type"), schema.relations)
        relation = schema.relations.get(relation_type) if relation_type is not None else None
        source_id, target_id = raw.get("source_id"), raw.get("target_id")
        source = entity_map.get(source_id) if isinstance(source_id, str) else None
        target = entity_map.get(target_id) if isinstance(target_id, str) else None
        if relation is None or source is None or target is None or source_id == target_id:
            result.add("invalid_relation_endpoint_or_type", location, "关系类型未知、端点不存在或关系自指")
            result.rejected_count += 1
            continue
        source_type = resolve_type_name(source.get("type"), schema.entities)
        target_type = resolve_type_name(target.get("type"), schema.entities)
        if source_type not in relation.source_types or target_type not in relation.target_types:
            result.add("invalid_relation_direction", location, "主宾实体类型不符合关系方向约束")
            result.rejected_count += 1
            continue
        provenance = resolve_provenance(raw, text)
        if not _has_evidence(provenance, text):
            result.add("invalid_provenance", location, "provenance 必须为当前 chunk 的连续原文")
            result.rejected_count += 1
            continue
        attributes = _qualifiers(raw.get("attributes", {}), relation, entity_map, text, result, location)
        if attributes is None:
            result.rejected_count += 1
            continue
        error = _business_error(relation, source, target, attributes, result.records)
        if error:
            result.add("relation_business_constraint", location, error)
            result.rejected_count += 1
            continue
        endpoints = sorted((source_id, target_id)) if relation.symmetric else [source_id, target_id]
        signature = json.dumps([relation_type, endpoints, attributes], sort_keys=True, ensure_ascii=False)
        if signature in signatures:
            result.add("duplicate_relation", location, "忽略同范围和限定条件下的重复关系", "warning")
            continue
        signatures.add(signature)
        result.records.append({
            "id": f"relation_{len(result.records) + 1}", "type": relation.zh_name,
            "relation_name": relation.zh_name, "type_zh": relation.zh_name,
            "type_en": relation.name, "relation_name_en": relation.name,
            "source_id": source_id, "source_name": source["name"],
            "source_type": schema.entities[source_type].zh_name, "source_type_en": source_type,
            "target_id": target_id, "target_name": target["name"],
            "target_type": schema.entities[target_type].zh_name, "target_type_en": target_type,
            "attributes": attributes, "provenance": provenance.strip(),
        })
    return result
