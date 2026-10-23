"""读取 schema.md 的实体、属性和有向关系，供提示词与校验器共用。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_SCHEMA_PATH = Path(__file__).with_name("schema.md")


def _codes(text: str) -> tuple[str, ...]:
    """提取 Markdown 行内代码中的机器字段或枚举值。"""
    return tuple(re.findall(r"`([^`]+)`", text))


@dataclass(frozen=True)
class PropertyDefinition:
    """保存一个属性的名称、数据类型、枚举和规范完整性要求。"""

    name: str
    description: str
    value_type: str
    required: bool = False
    enum: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntityDefinition:
    """保存核心实体定义及其可抽取属性。"""

    name: str
    zh_name: str
    kind: str
    description: str
    examples: str
    properties: dict[str, PropertyDefinition] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationDefinition:
    """保存关系的方向、端点类型、必需限定词及业务约束。"""

    name: str
    zh_name: str
    source_types: tuple[str, ...]
    target_types: tuple[str, ...]
    required_qualifiers: tuple[str, ...]
    description: str
    constraints: str
    symmetric: bool = False


def resolve_type_name(
    value: Any, definitions: dict[str, EntityDefinition] | dict[str, RelationDefinition],
) -> str | None:
    """将本体中的中文类型或英文机器名解析为唯一机器名，未知或歧义类型返回空值。"""
    if not isinstance(value, str):
        return None
    if value in definitions:
        return value
    matches = [definition.name for definition in definitions.values() if definition.zh_name == value]
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class MarkdownSchema:
    """保存同一次读取产生的 Schema 快照，防止提示词与校验规则漂移。"""

    entities: dict[str, EntityDefinition]
    relations: dict[str, RelationDefinition]
    common_properties: dict[str, PropertyDefinition]
    entity_rules: str
    measurement_rules: str
    fingerprint: str
    path: Path

    def candidate_relations(self, entities: list[dict[str, Any]]) -> list[RelationDefinition]:
        """只返回当前实体集中存在两个不同合法端点的关系定义。"""
        return [
            relation for relation in self.relations.values()
            if any(
                source["id"] != target["id"]
                and resolve_type_name(source.get("type"), self.entities) in relation.source_types
                and resolve_type_name(target.get("type"), self.entities) in relation.target_types
                for source in entities for target in entities
            )
        ]


def _table_rows(text: str) -> list[list[str]]:
    """读取简单 Markdown 表格，忽略标题行和分隔行。"""
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.startswith("|") and "`" in line
    ]


def _properties(text: str) -> dict[str, PropertyDefinition]:
    """从五列表格读取属性定义，枚举只接受规范中给出的机器值。"""
    result: dict[str, PropertyDefinition] = {}
    allowed_types = {"字符串", "字符串列表", "浮点数", "布尔值", "枚举", "对象"}
    for cells in _table_rows(text):
        if len(cells) != 5 or len(_codes(cells[0])) != 1:
            raise ValueError("Schema 属性表必须包含五列和唯一字段名")
        name = _codes(cells[0])[0]
        if cells[2] not in allowed_types or name in result:
            raise ValueError(f"不支持或重复的 Schema 属性定义：{name}")
        enum = _codes(cells[4]) if cells[2] == "枚举" else ()
        # 中文说明：公共字段使用普通文本列举枚举，不能只识别反引号。
        if cells[2] == "枚举" and not enum and " / " in cells[4]:
            enum = tuple(cells[4].split(" / "))
        result[name] = PropertyDefinition(name, cells[1], cells[2], cells[3] in {"必填", "规范要求"}, enum)
    return result


def _section(text: str, start: str, end: str | None = None) -> str:
    """按规范标题截取必要章节，缺失或重名时立即报错。"""
    if text.count(start) != 1 or (end is not None and text.count(end) != 1):
        raise ValueError(f"Schema 缺失或重复章节：{start}")
    section = text.split(start, 1)[1]
    return section.split(end, 1)[0] if end else section


def load_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> MarkdownSchema:
    """直接解析本目录规范；实体属性或关系定义不完整时拒绝静默降级。"""
    source = Path(path).resolve()
    text = source.read_text(encoding="utf-8-sig")
    entities: dict[str, EntityDefinition] = {}
    entity_table = _section(text, "## 2. 核心概念实体", "### 2.1 实体类型的合并边界")
    for cells in _table_rows(entity_table):
        if len(cells) != 6 or not _codes(cells[1]) or not _codes(cells[3]):
            raise ValueError("核心实体表结构不完整")
        name = _codes(cells[1])[0]
        if name in entities:
            raise ValueError(f"重复实体定义：{name}")
        entities[name] = EntityDefinition(name, cells[2], _codes(cells[3])[0], cells[4], cells[5])

    common = _properties(_section(text, "### 3.1 所有实体的公共字段", "### 3.2 按实体类型列出的专属属性"))
    property_text = _section(text, "### 3.2 按实体类型列出的专属属性", "### 3.3 属性建模说明")
    sections = re.split(r"(?m)^##### `([^`]+)`[^\n]*\n", property_text)
    property_names = set(sections[1::2])
    if property_names != set(entities) or len(sections[1::2]) != len(entities):
        raise ValueError("核心实体与属性表的类型不一致")
    for name, body in zip(sections[1::2], sections[2::2]):
        entities[name].properties.update(_properties(body))

    relation_table = _section(text, "### 4.1 关系总览", "### 4.2 每类关系的语义与约束")
    relation_details = _section(text, "### 4.2 每类关系的语义与约束", "## 6. 测量属性与单位规范")
    sections = re.split(r"(?m)^##### `([^`]+)`[^\n]*\n", relation_details)
    details = dict(zip(sections[1::2], sections[2::2]))
    relations: dict[str, RelationDefinition] = {}
    for cells in _table_rows(relation_table):
        if len(cells) != 5 or len(_codes(cells[0])) != 1:
            raise ValueError("关系总览表结构不完整")
        name = _codes(cells[0])[0]
        body = details.get(name, "")
        description = re.search(r"\*\*定义\*\*：([^\n]+)", body)
        constraints = re.search(r"\*\*业务约束\*\*：([^\n]+)", body)
        source_types, target_types = _codes(cells[2]), _codes(cells[3])
        if (name in relations or not description or not constraints or not source_types or not target_types
                or not set(source_types + target_types) <= set(entities)):
            raise ValueError(f"关系定义不完整、重复或引用未知实体：{name}")
        relations[name] = RelationDefinition(
            name, cells[1], source_types, target_types, _codes(cells[4]),
            description.group(1), constraints.group(1), "**对称性**：是" in body,
        )
    if not entities or not relations or set(details) != set(relations):
        raise ValueError("Schema 为空或关系总览与详细定义不一致")
    return MarkdownSchema(
        entities, relations, common,
        text.split("## 3. 实体属性", 1)[0] + "\n" + _section(text, "### 3.3 属性建模说明", "## 4. 实体关系"),
        _section(text, "## 6. 测量属性与单位规范"),
        hashlib.sha256(text.encode("utf-8")).hexdigest(), source,
    )
