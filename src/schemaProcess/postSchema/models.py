"""定义 postSchema 两级判定、新增概念和计划序列化所需的数据模型。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.schemaProcess.models import clamp_confidence


SCHEMA_NAME_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9]{1,63}$")


def _text(value: Any) -> str:
    """把外部值转换为去除首尾空白的文本。"""

    return str(value or "").strip()


def _text_list(value: Any) -> tuple[str, ...]:
    """把模型返回的单值或数组统一为去重后的非空文本元组。"""

    values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
    return tuple(dict.fromkeys(_text(item) for item in values if _text(item)))


@dataclass(frozen=True)
class NewConceptProposal:
    """描述一个需要新增到 Neo4j 的二级 ``EntityConcept``。"""

    schema: str
    zh_name: str
    category: str
    description: str
    examples: tuple[str, ...]
    source_raw_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """校验新概念满足 Schema 键名和必填语义字段约束。"""

        if not SCHEMA_NAME_PATTERN.fullmatch(self.schema):
            raise ValueError(f"新增概念 schema 必须是 PascalCase 英文标识：{self.schema!r}")
        if not self.zh_name or not self.category or not self.description or not self.examples:
            raise ValueError(f"新增概念字段不完整：{self.schema}")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        category: str | None = None,
        source_raw_types: Sequence[str] = (),
    ) -> "NewConceptProposal":
        """从 LLM 或计划 JSON 构造并规范化新增概念。"""

        return cls(
            schema=_text(value.get("schema")),
            zh_name=_text(value.get("zhName") or value.get("zh_name")),
            category=_text(category or value.get("category")),
            description=_text(value.get("description")),
            examples=_text_list(value.get("examples")),
            source_raw_types=_text_list(value.get("source_raw_types") or source_raw_types),
        )

    def to_dict(self) -> dict[str, Any]:
        """输出与 Neo4j 属性及计划文件兼容的 JSON 对象。"""

        return {
            "schema": self.schema,
            "zhName": self.zh_name,
            "category": self.category,
            "description": self.description,
            "examples": list(self.examples),
            "source_raw_types": list(self.source_raw_types),
        }


@dataclass(frozen=True)
class PostSchemaDecision:
    """保存一个未映射 ``raw_type`` 的一级类别和二级概念判定。"""

    raw_type: str
    category: str
    action: str
    selected_schema: str | None
    confidence: float
    reason: str
    new_concept: NewConceptProposal | None = None

    def __post_init__(self) -> None:
        """保证复用与新增两种动作携带互斥且完整的数据。"""

        if self.action not in {"MAP_EXISTING", "CREATE_NEW"}:
            raise ValueError(f"不支持的 postSchema 动作：{self.action}")
        if self.action == "MAP_EXISTING" and (not self.selected_schema or self.new_concept is not None):
            raise ValueError(f"MAP_EXISTING 必须且只能提供 selected_schema：{self.raw_type}")
        if self.action == "CREATE_NEW" and (self.selected_schema or self.new_concept is None):
            raise ValueError(f"CREATE_NEW 必须且只能提供 new_concept：{self.raw_type}")

    @property
    def effective_schema(self) -> str:
        """返回该判定最终写入文档图的 Schema 英文类型。"""

        if self.selected_schema:
            return self.selected_schema
        if self.new_concept:
            return self.new_concept.schema
        raise RuntimeError(f"判定没有有效 Schema：{self.raw_type}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PostSchemaDecision":
        """从计划 JSON 恢复一个经过校验的实体类型判定。"""

        action = _text(value.get("action")).upper()
        raw_type = _text(value.get("raw_type"))
        category = _text(value.get("category"))
        proposal_value = value.get("new_concept")
        proposal = None
        if action == "CREATE_NEW" and isinstance(proposal_value, Mapping):
            proposal = NewConceptProposal.from_mapping(
                proposal_value,
                category=category,
                source_raw_types=(raw_type,),
            )
        return cls(
            raw_type=raw_type,
            category=category,
            action=action,
            selected_schema=_text(value.get("selected_schema")) or None,
            confidence=clamp_confidence(value.get("confidence")),
            reason=_text(value.get("reason")) or "模型未提供理由",
            new_concept=proposal,
        )

    def to_dict(self) -> dict[str, Any]:
        """输出可复用计划中的实体类型判定对象。"""

        return {
            "raw_type": self.raw_type,
            "category": self.category,
            "action": self.action,
            "selected_schema": self.selected_schema,
            "confidence": round(clamp_confidence(self.confidence), 6),
            "reason": self.reason,
            "new_concept": self.new_concept.to_dict() if self.new_concept else None,
        }


__all__ = ["NewConceptProposal", "PostSchemaDecision", "SCHEMA_NAME_PATTERN"]
