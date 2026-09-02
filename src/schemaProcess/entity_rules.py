"""高精度、有优先级且可配置的实体规则映射。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from config.model_config import PROJECT_ROOT

from .models import SchemaSnapshot, clamp_confidence


DEFAULT_RULES_PATH = PROJECT_ROOT / "config" / "schema_alignment_rules.json"


def normalize_type_name(value: Any) -> str:
    """归一化开放抽取类型，兼容 snake_case、连字符和 Schema 驼峰名。"""

    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())


@dataclass(frozen=True)
class RuleMatch:
    schema_type: str
    confidence: float
    reason: str
    rule_id: str


@dataclass(frozen=True)
class _NameRule:
    rule_id: str
    pattern: re.Pattern[str]
    schema_type: str
    confidence: float


class EntityRuleMapper:
    """先匹配专业特异名称，再匹配可靠原始类型别名。"""

    def __init__(
        self,
        snapshot: SchemaSnapshot,
        rules_path: str | Path = DEFAULT_RULES_PATH,
    ) -> None:
        self.snapshot = snapshot
        self.rules_path = Path(rules_path)
        payload = json.loads(self.rules_path.read_text(encoding="utf-8"))
        self.raw_type_aliases = {
            normalize_type_name(raw_type): str(schema_type)
            for raw_type, schema_type in (payload.get("raw_type_aliases") or {}).items()
        }
        self.name_rules = [
            _NameRule(
                rule_id=str(item["id"]),
                pattern=re.compile(str(item["pattern"]), flags=re.IGNORECASE),
                schema_type=str(item["schema"]),
                confidence=clamp_confidence(item.get("confidence", 1.0)),
            )
            for item in payload.get("name_rules", [])
        ]

    def map_entity(self, entity: Mapping[str, Any]) -> RuleMatch | None:
        """返回第一个有效高精度命中；配置引用不存在的概念时忽略该规则。"""

        name = str(entity.get("name") or entity.get("official_name") or "").strip()
        for rule in self.name_rules:
            if rule.schema_type in self.snapshot.concepts_by_schema and rule.pattern.search(name):
                return RuleMatch(
                    schema_type=rule.schema_type,
                    confidence=rule.confidence,
                    reason=f"实体名称 {name!r} 命中优先级规则 {rule.rule_id!r}",
                    rule_id=rule.rule_id,
                )

        raw_type = entity.get("raw_type", entity.get("type"))
        normalized = normalize_type_name(raw_type)

        # 已经使用标准 Schema 名时直接接受，但仍保留原始 type/raw_type 字段。
        canonical_by_normalized = {
            normalize_type_name(schema): schema for schema in self.snapshot.concepts_by_schema
        }
        schema_type = canonical_by_normalized.get(normalized)
        if schema_type:
            return RuleMatch(
                schema_type=schema_type,
                confidence=1.0,
                reason=f"原始类型 {raw_type!r} 与 Schema 标识精确一致",
                rule_id="exact_schema_type",
            )

        schema_type = self.raw_type_aliases.get(normalized)
        if schema_type and schema_type in self.snapshot.concepts_by_schema:
            return RuleMatch(
                schema_type=schema_type,
                confidence=0.98,
                reason=f"原始类型 {raw_type!r} 命中高精度类型别名",
                rule_id="raw_type_alias",
            )
        return None


__all__ = ["DEFAULT_RULES_PATH", "EntityRuleMapper", "RuleMatch", "normalize_type_name"]
