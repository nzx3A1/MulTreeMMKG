"""加载 Stage 06 作用域、类型兼容和关系融合规则。

规则独立于算法代码，便于后续根据人工标注样本调整类型边界，而不用修改程序。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = PROJECT_ROOT / "config" / "multimodal_alignment_rules.json"


def _normalized_type(value: Any) -> str:
    """规范化配置和实体中的类型字符串。"""

    return str(value or "").strip().lower()


def _pair_set(values: Any) -> frozenset[frozenset[str]]:
    """把无向类型对数组转换为便于查询的不可变集合。"""

    result: set[frozenset[str]] = set()
    for value in values or []:
        if isinstance(value, list) and len(value) == 2:
            pair = frozenset(_normalized_type(item) for item in value if _normalized_type(item))
            if pair:
                result.add(pair)
    return frozenset(result)


@dataclass(frozen=True)
class AlignmentRules:
    """封装实体作用域、类型边界和关系限定字段规则。"""

    chunk_local_raw_types: frozenset[str]
    chunk_local_prefixes: tuple[str, ...]
    compatible_type_pairs: frozenset[frozenset[str]]
    incompatible_type_pairs: frozenset[frozenset[str]]
    identity_attribute_keys: frozenset[str]
    identifier_schema_types_zh: frozenset[str]
    abbreviation_suffixes: tuple[str, ...]
    name_replacements: dict[str, str]
    relation_evidence_attribute_keys: frozenset[str]
    absorb_self_loop_relations: frozenset[str]
    symmetric_relations: frozenset[str]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_RULES_PATH) -> "AlignmentRules":
        """从 JSON 文件加载规则并完成大小写规范化。"""

        rules_path = Path(path)
        with rules_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, Mapping):
            raise TypeError("多模态对齐规则必须是 JSON 对象")
        return cls(
            chunk_local_raw_types=frozenset(
                _normalized_type(item) for item in payload.get("chunk_local_raw_types", [])
            ),
            chunk_local_prefixes=tuple(
                _normalized_type(item) for item in payload.get("chunk_local_prefixes", [])
            ),
            compatible_type_pairs=_pair_set(payload.get("compatible_schema_type_zh")),
            incompatible_type_pairs=_pair_set(payload.get("incompatible_schema_type_zh")),
            identity_attribute_keys=frozenset(
                _normalized_type(item) for item in payload.get("identity_attribute_keys", [])
            ),
            identifier_schema_types_zh=frozenset(
                _normalized_type(item)
                for item in payload.get("identifier_schema_type_zh", [])
            ),
            abbreviation_suffixes=tuple(
                sorted(
                    (str(item).strip().lower() for item in payload.get("abbreviation_suffixes", [])),
                    key=lambda value: (-len(value), value),
                )
            ),
            name_replacements={
                str(key): str(value)
                for key, value in (payload.get("name_replacements") or {}).items()
            },
            relation_evidence_attribute_keys=frozenset(
                _normalized_type(item)
                for item in payload.get("relation_evidence_attribute_keys", [])
            ),
            absorb_self_loop_relations=frozenset(
                _normalized_type(item) for item in payload.get("absorb_self_loop_relations", [])
            ),
            symmetric_relations=frozenset(
                _normalized_type(item) for item in payload.get("symmetric_relations", [])
            ),
        )

    def entity_scope(self, entity: Mapping[str, Any]) -> str:
        """依据原始类型识别文档级实体或 Chunk 局部结构实体。"""

        candidates = {
            _normalized_type(entity.get("raw_type")),
            _normalized_type(entity.get("type")),
            _normalized_type(entity.get("schema_type")),
        }
        if any(candidate in self.chunk_local_raw_types for candidate in candidates if candidate):
            return "chunk_local"
        if any(
            candidate.startswith(prefix)
            for candidate in candidates
            for prefix in self.chunk_local_prefixes
            if candidate and prefix
        ):
            return "chunk_local"
        return "document_global"

    def type_compatibility(
        self,
        left_type_zh: str,
        right_type_zh: str,
        left_schema: str = "",
        right_schema: str = "",
        left_raw: str = "",
        right_raw: str = "",
    ) -> str:
        """返回 exact、compatible、incompatible 或 unknown 类型关系。"""

        left_zh = _normalized_type(left_type_zh)
        right_zh = _normalized_type(right_type_zh)
        if left_zh and right_zh:
            if left_zh == right_zh:
                return "exact"
            pair = frozenset((left_zh, right_zh))
            if pair in self.incompatible_type_pairs:
                return "incompatible"
            if pair in self.compatible_type_pairs:
                return "compatible"
            return "unknown"

        left_schema_value = _normalized_type(left_schema)
        right_schema_value = _normalized_type(right_schema)
        if left_schema_value and left_schema_value == right_schema_value:
            return "exact"
        left_raw_value = _normalized_type(left_raw)
        right_raw_value = _normalized_type(right_raw)
        if left_raw_value and left_raw_value == right_raw_value:
            return "exact"
        return "unknown"


__all__ = ["AlignmentRules", "DEFAULT_RULES_PATH"]
