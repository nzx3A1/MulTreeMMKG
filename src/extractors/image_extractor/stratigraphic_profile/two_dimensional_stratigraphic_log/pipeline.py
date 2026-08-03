"""二维平面地层—测井图的结构校验和确定性空间关系流水线。"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from ...schema_models import ImageExtractionTask


TWO_DIMENSIONAL_STRATIGRAPHIC_LOG_SCHEMA_VERSION = (
    "two_dimensional_stratigraphic_log.v1"
)

ALLOWED_RELATION_TYPES = frozenset(
    {
        "part_of",
        "contains",
        "directly_overlies",
        "directly_underlies",
        "intersects",
        "cuts_through",
        "offsets",
        "correlates_with",
        "lateral_transition_to",
        "tracks_along",
        "adjusted_from",
        "located_in",
        "bounded_by",
        "has_lithology",
        "has_log_curve",
        "contains_reservoir",
        "contains_shale",
        "characterizes",
        "related_to",
    }
)

INVERSE_RELATIONS = {
    "part_of": "contains",
    "contains": "part_of",
    "directly_overlies": "directly_underlies",
    "directly_underlies": "directly_overlies",
}


def is_two_dimensional_stratigraphic_log_payload(payload: Mapping[str, Any]) -> bool:
    """中文说明：只接受二维专用版本、实体数组和关系数组齐全的模型结果。"""

    return (
        str(payload.get("schema_version") or "")
        == TWO_DIMENSIONAL_STRATIGRAPHIC_LOG_SCHEMA_VERSION
        and isinstance(payload.get("entities"), list)
        and isinstance(payload.get("relations"), list)
    )


def _items(value: Any) -> list[Any]:
    """中文说明：仅把真实数组转换为列表，避免字符串被错误按字符处理。"""

    return list(value) if isinstance(value, list) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    """中文说明：把非对象值降级为空对象，便于集中生成可审计的丢弃记录。"""

    return value if isinstance(value, Mapping) else {}


def _confidence(value: Any, default: float = 0.8) -> float:
    """中文说明：把模型置信度限制在零到一范围。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


def _optional_int(value: Any) -> int | None:
    """中文说明：规范可选顺序字段，无效值保留为空而不是猜测顺序。"""

    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_bbox(value: Any) -> list[float]:
    """中文说明：校验千分制边界框并裁剪轻微越界坐标，完全无效时返回空数组。"""

    if not isinstance(value, list) or len(value) != 4:
        return []
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return []
    x0, y0, x1, y1 = (
        max(0.0, min(1000.0, coordinate)) for coordinate in (x0, y0, x1, y1)
    )
    if x1 <= x0 or y1 <= y0:
        return []
    return [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]


class TwoDimensionalStratigraphicLogPipeline:
    """把二维 VLM JSON 规范为可由 Graph 装配器直接消费的中间结果。"""

    def run(
        self,
        task: ImageExtractionTask,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """中文说明：规范实体后，从层序和横向组确定性生成上下、左右及相邻关系。"""

        if not is_two_dimensional_stratigraphic_log_payload(payload):
            raise ValueError(
                "二维平面地层—测井图响应必须符合 two_dimensional_stratigraphic_log.v1"
            )

        dropped: list[dict[str, Any]] = []
        uncertainties = [
            str(item).strip()
            for item in _items(payload.get("uncertainties"))
            if str(item).strip()
        ]
        entities = self._normalize_entities(payload, dropped, uncertainties)
        entity_by_id = {item["id"]: item for item in entities}
        if not entity_by_id:
            raise ValueError("二维平面地层—测井图没有可用实体")

        sequences = self._normalize_sequences(payload, entity_by_id, dropped)
        spatial_groups = self._normalize_spatial_groups(payload, entity_by_id, dropped)

        relation_by_signature: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._add_parent_relations(entities, entity_by_id, relation_by_signature, dropped)
        self._add_model_relations(
            payload,
            entity_by_id,
            relation_by_signature,
            dropped,
        )
        self._derive_vertical_relations(
            sequences,
            relation_by_signature,
        )

        relations = list(relation_by_signature.values())
        connected_entity_ids = {
            str(endpoint)
            for relation in relations
            for endpoint in (relation["source_id"], relation["target_id"])
        }
        pruned_isolated_entities = [
            {
                "id": entity["id"],
                "name": entity["name"],
                "type": entity["type"],
                "reason": "no_domain_relation",
            }
            for entity in entities
            if entity["id"] not in connected_entity_ids
        ]
        # 中文说明：Chunk 根节点由 Graph 阶段创建；其余无业务边实体在这里删除，避免只靠根包含边伪装成非孤立节点。
        entities = [
            entity for entity in entities if entity["id"] in connected_entity_ids
        ]
        retained_entity_ids = {entity["id"] for entity in entities}
        sequences = [
            {
                **sequence,
                "ordered_unit_ids_top_to_bottom": [
                    member_id
                    for member_id in sequence["ordered_unit_ids_top_to_bottom"]
                    if member_id in retained_entity_ids
                ],
            }
            for sequence in sequences
        ]
        sequences = [
            sequence
            for sequence in sequences
            if len(sequence["ordered_unit_ids_top_to_bottom"]) >= 2
        ]
        spatial_groups = [
            {
                **group,
                "ordered_member_ids_left_to_right": [
                    member_id
                    for member_id in group["ordered_member_ids_left_to_right"]
                    if member_id in retained_entity_ids
                ],
            }
            for group in spatial_groups
        ]
        spatial_groups = [
            group
            for group in spatial_groups
            if len(group["ordered_member_ids_left_to_right"]) >= 2
        ]
        vertical_types = {"directly_overlies", "directly_underlies"}
        return {
            "schema_version": TWO_DIMENSIONAL_STRATIGRAPHIC_LOG_SCHEMA_VERSION,
            "diagram_type": str(payload.get("diagram_type") or "other_2d"),
            "diagram_name": str(payload.get("diagram_name") or task.caption or "二维平面地层—测井图"),
            "coordinate_system": dict(_mapping(payload.get("coordinate_system"))),
            "entities": entities,
            "stratigraphic_sequences": sequences,
            "spatial_groups": spatial_groups,
            "relations": relations,
            "quality": {
                "entity_count": len(entities),
                "relation_count": len(relations),
                "vertical_relation_count": sum(
                    item["type"] in vertical_types for item in relations
                ),
                "horizontal_relation_count": 0,
                "explicit_relation_count": sum(
                    bool(item.get("explicit")) for item in relations
                ),
                "derived_relation_count": sum(
                    not bool(item.get("explicit")) for item in relations
                ),
                "dropped_relations": dropped,
                "pruned_isolated_entity_count": len(pruned_isolated_entities),
                "pruned_isolated_entities": pruned_isolated_entities,
                "uncertainties": uncertainties,
            },
        }

    @staticmethod
    def _normalize_entities(
        payload: Mapping[str, Any],
        dropped: list[dict[str, Any]],
        uncertainties: list[str],
    ) -> list[dict[str, Any]]:
        """中文说明：统一实体、父层、千分制坐标和证据字段，并删除重复或无名实体。"""

        entities: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, raw in enumerate(_items(payload.get("entities"))):
            item = _mapping(raw)
            local_id = str(item.get("id") or "").strip()
            name = str(item.get("name") or "").strip()
            if not local_id or not name:
                dropped.append(
                    {"kind": "entity", "index": index, "reason": "missing_id_or_name"}
                )
                continue
            if local_id in seen_ids:
                dropped.append(
                    {"kind": "entity", "id": local_id, "reason": "duplicate_entity_id"}
                )
                continue
            seen_ids.add(local_id)
            position = _mapping(item.get("position"))
            evidence = str(item.get("evidence") or "").strip()
            if not evidence:
                evidence = f"图中可见实体标签：{name}"
                uncertainties.append(f"实体 {name} 的模型证据为空，已回退为图中标签证据")
            raw_attributes = _mapping(item.get("attributes"))
            entities.append(
                {
                    "id": local_id,
                    "name": name,
                    "type": str(item.get("type") or "other").strip() or "other",
                    "parent_id": str(item.get("parent_id") or "").strip(),
                    "position": {
                        "bbox": _normalize_bbox(position.get("bbox")),
                        "x_order": _optional_int(position.get("x_order")),
                        "y_order": _optional_int(position.get("y_order")),
                        "top_value": position.get("top_value"),
                        "bottom_value": position.get("bottom_value"),
                        "unit": str(position.get("unit") or ""),
                    },
                    "attributes": dict(raw_attributes),
                    "evidence": evidence,
                    "confidence": _confidence(item.get("confidence")),
                }
            )
        return entities

    @staticmethod
    def _normalize_sequences(
        payload: Mapping[str, Any],
        entity_by_id: Mapping[str, Mapping[str, Any]],
        dropped: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """中文说明：校验每条从上到下序列的端点并去除重复成员。"""

        sequences: list[dict[str, Any]] = []
        for index, raw in enumerate(_items(payload.get("stratigraphic_sequences"))):
            item = _mapping(raw)
            member_ids = list(
                dict.fromkeys(
                    str(value).strip()
                    for value in _items(item.get("ordered_unit_ids_top_to_bottom"))
                    if str(value).strip()
                )
            )
            unknown = [member_id for member_id in member_ids if member_id not in entity_by_id]
            if unknown:
                dropped.append(
                    {
                        "kind": "stratigraphic_sequence",
                        "index": index,
                        "reason": "unresolved_member",
                        "member_ids": unknown,
                    }
                )
                member_ids = [member_id for member_id in member_ids if member_id in entity_by_id]
            if len(member_ids) < 2:
                dropped.append(
                    {
                        "kind": "stratigraphic_sequence",
                        "index": index,
                        "reason": "fewer_than_two_members",
                    }
                )
                continue
            sequences.append(
                {
                    "id": str(item.get("id") or f"sequence_{index}"),
                    "context_id": str(item.get("context_id") or "regional"),
                    "ordered_unit_ids_top_to_bottom": member_ids,
                    "evidence": str(item.get("evidence") or "图中层位从上到下排列"),
                    "confidence": _confidence(item.get("confidence"), 0.85),
                    "derived_from_entity_order": False,
                }
            )
        return sequences

    @staticmethod
    def _infer_missing_sequences(
        entities: list[Mapping[str, Any]],
        sequences: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """中文说明：当模型只填写 y_order 时，按父层上下文补建不与显式序列重复的层序。"""

        covered = {
            str(member_id)
            for sequence in sequences
            for member_id in sequence.get("ordered_unit_ids_top_to_bottom", [])
        }
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for entity in entities:
            if entity.get("type") not in {"stratigraphic_unit", "horizon"}:
                continue
            if entity["id"] in covered or entity["position"].get("y_order") is None:
                continue
            groups[str(entity.get("parent_id") or "regional")].append(entity)
        inferred: list[dict[str, Any]] = []
        for context_id, members in groups.items():
            ordered = sorted(members, key=lambda item: int(item["position"]["y_order"]))
            if len(ordered) < 2:
                continue
            inferred.append(
                {
                    "id": f"inferred_sequence_{len(inferred)}",
                    "context_id": context_id,
                    "ordered_unit_ids_top_to_bottom": [item["id"] for item in ordered],
                    "evidence": "各地层实体的 position.y_order 表示图中从上到下顺序",
                    "confidence": min(
                        0.85, *(float(item.get("confidence") or 0.8) for item in ordered)
                    ),
                    "derived_from_entity_order": True,
                }
            )
        return inferred

    @staticmethod
    def _normalize_spatial_groups(
        payload: Mapping[str, Any],
        entity_by_id: Mapping[str, Mapping[str, Any]],
        dropped: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """中文说明：校验井、构造区和相带的从左到右排列组。"""

        groups: list[dict[str, Any]] = []
        for index, raw in enumerate(_items(payload.get("spatial_groups"))):
            item = _mapping(raw)
            member_ids = list(
                dict.fromkeys(
                    str(value).strip()
                    for value in _items(item.get("ordered_member_ids_left_to_right"))
                    if str(value).strip()
                )
            )
            unknown = [member_id for member_id in member_ids if member_id not in entity_by_id]
            if unknown:
                dropped.append(
                    {
                        "kind": "spatial_group",
                        "index": index,
                        "reason": "unresolved_member",
                        "member_ids": unknown,
                    }
                )
                member_ids = [member_id for member_id in member_ids if member_id in entity_by_id]
            if len(member_ids) < 2:
                dropped.append(
                    {
                        "kind": "spatial_group",
                        "index": index,
                        "reason": "fewer_than_two_members",
                    }
                )
                continue
            groups.append(
                {
                    "id": str(item.get("id") or f"spatial_group_{index}"),
                    "kind": str(item.get("kind") or "other"),
                    "ordered_member_ids_left_to_right": member_ids,
                    "evidence": str(item.get("evidence") or "图中对象从左到右排列"),
                    "confidence": _confidence(item.get("confidence"), 0.85),
                    "derived_from_entity_order": False,
                }
            )
        return groups

    @staticmethod
    def _infer_missing_spatial_groups(
        entities: list[Mapping[str, Any]],
        groups: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """中文说明：当模型只填写 x_order 时，按同类实体补建横向顺序组。"""

        covered = {
            str(member_id)
            for group in groups
            for member_id in group.get("ordered_member_ids_left_to_right", [])
        }
        by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        eligible = {"well", "structure_zone", "sedimentary_facies"}
        for entity in entities:
            if entity.get("type") not in eligible:
                continue
            if entity["id"] in covered or entity["position"].get("x_order") is None:
                continue
            by_type[str(entity["type"])].append(entity)
        inferred: list[dict[str, Any]] = []
        for entity_type, members in by_type.items():
            ordered = sorted(members, key=lambda item: int(item["position"]["x_order"]))
            if len(ordered) < 2:
                continue
            inferred.append(
                {
                    "id": f"inferred_spatial_group_{entity_type}",
                    "kind": entity_type,
                    "ordered_member_ids_left_to_right": [item["id"] for item in ordered],
                    "evidence": "各实体的 position.x_order 表示图中从左到右顺序",
                    "confidence": min(
                        0.85, *(float(item.get("confidence") or 0.8) for item in ordered)
                    ),
                    "derived_from_entity_order": True,
                }
            )
        return inferred

    def _add_parent_relations(
        self,
        entities: list[Mapping[str, Any]],
        entity_by_id: Mapping[str, Mapping[str, Any]],
        relation_by_signature: dict[tuple[str, str, str], dict[str, Any]],
        dropped: list[dict[str, Any]],
    ) -> None:
        """中文说明：把实体 parent_id 转换为双向的属于和包含关系。"""

        for entity in entities:
            parent_id = str(entity.get("parent_id") or "")
            if not parent_id:
                continue
            if parent_id not in entity_by_id:
                dropped.append(
                    {
                        "source_id": entity["id"],
                        "type": "part_of",
                        "target_id": parent_id,
                        "reason": "unresolved_parent",
                    }
                )
                continue
            relation = {
                "source_id": entity["id"],
                "type": "part_of",
                "target_id": parent_id,
                "dimension": "stratigraphic",
                "explicit": True,
                "basis": "entity.parent_id",
                "evidence": entity["evidence"],
                "confidence": entity["confidence"],
            }
            self._upsert_with_inverse(relation_by_signature, relation)

    def _add_model_relations(
        self,
        payload: Mapping[str, Any],
        entity_by_id: Mapping[str, Mapping[str, Any]],
        relation_by_signature: dict[tuple[str, str, str], dict[str, Any]],
        dropped: list[dict[str, Any]],
    ) -> None:
        """中文说明：仅接收端点存在、类型受控且非自环的视觉关系。"""

        for index, raw in enumerate(_items(payload.get("relations"))):
            item = _mapping(raw)
            source_id = str(item.get("source_id") or item.get("source") or "").strip()
            target_id = str(item.get("target_id") or item.get("target") or "").strip()
            relation_type = str(item.get("type") or "").strip()
            reason = ""
            if source_id not in entity_by_id or target_id not in entity_by_id:
                reason = "unresolved_endpoint"
            elif source_id == target_id:
                reason = "self_relation"
            elif relation_type not in ALLOWED_RELATION_TYPES:
                reason = "unsupported_relation_type"
            if reason:
                dropped.append({**dict(item), "index": index, "reason": reason})
                continue
            relation = {
                "source_id": source_id,
                "type": relation_type,
                "target_id": target_id,
                "dimension": str(item.get("dimension") or "semantic"),
                "explicit": bool(item.get("explicit", True)),
                "basis": str(item.get("basis") or "图中关系标注"),
                "evidence": str(item.get("evidence") or "图中对象之间的可见关系"),
                "confidence": _confidence(item.get("confidence")),
            }
            self._upsert_with_inverse(relation_by_signature, relation)

    def _derive_vertical_relations(
        self,
        sequences: list[Mapping[str, Any]],
        relation_by_signature: dict[tuple[str, str, str], dict[str, Any]],
    ) -> None:
        """中文说明：只连接每条显式层序中相邻节点，生成直接上覆及其直接下伏逆向边。"""

        for sequence in sequences:
            member_ids = list(sequence["ordered_unit_ids_top_to_bottom"])
            for upper_id, lower_id in zip(member_ids, member_ids[1:]):
                relation = {
                    "source_id": upper_id,
                    "type": "directly_overlies",
                    "target_id": lower_id,
                    "dimension": "stratigraphic",
                    "explicit": False,
                    "basis": "adjacent_in_ordered_unit_ids_top_to_bottom",
                    "evidence": str(sequence["evidence"]),
                    "confidence": _confidence(sequence.get("confidence"), 0.85),
                    "context_id": sequence.get("context_id"),
                    "sequence_id": sequence.get("id"),
                }
                self._upsert_with_inverse(relation_by_signature, relation)

    @staticmethod
    def _upsert(
        relation_by_signature: dict[tuple[str, str, str], dict[str, Any]],
        relation: Mapping[str, Any],
    ) -> None:
        """中文说明：按三元组去重，显式视觉关系优先覆盖程序推导关系。"""

        normalized = dict(relation)
        signature = (
            str(normalized["source_id"]),
            str(normalized["type"]),
            str(normalized["target_id"]),
        )
        existing = relation_by_signature.get(signature)
        if existing is None or (
            bool(normalized.get("explicit")) and not bool(existing.get("explicit"))
        ):
            relation_by_signature[signature] = normalized

    def _upsert_with_inverse(
        self,
        relation_by_signature: dict[tuple[str, str, str], dict[str, Any]],
        relation: Mapping[str, Any],
    ) -> None:
        """中文说明：写入方向关系，并为有稳定逆类型的关系同步生成反向边。"""

        self._upsert(relation_by_signature, relation)
        inverse_type = INVERSE_RELATIONS.get(str(relation.get("type") or ""))
        if not inverse_type:
            return
        inverse = dict(relation)
        inverse.update(
            {
                "source_id": relation["target_id"],
                "type": inverse_type,
                "target_id": relation["source_id"],
                "inverse_of": str(relation.get("type") or ""),
            }
        )
        self._upsert(relation_by_signature, inverse)
