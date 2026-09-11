"""地图空间中间结果规范化与 Graph 公开入口。

本文件连接分阶段 VLM 输出、图例落地和确定性关系装配器。
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

from model import Graph

from ..schema_models import ImageExtractionTask
from .assembler import DeterministicRelationAssembler
from .geometry import normalize_geometry
from .relation_schema import RELATION_ZH, TYPE_ZH, canonical_entity_type, slug
from .stages import LegendGrounder


def _confidence(value: Any, default: float = 0.7) -> float:
    """把置信度规范到 0~1。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


def _items(value: Any) -> list[Any]:
    """只接受列表，避免把字符串错误拆成记录。"""

    return list(value) if isinstance(value, list) else []


def _normalize_map(task: ImageExtractionTask, value: Any) -> dict[str, Any]:
    """规范地图根实体并补充分类信息。"""

    raw = dict(value) if isinstance(value, Mapping) else {}
    name = str(raw.get("name") or raw.get("title") or raw.get("theme") or task.caption or "地图与平面空间图")
    attributes = dict(raw.get("attributes")) if isinstance(raw.get("attributes"), Mapping) else {}
    for key, item in raw.items():
        if key not in {"id", "name", "title", "theme", "type", "type_zh", "attributes", "evidence", "provenance", "confidence"}:
            attributes.setdefault(key, deepcopy(item))
    if raw.get("theme"):
        attributes.setdefault("description", str(raw["theme"]))
    attributes.setdefault("classification_code", task.classification_code)
    confidence = _confidence(raw.get("confidence", attributes.get("confidence")))
    attributes["confidence"] = confidence
    return {
        "id": str(raw.get("id") or "map"),
        "name": name,
        "type": "map_spatial",
        "type_zh": str(raw.get("type_zh") or task.classification_type or "地图与平面空间图"),
        "attributes": attributes,
        "evidence": str(raw.get("evidence") or raw.get("provenance") or task.caption or "整幅图"),
        "confidence": confidence,
    }


def _normalize_entity(value: Mapping[str, Any], index: int) -> dict[str, Any]:
    """规范实体类型、属性、证据与几何，同时保留未知原类型。"""

    raw = deepcopy(dict(value))
    canonical_type, original_type = canonical_entity_type(raw.get("type"))
    attributes = dict(raw.get("attributes")) if isinstance(raw.get("attributes"), Mapping) else {}
    if canonical_type == "UNMAPPED" and original_type:
        attributes.setdefault("raw_type", original_type)
    geometry = normalize_geometry(raw.get("geometry"))
    for key, item in raw.items():
        if key not in {"id", "name", "type", "type_zh", "aliases", "attributes", "geometry", "evidence", "provenance", "confidence"}:
            attributes.setdefault(key, deepcopy(item))
    confidence = _confidence(raw.get("confidence", attributes.get("confidence")))
    attributes["confidence"] = confidence
    result = {
        "id": str(raw.get("id") or f"entity_{index:03d}"),
        "name": str(raw.get("name") or f"未命名对象{index}"),
        "type": canonical_type,
        "type_zh": str(raw.get("type_zh") or TYPE_ZH.get(canonical_type) or "未映射空间对象"),
        "aliases": [str(item) for item in _items(raw.get("aliases")) if str(item).strip()],
        "attributes": attributes,
        "evidence": str(raw.get("evidence") or raw.get("provenance") or ""),
        "confidence": confidence,
    }
    if geometry.get("kind") != "unknown":
        result["geometry"] = geometry
    return result


def _normalize_legends(value: Any) -> list[dict[str, Any]]:
    """规范图例 ID、语义类型与视觉编码。"""

    legends: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, item in enumerate(_items(value), start=1):
        if not isinstance(item, Mapping):
            continue
        record = deepcopy(dict(item))
        legend_id = str(record.get("legend_id") or record.get("id") or f"legend_{index:03d}")
        base_id = legend_id
        suffix = 2
        while legend_id in used_ids:
            legend_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(legend_id)
        canonical_type, raw_type = canonical_entity_type(record.get("semantic_type") or record.get("type"))
        record["legend_id"] = legend_id
        record["label"] = str(record.get("label") or record.get("name") or legend_id)
        record["semantic_type"] = canonical_type
        record["type_zh"] = str(record.get("type_zh") or TYPE_ZH.get(canonical_type) or "未映射空间对象")
        record["visual_encoding"] = dict(record.get("visual_encoding")) if isinstance(record.get("visual_encoding"), Mapping) else {}
        record["confidence"] = _confidence(record.get("confidence"))
        if canonical_type == "UNMAPPED" and raw_type:
            record["raw_type"] = raw_type
        legends.append(record)
    return legends


def _normalize_spatial_mapping(value: Any, entities: list[dict[str, Any]]) -> dict[str, Any]:
    """规范空间候选、优先级、候选类型与锚点。"""

    raw = deepcopy(dict(value)) if isinstance(value, Mapping) else {}
    candidates: list[dict[str, Any]] = []
    for item in _items(raw.get("candidate_entities") or raw.get("spatial_relation_candidates")):
        if isinstance(item, Mapping):
            entity_id = str(item.get("entity_id") or item.get("id") or "")
            if entity_id:
                candidates.append({**dict(item), "entity_id": entity_id, "priority": _confidence(item.get("priority"), 0.5)})
        elif str(item).strip():
            candidates.append({"entity_id": str(item), "priority": 0.5})
    for entity in entities:
        attributes = entity.get("attributes") or {}
        if attributes.get("spatial_mapping_candidate"):
            candidates.append({
                "entity_id": entity["id"],
                "priority": _confidence(attributes.get("spatial_mapping_priority"), 0.5),
            })
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        by_id.setdefault(candidate["entity_id"], candidate)
    candidate_types: list[str] = []
    for raw_type in raw.get("candidate_entity_types") or ["place"]:
        canonical, _ = canonical_entity_type(raw_type)
        if canonical != "UNMAPPED":
            candidate_types.append(canonical)
    return {
        "enabled": bool(raw.get("enabled", True)),
        "candidate_entity_types": list(dict.fromkeys(candidate_types or ["place"])),
        "candidate_entities": list(by_id.values()),
        "spatial_anchor": str(raw.get("spatial_anchor") or raw.get("anchor_entity") or ""),
    }


def _normalize_relations(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """合并新旧关系候选字段，最终统一为小写候选记录。"""

    values = [
        *_items(payload.get("relation_candidates")),
        *_items(payload.get("semantic_relations")),
        *_items(payload.get("spatial_relations")),
    ]
    relations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in values:
        if not isinstance(raw, Mapping):
            continue
        item = deepcopy(dict(raw))
        source_id = str(item.get("source_id") or item.get("source") or "")
        target_id = str(item.get("target_id") or item.get("target") or "")
        relation_type = slug(item.get("type") or item.get("relation"))
        key = (source_id, relation_type, target_id)
        if not all(key) or key in seen:
            continue
        seen.add(key)
        item.update({
            "source_id": source_id,
            "target_id": target_id,
            "type": relation_type,
            "evidence": str(item.get("evidence") or ""),
            "confidence": _confidence(item.get("confidence")),
        })
        relations.append(item)
    return relations


def normalize_map_spatial_result(task: ImageExtractionTask, payload: Mapping[str, Any]) -> dict[str, Any]:
    """把三阶段或兼容旧字段的响应规范为 map_spatial.v3 中间结果。"""

    result = deepcopy(dict(payload))
    entities = [
        _normalize_entity(item, index)
        for index, item in enumerate(_items(result.get("entities")), start=1)
        if isinstance(item, Mapping)
    ]
    normalized = {
        "schema_version": "map_spatial.v3",
        "map": _normalize_map(task, result.get("map")),
        "legend_items": _normalize_legends(result.get("legend_items")),
        "entities": entities,
        "legend_bindings": [dict(item) for item in _items(result.get("legend_bindings")) if isinstance(item, Mapping)],
        "relation_candidates": _normalize_relations(result),
        "spatial_mapping": _normalize_spatial_mapping(
            result.get("spatial_mapping") or {"spatial_relation_candidates": result.get("spatial_relation_candidates")},
            entities,
        ),
        "uncertainties": [str(item) for item in _items(result.get("uncertainties")) if str(item).strip()],
    }
    return LegendGrounder().ground(normalized)


class MapSpatialGraphBuilder:
    """公开的确定性地图 Graph 构建器。"""

    def __init__(self, task: ImageExtractionTask, visual: Mapping[str, Any]) -> None:
        """保存任务，并确保调用方输入先经过规范化。"""

        self.task = task
        self.visual = normalize_map_spatial_result(task, visual)

    def build(self) -> Graph:
        """调用最终装配器生成 Graph。"""

        return DeterministicRelationAssembler(self.task, self.visual).build()


def build_map_spatial_graph(task: ImageExtractionTask, visual: Mapping[str, Any]) -> Graph:
    """规范化地图中间结果并确定性装配统一 Graph。"""

    return MapSpatialGraphBuilder(task, visual).build()


__all__ = [
    "RELATION_ZH",
    "TYPE_ZH",
    "MapSpatialGraphBuilder",
    "build_map_spatial_graph",
    "normalize_map_spatial_result",
]
