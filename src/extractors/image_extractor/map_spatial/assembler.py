"""把地图中间结果确定性装配为统一知识图。

本文件负责关系白名单、几何确认、稀疏空间链、去重与 Graph 输出。
"""
from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from model import Entity, Graph, Relation
from model.base import SourceModality

from ..schema_models import ImageExtractionTask
from .geometry import build_spatial_chain, geometry_within, line_crosses_polygon, relation_matches_geometry
from .relation_schema import DIRECTION_RELATIONS, RELATION_ZH, RelationSchemaGenerator, slug


_GEOMETRY_VALIDATED_RELATIONS = {
    "located_in",
    "located_near",
    "near_boundary_of",
    "near_or_within",
    "outside_erosion_boundary",
    "adjacent_to",
    "surrounds",
    "contained_in",
    "overlaps",
    "along_boundary_of",
    "crosses",
}


def _confidence(value: Any, default: float = 0.7) -> float:
    """把置信度约束到 0~1。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


def _clean(value: Any) -> Any:
    """递归清理不可序列化的非有限浮点数。"""

    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _local_id(value: Any, fallback: str) -> str:
    """保留可读本地标识，并对空值生成回退标识。"""

    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "").strip()).strip("_")
    return normalized or fallback


class DeterministicRelationAssembler:
    """第六阶段：程序确定端点、拓扑、方向与最终关系集合。"""

    def __init__(self, task: ImageExtractionTask, state: Mapping[str, Any]) -> None:
        """保存单图任务与已规范化中间结果。"""

        self.task = task
        self.state = deepcopy(dict(state))
        self.entities: list[Entity] = []
        self.relations: list[Relation] = []
        self.entity_records: dict[str, dict[str, Any]] = {}
        self.graph_ids: dict[str, str] = {}
        self.relation_keys: set[tuple[str, str, str]] = set()
        self.dropped_relations: list[dict[str, Any]] = []

    def build(self) -> Graph:
        """装配实体、语义候选、几何关系与空间方向链。"""

        self._build_entities()
        spatial_ids, spatial_config = self._spatial_candidates()
        relation_schema = RelationSchemaGenerator().generate(
            list(self.entity_records.values()),
            spatial_candidate_ids=spatial_ids,
        )
        legend_binding_relation_count = self._add_legend_binding_relations()
        self._add_vlm_candidates(relation_schema, set(spatial_ids))
        self._add_program_geometry_relations(relation_schema)
        generated_edges = self._add_spatial_directions(spatial_ids, spatial_config)

        graph = Graph.from_chunk(
            document_id=self.task.document_id,
            chunk_id=self.task.chunk_id,
            modality=SourceModality.IMAGE,
            entities=self.entities,
            relations=self.relations,
            raw_response=self.state,
            stage="stage_04_image_map_spatial_extraction",
        )
        graph.metadata.extra.update({
            "status": "completed" if self.entities else "empty",
            "extractor_kind": "map_spatial",
            "extractor_name": "地图与平面空间抽取器",
            "image_id": self.task.image_id,
            "image_index": self.task.image_index,
            "image_path": self.task.image_path,
            "source_image_path": self.task.image_path,
            "classification_code": self.task.classification_code,
            "classification_type": self.task.classification_type,
            "model_called": True,
            "schema_version": "map_spatial.v3",
            "algorithm": "图例解析→实体定位→图例绑定→关系Schema→VLM语义候选→程序确定性建图",
            "legend_items": _clean(self.state.get("legend_items") or []),
            "legend_bindings": _clean(self.state.get("legend_bindings") or []),
            "relation_generation": {
                "legend_driven": True,
                "deterministic_mapping": True,
                "relation_schema_candidates": relation_schema["groups"],
                "legend_binding_mapping": {
                    "enabled": True,
                    "generated_by_program": True,
                    "binding_count": len(self.state.get("legend_bindings") or []),
                    "generated_relation_count": legend_binding_relation_count,
                },
                "spatial_direction_mapping": {
                    "enabled": bool(spatial_config.get("enabled", True)),
                    "candidate_selection_by_vlm": True,
                    "direction_calculated_by_program": True,
                    "directed": True,
                    "max_out_degree": 1,
                    "max_in_degree": 1,
                    "allow_inverse_duplicate": False,
                    "allow_cycle": False,
                    "candidate_entities": [self.entity_records[entity_id]["name"] for entity_id in spatial_ids],
                    "generated_edges": generated_edges,
                    "generated_edge_count": len(generated_edges),
                },
            },
            "dropped_relations": self.dropped_relations,
            "uncertainties": list(dict.fromkeys(str(item) for item in self.state.get("uncertainties") or [] if str(item).strip())),
        })
        return graph

    def _add_legend_binding_relations(self) -> int:
        """把已验证的图例绑定自动转换为图例节点与图片实体之间的关系。"""

        legend_by_id = {
            str(item.get("legend_id") or ""): item
            for item in self.state.get("legend_items") or []
            if isinstance(item, Mapping) and item.get("legend_id")
        }
        before_count = len(self.relations)
        for raw_binding in self.state.get("legend_bindings") or []:
            if not isinstance(raw_binding, Mapping):
                continue
            binding = dict(raw_binding)
            legend_id = str(binding.get("legend_id") or "")
            legend = legend_by_id.get(legend_id, {})
            target_id = str(binding.get("legend_entity_id") or legend.get("materialized_entity_id") or "")
            target = self.entity_records.get(target_id)
            if not target:
                continue
            for source_value in binding.get("entity_ids") or []:
                source_id = str(source_value)
                source = self.entity_records.get(source_id)
                if not source or source_id == target_id:
                    continue
                requested_relation = slug(binding.get("binding_relation") or legend.get("binding_relation"))
                if requested_relation in RELATION_ZH:
                    relation_type = requested_relation
                elif target.get("type") == "lithology" and source.get("type") in {"depositional_zone", "sedimentary_microfacies"}:
                    relation_type = "has_dominant_lithology"
                else:
                    relation_type = "instance_of"
                source_confidence = (source.get("attributes") or {}).get("confidence", source.get("confidence"))
                target_confidence = (target.get("attributes") or {}).get("confidence", target.get("confidence"))
                confidence = min(
                    _confidence(binding.get("confidence"), 1.0),
                    _confidence(source_confidence),
                    _confidence(target_confidence),
                )
                self._add_relation(source_id, relation_type, target_id, {
                    "attributes": {
                        "legend_id": legend_id,
                        "legend_label": str(legend.get("label") or target.get("name") or legend_id),
                        "visual_encoding": _clean(legend.get("visual_encoding") or {}),
                    },
                    "mapping_basis": "legend_grounding",
                    "confidence": confidence,
                    "evidence": (
                        f"图中实体“{source.get('name')}”的视觉编码与图例“"
                        f"{legend.get('label') or target.get('name')}”绑定。"
                    ),
                }, inferred=True)
        return len(self.relations) - before_count

    def _build_entities(self) -> None:
        """按示例契约生成小写类型实体和可读稳定 ID。"""

        records: list[dict[str, Any]] = []
        map_record = self.state.get("map")
        if isinstance(map_record, Mapping):
            records.append(dict(map_record))
        records.extend(dict(item) for item in self.state.get("entities") or [] if isinstance(item, Mapping))

        used_local_ids: set[str] = set()
        for index, raw in enumerate(records, start=1):
            local_id = _local_id(raw.get("id"), f"entity_{index:03d}")
            base_id = local_id
            suffix = 2
            while local_id in used_local_ids:
                local_id = f"{base_id}_{suffix}"
                suffix += 1
            used_local_ids.add(local_id)
            raw["id"] = local_id
            raw_type = str(raw.get("type") or "UNMAPPED")
            graph_id = f"{self.task.image_id}:map:entity:{local_id}"
            attributes = dict(raw.get("attributes")) if isinstance(raw.get("attributes"), Mapping) else {}
            if raw.get("geometry"):
                attributes["geometry"] = _clean(raw["geometry"])
            for key, value in raw.items():
                if key not in {"id", "name", "type", "type_zh", "aliases", "attributes", "geometry", "evidence", "provenance", "confidence"}:
                    attributes.setdefault(key, _clean(value))
            attributes["confidence"] = _confidence(raw.get("confidence", attributes.get("confidence")))
            name = str(raw.get("name") or local_id)
            evidence = str(raw.get("evidence") or raw.get("provenance") or self.task.caption or "整幅图")
            entity = Entity(
                id=graph_id,
                name=name,
                type=raw_type,
                type_zh=str(raw.get("type_zh") or "未映射空间对象"),
                aliases=[str(value) for value in raw.get("aliases") or [] if str(value).strip()],
                attributes=attributes,
                provenance=evidence,
                metadata={
                    "source_modality": "image",
                    "local_id": local_id,
                    "visual_evidence": evidence,
                },
            )
            self.entities.append(entity)
            self.entity_records[local_id] = {
                **raw,
                "id": local_id,
                "name": name,
                "type": raw_type,
                "attributes": attributes,
                "geometry": attributes.get("geometry") or {},
            }
            self.graph_ids[local_id] = graph_id

    def _spatial_candidates(self) -> tuple[list[str], dict[str, Any]]:
        """读取 VLM 候选并验证类型、实体引用和坐标。"""

        config = dict(self.state.get("spatial_mapping")) if isinstance(self.state.get("spatial_mapping"), Mapping) else {}
        if not bool(config.get("enabled", True)):
            return [], config
        allowed_types = {str(value) for value in config.get("candidate_entity_types") or ["place"]}
        raw_candidates = config.get("candidate_entities") or []
        candidate_ids: list[str] = []
        priorities: dict[str, float] = {}
        for raw in raw_candidates:
            if isinstance(raw, Mapping):
                entity_id = str(raw.get("entity_id") or "")
                priorities[entity_id] = _confidence(raw.get("priority"), 0.5)
            else:
                entity_id = str(raw)
            entity = self.entity_records.get(entity_id)
            if entity and entity.get("type") in allowed_types and entity.get("geometry"):
                candidate_ids.append(entity_id)
        for entity_id, entity in self.entity_records.items():
            attributes = entity.get("attributes") or {}
            if attributes.get("spatial_mapping_candidate") and entity.get("type") in allowed_types and entity.get("geometry"):
                candidate_ids.append(entity_id)
                priorities.setdefault(entity_id, _confidence(attributes.get("spatial_mapping_priority"), 0.5))
        candidate_ids = list(dict.fromkeys(candidate_ids))
        for entity_id in candidate_ids:
            self.entity_records[entity_id]["spatial_mapping_priority"] = priorities.get(entity_id, 0.5)
            self.entity_records[entity_id]["attributes"]["spatial_mapping_candidate"] = True
            self.entity_records[entity_id]["attributes"]["spatial_mapping_priority"] = priorities.get(entity_id, 0.5)
            graph_id = self.graph_ids[entity_id]
            entity = next(item for item in self.entities if item.id == graph_id)
            entity.attributes["spatial_mapping_candidate"] = True
            entity.attributes["spatial_mapping_priority"] = priorities.get(entity_id, 0.5)
        return candidate_ids, config

    def _add_vlm_candidates(self, relation_schema: Mapping[str, Any], spatial_ids: set[str]) -> None:
        """校验并装配 VLM 选择的白名单关系候选。"""

        _ = relation_schema
        for raw in self.state.get("relation_candidates") or []:
            if not isinstance(raw, Mapping):
                continue
            candidate = dict(raw)
            source_id = str(candidate.get("source_id") or candidate.get("source") or "")
            target_id = str(candidate.get("target_id") or candidate.get("target") or "")
            relation_type = slug(candidate.get("type") or candidate.get("relation"))
            source = self.entity_records.get(source_id)
            target = self.entity_records.get(target_id)
            reason = ""
            if not source or not target or source_id == target_id:
                reason = "unresolved_endpoint"
            elif relation_type in DIRECTION_RELATIONS:
                reason = "direction_must_be_calculated_by_program"
            elif not RelationSchemaGenerator.is_allowed(
                str(source.get("type")),
                str(target.get("type")),
                relation_type,
                both_spatial_candidates=source_id in spatial_ids and target_id in spatial_ids,
            ):
                reason = "relation_not_allowed_by_type_schema"
            elif relation_type in _GEOMETRY_VALIDATED_RELATIONS and not relation_matches_geometry(
                relation_type,
                source.get("geometry"),
                target.get("geometry"),
            ):
                reason = "geometry_not_confirmed"
            if reason:
                self.dropped_relations.append({
                    "source_id": source_id,
                    "type": relation_type,
                    "target_id": target_id,
                    "reason": reason,
                })
                continue
            self._add_relation(source_id, relation_type, target_id, candidate, inferred=False)

    def _add_program_geometry_relations(self, relation_schema: Mapping[str, Any]) -> None:
        """根据点落面和线穿面自动生成无需 VLM 判断的关系。"""

        for pair in relation_schema.get("pairs") or []:
            if not isinstance(pair, Mapping):
                continue
            source_id = str(pair.get("source_id") or "")
            target_id = str(pair.get("target_id") or "")
            source = self.entity_records.get(source_id)
            target = self.entity_records.get(target_id)
            if not source or not target:
                continue
            allowed = set(pair.get("allowed_relations") or [])
            if "located_in" in allowed and geometry_within(source.get("geometry"), target.get("geometry")):
                self._add_relation(source_id, "located_in", target_id, {
                    "mapping_basis": "point_or_geometry_in_polygon",
                    "evidence": "由实体几何确定：源对象位于目标区域内。",
                    "confidence": min(_confidence(source.get("confidence")), _confidence(target.get("confidence"))),
                }, inferred=True)
            if "crosses" in allowed and line_crosses_polygon(source.get("geometry"), target.get("geometry")):
                self._add_relation(source_id, "crosses", target_id, {
                    "mapping_basis": "line_polygon_intersection",
                    "evidence": "由实体几何确定：线对象穿过目标区域。",
                    "confidence": min(_confidence(source.get("confidence")), _confidence(target.get("confidence"))),
                }, inferred=True)

    def _add_spatial_directions(self, spatial_ids: Sequence[str], config: Mapping[str, Any]) -> list[dict[str, str]]:
        """构造无环且每节点至多一入一出的单向空间链。"""

        edges = build_spatial_chain(
            list(self.entity_records.values()),
            spatial_ids,
            anchor_id=str(config.get("spatial_anchor") or config.get("anchor_entity") or ""),
        )
        generated: list[dict[str, str]] = []
        for edge in edges:
            source_id, target_id, relation_type = edge["source_id"], edge["target_id"], edge["type"]
            self._add_relation(source_id, relation_type, target_id, {
                **edge,
                "direction": RELATION_ZH.get(relation_type, relation_type),
                "relation_group": "relative_position",
                "selection_mode": "sparse_spatial_chain",
                "evidence": (
                    f"根据图中实体中心位置，{self.entity_records[source_id]['name']}位于"
                    f"{self.entity_records[target_id]['name']}{RELATION_ZH.get(relation_type, relation_type)}方向。"
                ),
            }, inferred=True)
            generated.append({
                "source": str(self.entity_records[source_id]["name"]),
                "relation": relation_type,
                "target": str(self.entity_records[target_id]["name"]),
            })
        return generated

    def _add_relation(
        self,
        source_local_id: str,
        relation_type: str,
        target_local_id: str,
        raw: Mapping[str, Any],
        *,
        inferred: bool,
    ) -> None:
        """按端点与类型去重并生成示例一致的关系字段。"""

        key = (source_local_id, relation_type, target_local_id)
        if (
            key in self.relation_keys
            or source_local_id == target_local_id
            or source_local_id not in self.graph_ids
            or target_local_id not in self.graph_ids
        ):
            return
        self.relation_keys.add(key)
        source = self.entity_records[source_local_id]
        target = self.entity_records[target_local_id]
        attributes = dict(raw.get("attributes")) if isinstance(raw.get("attributes"), Mapping) else {}
        for name in ("mapping_basis", "direction", "relation_group", "selection_mode", "value_range", "values", "unit"):
            if raw.get(name) not in (None, ""):
                attributes[name] = _clean(raw[name])
        attributes.setdefault("mapping_basis", "deterministic_geometry" if inferred else "vlm_semantic_candidate")
        attributes["confidence"] = _confidence(raw.get("confidence"))
        relation_name = RELATION_ZH.get(relation_type, relation_type)
        relation_id = f"{self.task.image_id}:map:relation:r{len(self.relations) + 1:03d}"
        evidence = str(raw.get("evidence") or self.task.caption or "图中直接可见")
        self.relations.append(Relation(
            id=relation_id,
            type=relation_type,
            relation_name=relation_name,
            type_zh=relation_name,
            source_id=self.graph_ids[source_local_id],
            source_name=str(source["name"]),
            source_type=str(source["type"]),
            target_id=self.graph_ids[target_local_id],
            target_name=str(target["name"]),
            target_type=str(target["type"]),
            attributes=attributes,
            provenance=evidence,
            metadata={"source_modality": "image", "visual_evidence": evidence},
        ))


__all__ = ["DeterministicRelationAssembler"]
