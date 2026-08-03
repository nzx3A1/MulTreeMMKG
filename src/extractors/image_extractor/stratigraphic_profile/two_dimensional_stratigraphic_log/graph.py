"""二维平面地层—测井图中间结果的确定性 Graph 装配器。"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from model import Entity, Graph, Relation
from model.base import SourceModality

from ...schema_models import ImageExtractionTask


RELATION_ZH = {
    "part_of": "属于",
    "contains": "包含",
    "directly_overlies": "直接上覆",
    "directly_underlies": "直接下伏",
    "above": "位于上方",
    "below": "位于下方",
    "left_of": "位于左侧",
    "right_of": "位于右侧",
    "adjacent_to": "相邻",
    "intersects": "相交或钻遇",
    "cuts_through": "切穿",
    "offsets": "错断",
    "correlates_with": "可对比于",
    "lateral_transition_to": "横向过渡为",
    "tracks_along": "沿其延伸",
    "adjusted_from": "调整自",
    "located_in": "位于",
    "bounded_by": "受其边界约束",
    "has_lithology": "具有岩性",
    "has_log_curve": "具有测井曲线",
    "contains_reservoir": "包含储层",
    "contains_shale": "包含页岩段",
    "characterizes": "表征",
    "related_to": "相关",
}


def _confidence(value: Any, default: float = 0.8) -> float:
    """中文说明：将写入实体和关系的置信度限制在零到一。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


def _slug(value: str) -> str:
    """中文说明：生成稳定 ID 中便于阅读且不含特殊符号的片段。"""

    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value.strip()).strip("-")
    return cleaned[:48] or "item"


def _stable_id(task: ImageExtractionTask, kind: str, local_id: str) -> str:
    """中文说明：以图片任务和局部 ID 生成跨运行稳定、跨图不冲突的标识。"""

    digest = hashlib.sha1(
        f"{task.image_id}|2d-strat-log|{kind}|{local_id}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{task.image_id}:2d-strat-log:{kind}:{_slug(local_id)}:{digest}"


class TwoDimensionalStratigraphicLogGraphBuilder:
    """把二维中间结果转换为统一实体关系图并强制注入逐图来源。"""

    def __init__(self, task: ImageExtractionTask, intermediate: Mapping[str, Any]) -> None:
        """中文说明：保存单图任务和中间结果，并初始化局部 ID 到 Graph ID 的索引。"""

        self.task = task
        self.intermediate = intermediate
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}
        self.aliases: dict[str, str] = {}

    def build(self) -> Graph:
        """中文说明：先注册剖面根节点和全部图元，再装配图元关系且保持事件数组为空。"""

        diagram_evidence = (
            self.task.caption
            or str(self.intermediate.get("diagram_name") or "")
            or "来源二维平面地层—测井图"
        )
        diagram_id = self._register_entity(
            "__diagram__",
            str(self.intermediate.get("diagram_name") or "二维平面地层—测井图"),
            "two_dimensional_stratigraphic_log",
            attributes={
                "diagram_type": self.intermediate.get("diagram_type"),
                "coordinate_system": dict(self.intermediate.get("coordinate_system") or {}),
            },
            evidence=diagram_evidence,
            confidence=1.0,
        )

        for raw in self.intermediate.get("entities", []):
            item = raw if isinstance(raw, Mapping) else {}
            local_id = str(item.get("id") or "")
            if not local_id:
                continue
            entity_id = self._register_entity(
                local_id,
                str(item.get("name") or local_id),
                str(item.get("type") or "other"),
                attributes={
                    **dict(item.get("attributes") or {}),
                    "parent_id": item.get("parent_id") or "",
                    "position": dict(item.get("position") or {}),
                    "confidence": _confidence(item.get("confidence")),
                },
                evidence=str(item.get("evidence") or ""),
                confidence=_confidence(item.get("confidence")),
            )
            self._add_relation(
                diagram_id,
                "contains",
                entity_id,
                dimension="membership",
                explicit=False,
                basis="diagram_membership",
                evidence=str(item.get("evidence") or diagram_evidence),
                confidence=_confidence(item.get("confidence")),
            )

        for raw in self.intermediate.get("relations", []):
            item = raw if isinstance(raw, Mapping) else {}
            source_id = self.aliases.get(str(item.get("source_id") or ""))
            target_id = self.aliases.get(str(item.get("target_id") or ""))
            if not source_id or not target_id:
                continue
            reserved = {
                "source_id",
                "type",
                "target_id",
                "dimension",
                "explicit",
                "basis",
                "evidence",
                "confidence",
            }
            self._add_relation(
                source_id,
                str(item.get("type") or "related_to"),
                target_id,
                dimension=str(item.get("dimension") or "semantic"),
                explicit=bool(item.get("explicit")),
                basis=str(item.get("basis") or "二维图内关系"),
                evidence=str(item.get("evidence") or ""),
                confidence=_confidence(item.get("confidence")),
                attributes={key: value for key, value in item.items() if key not in reserved},
            )

        graph = Graph.from_chunk(
            document_id=self.task.document_id,
            chunk_id=self.task.chunk_id,
            modality=SourceModality.IMAGE,
            entities=self.entities.values(),
            relations=self.relations.values(),
            events=(),
            raw_response={"structured_intermediate_result": dict(self.intermediate)},
            stage="stage_04_image_two_dimensional_stratigraphic_log",
        )
        quality = dict(self.intermediate.get("quality") or {})
        graph.metadata.extra.update(
            {
                "status": "completed" if len(self.entities) > 1 else "empty",
                "extractor_kind": "stratigraphic_profile",
                "extractor_name": "二维平面地层—测井图抽取器",
                "image_id": self.task.image_id,
                "image_index": self.task.image_index,
                "image_path": self.task.image_path,
                "source_image_path": self.task.image_path,
                "classification_code": self.task.classification_code,
                "classification_type": self.task.classification_type,
                "events_extracted": False,
                "algorithm": "二维图元抽取 + 层序/横向顺序确定性关系装配",
                "diagram_type": self.intermediate.get("diagram_type"),
                "coordinate_system": dict(self.intermediate.get("coordinate_system") or {}),
                "stratigraphic_sequences": list(
                    self.intermediate.get("stratigraphic_sequences") or []
                ),
                "spatial_groups": list(self.intermediate.get("spatial_groups") or []),
                "quality": quality,
                "uncertainties": list(quality.get("uncertainties") or []),
                "dropped_relations": list(quality.get("dropped_relations") or []),
            }
        )
        return graph

    def _source_metadata(self, evidence: str) -> dict[str, Any]:
        """中文说明：给每个节点和边写入图片路径、图片任务、Chunk 与视觉证据。"""

        return {
            "source_modality": "image",
            "source_image_path": self.task.image_path,
            "source_image_id": self.task.image_id,
            "source_chunk_id": self.task.chunk_id,
            "image_index": self.task.image_index,
            "visual_evidence": evidence,
        }

    def _register_entity(
        self,
        local_id: str,
        name: str,
        entity_type: str,
        *,
        attributes: Mapping[str, Any],
        evidence: str,
        confidence: float,
    ) -> str:
        """中文说明：注册实体并建立模型局部 ID 到统一 Graph ID 的稳定映射。"""

        graph_id = _stable_id(self.task, "entity", local_id)
        clean_evidence = evidence or f"图中可见实体：{name}"
        self.entities[graph_id] = Entity(
            id=graph_id,
            name=name,
            type=entity_type,
            attributes={**dict(attributes), "confidence": _confidence(confidence)},
            provenance=clean_evidence,
            metadata=self._source_metadata(clean_evidence),
        )
        self.aliases[local_id] = graph_id
        return graph_id

    def _add_relation(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        *,
        dimension: str,
        explicit: bool,
        basis: str,
        evidence: str,
        confidence: float,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """中文说明：创建按三元组去重的关系，并保留推导依据、显式性和逐图来源。"""

        local_id = f"{source_id}|{relation_type}|{target_id}"
        relation_id = _stable_id(self.task, "relation", local_id)
        if relation_id in self.relations:
            return
        source = self.entities[source_id]
        target = self.entities[target_id]
        clean_evidence = evidence or f"图中{source.name}与{target.name}的相对关系"
        relation_name = RELATION_ZH.get(relation_type, relation_type)
        self.relations[relation_id] = Relation(
            id=relation_id,
            type=relation_type,
            relation_name=relation_name,
            type_zh=relation_name,
            source_id=source_id,
            source_name=source.name,
            source_type=source.type,
            target_id=target_id,
            target_name=target.name,
            target_type=target.type,
            attributes={
                **dict(attributes or {}),
                "dimension": dimension,
                "explicit": explicit,
                "inference_basis": basis,
                "confidence": _confidence(confidence),
            },
            provenance=clean_evidence,
            metadata=self._source_metadata(clean_evidence),
        )


def build_two_dimensional_stratigraphic_log_graph(
    task: ImageExtractionTask,
    intermediate: Mapping[str, Any],
) -> Graph:
    """中文说明：公开二维图谱装配入口，供统一调度、测试和离线重建共同使用。"""

    return TwoDimensionalStratigraphicLogGraphBuilder(task, intermediate).build()
