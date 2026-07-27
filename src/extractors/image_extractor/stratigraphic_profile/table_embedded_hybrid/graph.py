"""表格嵌入混合地层图中间结果的确定性知识图谱装配器。"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping

from model import Entity, Graph, Relation
from model.base import SourceModality

from ...schema_models import ImageExtractionTask


RELATION_NAMES = {
    "contains": "包含",
    "part_of": "属于",
    "directly_overlies": "直接上覆",
    "has_lithology": "具有岩性",
    "has_sedimentary_facies": "具有沉积相",
    "contains_reservoir": "包含储层",
    "contains_oil_layer": "包含油层",
    "contains_geological_feature": "包含地质特征",
    "contains_interval": "包含井段",
    "characterizes": "表征",
    "has_log_curve": "具有测井曲线",
    "aligned_with": "深度对齐于",
    "located_in": "位于",
    "higher_response_than": "响应高于",
}

ENTITY_TYPE_BY_FIELD = {
    "stratigraphic_intervals": "stratigraphic_unit",
    "reference_intervals": "depth_interval",
    "lithology_intervals": "lithology_interval",
    "facies_intervals": "sedimentary_facies_interval",
    "reservoir_intervals": "reservoir_interval",
    "oil_layer_intervals": "oil_layer_interval",
    "geological_feature_intervals": "geological_feature_interval",
    "curve_observations": "curve_response_interval",
    "curve_tracks": "log_curve",
    "point_markers": "well",
}


def _slug(value: str) -> str:
    """中文说明：把局部标识转换为稳定图谱 ID 中可读且安全的片段。"""

    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value.strip()).strip("-")
    return cleaned[:52] or "item"


def _stable_id(task: ImageExtractionTask, kind: str, local_id: str) -> str:
    """中文说明：结合来源图片生成跨运行稳定、跨图片不冲突的节点或关系 ID。"""

    digest = hashlib.sha1(f"{task.image_id}|table-hybrid|{kind}|{local_id}".encode("utf-8")).hexdigest()[:10]
    return f"{task.image_id}:table-hybrid:{kind}:{_slug(local_id)}:{digest}"


def _confidence(value: Any, default: float = 0.8) -> float:
    """中文说明：规范模型图元和对齐关系置信度，防止异常响应产生越界分数。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


class TableEmbeddedHybridGraphBuilder:
    """把中间结果中的区间、曲线和井标注装配为统一 Graph。"""

    def __init__(self, task: ImageExtractionTask, intermediate: Mapping[str, Any]) -> None:
        """中文说明：初始化来源元数据、节点索引和关系去重容器。"""

        self.task = task
        self.intermediate = intermediate
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}
        self.local_to_graph: dict[str, str] = {}

    def build(self) -> Graph:
        """中文说明：先注册剖面及全部专用图元，再按确定性规则装配包含、层序和深度对齐关系。"""

        diagram = self.intermediate.get("diagram")
        if not isinstance(diagram, Mapping):
            raise ValueError("中间结果缺少 diagram")
        profile_local_id = str(diagram.get("id") or "profile")
        profile_id = self._add_entity(
            profile_local_id,
            str(diagram.get("name") or "表格嵌入混合地层图"),
            "stratigraphic_profile",
            attributes={
                "width": diagram.get("width"),
                "height": diagram.get("height"),
                "subtype": self.intermediate.get("subtype"),
                "coordinate_system": self.intermediate.get("coordinate_system"),
            },
            evidence=self.task.caption or "整张混合地层图的可见版面",
            confidence=1.0,
        )
        parsed = self.intermediate.get("parsed")
        if not isinstance(parsed, Mapping):
            raise ValueError("中间结果缺少 parsed")
        for field, entity_type in ENTITY_TYPE_BY_FIELD.items():
            for item in self._records(parsed.get(field)):
                local_id = str(item.get("id") or "").strip()
                if not local_id:
                    continue
                graph_id = self._add_entity_from_record(local_id, item, entity_type)
                self._add_relation(
                    profile_id,
                    "contains" if entity_type != "log_curve" else "has_log_curve",
                    graph_id,
                    evidence=str(item.get("evidence") or "图内轨道可见对象"),
                    confidence=_confidence(item.get("confidence"), 0.9),
                    explicit=True,
                    basis="visible_track_membership",
                )
        for item in self._records(parsed.get("objects")):
            local_id = str(item.get("id") or "").strip()
            if not local_id:
                continue
            graph_id = self._add_entity_from_record(
                local_id,
                item,
                str(item.get("entity_type") or "geological_object"),
            )
            self._add_relation(
                profile_id,
                "contains",
                graph_id,
                evidence=str(item.get("evidence") or "图内可见对象"),
                confidence=_confidence(item.get("confidence"), 0.9),
                explicit=True,
                basis="visible_diagram_membership",
            )
        for item in self._records(self.intermediate.get("alignment_relations")):
            source_id = self.local_to_graph.get(str(item.get("source_id") or ""))
            target_id = self.local_to_graph.get(str(item.get("target_id") or ""))
            if not source_id or not target_id:
                continue
            attributes = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "source_id",
                    "target_id",
                    "relation_type",
                    "evidence",
                    "confidence",
                    "explicit",
                    "basis",
                    "attributes",
                }
            }
            if isinstance(item.get("attributes"), Mapping):
                attributes.update(dict(item["attributes"]))
            self._add_relation(
                source_id,
                str(item.get("relation_type") or "aligned_with"),
                target_id,
                evidence=str(item.get("evidence") or "公共纵轴确定性对齐"),
                confidence=_confidence(item.get("confidence")),
                explicit=bool(item.get("explicit", False)),
                basis=str(item.get("basis") or "deterministic_alignment"),
                attributes=attributes,
            )
        graph = Graph.from_chunk(
            document_id=self.task.document_id,
            chunk_id=self.task.chunk_id,
            modality=SourceModality.IMAGE,
            entities=self.entities.values(),
            relations=self.relations.values(),
            events=(),
            raw_response={"structured_intermediate_result": dict(self.intermediate)},
            stage="stage_04_image_table_embedded_hybrid_extraction",
        )
        graph.metadata.extra.update(
            {
                "status": "completed" if self.entities else "empty",
                "extractor_kind": "stratigraphic_profile",
                "extractor_name": "表格—图像嵌入混合型地层图抽取器",
                "stratigraphic_subtype": "table_embedded_hybrid",
                "image_id": self.task.image_id,
                "image_index": self.task.image_index,
                "image_path": self.task.image_path,
                "source_image_path": self.task.image_path,
                "classification_code": self.task.classification_code,
                "classification_type": self.task.classification_type,
                "model_called": True,
                "events_extracted": False,
                "algorithm": self.intermediate.get("algorithm"),
                "coordinate_system": self.intermediate.get("coordinate_system"),
                "track_count": len(self.intermediate.get("tracks", [])),
                "quality": self.intermediate.get("quality"),
            }
        )
        reference_errors = graph.validate_references()
        if reference_errors:
            raise ValueError(f"确定性 Graph 引用校验失败：{reference_errors}")
        return graph

    @staticmethod
    def _records(value: Any) -> Iterable[Mapping[str, Any]]:
        """中文说明：只迭代对象数组，忽略无法成为图谱记录的脏值。"""

        if not isinstance(value, list):
            return ()
        return (item for item in value if isinstance(item, Mapping))

    def _source_metadata(self, evidence: str, record: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """中文说明：为每个节点和关系强制写入精确来源图片及其视觉证据。"""

        metadata = {
            "source_modality": "image",
            "source_image_path": self.task.image_path,
            "source_image_id": self.task.image_id,
            "source_chunk_id": self.task.chunk_id,
            "image_index": self.task.image_index,
            "visual_evidence": evidence,
        }
        if record:
            for key in ("top_y", "bottom_y", "pixel_y", "top_value", "bottom_value", "vertical_value"):
                if key in record:
                    metadata[key] = record[key]
        return metadata

    def _add_entity_from_record(
        self,
        local_id: str,
        item: Mapping[str, Any],
        entity_type: str,
    ) -> str:
        """中文说明：把一个规范化图元注册为节点，并保留深度、轨道及专用解析属性。"""

        reserved = {"id", "name", "evidence", "confidence", "attributes"}
        attributes = {key: value for key, value in item.items() if key not in reserved}
        raw_attributes = item.get("attributes")
        if isinstance(raw_attributes, Mapping):
            attributes.update(dict(raw_attributes))
        return self._add_entity(
            local_id,
            str(item.get("name") or local_id),
            entity_type,
            attributes=attributes,
            evidence=str(item.get("evidence") or "图内专用轨道可见图元"),
            confidence=_confidence(item.get("confidence")),
            record=item,
        )

    def _add_entity(
        self,
        local_id: str,
        name: str,
        entity_type: str,
        *,
        attributes: Mapping[str, Any],
        evidence: str,
        confidence: float,
        record: Mapping[str, Any] | None = None,
    ) -> str:
        """中文说明：使用局部 ID 去重实体，并建立对齐关系可解析的局部到图 ID 映射。"""

        existing = self.local_to_graph.get(local_id)
        if existing:
            return existing
        graph_id = _stable_id(self.task, "entity", local_id)
        self.entities[graph_id] = Entity(
            id=graph_id,
            name=name,
            type=entity_type,
            attributes={**dict(attributes), "confidence": _confidence(confidence)},
            provenance=evidence,
            metadata=self._source_metadata(evidence, record),
        )
        self.local_to_graph[local_id] = graph_id
        return graph_id

    def _add_relation(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        *,
        evidence: str,
        confidence: float,
        explicit: bool,
        basis: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """中文说明：创建带推理边界、置信度和逐图来源的去重关系。"""

        local_id = f"{source_id}|{relation_type}|{target_id}"
        relation_id = _stable_id(self.task, "relation", local_id)
        if relation_id in self.relations:
            return
        source = self.entities[source_id]
        target = self.entities[target_id]
        relation_name = RELATION_NAMES.get(relation_type, relation_type)
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
                "explicit": explicit,
                "inference_basis": basis,
                "confidence": _confidence(confidence),
            },
            provenance=evidence,
            metadata=self._source_metadata(evidence),
        )


def build_table_embedded_hybrid_graph(
    task: ImageExtractionTask,
    intermediate: Mapping[str, Any],
) -> Graph:
    """中文说明：公开确定性图谱装配入口，将真实 VLM 中间结果转换为统一 Graph。"""

    return TableEmbeddedHybridGraphBuilder(task, intermediate).build()
