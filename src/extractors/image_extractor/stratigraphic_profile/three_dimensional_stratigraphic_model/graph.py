"""三维地层建模图中间结果的确定性 Graph 装配器。"""
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
    "has_lithology": "具有岩性",
    "directly_overlies": "直接上覆",
    "directly_underlies": "直接下伏",
    "located_in": "位于",
    "cuts_through": "切穿",
    "offsets": "错断",
    "intersects": "相交或钻遇",
    "adjacent_to": "邻接",
    "flows_to": "流向",
    "transports": "输导",
    "generates": "生成",
    "supplies_hydrocarbons_to": "向其供烃",
    "controls": "控制",
    "dolomitizes": "使其白云石化",
    "acts_as_source_rock": "充当烃源岩",
    "acts_as_reservoir": "充当储集层",
    "acts_as_seal": "充当盖层",
}


def _slug(value: str) -> str:
    """中文说明：把局部标识转换为稳定图谱 ID 中安全且可读的片段。"""

    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value.strip()).strip("-")
    return cleaned[:52] or "item"


def _stable_id(task: ImageExtractionTask, kind: str, local_id: str) -> str:
    """中文说明：结合来源图片和三维子类型生成跨运行稳定且跨图不冲突的 ID。"""

    digest = hashlib.sha1(f"{task.image_id}|3d-strat|{kind}|{local_id}".encode("utf-8")).hexdigest()[:10]
    return f"{task.image_id}:3d-strat:{kind}:{_slug(local_id)}:{digest}"


def _confidence(value: Any, default: float = 0.8) -> float:
    """中文说明：规范实体和关系置信度，保证结果始终位于零到一。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


class ThreeDimensionalStratigraphicModelGraphBuilder:
    """把三维模型、地层、构造对象及上下文三元组装配为统一 Graph。"""

    def __init__(self, task: ImageExtractionTask, intermediate: Mapping[str, Any]) -> None:
        """中文说明：初始化来源任务、局部 ID 索引以及实体关系去重容器。"""

        self.task = task
        self.intermediate = intermediate
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}
        self.local_to_graph: dict[str, str] = {}

    def build(self) -> Graph:
        """中文说明：先注册三维模型与全部图元，再写入包含、上下、空间和上下文关系。"""

        model = self.intermediate.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("三维中间结果缺少 model")
        model_id = self._add_entity(model)
        entities_by_field = self.intermediate.get("entities")
        if not isinstance(entities_by_field, Mapping):
            raise ValueError("三维中间结果缺少 entities")
        for records in entities_by_field.values():
            for record in self._records(records):
                entity_id = self._add_entity(record)
                self._add_relation(
                    model_id,
                    "contains",
                    entity_id,
                    evidence=str(record.get("evidence") or "三维模型中的可见对象"),
                    visual_anchor_evidence=str(record.get("evidence") or "三维模型中的可见对象"),
                    context_evidence="",
                    confidence=_confidence(record.get("confidence"), 0.9),
                    explicit=True,
                    basis="visible_model_membership",
                    evidence_scope="visual",
                    attributes={},
                )
        for relation in self._records(self.intermediate.get("relations")):
            source_id = self.local_to_graph.get(str(relation.get("source_id") or ""))
            target_id = self.local_to_graph.get(str(relation.get("target_id") or ""))
            if not source_id or not target_id:
                continue
            self._add_relation(
                source_id,
                str(relation.get("relation_type") or "related_to"),
                target_id,
                evidence=str(relation.get("evidence") or "三维图内关系"),
                visual_anchor_evidence=str(relation.get("visual_anchor_evidence") or relation.get("evidence") or "三维图内对象锚点"),
                context_evidence=str(relation.get("context_evidence") or ""),
                confidence=_confidence(relation.get("confidence")),
                explicit=bool(relation.get("explicit", False)),
                basis=str(relation.get("basis") or "deterministic_three_dimensional_relation"),
                evidence_scope=str(relation.get("evidence_scope") or "visual"),
                attributes=dict(relation.get("attributes")) if isinstance(relation.get("attributes"), Mapping) else {},
            )

        graph = Graph.from_chunk(
            document_id=self.task.document_id,
            chunk_id=self.task.chunk_id,
            modality=SourceModality.IMAGE,
            entities=self.entities.values(),
            relations=self.relations.values(),
            events=(),
            raw_response={
                "visual": self.intermediate.get("raw_visual_payload"),
                "reviewed_visual": self.intermediate.get("reviewed_visual_payload"),
                "structured_intermediate_result": dict(self.intermediate),
            },
            stage="stage_04_image_three_dimensional_stratigraphic_model_extraction",
        )
        quality = self.intermediate.get("quality") if isinstance(self.intermediate.get("quality"), Mapping) else {}
        graph.metadata.extra.update(
            {
                "status": "completed" if self.entities else "empty",
                "extractor_kind": "stratigraphic_profile",
                "extractor_name": "三维地层建模图抽取器",
                "stratigraphic_subtype": "three_dimensional_stratigraphic_model",
                "image_id": self.task.image_id,
                "image_index": self.task.image_index,
                "image_path": self.task.image_path,
                "source_image_path": self.task.image_path,
                "classification_code": self.task.classification_code,
                "classification_type": self.task.classification_type,
                "model_called": True,
                "events_extracted": False,
                "algorithm": self.intermediate.get("algorithm"),
                "quality": dict(quality),
                "vertical_relation_count": quality.get("vertical_relation_count", 0),
                "context_relation_count": quality.get("context_relation_count", 0),
            }
        )
        reference_errors = graph.validate_references()
        if reference_errors:
            raise ValueError(f"三维 Graph 引用校验失败：{reference_errors}")
        return graph

    @staticmethod
    def _records(value: Any) -> Iterable[Mapping[str, Any]]:
        """中文说明：只迭代对象数组，避免脏字符串或空值进入装配流程。"""

        if not isinstance(value, list):
            return ()
        return (item for item in value if isinstance(item, Mapping))

    def _source_metadata(
        self,
        visual_evidence: str,
        *,
        evidence_scope: str,
        context_evidence: str = "",
        image_region: str = "",
        source_kind: str = "visual",
    ) -> dict[str, Any]:
        """中文说明：强制记录来源图、Chunk、图内锚点以及可选正文证据，禁止混淆证据边界。"""

        return {
            "source_modality": "image",
            "source_image_path": self.task.image_path,
            "source_image_id": self.task.image_id,
            "source_chunk_id": self.task.chunk_id,
            "image_index": self.task.image_index,
            "image_region": image_region,
            "visual_evidence": visual_evidence,
            "context_evidence": context_evidence,
            "evidence_scope": evidence_scope,
            "source_kind": source_kind,
        }

    def _add_entity(self, record: Mapping[str, Any]) -> str:
        """中文说明：注册一个模型或图元实体，保留视觉与上下文双重证据来源。"""

        local_id = str(record.get("id") or "").strip()
        name = str(record.get("name") or local_id).strip()
        if not local_id or not name:
            raise ValueError("三维实体缺少 id 或 name")
        existing = self.local_to_graph.get(local_id)
        if existing:
            return existing
        graph_id = _stable_id(self.task, "entity", local_id)
        evidence = str(record.get("evidence") or "图中可见的三维对象")
        context_evidence = str(record.get("context_evidence") or "")
        source_kind = str(record.get("source_kind") or "visual")
        attributes = dict(record.get("attributes")) if isinstance(record.get("attributes"), Mapping) else {}
        attributes.update(
            {
                "local_id": local_id,
                "source_kind": source_kind,
                "confidence": _confidence(record.get("confidence")),
            }
        )
        self.entities[graph_id] = Entity(
            id=graph_id,
            name=name,
            type=str(record.get("entity_type") or "geological_object"),
            attributes=attributes,
            provenance=context_evidence if source_kind == "visual_context" and context_evidence else evidence,
            metadata=self._source_metadata(
                evidence,
                evidence_scope="visual_context" if source_kind == "visual_context" else "visual",
                context_evidence=context_evidence,
                image_region=str(record.get("image_region") or ""),
                source_kind=source_kind,
            ),
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
        visual_anchor_evidence: str,
        context_evidence: str,
        confidence: float,
        explicit: bool,
        basis: str,
        evidence_scope: str,
        attributes: Mapping[str, Any],
    ) -> None:
        """中文说明：创建去重关系，并明确区分视觉事实、层序推导和正文支持三元组。"""

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
                **dict(attributes),
                "explicit": explicit,
                "inference_basis": basis,
                "evidence_scope": evidence_scope,
                "confidence": _confidence(confidence),
            },
            provenance=context_evidence if evidence_scope == "context" and context_evidence else evidence,
            metadata=self._source_metadata(
                visual_anchor_evidence or evidence,
                evidence_scope=evidence_scope,
                context_evidence=context_evidence,
            ),
        )


def build_three_dimensional_stratigraphic_model_graph(
    task: ImageExtractionTask,
    intermediate: Mapping[str, Any],
) -> Graph:
    """中文说明：公开确定性图谱装配入口，供调度器、批处理和离线测试共同使用。"""

    return ThreeDimensionalStratigraphicModelGraphBuilder(task, intermediate).build()
