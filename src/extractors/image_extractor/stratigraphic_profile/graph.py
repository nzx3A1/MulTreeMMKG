"""地层—剖面—测井视觉结果的确定性 Graph 装配器。"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from model import Entity, Graph, Relation
from model.base import SourceModality

from ..schema_models import ImageExtractionTask


RELATION_ZH = {
    "part_of": "属于",
    "contains": "包含",
    "has_lithology": "具有岩性",
    "contains_interval": "包含井段",
    "directly_overlies": "直接上覆",
    "directly_underlies": "直接下伏",
    "above": "位于上方",
    "below": "位于下方",
    "adjacent_to": "邻接",
    "lateral_transition_to": "横向过渡为",
    "correlates_with": "可对比于",
    "intersects": "钻遇",
    "has_log_curve": "具有测井曲线",
    "measured_in": "测量于",
    "tracks_along": "沿其延伸",
    "deviates_from": "偏离",
    "adjusted_from": "调整自",
    "located_in": "位于",
    "cuts_through": "切穿",
    "offsets": "错断",
    "contains_reservoir": "包含储层",
    "contains_oil_layer": "包含油层",
    "bounded_by": "受其边界约束",
    "acts_as_source_rock": "充当烃源岩",
    "acts_as_reservoir": "充当储集层",
    "acts_as_seal": "充当盖层",
    "higher_response_than": "响应高于",
    "lower_response_than": "响应低于",
    "characterizes": "表征",
    "dolomitizes": "使其白云石化",
    "flows_to": "流向",
}


def _items(value: Any) -> list[Any]:
    """中文说明：只展开列表字段，防止异常字符串被按字符迭代。"""

    return list(value) if isinstance(value, list) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    """中文说明：把非对象字段降级为空对象，保证 mock 与真实模型脏响应都可处理。"""

    return value if isinstance(value, Mapping) else {}


def _confidence(value: Any, default: float = 0.75) -> float:
    """中文说明：将模型置信度规范到 0 到 1。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


def _slug(value: str) -> str:
    """中文说明：生成稳定 ID 中可读的安全片段。"""

    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value.strip()).strip("-")
    return cleaned[:48] or "item"


def _stable_id(task: ImageExtractionTask, kind: str, local_id: str) -> str:
    """中文说明：使用来源图片和局部 ID 生成跨运行稳定且不会跨图冲突的标识。"""

    digest = hashlib.sha1(f"{task.image_id}|{kind}|{local_id}".encode("utf-8")).hexdigest()[:10]
    return f"{task.image_id}:strat:{kind}:{_slug(local_id)}:{digest}"


class StratigraphicProfileGraphBuilder:
    """把地层剖面 VLM/LLM JSON 转为统一 Graph，并强制注入逐图来源。"""

    def __init__(self, task: ImageExtractionTask, visual: Mapping[str, Any], audit: Mapping[str, Any]) -> None:
        """中文说明：初始化实体、关系、别名和被丢弃关系的去重容器。"""

        self.task = task
        self.visual = visual
        self.audit = audit
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}
        self.aliases: dict[str, str] = {}
        self.unit_records: list[tuple[int, str, str, float]] = []
        self.parent_links: list[tuple[str, str, str, float]] = []
        self.parent_by_child: dict[str, str] = {}
        self.dropped_relations: list[dict[str, Any]] = []

    def build(self) -> Graph:
        """中文说明：按对象类别装配节点，再补模型关系、层级边与可靠的相邻层序边。"""

        self._add_lithologies()
        self._add_units()
        self._add_records("wells", "well")
        self._add_records("log_curves", "log_curve")
        self._add_records("zones", "zone")
        self._add_records("entities", "geological_object")
        self._add_parent_relations()
        self._add_curve_relations()
        self._add_model_relations()
        self._derive_adjacent_unit_relations()

        graph = Graph.from_chunk(
            document_id=self.task.document_id,
            chunk_id=self.task.chunk_id,
            modality=SourceModality.IMAGE,
            entities=self.entities.values(),
            relations=self.relations.values(),
            raw_response={"visual": dict(self.visual), "relation_audit": dict(self.audit)},
            stage="stage_04_image_stratigraphic_profile_extraction",
        )
        graph.metadata.extra.update(
            {
                "status": "completed" if self.entities else "empty",
                "extractor_kind": "stratigraphic_profile",
                "extractor_name": "地层—剖面—测井抽取器",
                "image_id": self.task.image_id,
                "image_index": self.task.image_index,
                "image_path": self.task.image_path,
                "source_image_path": self.task.image_path,
                "classification_code": self.task.classification_code,
                "classification_type": self.task.classification_type,
                "model_called": True,
                "algorithm": "VLM逐图证据抽取 + LLM关系审查 + 确定性层序装配",
                "diagram_type": str(self.visual.get("diagram_type") or ""),
                "coordinate_system": dict(_mapping(self.visual.get("coordinate_system"))),
                "uncertainties": [str(item) for item in _items(self.visual.get("uncertainties"))],
                "audit_warnings": [str(item) for item in _items(self.audit.get("warnings"))],
                "dropped_relations": self.dropped_relations,
            }
        )
        return graph

    def _source_metadata(self, evidence: str, image_region: str = "") -> dict[str, Any]:
        """中文说明：为每个实体和关系写入可机器检索的来源图片、Chunk 与图内区域。"""

        return {
            "source_modality": "image",
            "source_image_path": self.task.image_path,
            "source_image_id": self.task.image_id,
            "source_chunk_id": self.task.chunk_id,
            "image_index": self.task.image_index,
            "image_region": image_region,
            "visual_evidence": evidence,
        }

    def _register_entity(
        self,
        local_id: str,
        name: str,
        entity_type: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        evidence: str = "",
        image_region: str = "",
        confidence: float = 0.75,
    ) -> str:
        """中文说明：注册实体并让模型局部 ID 与名称都能解析为统一图 ID。"""

        clean_local = str(local_id or name).strip()
        clean_name = str(name or clean_local).strip()
        existing = self.aliases.get(clean_local) or self.aliases.get(clean_name)
        if existing:
            self.aliases[clean_local] = existing
            self.aliases[clean_name] = existing
            return existing
        graph_id = _stable_id(self.task, "entity", clean_local)
        self.entities[graph_id] = Entity(
            id=graph_id,
            name=clean_name,
            type=entity_type or "geological_object",
            attributes={**dict(attributes or {}), "confidence": _confidence(confidence)},
            provenance=evidence or self.task.caption or f"来源图片：{self.task.image_path}",
            metadata=self._source_metadata(evidence, image_region),
        )
        self.aliases[clean_local] = graph_id
        self.aliases[clean_name] = graph_id
        return graph_id

    def _add_lithologies(self) -> None:
        """中文说明：把图例岩性注册为可复用实体。"""

        for raw in _items(self.visual.get("lithologies")):
            item = _mapping(raw)
            local_id = str(item.get("id") or item.get("name") or "")
            name = str(item.get("name") or local_id)
            if not local_id or not name:
                continue
            self._register_entity(
                local_id,
                name,
                "lithology",
                attributes={"visual_pattern": item.get("visual_pattern", "")},
                evidence=str(item.get("evidence") or ""),
                image_region=str(item.get("image_region") or ""),
                confidence=_confidence(item.get("confidence")),
            )

    def _add_units(self) -> None:
        """中文说明：注册地层单元、深度和岩性组成，并暂存父层级引用。"""

        for raw in _items(self.visual.get("stratigraphic_units")):
            item = _mapping(raw)
            local_id = str(item.get("id") or item.get("name") or "")
            name = str(item.get("name") or local_id)
            if not local_id or not name:
                continue
            try:
                order = int(item.get("order_bottom_to_top"))
            except (TypeError, ValueError):
                order = len(self.unit_records)
            evidence = str(item.get("evidence") or "")
            confidence = _confidence(item.get("confidence"))
            unit_id = self._register_entity(
                local_id,
                name,
                "stratigraphic_unit",
                attributes={
                    "rank": item.get("rank", ""),
                    "order_bottom_to_top": order,
                    "top_depth_m": item.get("top_depth_m"),
                    "bottom_depth_m": item.get("bottom_depth_m"),
                    "geometry": item.get("geometry", ""),
                    "parent_unit_id": item.get("parent_unit_id", ""),
                },
                evidence=evidence,
                image_region=str(item.get("image_region") or ""),
                confidence=confidence,
            )
            self.unit_records.append((order, unit_id, name, confidence))
            if item.get("parent_unit_id"):
                self.parent_links.append((unit_id, str(item.get("parent_unit_id")), evidence, confidence))
            for raw_lith in _items(item.get("lithologies")):
                lith = _mapping(raw_lith)
                lith_ref = str(lith.get("id") or lith.get("lithology_id") or lith.get("name") or "")
                lith_id = self.aliases.get(lith_ref)
                if not lith_id:
                    continue
                self._add_relation(
                    unit_id,
                    "has_lithology",
                    lith_id,
                    "stratigraphic",
                    True,
                    "层内填充与图例匹配",
                    str(lith.get("evidence") or evidence),
                    _confidence(lith.get("confidence"), confidence),
                    attributes={"role": lith.get("role", "")},
                )

    def _add_records(self, field: str, default_type: str) -> None:
        """中文说明：统一注册井、测井曲线、区带和其他地质对象，保留其专用属性。"""

        for raw in _items(self.visual.get(field)):
            item = _mapping(raw)
            local_id = str(item.get("id") or item.get("name") or "")
            name = str(item.get("name") or local_id)
            if not local_id or not name:
                continue
            reserved = {"id", "name", "type", "evidence", "image_region", "confidence", "attributes"}
            attributes = {key: value for key, value in item.items() if key not in reserved}
            attributes.update(dict(_mapping(item.get("attributes"))))
            self._register_entity(
                local_id,
                name,
                str(item.get("type") or default_type),
                attributes=attributes,
                evidence=str(item.get("evidence") or ""),
                image_region=str(item.get("image_region") or ""),
                confidence=_confidence(item.get("confidence")),
            )

    def _add_parent_relations(self) -> None:
        """中文说明：在全部层段注册后解析子层属于父层的层级关系。"""

        for child_id, parent_ref, evidence, confidence in self.parent_links:
            parent_id = self.aliases.get(parent_ref)
            if parent_id:
                self.parent_by_child[child_id] = parent_id
                self._add_relation(child_id, "part_of", parent_id, "stratigraphic", True, "图内地层层级", evidence, confidence)
            else:
                self.dropped_relations.append({"source": child_id, "type": "part_of", "target": parent_ref, "reason": "unresolved_parent"})

    def _add_curve_relations(self) -> None:
        """中文说明：根据曲线记录的 well_id 自动补齐井到测井曲线的归属边。"""

        for raw in _items(self.visual.get("log_curves")):
            item = _mapping(raw)
            curve_id = self.aliases.get(str(item.get("id") or item.get("name") or ""))
            well_id = self.aliases.get(str(item.get("well_id") or ""))
            if curve_id and well_id:
                evidence = str(item.get("evidence") or "")
                self._add_relation(well_id, "has_log_curve", curve_id, "logging", True, "曲线道表头与井柱", evidence, _confidence(item.get("confidence")))

    def _add_model_relations(self) -> None:
        """中文说明：应用视觉关系与审查新增关系，并过滤审查明确否决的边。"""

        rejected = {
            (str(item.get("source")), str(item.get("type")), str(item.get("target")))
            for raw in _items(self.audit.get("reject_relations"))
            if (item := _mapping(raw))
        }
        for raw in [*_items(self.visual.get("relations")), *_items(self.audit.get("added_relations"))]:
            item = _mapping(raw)
            signature = (str(item.get("source")), str(item.get("type")), str(item.get("target")))
            if signature in rejected:
                self.dropped_relations.append({**dict(item), "reason": "rejected_by_audit"})
                continue
            source_id = self.aliases.get(signature[0])
            target_id = self.aliases.get(signature[2])
            if not source_id or not target_id:
                self.dropped_relations.append({**dict(item), "reason": "unresolved_endpoint"})
                continue
            self._add_relation(
                source_id,
                signature[1] or "related_to",
                target_id,
                str(item.get("dimension") or "semantic"),
                bool(item.get("explicit", True)),
                str(item.get("basis") or "图内证据关系"),
                str(item.get("evidence") or ""),
                _confidence(item.get("confidence")),
            )

    def _derive_adjacent_unit_relations(self) -> None:
        """中文说明：分别在顶级层段和每组同父子层内推导相邻层序，避免跨层级错误连接。"""

        groups: dict[str | None, list[tuple[int, str, str, float]]] = {}
        for record in self.unit_records:
            parent_id = self.parent_by_child.get(record[1])
            groups.setdefault(parent_id, []).append(record)
        for siblings in groups.values():
            ordered = sorted(siblings, key=lambda record: record[0])
            for lower, upper in zip(ordered, ordered[1:]):
                _, lower_id, lower_name, lower_conf = lower
                _, upper_id, upper_name, upper_conf = upper
                confidence = min(lower_conf, upper_conf, 0.9)
                evidence = f"图中层序显示{upper_name}位于{lower_name}之上"
                self._add_relation(upper_id, "directly_overlies", lower_id, "stratigraphic", False, "order_bottom_to_top", evidence, confidence)
                self._add_relation(lower_id, "directly_underlies", upper_id, "stratigraphic", False, "order_bottom_to_top", evidence, confidence)

    def _add_relation(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        dimension: str,
        explicit: bool,
        basis: str,
        evidence: str,
        confidence: float,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """中文说明：创建去重关系，并像实体一样强制记录其来源图片和图内证据。"""

        local_id = f"{source_id}|{relation_type}|{target_id}"
        relation_id = _stable_id(self.task, "relation", local_id)
        if relation_id in self.relations:
            return
        source = self.entities[source_id]
        target = self.entities[target_id]
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
            provenance=evidence or self.task.caption or f"来源图片：{self.task.image_path}",
            metadata=self._source_metadata(evidence),
        )


def build_stratigraphic_profile_graph(
    task: ImageExtractionTask,
    visual: Mapping[str, Any],
    audit: Mapping[str, Any] | None = None,
) -> Graph:
    """中文说明：公开 Graph 装配入口，供真实抽取、mock 测试和离线重建共同使用。"""

    return StratigraphicProfileGraphBuilder(task, visual, audit or {}).build()
