"""地图空间 VLM 结果的规范化与确定性 Graph 装配。"""
from __future__ import annotations

import hashlib
import math
import re
from copy import deepcopy
from typing import Any, Mapping

from model import Entity, Graph, Relation
from model.base import SourceModality

from ..schema_models import ImageExtractionTask
from .geometry import reconstruct_map_geometry


ENTITY_TYPE_MAP = {
    "geographic_location": ("GeographicLocation", "地理位置"),
    "geographiclocation": ("GeographicLocation", "地理位置"),
    "location": ("GeographicLocation", "地理位置"),
    "basin": ("Basin", "盆地"),
    "study_area": ("StudyArea", "研究区"),
    "studyarea": ("StudyArea", "研究区"),
    "block": ("Block", "区块"),
    "well": ("Well", "井"),
    "oil_gas_field": ("OilGasField", "油气田"),
    "oilgasfield": ("OilGasField", "油气田"),
    "oilfield": ("OilGasField", "油气田"),
    "gas_field": ("OilGasField", "油气田"),
    "prospect_area": ("ProspectArea", "有利区/预测区"),
    "lithology": ("Lithology", "岩性"),
    "lithofacies": ("Lithofacies", "岩相"),
    "sedimentary_facies": ("SedimentaryFacies", "沉积相"),
    "sedimentaryfacies": ("SedimentaryFacies", "沉积相"),
    "subfacies": ("Subfacies", "亚相"),
    "microfacies": ("Microfacies", "微相"),
    "paleogeographic_unit": ("PaleogeographicUnit", "古地理单元"),
    "paleogeographicunit": ("PaleogeographicUnit", "古地理单元"),
    "structural_unit": ("StructuralUnit", "构造单元"),
    "structuralunit": ("StructuralUnit", "构造单元"),
    "depression": ("Depression", "凹陷/坳陷"),
    "sag": ("Sag", "洼陷"),
    "uplift": ("Uplift", "隆起"),
    "slope": ("Slope", "斜坡"),
    "structural_belt": ("StructuralBelt", "构造带"),
    "structuralbelt": ("StructuralBelt", "构造带"),
    "fault": ("Fault", "断层"),
    "fault_zone": ("FaultZone", "断裂带"),
    "province": ("Province", "省级行政区"),
    "city": ("City", "城市"),
    "county": ("County", "县级行政区"),
    "town": ("Town", "乡镇"),
}

STRATIGRAPHIC_TYPE_MAP = {
    "formation": ("Formation", "组"),
    "member": ("StratigraphicMember", "段"),
    "stratigraphic_member": ("StratigraphicMember", "段"),
    "stratigraphicmember": ("StratigraphicMember", "段"),
    "submember": ("SubMember", "亚段/小层"),
    "bed": ("Bed", "层/单层"),
    "reservoir_interval": ("ReservoirInterval", "目的层/储层段"),
    "reservoirinterval": ("ReservoirInterval", "目的层/储层段"),
    "unknown": ("StratigraphicUnit", "地层/层位"),
}

SECONDARY_TYPE_MAP = {
    "facies_transition": ("FaciesTransition", "相带过渡"),
    "spatial_distribution": ("SpatialDistribution", "空间展布"),
    "high_value_zone": ("HighValueZone", "高值区"),
    "low_value_zone": ("LowValueZone", "低值区"),
    "section_line": ("SectionLine", "剖面线"),
    "boundary": ("Boundary", "边界"),
    "stratigraphic_erosion_line": ("StratigraphicErosionLine", "地层剥蚀线"),
    "relative_direction": ("RelativeDirection", "相对方向"),
    "spatial_trend": ("SpatialTrend", "空间趋势"),
}

RELATION_ZH = {
    "DEPICTS": "描绘",
    "HAS_THEME": "具有地图主题",
    "LOCATED_IN": "位于",
    "WITHIN": "位于内部",
    "CONTAINS": "包含",
    "COVERS": "覆盖",
    "OVERLAPS": "重叠",
    "INTERSECTS": "相交",
    "CROSSES": "穿过",
    "TOUCHES": "接触",
    "ADJACENT_TO": "邻接",
    "NEAR": "邻近",
    "DISTRIBUTED_IN": "分布于",
    "DEVELOPED_IN": "发育于",
    "PART_OF": "隶属于",
    "HAS_LITHOLOGY": "具有岩性",
    "HAS_FACIES": "具有沉积相",
    "HAS_GEOLOGICAL_ATTRIBUTE": "具有地质属性",
    "HAS_VALUE": "具有属性值",
    "HAS_SPATIAL_FEATURE": "具有空间特征",
    "TRANSITIONS_TO": "过渡为",
    "HAS_RELATIVE_DIRECTION": "具有相对方向",
    "HAS_SPATIAL_TREND": "具有空间趋势",
    "NORTH_OF": "位于北侧",
    "SOUTH_OF": "位于南侧",
    "EAST_OF": "位于东侧",
    "WEST_OF": "位于西侧",
    "NORTHEAST_OF": "位于东北侧",
    "NORTHWEST_OF": "位于西北侧",
    "SOUTHEAST_OF": "位于东南侧",
    "SOUTHWEST_OF": "位于西南侧",
    "PAGE_LEFT_OF": "位于页面左侧",
    "PAGE_RIGHT_OF": "位于页面右侧",
    "PAGE_ABOVE": "位于页面上方",
    "PAGE_BELOW": "位于页面下方",
    "PAGE_ABOVE_LEFT_OF": "位于页面左上方",
    "PAGE_ABOVE_RIGHT_OF": "位于页面右上方",
    "PAGE_BELOW_LEFT_OF": "位于页面左下方",
    "PAGE_BELOW_RIGHT_OF": "位于页面右下方",
}

_SPATIAL_RELATION_ALIASES = {
    "LOCATED_IN": "LOCATED_IN",
    "LOCATEDIN": "LOCATED_IN",
    "WITHIN": "WITHIN",
    "INSIDE": "WITHIN",
    "CONTAINS": "CONTAINS",
    "CONTAIN": "CONTAINS",
    "OVERLAPS": "OVERLAPS",
    "OVERLAP": "OVERLAPS",
    "ADJACENT_TO": "ADJACENT_TO",
    "ADJACENT": "ADJACENT_TO",
    "TOUCHES": "TOUCHES",
    "TOUCH": "TOUCHES",
    "INTERSECTS": "INTERSECTS",
    "INTERSECT": "INTERSECTS",
    "CROSSES": "CROSSES",
    "CROSS": "CROSSES",
    "COVERS": "COVERS",
    "COVER": "COVERS",
    "NEAR": "NEAR",
    "NORTH_OF": "NORTH_OF",
    "SOUTH_OF": "SOUTH_OF",
    "EAST_OF": "EAST_OF",
    "WEST_OF": "WEST_OF",
    "NORTHEAST_OF": "NORTHEAST_OF",
    "NORTHWEST_OF": "NORTHWEST_OF",
    "SOUTHEAST_OF": "SOUTHEAST_OF",
    "SOUTHWEST_OF": "SOUTHWEST_OF",
    "PAGE_LEFT_OF": "PAGE_LEFT_OF",
    "PAGE_RIGHT_OF": "PAGE_RIGHT_OF",
    "PAGE_ABOVE": "PAGE_ABOVE",
    "PAGE_BELOW": "PAGE_BELOW",
    "PAGE_ABOVE_LEFT_OF": "PAGE_ABOVE_LEFT_OF",
    "PAGE_ABOVE_RIGHT_OF": "PAGE_ABOVE_RIGHT_OF",
    "PAGE_BELOW_LEFT_OF": "PAGE_BELOW_LEFT_OF",
    "PAGE_BELOW_RIGHT_OF": "PAGE_BELOW_RIGHT_OF",
}

_SEMANTIC_RELATION_ALIASES = {
    "DISTRIBUTED_IN": "DISTRIBUTED_IN",
    "OCCURS_IN": "DISTRIBUTED_IN",
    "DEVELOPED_IN": "DEVELOPED_IN",
    "PART_OF": "PART_OF",
    "BELONGS_TO": "PART_OF",
    "HAS_LITHOLOGY": "HAS_LITHOLOGY",
    "HAS_FACIES": "HAS_FACIES",
}
_SYMMETRIC_RELATIONS = {"OVERLAPS", "ADJACENT_TO", "TOUCHES", "INTERSECTS", "NEAR"}
_LITHOLOGY_ENTITY_TYPES = {"lithology", "lithofacies"}
_FACIES_ENTITY_TYPES = {"sedimentary_facies", "subfacies", "microfacies", "paleogeographic_unit"}
_STRUCTURAL_ENTITY_TYPES = {"structural_unit", "depression", "sag", "uplift", "slope", "structural_belt"}
_CONTAINMENT_RELATIONS = {"LOCATED_IN", "WITHIN", "CONTAINS", "DISTRIBUTED_IN", "DEVELOPED_IN", "PART_OF"}
_PROXIMITY_RELATIONS = {"ADJACENT_TO", "NEAR"}
_DIRECTION_RELATIONS = {
    "NORTH_OF", "SOUTH_OF", "EAST_OF", "WEST_OF",
    "NORTHEAST_OF", "NORTHWEST_OF", "SOUTHEAST_OF", "SOUTHWEST_OF",
    "PAGE_LEFT_OF", "PAGE_RIGHT_OF", "PAGE_ABOVE", "PAGE_BELOW",
    "PAGE_ABOVE_LEFT_OF", "PAGE_ABOVE_RIGHT_OF",
    "PAGE_BELOW_LEFT_OF", "PAGE_BELOW_RIGHT_OF",
}
_EVIDENCE_SCOPES = {"visual", "caption", "context"}
_DIRECTION_RELATION_ALIASES = {
    "north": "NORTH_OF",
    "south": "SOUTH_OF",
    "east": "EAST_OF",
    "west": "WEST_OF",
    "northeast": "NORTHEAST_OF",
    "northwest": "NORTHWEST_OF",
    "southeast": "SOUTHEAST_OF",
    "southwest": "SOUTHWEST_OF",
    "page_left": "PAGE_LEFT_OF",
    "page_right": "PAGE_RIGHT_OF",
    "page_above": "PAGE_ABOVE",
    "page_below": "PAGE_BELOW",
}


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _confidence(value: Any, default: float = 0.7) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


def _number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", _text(value).lower()).strip("_")


def _evidence_scope(value: Any) -> str:
    scope = _text(value).lower()
    return scope if scope in _EVIDENCE_SCOPES else "visual"


def _deduplicate_ids(records: list[dict[str, Any]], prefix: str, used: set[str]) -> None:
    for index, item in enumerate(records, start=1):
        base = _text(item.get("id")) or f"{prefix}_{index}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        item["id"] = candidate
        used.add(candidate)


def _normalize_record(item: Mapping[str, Any]) -> dict[str, Any]:
    record = _sanitize(dict(item))
    record["name"] = _text(record.get("name"))
    record["evidence"] = _text(record.get("evidence"))
    record["evidence_scope"] = _evidence_scope(record.get("evidence_scope"))
    record["confidence"] = _confidence(record.get("confidence"))
    return record


def _relation_key(value: Any) -> str:
    """把模型关系名转换为稳定的大写下划线格式。"""

    return re.sub(r"[^A-Z]+", "_", _text(value).upper()).strip("_")


def _semanticize_membership(relation: dict[str, Any], entity_types: Mapping[str, str]) -> dict[str, Any]:
    """阻止把岩性、沉积相和构造隶属关系泛化为 LOCATED_IN。"""

    relation_type = _text(relation.get("type")).upper()
    source_id = _text(relation.get("source_id"))
    target_id = _text(relation.get("target_id"))
    if relation_type == "CONTAINS":
        source_id, target_id = target_id, source_id
    if relation_type not in {"LOCATED_IN", "WITHIN", "CONTAINS"}:
        return relation
    source_type = entity_types.get(source_id, "")
    if source_type in _LITHOLOGY_ENTITY_TYPES:
        semantic_type, dimension = "DISTRIBUTED_IN", "semantic"
    elif source_type in _FACIES_ENTITY_TYPES:
        semantic_type, dimension = "DEVELOPED_IN", "semantic"
    elif source_type in _STRUCTURAL_ENTITY_TYPES:
        semantic_type, dimension = "PART_OF", "structural"
    else:
        return relation
    converted = dict(relation)
    converted.update({
        "source_id": source_id,
        "target_id": target_id,
        "type": semantic_type,
        "dimension": dimension,
        "original_relation_type": relation_type,
        "normalization_rule": "entity_type_aware_membership",
    })
    return converted


def _normalize_relations(
    values: Any,
    aliases: Mapping[str, str],
    *,
    default_dimension: str,
) -> list[dict[str, Any]]:
    """规范关系端点、类型和公共证据字段，并在同组内去重。"""

    relations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in _items(values):
        if not isinstance(raw, Mapping):
            continue
        item = _normalize_record(raw)
        source_id = _text(item.get("source_id") or item.get("source"))
        target_id = _text(item.get("target_id") or item.get("target"))
        relation_type = aliases.get(_relation_key(item.get("type")))
        if not source_id or not target_id or source_id == target_id or not relation_type:
            continue
        endpoints = tuple(sorted((source_id, target_id))) if relation_type in _SYMMETRIC_RELATIONS else (source_id, target_id)
        key = (endpoints[0], relation_type, endpoints[1])
        if key in seen:
            continue
        seen.add(key)
        item.update({
            "source_id": source_id,
            "target_id": target_id,
            "type": relation_type,
            "dimension": _text(item.get("dimension")) or default_dimension,
            "explicit": bool(item.get("explicit", True)),
        })
        relations.append(item)
    return relations


def _measurement_distances(reconstruction: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    """建立无向实体对到几何边界距离的索引。"""

    distances: dict[tuple[str, str], float] = {}
    for item in _items(reconstruction.get("measurements")):
        if not isinstance(item, Mapping):
            continue
        source_id, target_id = _text(item.get("source_id")), _text(item.get("target_id"))
        distance = _number(item.get("edge_distance_normalized"))
        if source_id and target_id and distance is not None:
            distances[tuple(sorted((source_id, target_id)))] = float(distance)
    return distances


def _sparsify_proximity_relations(
    relations: list[dict[str, Any]],
    reconstruction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """对邻接和邻近关系分别求最小生成森林，消除 A-B-C 三角冗余边。"""

    distances = _measurement_distances(reconstruction)
    kept = [relation for relation in relations if relation.get("type") not in _PROXIMITY_RELATIONS]
    for relation_type in _PROXIMITY_RELATIONS:
        candidates = [relation for relation in relations if relation.get("type") == relation_type]
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            """查找并压缩并查集中的根节点。"""

            parent.setdefault(node, node)
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        def union(first: str, second: str) -> bool:
            """合并两个连通分量，已连通时返回假。"""

            first_root, second_root = find(first), find(second)
            if first_root == second_root:
                return False
            parent[second_root] = first_root
            return True

        candidates.sort(
            key=lambda relation: (
                distances.get(tuple(sorted((_text(relation.get("source_id")), _text(relation.get("target_id"))))), math.inf),
                -_confidence(relation.get("confidence")),
            )
        )
        for relation in candidates:
            source_id, target_id = _text(relation.get("source_id")), _text(relation.get("target_id"))
            if source_id and target_id and union(source_id, target_id):
                kept.append(relation)
    return kept


def _sparsify_direction_relations(
    relations: list[dict[str, Any]],
    reconstruction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """每个源实体只保留一条最近且置信度最高的方位关系。"""

    distances = _measurement_distances(reconstruction)
    kept = [relation for relation in relations if relation.get("type") not in _DIRECTION_RELATIONS]
    by_source: dict[str, list[dict[str, Any]]] = {}
    for relation in relations:
        if relation.get("type") in _DIRECTION_RELATIONS:
            by_source.setdefault(_text(relation.get("source_id")), []).append(relation)
    for candidates in by_source.values():
        candidates.sort(
            key=lambda relation: (
                distances.get(tuple(sorted((_text(relation.get("source_id")), _text(relation.get("target_id"))))), math.inf),
                -_confidence(relation.get("confidence")),
            )
        )
        kept.append(candidates[0])
    return kept


def _canonical_containment_edge(relation: Mapping[str, Any]) -> tuple[str, str]:
    """把 CONTAINS 反向转换为统一的子对象到父容器方向。"""

    source_id, target_id = _text(relation.get("source_id")), _text(relation.get("target_id"))
    return (target_id, source_id) if relation.get("type") == "CONTAINS" else (source_id, target_id)


def _reduce_transitive_containment(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """删除可由更近一级容器路径推出的跨级包含关系。"""

    containment_candidates = [relation for relation in relations if relation.get("type") in _CONTAINMENT_RELATIONS]
    others = [relation for relation in relations if relation.get("type") not in _CONTAINMENT_RELATIONS]
    by_edge: dict[tuple[str, str], dict[str, Any]] = {}
    semantic_priority = {"DISTRIBUTED_IN", "DEVELOPED_IN", "PART_OF"}
    for relation in containment_candidates:
        edge = _canonical_containment_edge(relation)
        previous = by_edge.get(edge)
        score = (relation.get("type") in semantic_priority, bool(relation.get("explicit")), _confidence(relation.get("confidence")))
        previous_score = (
            previous.get("type") in semantic_priority,
            bool(previous.get("explicit")),
            _confidence(previous.get("confidence")),
        ) if previous else (False, False, -1.0)
        if previous is None or score > previous_score:
            by_edge[edge] = relation
    containment = list(by_edge.values())
    edges = [_canonical_containment_edge(relation) for relation in containment]

    def has_alternative_path(source_id: str, target_id: str, excluded_index: int) -> bool:
        """检查排除当前边后是否仍存在源到目标的有向路径。"""

        adjacency: dict[str, set[str]] = {}
        for index, (source, target) in enumerate(edges):
            if index != excluded_index:
                adjacency.setdefault(source, set()).add(target)
        pending = list(adjacency.get(source_id, set()))
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target_id:
                return True
            if current not in visited:
                visited.add(current)
                pending.extend(adjacency.get(current, set()) - visited)
        return False

    kept_containment = [
        relation
        for index, relation in enumerate(containment)
        if not has_alternative_path(*edges[index], index)
    ]
    return others + kept_containment


def _reduce_relation_density(
    spatial_relations: list[dict[str, Any]],
    semantic_relations: list[dict[str, Any]],
    reconstruction: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """统一执行邻接森林、方位限流和跨级包含约简。"""

    sparse_spatial = _sparsify_proximity_relations(spatial_relations, reconstruction)
    sparse_spatial = _sparsify_direction_relations(sparse_spatial, reconstruction)
    combined = _reduce_transitive_containment(sparse_spatial + semantic_relations)
    return (
        [relation for relation in combined if relation.get("dimension") == "spatial"],
        [relation for relation in combined if relation.get("dimension") != "spatial"],
    )


def normalize_map_spatial_result(
    task: ImageExtractionTask,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """清洗模型 JSON，补全实体类型并重建可计算地图几何。"""

    result = _sanitize(deepcopy(dict(payload)))
    result["schema_version"] = "map_spatial.v2"
    map_item = _normalize_record(_mapping(result.get("map")))
    map_item["id"] = _text(map_item.get("id")) or "map"
    map_item["title"] = _text(map_item.get("title")) or task.caption.splitlines()[0].strip() or "地图"
    map_item["theme"] = _text(map_item.get("theme")) or map_item["title"]
    result["map"] = map_item

    used_ids = {map_item["id"]}
    units = [
        _normalize_record(item)
        for raw in _items(result.get("stratigraphic_units"))
        if isinstance(raw, Mapping) and (item := _mapping(raw)) and _text(item.get("name"))
    ]
    _deduplicate_ids(units, "strat", used_ids)
    for item in units:
        rank = _slug(item.get("rank"))
        item["rank"] = rank if rank in STRATIGRAPHIC_TYPE_MAP else "unknown"
    result["stratigraphic_units"] = units

    entities = [
        _normalize_record(item)
        for raw in _items(result.get("entities"))
        if isinstance(raw, Mapping) and (item := _mapping(raw)) and _text(item.get("name"))
    ]
    _deduplicate_ids(entities, "object", used_ids)
    for item in entities:
        item["type"] = _slug(item.get("type")) or "geographic_location"
        mapped_type, type_zh = ENTITY_TYPE_MAP.get(item["type"], ("GeologicalSpatialEntity", "地质空间对象"))
        item["entity_type"] = mapped_type
        item["entity_type_zh"] = type_zh
        item["geometry"] = _mapping(item.get("geometry"))
        item["attributes"] = _mapping(item.get("attributes"))
    result["entities"] = entities

    attributes = [
        _normalize_record(item)
        for raw in _items(result.get("geological_attributes"))
        if isinstance(raw, Mapping) and (item := _mapping(raw)) and _text(item.get("name"))
    ]
    _deduplicate_ids(attributes, "attribute", used_ids)
    for item in attributes:
        item["subject_ids"] = [_text(value) for value in _items(item.get("subject_ids")) if _text(value)]
        item["value"] = _number(item.get("value")) if not isinstance(item.get("value"), str) else _text(item.get("value"))
        item["minimum"] = _number(item.get("minimum"))
        item["maximum"] = _number(item.get("maximum"))
        item["unit"] = _text(item.get("unit"))
    result["geological_attributes"] = attributes

    features = [
        _normalize_record(item)
        for raw in _items(result.get("secondary_features"))
        if isinstance(raw, Mapping) and (item := _mapping(raw))
    ]
    _deduplicate_ids(features, "feature", used_ids)
    kept_features: list[dict[str, Any]] = []
    for item in features:
        feature_type = _slug(item.get("type"))
        if feature_type not in SECONDARY_TYPE_MAP:
            continue
        item["type"] = feature_type
        item["name"] = _text(item.get("name")) or SECONDARY_TYPE_MAP[feature_type][1]
        item["subject_ids"] = [_text(value) for value in _items(item.get("subject_ids")) if _text(value)]
        item["source_id"] = _text(item.get("source_id"))
        item["target_id"] = _text(item.get("target_id"))
        item["direction"] = _slug(item.get("direction"))
        item["geometry"] = _mapping(item.get("geometry"))
        kept_features.append(item)
    result["secondary_features"] = kept_features

    entity_types = {item["id"]: item["type"] for item in entities}
    spatial_relations = _normalize_relations(
        result.get("spatial_relations"),
        _SPATIAL_RELATION_ALIASES,
        default_dimension="spatial",
    )
    semantic_relations = _normalize_relations(
        result.get("semantic_relations"),
        _SEMANTIC_RELATION_ALIASES,
        default_dimension="semantic",
    )
    kept_spatial: list[dict[str, Any]] = []
    for relation in spatial_relations:
        converted = _semanticize_membership(relation, entity_types)
        if converted.get("dimension") == "spatial":
            kept_spatial.append(converted)
        else:
            semantic_relations.append(converted)

    reconstruction = reconstruct_map_geometry(task, result)
    for relation in reconstruction["computed_relations"]:
        converted = _semanticize_membership(relation, entity_types)
        target = kept_spatial if converted.get("dimension") == "spatial" else semantic_relations
        target.append(converted)
    kept_spatial, semantic_relations = _reduce_relation_density(
        kept_spatial,
        semantic_relations,
        reconstruction,
    )
    result["spatial_relations"] = _normalize_relations(kept_spatial, _SPATIAL_RELATION_ALIASES, default_dimension="spatial")
    result["semantic_relations"] = _normalize_relations(semantic_relations, _SEMANTIC_RELATION_ALIASES, default_dimension="semantic")
    result["geometry_reconstruction"] = reconstruction
    result["georeference"] = reconstruction["georeference"]
    result["uncertainties"] = [_text(value) for value in _items(result.get("uncertainties")) if _text(value)]
    return result


def _stable_id(task: ImageExtractionTask, kind: str, local_id: str) -> str:
    digest = hashlib.sha1(f"{task.image_id}|map_spatial|{kind}|{local_id}".encode("utf-8")).hexdigest()[:12]
    return f"{task.image_id}:map:{kind}:{digest}"


class MapSpatialGraphBuilder:
    """把规范化地图对象转换为带证据、稳定 ID 和有效引用的知识图。"""

    def __init__(self, task: ImageExtractionTask, visual: Mapping[str, Any]) -> None:
        self.task = task
        self.visual = dict(visual)
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}
        self.aliases: dict[str, str] = {}
        self.dropped_relations: list[dict[str, Any]] = []
        self.map_id = ""

    def build(self) -> Graph:
        self._add_map()
        self._add_stratigraphic_units()
        self._add_primary_entities()
        self._add_geological_attributes()
        self._add_secondary_features()
        self._add_semantic_relations()
        self._add_spatial_relations()
        graph = Graph.from_chunk(
            document_id=self.task.document_id,
            chunk_id=self.task.chunk_id,
            modality=SourceModality.IMAGE,
            entities=self.entities.values(),
            relations=self.relations.values(),
            raw_response=self.visual,
            stage="stage_04_image_map_spatial_extraction",
        )
        graph.metadata.extra.update(
            {
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
                "schema_version": "map_spatial.v2",
                "algorithm": "VLM证据抽取 + 类型约束语义关系 + 可计算几何重建 + 确定性Graph装配",
                "primary_entity_count": len(_items(self.visual.get("stratigraphic_units"))) + len(_items(self.visual.get("entities"))),
                "geological_attribute_count": len(_items(self.visual.get("geological_attributes"))),
                "secondary_feature_count": len(_items(self.visual.get("secondary_features"))),
                "spatial_relation_count": sum(
                    relation.type in _SPATIAL_RELATION_ALIASES.values()
                    for relation in self.relations.values()
                ),
                "semantic_relation_count": sum(
                    relation.type in _SEMANTIC_RELATION_ALIASES.values()
                    for relation in self.relations.values()
                ),
                "geometry_reconstruction": _mapping(self.visual.get("geometry_reconstruction")),
                "dropped_relations": self.dropped_relations,
                "uncertainties": _items(self.visual.get("uncertainties")),
            }
        )
        return graph

    def _entity(
        self,
        local_id: str,
        name: str,
        entity_type: str,
        type_zh: str,
        item: Mapping[str, Any],
    ) -> str:
        graph_id = _stable_id(self.task, "entity", local_id)
        evidence = _text(item.get("evidence")) or self.task.caption
        attributes = {
            key: _sanitize(value)
            for key, value in item.items()
            if key not in {"id", "name", "title", "evidence", "confidence", "evidence_scope"}
        }
        attributes.update(
            {
                "confidence": _confidence(item.get("confidence")),
                "evidence_scope": _evidence_scope(item.get("evidence_scope")),
                "entity_type": entity_type,
                "entity_type_zh": type_zh,
            }
        )
        self.entities[graph_id] = Entity(
            id=graph_id,
            name=name or local_id,
            type=entity_type,
            type_zh=type_zh,
            attributes=attributes,
            provenance=evidence,
            metadata={
                "source_modality": "image",
                "local_id": local_id,
                "visual_evidence": evidence,
                "evidence_scope": attributes["evidence_scope"],
                "entity_type": entity_type,
                "entity_type_zh": type_zh,
            },
        )
        self.aliases[local_id] = graph_id
        if name:
            self.aliases.setdefault(name, graph_id)
        return graph_id

    def _resolve(self, value: Any) -> str | None:
        return self.aliases.get(_text(value))

    def _relation(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        *,
        local_id: str,
        item: Mapping[str, Any],
        inferred: bool = False,
    ) -> None:
        if source_id not in self.entities or target_id not in self.entities or source_id == target_id:
            return
        symmetric_key = "|".join(sorted((source_id, target_id))) if relation_type in _SYMMETRIC_RELATIONS else f"{source_id}|{target_id}"
        relation_id = _stable_id(self.task, "relation", f"{relation_type}|{symmetric_key}")
        if relation_id in self.relations:
            return
        source = self.entities[source_id]
        target = self.entities[target_id]
        evidence = _text(item.get("evidence")) or self.task.caption
        dimension = _text(item.get("dimension"))
        is_spatial = dimension == "spatial" or (
            not dimension
            and (
            relation_type in _SPATIAL_RELATION_ALIASES.values()
            or relation_type.endswith("_OF")
            or relation_type in {"PAGE_ABOVE", "PAGE_BELOW", "TRANSITIONS_TO", "HAS_RELATIVE_DIRECTION", "HAS_SPATIAL_TREND"}
            )
        )
        relation_attributes = {
            key: _sanitize(value)
            for key, value in item.items()
            if key
            not in {
                "id", "name", "source", "source_id", "target", "target_id", "type",
                "evidence", "evidence_scope", "confidence", "explicit", "dimension",
            }
        }
        relation_attributes.update(
            {
                "dimension": dimension or ("spatial" if is_spatial else "semantic"),
                "explicit": bool(item.get("explicit", not inferred)),
                "inferred": inferred or bool(item.get("computed", False)),
                "confidence": _confidence(item.get("confidence")),
                "evidence_scope": _evidence_scope(item.get("evidence_scope")),
            }
        )
        self.relations[relation_id] = Relation(
            id=relation_id,
            type=relation_type,
            relation_name=RELATION_ZH.get(relation_type, relation_type),
            type_zh=RELATION_ZH.get(relation_type, relation_type),
            source_id=source_id,
            source_name=source.name,
            source_type=source.type,
            target_id=target_id,
            target_name=target.name,
            target_type=target.type,
            attributes=relation_attributes,
            provenance=evidence,
            metadata={"source_modality": "image", "visual_evidence": evidence, "local_id": local_id},
        )

    def _depicts(self, target_id: str, local_id: str, item: Mapping[str, Any]) -> None:
        self._relation(self.map_id, "DEPICTS", target_id, local_id=f"depicts:{local_id}", item=item, inferred=True)

    def _add_map(self) -> None:
        item = _mapping(self.visual.get("map"))
        local_id = _text(item.get("id")) or "map"
        name = _text(item.get("title")) or _text(item.get("theme")) or self.task.caption or "地图"
        self.map_id = self._entity(local_id, name, "GeologicalMap", "地质图", item)
        theme = _text(item.get("theme"))
        if theme:
            theme_id = self._entity(f"{local_id}:theme", theme, "MapTheme", "地图主题", item)
            self._relation(self.map_id, "HAS_THEME", theme_id, local_id=f"{local_id}:theme", item=item)

    def _add_stratigraphic_units(self) -> None:
        for item in _items(self.visual.get("stratigraphic_units")):
            unit = _mapping(item)
            local_id = _text(unit.get("id"))
            if not local_id:
                continue
            entity_type, type_zh = STRATIGRAPHIC_TYPE_MAP.get(_slug(unit.get("rank")), STRATIGRAPHIC_TYPE_MAP["unknown"])
            entity_id = self._entity(local_id, _text(unit.get("name")), entity_type, type_zh, unit)
            self._depicts(entity_id, local_id, unit)

    def _add_primary_entities(self) -> None:
        for item in _items(self.visual.get("entities")):
            record = _mapping(item)
            local_id = _text(record.get("id"))
            if not local_id:
                continue
            raw_type = _slug(record.get("type"))
            entity_type, type_zh = ENTITY_TYPE_MAP.get(raw_type, ("GeologicalSpatialEntity", "地质空间对象"))
            entity_id = self._entity(local_id, _text(record.get("name")), entity_type, type_zh, record)
            self._depicts(entity_id, local_id, record)

    def _attribute_value_text(self, item: Mapping[str, Any]) -> str:
        unit = _text(item.get("unit"))
        value = item.get("value")
        minimum = item.get("minimum")
        maximum = item.get("maximum")
        if value not in (None, ""):
            return f"{value}{unit}"
        if minimum is not None and maximum is not None:
            return f"{minimum}–{maximum}{unit}"
        if minimum is not None:
            return f"≥{minimum}{unit}"
        if maximum is not None:
            return f"≤{maximum}{unit}"
        return _text(item.get("qualifier"))

    def _add_geological_attributes(self) -> None:
        for item in _items(self.visual.get("geological_attributes")):
            record = _mapping(item)
            local_id = _text(record.get("id"))
            if not local_id:
                continue
            attribute_id = self._entity(local_id, _text(record.get("name")), "GeologicalAttribute", "地质属性", record)
            self._depicts(attribute_id, local_id, record)
            subjects = [self._resolve(value) for value in _items(record.get("subject_ids"))]
            subjects = [value for value in subjects if value]
            if not subjects:
                subjects = [self.map_id]
            for index, subject_id in enumerate(dict.fromkeys(subjects), start=1):
                self._relation(subject_id, "HAS_GEOLOGICAL_ATTRIBUTE", attribute_id, local_id=f"{local_id}:subject:{index}", item=record)
            value_text = self._attribute_value_text(record)
            if value_text:
                value_record = {
                    "value": record.get("value"),
                    "minimum": record.get("minimum"),
                    "maximum": record.get("maximum"),
                    "unit": record.get("unit"),
                    "qualifier": record.get("qualifier"),
                    "evidence": record.get("evidence"),
                    "evidence_scope": record.get("evidence_scope"),
                    "confidence": record.get("confidence"),
                }
                value_id = self._entity(f"{local_id}:value", value_text, "AttributeValue", "属性值/范围", value_record)
                self._relation(attribute_id, "HAS_VALUE", value_id, local_id=f"{local_id}:value", item=record)

    def _add_secondary_features(self) -> None:
        for item in _items(self.visual.get("secondary_features")):
            record = _mapping(item)
            local_id = _text(record.get("id"))
            feature_type = _slug(record.get("type"))
            if not local_id or feature_type not in SECONDARY_TYPE_MAP:
                continue
            entity_type, type_zh = SECONDARY_TYPE_MAP[feature_type]
            feature_id = self._entity(local_id, _text(record.get("name")) or type_zh, entity_type, type_zh, record)
            self._depicts(feature_id, local_id, record)
            for index, subject_ref in enumerate(_items(record.get("subject_ids")), start=1):
                if subject_id := self._resolve(subject_ref):
                    relation_type = "HAS_SPATIAL_TREND" if feature_type == "spatial_trend" else "HAS_SPATIAL_FEATURE"
                    self._relation(subject_id, relation_type, feature_id, local_id=f"{local_id}:subject:{index}", item=record)

            source_id = self._resolve(record.get("source_id"))
            target_id = self._resolve(record.get("target_id"))
            if source_id and target_id and feature_type == "facies_transition":
                self._relation(source_id, "TRANSITIONS_TO", target_id, local_id=f"{local_id}:transition", item=record)
            if source_id and target_id and feature_type == "relative_direction":
                direction = _slug(record.get("direction"))
                relation_type = _DIRECTION_RELATION_ALIASES.get(
                    direction,
                    direction.upper() if direction else "HAS_RELATIVE_DIRECTION",
                )
                self._relation(source_id, relation_type, target_id, local_id=f"{local_id}:direction", item=record)

    def _add_spatial_relations(self) -> None:
        for item in _items(self.visual.get("spatial_relations")):
            record = _mapping(item)
            source_id = self._resolve(record.get("source_id"))
            target_id = self._resolve(record.get("target_id"))
            relation_type = _text(record.get("type")).upper()
            if not source_id or not target_id or relation_type not in _SPATIAL_RELATION_ALIASES.values():
                self.dropped_relations.append(
                    {
                        "source_id": record.get("source_id"),
                        "type": relation_type,
                        "target_id": record.get("target_id"),
                        "reason": "unresolved_endpoint_or_unsupported_relation",
                    }
                )
                continue
            self._relation(source_id, relation_type, target_id, local_id=f"spatial:{len(self.relations)}", item=record)

    def _add_semantic_relations(self) -> None:
        """装配类型约束后的岩性、沉积相和构造隶属关系。"""

        for item in _items(self.visual.get("semantic_relations")):
            record = _mapping(item)
            source_id = self._resolve(record.get("source_id"))
            target_id = self._resolve(record.get("target_id"))
            relation_type = _text(record.get("type")).upper()
            if not source_id or not target_id or relation_type not in _SEMANTIC_RELATION_ALIASES.values():
                self.dropped_relations.append(
                    {
                        "source_id": record.get("source_id"),
                        "type": relation_type,
                        "target_id": record.get("target_id"),
                        "reason": "unresolved_endpoint_or_unsupported_semantic_relation",
                    }
                )
                continue
            self._relation(source_id, relation_type, target_id, local_id=f"semantic:{len(self.relations)}", item=record)


def build_map_spatial_graph(task: ImageExtractionTask, visual: Mapping[str, Any]) -> Graph:
    """公开的地图 Graph 装配入口，供抽取器和离线测试复用。"""

    return MapSpatialGraphBuilder(task, visual).build()


__all__ = [
    "ENTITY_TYPE_MAP",
    "RELATION_ZH",
    "SECONDARY_TYPE_MAP",
    "STRATIGRAPHIC_TYPE_MAP",
    "MapSpatialGraphBuilder",
    "build_map_spatial_graph",
    "normalize_map_spatial_result",
]
