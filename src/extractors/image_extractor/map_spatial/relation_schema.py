"""地图实体类型与允许关系的确定性白名单。

本文件根据实体类型生成候选关系，不允许 VLM 自由创造关系名称。
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


DIRECTION_RELATIONS = (
    "north_of",
    "south_of",
    "east_of",
    "west_of",
    "northeast_of",
    "northwest_of",
    "southeast_of",
    "southwest_of",
)

TYPE_ALIASES = {
    "geographic_location": "place",
    "location": "place",
    "province": "place",
    "city": "place",
    "county": "place",
    "town": "place",
    "gas_field": "hydrocarbon_field",
    "oil_field": "hydrocarbon_field",
    "oil_gas_field": "hydrocarbon_field",
    "microfacies": "sedimentary_microfacies",
    "sedimentary_facies": "sedimentary_microfacies",
    "subfacies": "sedimentary_microfacies",
    "paleogeographic_unit": "depositional_zone",
    "section_line": "profile_line",
    "boundary": "geological_boundary",
    "stratigraphic_erosion_line": "geological_boundary",
    "high_value_zone": "reservoir_property_zone",
    "medium_value_zone": "reservoir_property_zone",
    "low_value_zone": "reservoir_property_zone",
}

TYPE_ZH = {
    "map_spatial": "地图与平面空间图",
    "place": "地名",
    "well": "井",
    "hydrocarbon_field": "油气田",
    "depositional_zone": "沉积古地理单元",
    "sedimentary_microfacies": "沉积微相",
    "lithology": "岩性",
    "paleogeographic_landmass": "古陆",
    "paleogeographic_landmass_class": "古陆图例类型",
    "profile_line": "地质剖面位置线",
    "map_boundary": "范围边界",
    "geological_boundary": "地质边界",
    "legend_class": "图例类别",
    "reservoir_property": "储层物性参数",
    "parameter_contour_set": "参数等值线",
    "reservoir_property_zone": "储层参数分区",
    "structural_unit": "构造单元",
    "UNMAPPED": "未映射空间对象",
}

RELATION_ZH = {
    "has_dominant_lithology": "主要岩性",
    "instance_of": "属于图例类别",
    "has_facies": "具有沉积微相",
    "maps_property": "表示储层参数",
    "contours_property": "等值线表示",
    "defines_zone": "限定参数区",
    "located_in": "位于",
    "located_near": "邻近",
    "near_boundary_of": "邻近边界",
    "near_or_within": "邻近或位于",
    "outside_erosion_boundary": "位于剥蚀线外",
    "adjacent_to": "相邻",
    "surrounds": "包围",
    "contained_in": "包含于",
    "transition_to": "过渡为",
    "overlaps": "重叠",
    "along_boundary_of": "沿边界分布",
    "crosses": "穿过",
    "high_value_of": "为高值区",
    "low_value_of": "为低值区",
    "north_of": "北",
    "south_of": "南",
    "east_of": "东",
    "west_of": "西",
    "northeast_of": "东北",
    "northwest_of": "西北",
    "southeast_of": "东南",
    "southwest_of": "西南",
}

RELATION_SCHEMA: dict[tuple[str, str], tuple[str, ...]] = {
    ("map_spatial", "sedimentary_microfacies"): ("has_facies",),
    ("map_spatial", "reservoir_property"): ("maps_property",),
    ("depositional_zone", "lithology"): ("has_dominant_lithology",),
    ("sedimentary_microfacies", "lithology"): ("has_dominant_lithology",),
    ("paleogeographic_landmass", "paleogeographic_landmass_class"): ("instance_of",),
    ("depositional_zone", "paleogeographic_landmass_class"): ("instance_of",),
    ("well", "sedimentary_microfacies"): ("located_in", "near_boundary_of"),
    ("place", "sedimentary_microfacies"): ("located_in",),
    ("well", "geological_boundary"): ("outside_erosion_boundary",),
    ("place", "geological_boundary"): ("outside_erosion_boundary",),
    ("well", "reservoir_property_zone"): ("located_in", "near_or_within"),
    ("place", "reservoir_property_zone"): ("located_in", "near_or_within"),
    ("parameter_contour_set", "reservoir_property"): ("contours_property",),
    ("parameter_contour_set", "reservoir_property_zone"): ("defines_zone",),
    ("reservoir_property_zone", "reservoir_property"): ("high_value_of", "low_value_of"),
    ("depositional_zone", "depositional_zone"): ("adjacent_to", "surrounds", "contained_in", "transition_to"),
    ("sedimentary_microfacies", "sedimentary_microfacies"): ("adjacent_to", "surrounds", "contained_in", "transition_to"),
    ("hydrocarbon_field", "depositional_zone"): ("located_in", "overlaps", "along_boundary_of"),
    ("hydrocarbon_field", "sedimentary_microfacies"): ("located_in", "overlaps", "along_boundary_of"),
    ("profile_line", "depositional_zone"): ("crosses",),
    ("profile_line", "sedimentary_microfacies"): ("crosses",),
    ("profile_line", "hydrocarbon_field"): ("crosses",),
    ("place", "place"): DIRECTION_RELATIONS,
}


def slug(value: Any) -> str:
    """把任意标识规范为小写下划线形式。"""

    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def canonical_entity_type(value: Any) -> tuple[str, str]:
    """规范实体类型；未知类型保留原值并标记为 UNMAPPED。"""

    raw_type = slug(value)
    canonical = TYPE_ALIASES.get(raw_type, raw_type)
    if canonical not in TYPE_ZH:
        return "UNMAPPED", raw_type
    return canonical, raw_type


class RelationSchemaGenerator:
    """根据当前实体类型组合生成允许的关系候选。"""

    def generate(
        self,
        entities: Sequence[Mapping[str, Any]],
        *,
        spatial_candidate_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        """生成分组白名单和具体实体对候选。"""

        candidate_set = {str(value) for value in spatial_candidate_ids}
        groups: dict[tuple[str, str], tuple[str, ...]] = {}
        pairs: list[dict[str, Any]] = []
        for source in entities:
            source_id = str(source.get("id") or "")
            source_type = str(source.get("type") or "UNMAPPED")
            for target in entities:
                target_id = str(target.get("id") or "")
                target_type = str(target.get("type") or "UNMAPPED")
                if not source_id or not target_id or source_id == target_id:
                    continue
                allowed = RELATION_SCHEMA.get((source_type, target_type), ())
                target_attributes = target.get("attributes") if isinstance(target.get("attributes"), Mapping) else {}
                if target_attributes.get("legend_item"):
                    legend_semantic_type = str(target_attributes.get("semantic_type") or target_type)
                    if target_type == "lithology" and source_type in {"depositional_zone", "sedimentary_microfacies"}:
                        allowed = tuple(dict.fromkeys((*allowed, "has_dominant_lithology")))
                    elif target_type == "legend_class" or target_type.endswith("_class") or source_type == legend_semantic_type:
                        allowed = tuple(dict.fromkeys((*allowed, "instance_of")))
                if source_id in candidate_set and target_id in candidate_set:
                    allowed = tuple(dict.fromkeys((*allowed, *DIRECTION_RELATIONS)))
                if not allowed:
                    continue
                groups[(source_type, target_type)] = allowed
                pairs.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_type": source_type,
                    "target_type": target_type,
                    "allowed_relations": list(allowed),
                })
        return {
            "groups": [
                {"source_type": source_type, "target_type": target_type, "allowed_relations": list(allowed)}
                for (source_type, target_type), allowed in groups.items()
            ],
            "pairs": pairs,
        }

    @staticmethod
    def is_allowed(source_type: str, target_type: str, relation_type: str, *, both_spatial_candidates: bool = False) -> bool:
        """检查具体类型对是否允许指定关系。"""

        relation = slug(relation_type)
        allowed = RELATION_SCHEMA.get((source_type, target_type), ())
        return relation in allowed or both_spatial_candidates and relation in DIRECTION_RELATIONS


__all__ = [
    "DIRECTION_RELATIONS",
    "RELATION_SCHEMA",
    "RELATION_ZH",
    "TYPE_ALIASES",
    "TYPE_ZH",
    "RelationSchemaGenerator",
    "canonical_entity_type",
    "slug",
]
