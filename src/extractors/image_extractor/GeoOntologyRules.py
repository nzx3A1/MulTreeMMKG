from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple


@dataclass(frozen=True)
class RelationRule:
    head_type: str
    tail_type: str
    relation_types: Tuple[str, ...]


class GeoOntologyRules:
    """Petroleum-geology ontology constraints for image extraction."""

    ENTITY_TYPES: Dict[str, str] = {
        "Stratum": "地层、层位、旋回",
        "Lithology": "岩性、矿物或岩石组合",
        "Reservoir": "储层、砂体、有效储集空间",
        "SourceRock": "烃源岩、煤层、有机质富集层",
        "Fault": "断层、断裂",
        "Structure": "构造单元、盆地、隆起、凹陷、区带",
        "Facies": "沉积相、微相、相带",
        "Curve": "曲线、图例曲线、测井响应",
        "Axis": "坐标轴、深度轴、比例尺",
        "Parameter": "地质、实验或测井参数",
        "Trend": "趋势、变化方向、异常段",
        "Process": "地质过程、沉积过程、成藏过程",
        "Hydrocarbon": "油气、烃类、流体",
        "EvidenceRegion": "图像证据区域或证据片段",
    }

    RELATION_RULES: Tuple[RelationRule, ...] = (
        RelationRule("Stratum", "Lithology", ("has_lithology",)),
        RelationRule("Stratum", "Reservoir", ("contains",)),
        RelationRule("Stratum", "Facies", ("contains", "has_facies")),
        RelationRule("Stratum", "Stratum", ("contains", "overlies", "underlies", "precedes")),
        RelationRule("Stratum", "Structure", ("located_in",)),
        RelationRule("Structure", "Fault", ("contains", "controlled_by")),
        RelationRule("Structure", "Stratum", ("contains", "located_in")),
        RelationRule("Structure", "Facies", ("contains", "has_facies")),
        RelationRule("Structure", "Structure", ("adjacent_to", "located_in", "contains")),
        RelationRule("Fault", "Reservoir", ("controls", "controlled_by")),
        RelationRule("Curve", "Parameter", ("quantifies",)),
        RelationRule("Curve", "Trend", ("has_trend",)),
        RelationRule("Axis", "Parameter", ("quantifies",)),
        RelationRule("Parameter", "Parameter", ("increases_with", "decreases_with", "positively_correlated_with", "negatively_correlated_with")),
        RelationRule("Process", "Process", ("precedes", "causes")),
        RelationRule("Process", "Reservoir", ("causes", "controls")),
        RelationRule("Process", "Facies", ("causes", "indicates")),
        RelationRule("Hydrocarbon", "Fault", ("migrates_along",)),
        RelationRule("Hydrocarbon", "Reservoir", ("accumulates_in",)),
        RelationRule("Facies", "Process", ("indicates",)),
        RelationRule("Facies", "Facies", ("adjacent_to", "precedes", "contains")),
        RelationRule("Lithology", "Reservoir", ("indicates", "has_component")),
        RelationRule("Lithology", "Lithology", ("interbedded_with", "adjacent_to")),
        RelationRule("EvidenceRegion", "Stratum", ("supports", "aligns_with")),
        RelationRule("EvidenceRegion", "Lithology", ("supports", "aligns_with")),
        RelationRule("EvidenceRegion", "Facies", ("supports", "aligns_with")),
        RelationRule("EvidenceRegion", "Process", ("supports", "aligns_with")),
    )

    RELATION_ALIASES: Dict[str, str] = {
        "包含": "contains",
        "包括": "contains",
        "含有": "contains",
        "位于": "located_in",
        "属于": "located_in",
        "邻接": "adjacent_to",
        "相邻": "adjacent_to",
        "主要岩性": "has_lithology",
        "岩性": "has_lithology",
        "发育相": "has_facies",
        "沉积相": "has_facies",
        "控制": "controls",
        "受控于": "controlled_by",
        "指示": "indicates",
        "量化": "quantifies",
        "表征": "quantifies",
        "趋势": "has_trend",
        "增加": "increases_with",
        "随之增加": "increases_with",
        "降低": "decreases_with",
        "随之降低": "decreases_with",
        "先于": "precedes",
        "导致": "causes",
        "迁移通道": "migrates_along",
        "聚集于": "accumulates_in",
        "互层": "interbedded_with",
        "支撑": "supports",
        "对齐": "aligns_with",
    }

    GENERIC_RELATIONS: Set[str] = {
        "supports",
        "aligns_with",
        "related_to",
        "located_in",
        "adjacent_to",
    }

    def relation_types(self) -> Set[str]:
        values: Set[str] = set(self.GENERIC_RELATIONS)
        for rule in self.RELATION_RULES:
            values.update(rule.relation_types)
        return values

    def entity_types(self) -> Set[str]:
        return set(self.ENTITY_TYPES)

    def normalize_entity_type(self, value: str) -> str:
        text = (value or "").strip()
        if text in self.ENTITY_TYPES:
            return text
        lowered = text.lower()
        for item in self.ENTITY_TYPES:
            if lowered == item.lower():
                return item
        return text or "EvidenceRegion"

    def normalize_relation_type(self, relation: str, relation_type: str = "") -> str:
        for value in (relation_type, relation):
            text = (value or "").strip()
            if not text:
                continue
            if text in self.relation_types():
                return text
            lowered = text.lower()
            for item in self.relation_types():
                if lowered == item.lower():
                    return item
            for alias, normalized in self.RELATION_ALIASES.items():
                if alias in text:
                    return normalized
        return "related_to"

    def is_allowed(self, head_type: str, relation_type: str, tail_type: str) -> bool:
        relation_type = self.normalize_relation_type("", relation_type)
        if relation_type in self.GENERIC_RELATIONS:
            return True

        head_type = self.normalize_entity_type(head_type)
        tail_type = self.normalize_entity_type(tail_type)
        for rule in self.RELATION_RULES:
            if rule.head_type == head_type and rule.tail_type == tail_type and relation_type in rule.relation_types:
                return True
        return False

    def best_effort_allowed(self, head_type: str, relation_type: str, tail_type: str) -> bool:
        relation_type = self.normalize_relation_type("", relation_type)
        if relation_type not in self.relation_types():
            return False
        if not head_type or not tail_type:
            return True
        return self.is_allowed(head_type, relation_type, tail_type)

    def compact_schema_text(self) -> str:
        entity_text = "\n".join(f"- {k}: {v}" for k, v in self.ENTITY_TYPES.items())
        relation_text = "\n".join(
            f"- {rule.head_type} -> {rule.tail_type}: {', '.join(rule.relation_types)}"
            for rule in self.RELATION_RULES
        )
        return f"实体类型:\n{entity_text}\n\n关系模板:\n{relation_text}"

    def infer_entity_type(self, name: str, context: str = "") -> str:
        text = name or ""
        if any(k in text for k in ("组", "段", "层", "旋回", "山1", "山2", "山3")):
            return "Stratum"
        if any(k in text for k in ("砂岩", "泥岩", "煤", "灰岩", "页岩", "岩性")):
            return "Lithology"
        if any(k in text for k in ("储层", "砂体", "孔隙", "渗透")):
            return "Reservoir"
        if any(k in text for k in ("盆地", "隆起", "凹陷", "斜坡", "构造", "区")):
            return "Structure"
        if any(k in text for k in ("断层", "断裂")):
            return "Fault"
        if any(k in text for k in ("河道", "扇", "洼地", "沼泽", "三角洲", "沉积相", "微相")):
            return "Facies"
        if any(k in text for k in ("曲线", "测井", "GR", "AC", "DEN")):
            return "Curve"
        if any(k in text for k in ("增加", "降低", "变薄", "萎缩", "推进", "趋势")):
            return "Trend"
        if any(k in text for k in ("沉积", "成藏", "迁移", "演化", "剥蚀", "供应")):
            return "Process"
        return "EvidenceRegion"

    def allowed_relation_hint(self, head_type: str, tail_type: str) -> List[str]:
        result: List[str] = []
        for rule in self.RELATION_RULES:
            if rule.head_type == head_type and rule.tail_type == tail_type:
                result.extend(rule.relation_types)
        return result


def unique_by_name(items: Iterable[dict]) -> List[dict]:
    seen = set()
    result: List[dict] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(item)
    return result
