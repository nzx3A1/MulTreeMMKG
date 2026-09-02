"""定量图表 VLM 结果的规范化与统一 Graph 装配。"""
from __future__ import annotations

import hashlib
import math
import re
from copy import deepcopy
from typing import Any, Mapping

from model import Entity, Graph, Relation
from model.base import SourceModality

from ..schema_models import ImageExtractionTask


RELATION_ZH = {
    "has_panel": "包含子图",
    "has_axis": "具有坐标轴",
    "has_series": "具有数据序列",
    "uses_x_axis": "使用横轴",
    "uses_y_axis": "使用纵轴",
    "exhibits_trend": "呈现趋势",
    "varies_along": "沿其变化",
    "has_correlation": "具有相关性",
    "relates_x_axis": "关联自变量轴",
    "relates_y_axis": "关联因变量轴",
    "has_threshold": "具有阈值",
    "defined_on": "定义于",
    "applies_to": "适用于",
    "exhibits_temporal_change": "呈现时间变化",
    "occurs_along": "发生于时间轴",
    "has_anomaly": "具有异常",
    "higher_than": "高于",
    "lower_than": "低于",
    "similar_to": "近似于",
    "converges_with": "趋近于",
    "diverges_from": "偏离于",
    "crosses": "与其交叉",
    "more_variable_than": "波动大于",
}

_TIME_PATTERN = re.compile(r"时间|日期|年份|年|月|日|age|time|date|year", re.IGNORECASE)


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
    """递归移除不可 JSON 化的非有限浮点值并保留常见容器。"""

    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _local_id(item: Mapping[str, Any], prefix: str, index: int) -> str:
    return _text(item.get("id")) or f"{prefix}_{index}"


def _deduplicate_ids(records: list[dict[str, Any]], prefix: str) -> None:
    used: set[str] = set()
    for index, item in enumerate(records, start=1):
        base = _local_id(item, prefix, index)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        item["id"] = candidate
        used.add(candidate)


def _caption_sample_size(caption: str, series_name: str) -> int | None:
    """从图注中补全与图例名称紧邻的 N，避免把总样本数错配到其他序列。"""

    name = series_name.strip()
    if not name:
        return None
    compact_caption = re.sub(r"\s+", "", caption)
    compact_name = re.sub(r"\s+", "", name)
    position = compact_caption.find(compact_name)
    if position < 0:
        return None
    window = compact_caption[position : position + len(compact_name) + 24]
    match = re.search(r"N\s*=\s*(\d+)", window, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _evidence_sample_size(series: Mapping[str, Any]) -> int | None:
    """仅接受序列名称或证据文本中明确出现的 N/n 样本量。"""

    evidence = " ".join((_text(series.get("name")), _text(series.get("evidence"))))
    match = re.search(r"(?:^|[\s（(,，;；:：])N\s*=\s*(\d+)", evidence, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def normalize_quantitative_chart_result(
    task: ImageExtractionTask,
    payload: Mapping[str, Any],
    *,
    max_points_per_series: int = 24,
) -> dict[str, Any]:
    """确定性清洗 VLM 输出、修复常见引用并限制密集数据点数量。"""

    result = _sanitize(deepcopy(dict(payload)))
    result["chart"] = _mapping(result.get("chart"))
    for field, prefix in (
        ("panels", "panel"),
        ("axes", "axis"),
        ("series", "series"),
        ("trends", "trend"),
        ("correlations", "correlation"),
        ("thresholds", "threshold"),
        ("temporal_changes", "time_change"),
        ("anomalies", "anomaly"),
    ):
        records = [_mapping(item) for item in _items(result.get(field)) if isinstance(item, Mapping)]
        _deduplicate_ids(records, prefix)
        result[field] = records
    result["comparisons"] = [
        _mapping(item) for item in _items(result.get("comparisons")) if isinstance(item, Mapping)
    ]
    result["uncertainties"] = [_text(item) for item in _items(result.get("uncertainties")) if _text(item)]

    panels = result["panels"]
    if not panels:
        panels.append({"id": "panel_1", "label": "", "title": "", "confidence": 1.0})
    panel_ids = {item["id"] for item in panels}
    default_panel_id = panels[0]["id"]

    axes = result["axes"]
    axis_ids = {item["id"] for item in axes}
    first_x = next((item["id"] for item in axes if _text(item.get("orientation")) in {"x", "secondary_x"}), "")
    first_y = next((item["id"] for item in axes if _text(item.get("orientation")) in {"y", "secondary_y"}), "")
    for axis in axes:
        if _text(axis.get("panel_id")) not in panel_ids:
            axis["panel_id"] = default_panel_id
        scale = _text(axis.get("scale")).lower()
        axis["scale"] = scale if scale in {"linear", "log10", "ln", "categorical", "datetime", "unknown"} else "unknown"
        axis["minimum"] = _number(axis.get("minimum"))
        axis["maximum"] = _number(axis.get("maximum"))
        axis["confidence"] = _confidence(axis.get("confidence"))
        axis["categories"] = _items(axis.get("categories"))
        axis["ticks"] = _items(axis.get("ticks"))

    point_limit = max(0, int(max_points_per_series))
    series_ids: set[str] = set()
    for series in result["series"]:
        series_ids.add(series["id"])
        if _text(series.get("panel_id")) not in panel_ids:
            series["panel_id"] = default_panel_id
        if _text(series.get("x_axis_id")) not in axis_ids and first_x:
            series["x_axis_id"] = first_x
        if _text(series.get("y_axis_id")) not in axis_ids and first_y:
            series["y_axis_id"] = first_y
        series["confidence"] = _confidence(series.get("confidence"))
        model_sample_size = _number(series.get("sample_size"))
        explicit_sample_size = _caption_sample_size(task.caption, _text(series.get("name")))
        if explicit_sample_size is None:
            explicit_sample_size = _evidence_sample_size(series)
        series["sample_size"] = explicit_sample_size
        if model_sample_size is not None and explicit_sample_size is None:
            result["uncertainties"].append(
                f"序列 {series.get('name') or series['id']} 缺少明确 N/n 证据，已忽略模型给出的 sample_size={model_sample_size}"
            )
        points = [_mapping(point) for point in _items(series.get("data_points")) if isinstance(point, Mapping)]
        series["data_points"] = points[:point_limit]
        if len(points) > point_limit:
            result["uncertainties"].append(
                f"序列 {series.get('name') or series['id']} 的代表点由 {len(points)} 个截断为 {point_limit} 个"
            )

    for field in ("trends", "correlations", "thresholds", "temporal_changes", "anomalies"):
        for item in result[field]:
            item["confidence"] = _confidence(item.get("confidence"))
            item["series_ids"] = [
                _text(series_id) for series_id in _items(item.get("series_ids")) if _text(series_id) in series_ids
            ]

    for correlation in result["correlations"]:
        correlation["coefficient"] = _number(correlation.get("coefficient"))
    for threshold in result["thresholds"]:
        threshold["value"] = _number(threshold.get("value"))
    for change in result["temporal_changes"]:
        change["magnitude"] = _number(change.get("magnitude"))
        change["rate"] = _number(change.get("rate"))

    time_axis_ids = {
        axis["id"]
        for axis in axes
        if axis.get("scale") == "datetime"
        or _TIME_PATTERN.search(" ".join((_text(axis.get("name")), _text(axis.get("variable")))))
    }
    kept_changes: list[dict[str, Any]] = []
    for change in result["temporal_changes"]:
        time_axis_id = _text(change.get("time_axis_id"))
        if time_axis_id in time_axis_ids:
            kept_changes.append(change)
        else:
            result["uncertainties"].append(
                f"已忽略缺少真实时间轴支撑的时间变化：{change.get('id', '')}"
            )
    result["temporal_changes"] = kept_changes
    return result


def _stable_id(task: ImageExtractionTask, kind: str, local_id: str) -> str:
    digest = hashlib.sha1(f"{task.image_id}|{kind}|{local_id}".encode("utf-8")).hexdigest()[:12]
    return f"{task.image_id}:chart:{kind}:{digest}"


class QuantitativeChartGraphBuilder:
    """将规范化的图表语义对象构造成实体和可校验关系。"""

    def __init__(self, task: ImageExtractionTask, visual: Mapping[str, Any]) -> None:
        self.task = task
        self.visual = dict(visual)
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}
        self.aliases: dict[str, str] = {}
        self.chart_id = ""

    def build(self) -> Graph:
        self._add_chart()
        self._add_panels()
        self._add_axes()
        self._add_series()
        self._add_features("trends", "chart_trend", "trend", "exhibits_trend")
        self._add_correlations()
        self._add_thresholds()
        self._add_temporal_changes()
        self._add_features("anomalies", "chart_anomaly", "anomaly", "has_anomaly")
        self._add_comparisons()
        graph = Graph.from_chunk(
            document_id=self.task.document_id,
            chunk_id=self.task.chunk_id,
            modality=SourceModality.IMAGE,
            entities=self.entities.values(),
            relations=self.relations.values(),
            raw_response=self.visual,
            stage="stage_04_image_quantitative_chart_extraction",
        )
        graph.metadata.extra.update(
            {
                "status": "completed" if self.entities else "empty",
                "extractor_kind": "quantitative_chart",
                "extractor_name": "定量图表与实验曲线抽取器",
                "image_id": self.task.image_id,
                "image_index": self.task.image_index,
                "image_path": self.task.image_path,
                "classification_code": self.task.classification_code,
                "classification_type": self.task.classification_type,
                "model_called": True,
                "algorithm": "VLM结构化识别 + 确定性引用修复 + Graph装配",
                "chart_type": _text(_mapping(self.visual.get("chart")).get("chart_type")),
                "panel_count": len(_items(self.visual.get("panels"))),
                "axis_count": len(_items(self.visual.get("axes"))),
                "series_count": len(_items(self.visual.get("series"))),
                "trend_count": len(_items(self.visual.get("trends"))),
                "correlation_count": len(_items(self.visual.get("correlations"))),
                "threshold_count": len(_items(self.visual.get("thresholds"))),
                "temporal_change_count": len(_items(self.visual.get("temporal_changes"))),
                "uncertainties": _items(self.visual.get("uncertainties")),
            }
        )
        return graph

    def _entity(
        self,
        local_id: str,
        name: str,
        entity_type: str,
        item: Mapping[str, Any],
    ) -> str:
        existing = self.aliases.get(local_id)
        if existing:
            return existing
        graph_id = _stable_id(self.task, "entity", local_id)
        evidence = _text(item.get("evidence")) or self.task.caption
        attributes = {
            key: _sanitize(value)
            for key, value in item.items()
            if key not in {"id", "name", "title", "label", "evidence", "confidence"}
        }
        attributes["confidence"] = _confidence(item.get("confidence"))
        self.entities[graph_id] = Entity(
            id=graph_id,
            name=name or local_id,
            type=entity_type,
            attributes=attributes,
            provenance=evidence,
            metadata={"source_modality": "image", "local_id": local_id, "visual_evidence": evidence},
        )
        self.aliases[local_id] = graph_id
        if name:
            self.aliases.setdefault(name, graph_id)
        return graph_id

    def _relation(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        *,
        local_id: str,
        attributes: Mapping[str, Any] | None = None,
        evidence: str = "",
        confidence: Any = 0.7,
    ) -> None:
        if source_id not in self.entities or target_id not in self.entities:
            return
        relation_id = _stable_id(self.task, "relation", f"{local_id}|{source_id}|{relation_type}|{target_id}")
        source = self.entities[source_id]
        target = self.entities[target_id]
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
            attributes={**dict(attributes or {}), "confidence": _confidence(confidence)},
            provenance=evidence or self.task.caption,
            metadata={"source_modality": "image", "visual_evidence": evidence},
        )

    def _resolve(self, value: Any) -> str | None:
        return self.aliases.get(_text(value))

    def _add_chart(self) -> None:
        chart = _mapping(self.visual.get("chart"))
        local_id = _text(chart.get("id")) or "chart"
        name = _text(chart.get("title")) or self.task.caption.splitlines()[0].strip() or "定量图表"
        self.chart_id = self._entity(local_id, name, "quantitative_chart", chart)

    def _add_panels(self) -> None:
        for item in _items(self.visual.get("panels")):
            panel = _mapping(item)
            local_id = _text(panel.get("id"))
            if not local_id:
                continue
            name = _text(panel.get("title")) or f"子图{_text(panel.get('label')) or local_id}"
            panel_id = self._entity(local_id, name, "chart_panel", panel)
            self._relation(self.chart_id, "has_panel", panel_id, local_id=local_id, evidence=_text(panel.get("evidence")), confidence=panel.get("confidence"))

    def _add_axes(self) -> None:
        for item in _items(self.visual.get("axes")):
            axis = _mapping(item)
            local_id = _text(axis.get("id"))
            if not local_id:
                continue
            orientation = _text(axis.get("orientation")) or "axis"
            variable = _text(axis.get("variable")) or _text(axis.get("name")) or local_id
            axis_id = self._entity(local_id, f"{orientation}轴：{variable}", "chart_axis", axis)
            panel_id = self._resolve(axis.get("panel_id"))
            owner = panel_id or self.chart_id
            self._relation(owner, "has_axis", axis_id, local_id=local_id, evidence=_text(axis.get("evidence")), confidence=axis.get("confidence"))

    def _add_series(self) -> None:
        for item in _items(self.visual.get("series")):
            series = _mapping(item)
            local_id = _text(series.get("id"))
            if not local_id:
                continue
            name = _text(series.get("name")) or local_id
            series_id = self._entity(local_id, name, "data_series", series)
            owner = self._resolve(series.get("panel_id")) or self.chart_id
            self._relation(owner, "has_series", series_id, local_id=local_id, evidence=_text(series.get("evidence")), confidence=series.get("confidence"))
            for field, relation_type in (("x_axis_id", "uses_x_axis"), ("y_axis_id", "uses_y_axis")):
                axis_id = self._resolve(series.get(field))
                if axis_id:
                    self._relation(series_id, relation_type, axis_id, local_id=f"{local_id}:{field}", evidence=_text(series.get("evidence")), confidence=series.get("confidence"))

    def _add_features(self, field: str, entity_type: str, name_prefix: str, relation_type: str) -> None:
        for item in _items(self.visual.get(field)):
            feature = _mapping(item)
            local_id = _text(feature.get("id"))
            if not local_id:
                continue
            label = _text(feature.get("label")) or _text(feature.get("description"))
            if not label:
                label = f"{name_prefix}:{_text(feature.get('direction') or feature.get('kind')) or local_id}"
            feature_id = self._entity(local_id, label, entity_type, feature)
            linked = False
            for series_local in _items(feature.get("series_ids")):
                series_id = self._resolve(series_local)
                if series_id:
                    self._relation(series_id, relation_type, feature_id, local_id=f"{local_id}:{series_local}", evidence=_text(feature.get("evidence")), confidence=feature.get("confidence"))
                    linked = True
            if not linked:
                self._relation(self.chart_id, relation_type, feature_id, local_id=local_id, evidence=_text(feature.get("evidence")), confidence=feature.get("confidence"))
            if field == "trends":
                for axis_local in {self.visual_axis_for_series(series_local, "x_axis_id") for series_local in _items(feature.get("series_ids"))}:
                    axis_id = self._resolve(axis_local)
                    if axis_id:
                        self._relation(feature_id, "varies_along", axis_id, local_id=f"{local_id}:{axis_local}", evidence=_text(feature.get("evidence")), confidence=feature.get("confidence"))

    def visual_axis_for_series(self, series_local_id: Any, field: str) -> str:
        for item in _items(self.visual.get("series")):
            series = _mapping(item)
            if _text(series.get("id")) == _text(series_local_id):
                return _text(series.get(field))
        return ""

    def _add_correlations(self) -> None:
        for item in _items(self.visual.get("correlations")):
            correlation = _mapping(item)
            local_id = _text(correlation.get("id"))
            if not local_id:
                continue
            direction = _text(correlation.get("direction")) or "unknown"
            strength = _text(correlation.get("strength")) or "unknown"
            correlation_id = self._entity(local_id, f"{strength} {direction} correlation", "chart_correlation", correlation)
            self._relation(self.chart_id, "has_correlation", correlation_id, local_id=local_id, evidence=_text(correlation.get("evidence")), confidence=correlation.get("confidence"))
            for field, relation_type in (("x_axis_id", "relates_x_axis"), ("y_axis_id", "relates_y_axis")):
                axis_id = self._resolve(correlation.get(field))
                if axis_id:
                    self._relation(correlation_id, relation_type, axis_id, local_id=f"{local_id}:{field}", evidence=_text(correlation.get("evidence")), confidence=correlation.get("confidence"))
            for series_local in _items(correlation.get("series_ids")):
                series_id = self._resolve(series_local)
                if series_id:
                    self._relation(series_id, "has_correlation", correlation_id, local_id=f"{local_id}:{series_local}", evidence=_text(correlation.get("evidence")), confidence=correlation.get("confidence"))

    def _add_thresholds(self) -> None:
        for item in _items(self.visual.get("thresholds")):
            threshold = _mapping(item)
            local_id = _text(threshold.get("id"))
            if not local_id:
                continue
            value = threshold.get("value")
            label = _text(threshold.get("label")) or f"阈值 {value if value is not None else local_id}"
            threshold_id = self._entity(local_id, label, "chart_threshold", threshold)
            self._relation(self.chart_id, "has_threshold", threshold_id, local_id=local_id, evidence=_text(threshold.get("evidence")), confidence=threshold.get("confidence"))
            axis_id = self._resolve(threshold.get("axis_id"))
            if axis_id:
                self._relation(threshold_id, "defined_on", axis_id, local_id=f"{local_id}:axis", evidence=_text(threshold.get("evidence")), confidence=threshold.get("confidence"))
            for series_local in _items(threshold.get("series_ids")):
                series_id = self._resolve(series_local)
                if series_id:
                    self._relation(threshold_id, "applies_to", series_id, local_id=f"{local_id}:{series_local}", evidence=_text(threshold.get("evidence")), confidence=threshold.get("confidence"))

    def _add_temporal_changes(self) -> None:
        for item in _items(self.visual.get("temporal_changes")):
            change = _mapping(item)
            local_id = _text(change.get("id"))
            if not local_id:
                continue
            name = _text(change.get("description")) or f"时间变化：{_text(change.get('direction')) or local_id}"
            change_id = self._entity(local_id, name, "temporal_change", change)
            linked = False
            for series_local in _items(change.get("series_ids")):
                series_id = self._resolve(series_local)
                if series_id:
                    self._relation(series_id, "exhibits_temporal_change", change_id, local_id=f"{local_id}:{series_local}", evidence=_text(change.get("evidence")), confidence=change.get("confidence"))
                    linked = True
            if not linked:
                self._relation(self.chart_id, "exhibits_temporal_change", change_id, local_id=local_id, evidence=_text(change.get("evidence")), confidence=change.get("confidence"))
            axis_id = self._resolve(change.get("time_axis_id"))
            if axis_id:
                self._relation(change_id, "occurs_along", axis_id, local_id=f"{local_id}:axis", evidence=_text(change.get("evidence")), confidence=change.get("confidence"))

    def _add_comparisons(self) -> None:
        for index, item in enumerate(_items(self.visual.get("comparisons")), start=1):
            comparison = _mapping(item)
            source_id = self._resolve(comparison.get("source_series_id"))
            target_id = self._resolve(comparison.get("target_series_id"))
            relation_type = _text(comparison.get("relation"))
            if not source_id or not target_id or relation_type not in RELATION_ZH:
                continue
            self._relation(
                source_id,
                relation_type,
                target_id,
                local_id=f"comparison_{index}",
                attributes={"scope": comparison.get("scope", ""), "description": comparison.get("description", "")},
                evidence=_text(comparison.get("evidence")),
                confidence=comparison.get("confidence"),
            )


def build_quantitative_chart_graph(task: ImageExtractionTask, visual: Mapping[str, Any]) -> Graph:
    """公开 Graph 装配入口。调用前应先执行结果规范化。"""

    return QuantitativeChartGraphBuilder(task, visual).build()


__all__ = [
    "RELATION_ZH",
    "build_quantitative_chart_graph",
    "normalize_quantitative_chart_result",
]
