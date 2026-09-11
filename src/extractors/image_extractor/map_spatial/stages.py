"""地图空间抽取器的分阶段 VLM 调用与图例落地逻辑。

本文件实现 LegendParser、MapEntityExtractor、LegendGrounder 和
SemanticRelationExtractor，便于逐阶段诊断模型响应。
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Callable, Mapping

from src.utils.llm_client import safe_json_loads

from ..schema_models import ImageExtractionTask
from .prompt import build_entity_prompt, build_legend_prompt, build_relation_prompt
from .relation_schema import TYPE_ZH, canonical_entity_type, slug


DescribeImage = Callable[..., Any]


def _response_mapping(response: Any, stage_name: str) -> dict[str, Any]:
    """把 VLM 文本或对象响应解析为非空 JSON 对象。"""

    payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
    if not isinstance(payload, Mapping) or not payload:
        raw = str(response or "")
        raise ValueError(f"{stage_name}响应不是非空 JSON 对象：字符数={len(raw)}")
    return dict(payload)


def _stage_max_tokens(stage: str, default: int) -> int:
    """读取分阶段 token 上限，并兼容原有总配置项。"""

    specific = f"MAP_SPATIAL_{stage.upper()}_MAX_TOKENS"
    return int(os.getenv(specific, os.getenv("MAP_SPATIAL_VLM_MAX_TOKENS", str(default))))


class LegendParser:
    """第一阶段：只解析图题和图例类别。"""

    def run(self, task: ImageExtractionTask, describe_image: DescribeImage) -> dict[str, Any]:
        """调用 VLM 获取地图信息与图例项。"""

        response = describe_image(
            task.image_path,
            build_legend_prompt(task),
            task_name=f"地图图例解析:{task.image_id}",
            response_format={"type": "json_object"},
            max_tokens=_stage_max_tokens("legend", 4096),
        )
        payload = _response_mapping(response, "图例解析")
        if "map" not in payload and "legend_items" not in payload:
            raise ValueError("图例解析响应缺少 map 和 legend_items")
        payload.setdefault("legend_items", [])
        payload.setdefault("uncertainties", [])
        return payload


class MapEntityExtractor:
    """第二阶段：依据图例抽取实例、几何、绑定和空间候选。"""

    def run(
        self,
        task: ImageExtractionTask,
        describe_image: DescribeImage,
        legend_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """调用 VLM 获取地图实体及归一化坐标。"""

        response = describe_image(
            task.image_path,
            build_entity_prompt(task, legend_result),
            task_name=f"地图实体抽取:{task.image_id}",
            response_format={"type": "json_object"},
            max_tokens=_stage_max_tokens("entity", 12288),
        )
        payload = _response_mapping(response, "实体抽取")
        if not isinstance(payload.get("entities"), list):
            raise ValueError("实体抽取响应缺少 entities 列表")
        payload.setdefault("legend_bindings", [])
        payload.setdefault("spatial_mapping", {})
        payload.setdefault("uncertainties", [])
        return payload


class LegendGrounder:
    """第三阶段：确定性校验图例与实体绑定，并补全视觉编码。"""

    _DEFAULT_EMIT_TYPES = {
        "lithology",
        "sedimentary_microfacies",
        "paleogeographic_landmass_class",
        "geological_boundary",
        "reservoir_property",
    }

    @classmethod
    def _legend_node_type(cls, legend: Mapping[str, Any]) -> tuple[str, str, str]:
        """确定图例节点类型，并把普通符号图例统一标记为 legend_class。"""

        canonical_type, raw_type = canonical_entity_type(legend.get("semantic_type"))
        if bool(legend.get("emit_entity")) or canonical_type in cls._DEFAULT_EMIT_TYPES:
            node_type = canonical_type if canonical_type != "UNMAPPED" else "legend_class"
            type_zh = str(legend.get("type_zh") or TYPE_ZH.get(node_type) or "图例类别")
        else:
            node_type = "legend_class"
            type_zh = "图例类别"
        return node_type, type_zh, raw_type

    def ground(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """校验显式或实体内嵌绑定，并把每个图例物化为可关联节点。"""

        result = deepcopy(dict(state))
        legends = [dict(item) for item in result.get("legend_items") or [] if isinstance(item, Mapping)]
        entities = [dict(item) for item in result.get("entities") or [] if isinstance(item, Mapping)]
        legend_by_id = {str(item.get("legend_id") or ""): item for item in legends if item.get("legend_id")}
        entity_by_id = {str(item.get("id") or ""): item for item in entities if item.get("id")}
        binding_by_legend: dict[str, dict[str, Any]] = {}

        for raw in result.get("legend_bindings") or []:
            if not isinstance(raw, Mapping):
                continue
            legend_id = str(raw.get("legend_id") or "")
            legend = legend_by_id.get(legend_id)
            if not legend:
                continue
            resolved_ids = [str(value) for value in raw.get("entity_ids") or [] if str(value) in entity_by_id]
            if not resolved_ids:
                continue
            binding = binding_by_legend.setdefault(legend_id, {"legend_id": legend_id, "entity_ids": []})
            binding.update({key: deepcopy(value) for key, value in raw.items() if key not in {"legend_id", "entity_ids"}})
            binding["entity_ids"] = list(dict.fromkeys([*binding["entity_ids"], *resolved_ids]))

        # 中文说明：兼容 VLM 把 legend_binding 直接写在实体属性中而没有单独输出绑定列表的情况。
        legend_id_by_label = {
            str(legend.get("label") or "").strip().casefold(): legend_id
            for legend_id, legend in legend_by_id.items()
            if str(legend.get("label") or "").strip()
        }
        for entity_id, entity in entity_by_id.items():
            attributes = dict(entity.get("attributes")) if isinstance(entity.get("attributes"), Mapping) else {}
            raw_binding = entity.get("legend_id") or attributes.get("legend_id") or entity.get("legend_binding") or attributes.get("legend_binding")
            values = raw_binding if isinstance(raw_binding, list) else [raw_binding]
            for value in values:
                text = str(value or "").strip()
                legend_id = text if text in legend_by_id else legend_id_by_label.get(text.casefold(), "")
                if not legend_id:
                    continue
                binding = binding_by_legend.setdefault(legend_id, {"legend_id": legend_id, "entity_ids": []})
                binding["entity_ids"] = list(dict.fromkeys([*binding["entity_ids"], entity_id]))

        bindings = list(binding_by_legend.values())
        for binding in bindings:
            legend_id = str(binding["legend_id"])
            legend = legend_by_id[legend_id]
            canonical_type, _ = canonical_entity_type(legend.get("semantic_type"))
            for entity_id in binding["entity_ids"]:
                entity = entity_by_id[entity_id]
                attributes = dict(entity.get("attributes")) if isinstance(entity.get("attributes"), Mapping) else {}
                attributes.setdefault("legend_binding", str(legend.get("label") or legend_id))
                if legend.get("visual_encoding") and "visual_encoding" not in attributes:
                    attributes["visual_encoding"] = deepcopy(legend["visual_encoding"])
                entity["attributes"] = attributes
                if str(entity.get("type") or "") in {"", "unmapped", "UNMAPPED"} and canonical_type != "UNMAPPED":
                    entity["type"] = canonical_type
                    entity["type_zh"] = str(legend.get("type_zh") or TYPE_ZH.get(canonical_type) or "未映射空间对象")

        for legend in legends:
            label = str(legend.get("label") or "").strip()
            if not label:
                continue
            legend_id = str(legend.get("legend_id") or "")
            existing = next(
                (
                    item for item in entities
                    if isinstance(item.get("attributes"), Mapping)
                    and str(item["attributes"].get("legend_id") or "") == legend_id
                    and bool(item["attributes"].get("legend_item"))
                ),
                None,
            )
            if existing:
                local_id = str(existing["id"])
            else:
                node_type, type_zh, raw_type = self._legend_node_type(legend)
                local_id = slug(legend.get("entity_id") or f"legend_{legend_id or label}") or f"legend_{len(entities) + 1}"
                base_id = local_id
                suffix = 2
                while local_id in entity_by_id:
                    local_id = f"{base_id}_{suffix}"
                    suffix += 1
                entity = {
                    "id": local_id,
                    "name": label,
                    "type": node_type,
                    "type_zh": type_zh,
                    "attributes": {
                        "legend_item": True,
                        "legend_id": legend_id,
                        "semantic_type": str(legend.get("semantic_type") or "UNMAPPED"),
                        "visual_encoding": deepcopy(legend.get("visual_encoding") or {}),
                        "confidence": legend.get("confidence", 0.7),
                        **({"raw_type": raw_type} if raw_type and str(legend.get("semantic_type")) == "UNMAPPED" else {}),
                    },
                    "evidence": str(legend.get("evidence") or f"图例“{label}”"),
                    "confidence": legend.get("confidence", 0.7),
                }
                entities.append(entity)
                entity_by_id[local_id] = entity
            legend["materialized_entity_id"] = local_id
            binding = binding_by_legend.get(legend_id)
            if binding:
                binding["legend_entity_id"] = local_id

        result["legend_items"] = legends
        result["entities"] = entities
        result["legend_bindings"] = bindings
        return result


class SemanticRelationExtractor:
    """第五阶段：让 VLM 只选择白名单内的语义关系候选。"""

    def run(
        self,
        task: ImageExtractionTask,
        describe_image: DescribeImage,
        state: Mapping[str, Any],
        relation_schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        """调用 VLM 获取稀疏关系候选，不接受其地点方向。"""

        response = describe_image(
            task.image_path,
            build_relation_prompt(task, state, relation_schema),
            task_name=f"地图语义关系候选:{task.image_id}",
            response_format={"type": "json_object"},
            max_tokens=_stage_max_tokens("relation", 8192),
        )
        payload = _response_mapping(response, "语义关系候选")
        payload.setdefault("relation_candidates", [])
        payload.setdefault("uncertainties", [])
        if not isinstance(payload["relation_candidates"], list):
            raise ValueError("语义关系候选响应的 relation_candidates 不是列表")
        return payload


__all__ = [
    "LegendGrounder",
    "LegendParser",
    "MapEntityExtractor",
    "SemanticRelationExtractor",
]
