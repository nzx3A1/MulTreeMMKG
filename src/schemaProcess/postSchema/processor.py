"""生成 postSchema 可审计计划，并据此重写 Stage 05 主图的实体和关系类型。"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from src.schemaProcess.models import SchemaConcept, SchemaRelation, SchemaSnapshot
from src.schemaProcess.relation_aligner import normalize_relation_name

from .llm_mapper import PostSchemaSelector
from .models import NewConceptProposal, PostSchemaDecision


LOGGER = logging.getLogger(__name__)


def payload_sha256(value: Any) -> str:
    """计算稳定 JSON 指纹，防止把旧计划应用到已变化的输入文件。"""

    content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _require_list(value: Any, label: str) -> list[Any]:
    """校验输入字段为 JSON 数组并返回普通列表。"""

    if not isinstance(value, list):
        raise TypeError(f"{label} 必须是数组")
    return value


def _entity_status_counts(entities: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """重新汇总单个文档子图中的实体对齐状态。"""

    return dict(
        sorted(
            Counter(
                str((item.get("schema_alignment") or {}).get("status") or "UNKNOWN")
                for item in entities
            ).items()
        )
    )


def _relation_status_counts(relations: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """重新汇总单个文档子图中的关系对齐状态。"""

    return dict(
        sorted(
            Counter(
                str((item.get("schema_alignment") or {}).get("status") or "UNKNOWN")
                for item in relations
            ).items()
        )
    )


def _find_exact_relation(
    relation: Mapping[str, Any],
    candidates: Sequence[SchemaRelation],
) -> SchemaRelation | None:
    """按原英文/中文关系字段查找唯一精确候选。"""

    raw_en = {
        normalize_relation_name(relation.get(field))
        for field in ("raw_relation", "type", "relation_name")
        if relation.get(field)
    }
    raw_zh = {
        str(relation.get(field) or "").strip()
        for field in ("type_zh", "relation_name")
        if relation.get(field)
    }
    matches = [
        item
        for item in candidates
        if normalize_relation_name(item.relation_en) in raw_en
        or (item.relation_zh and item.relation_zh in raw_zh)
    ]
    return matches[0] if len(matches) == 1 else None


class PostSchemaProcessor:
    """根据未映射报告完成实体补映射、关系重判定和最终类型物化。"""

    def __init__(self, snapshot: SchemaSnapshot, selector: PostSchemaSelector) -> None:
        """保存当前 Schema 快照与可注入选择器。"""

        self.snapshot = snapshot
        self.selector = selector

    @staticmethod
    def _validate_inputs(
        unmapped_report: Mapping[str, Any],
        graph_payload: Mapping[str, Any],
    ) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
        """校验报告和主图的核心数组结构及目标实体定位。"""

        groups = _require_list(unmapped_report.get("unmapped_entity_types"), "unmapped_entity_types")
        rows = _require_list(unmapped_report.get("entities"), "entities")
        graphs = _require_list(graph_payload.get("graphs"), "graphs")
        typed_groups = [item for item in groups if isinstance(item, Mapping)]
        typed_rows = [item for item in rows if isinstance(item, Mapping)]
        typed_graphs = [item for item in graphs if isinstance(item, Mapping)]
        if len(typed_groups) != len(groups) or len(typed_rows) != len(rows) or len(typed_graphs) != len(graphs):
            raise TypeError("未映射报告或主图数组中包含非对象条目")

        seen_locations: set[tuple[int, str]] = set()
        for row in typed_rows:
            try:
                graph_index = int(row.get("graph_index"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"未映射实体缺少有效 graph_index：{row.get('entity_id')}") from exc
            entity_id = str(row.get("entity_id") or "").strip()
            if not entity_id or graph_index < 0 or graph_index >= len(typed_graphs):
                raise ValueError(f"未映射实体定位无效：graph={graph_index}, id={entity_id!r}")
            entities = _require_list(typed_graphs[graph_index].get("entities"), f"graphs[{graph_index}].entities")
            matches = [item for item in entities if isinstance(item, Mapping) and str(item.get("id") or "") == entity_id]
            if len(matches) != 1:
                raise ValueError(f"主图中无法唯一定位未映射实体：graph={graph_index}, id={entity_id}")
            location = (graph_index, entity_id)
            if location in seen_locations:
                raise ValueError(f"未映射报告包含重复实体定位：{location}")
            seen_locations.add(location)
        return typed_groups, typed_rows, typed_graphs

    def _validate_entity_decisions(
        self,
        groups: Sequence[Mapping[str, Any]],
        decisions: Sequence[PostSchemaDecision],
    ) -> dict[str, PostSchemaDecision]:
        """保证每种 raw_type 恰有一个判定且引用当前 Schema 或合法新概念。"""

        expected_raw_types = [str(item.get("raw_type") or "").strip() for item in groups]
        decision_map: dict[str, PostSchemaDecision] = {}
        new_schemas: dict[str, NewConceptProposal] = {}
        for decision in decisions:
            if decision.raw_type in decision_map:
                raise ValueError(f"重复的实体类型判定：{decision.raw_type}")
            if decision.category not in {item.category for item in self.snapshot.concepts}:
                raise ValueError(f"判定引用不存在的一级类别：{decision.category}")
            if decision.action == "MAP_EXISTING":
                concept = self.snapshot.concepts_by_schema.get(decision.effective_schema)
                if concept is None or concept.category != decision.category:
                    raise ValueError(
                        f"实体类型 {decision.raw_type!r} 引用了类别外或不存在的概念：{decision.effective_schema}"
                    )
            else:
                proposal = decision.new_concept
                assert proposal is not None
                current = self.snapshot.concepts_by_schema.get(proposal.schema)
                if current is not None and (
                    current.category != proposal.category or current.zh_name != proposal.zh_name
                ):
                    raise ValueError(f"计划新增的 Schema 与当前同名概念冲突：{proposal.schema}")
                previous = new_schemas.get(proposal.schema)
                if previous is not None and previous != proposal:
                    raise ValueError(f"多个 raw_type 给出了冲突的新概念定义：{proposal.schema}")
                new_schemas[proposal.schema] = proposal
            decision_map[decision.raw_type] = decision
        if set(expected_raw_types) != set(decision_map) or len(expected_raw_types) != len(decision_map):
            missing = sorted(set(expected_raw_types) - set(decision_map))
            extra = sorted(set(decision_map) - set(expected_raw_types))
            raise ValueError(f"实体类型判定与报告不一致：缺少={missing}，多出={extra}")
        return decision_map

    @staticmethod
    def _prospective_entity_schemas(
        graphs: Sequence[Mapping[str, Any]],
        report_rows: Sequence[Mapping[str, Any]],
        decision_map: Mapping[str, PostSchemaDecision],
        entity_overrides: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, str | None]]:
        """在不修改主图的情况下计算每个端点补映射后的 Schema。"""

        targets = {
            (int(row["graph_index"]), str(row.get("entity_id") or "")): decision_map[
                str(row.get("raw_type") or "").strip()
            ].effective_schema
            for row in report_rows
        }
        targets.update(
            {
                (int(item["graph_index"]), str(item["entity_id"])): str(item["schema"])
                for item in entity_overrides
            }
        )
        result: list[dict[str, str | None]] = []
        for graph_index, graph in enumerate(graphs):
            endpoint_map: dict[str, str | None] = {}
            for entity in _require_list(graph.get("entities"), f"graphs[{graph_index}].entities"):
                if not isinstance(entity, Mapping):
                    raise TypeError(f"graphs[{graph_index}].entities 包含非对象条目")
                entity_id = str(entity.get("id") or "").strip()
                if not entity_id or entity_id in endpoint_map:
                    raise ValueError(f"graphs[{graph_index}] 实体 id 缺失或重复：{entity_id!r}")
                endpoint_map[entity_id] = targets.get(
                    (graph_index, entity_id),
                    str(entity.get("schema_type") or "").strip() or None,
                )
            result.append(endpoint_map)
        return result

    def _plan_relations(
        self,
        graphs: Sequence[Mapping[str, Any]],
        endpoint_schemas: Sequence[Mapping[str, str | None]],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """按补映射后的有向端点生成直接、精确或 LLM 关系替换计划。"""

        planned: list[dict[str, Any]] = []
        stats: Counter[str] = Counter()
        ambiguous_requests: list[
            tuple[Mapping[str, Any], str, str, Sequence[SchemaRelation]]
        ] = []
        ambiguous_locations: list[tuple[int, int]] = []
        ambiguous_candidates: list[Sequence[SchemaRelation]] = []

        for graph_index, graph in enumerate(graphs):
            relations = _require_list(graph.get("relations"), f"graphs[{graph_index}].relations")
            for relation_index, relation in enumerate(relations):
                if not isinstance(relation, Mapping):
                    raise TypeError(f"graphs[{graph_index}].relations[{relation_index}] 必须是对象")
                source_schema = endpoint_schemas[graph_index].get(str(relation.get("source_id") or ""))
                target_schema = endpoint_schemas[graph_index].get(str(relation.get("target_id") or ""))
                if not source_schema or not target_schema:
                    stats["endpoint_unmapped"] += 1
                    continue
                candidates = self.snapshot.relation_candidates(source_schema, target_schema)
                if not candidates:
                    stats["no_schema_relation"] += 1
                    continue

                current = str(relation.get("schema_relation") or "").strip()
                selected = next((item for item in candidates if item.relation_en == current), None)
                method = "existing_schema_relation"
                confidence = float((relation.get("schema_alignment") or {}).get("confidence") or 1.0)
                reason = "保留 Stage 05 已确定且端点仍合法的 Schema 关系"
                if selected is None:
                    selected = _find_exact_relation(relation, candidates)
                    method = "post_schema_exact_relation"
                    confidence = 0.99
                    reason = "原始关系英文或中文名称与补映射端点候选精确一致"
                if selected is None and len(candidates) == 1:
                    selected = candidates[0]
                    method = "post_schema_single_endpoint_candidate"
                    confidence = 0.98
                    reason = "补映射后的有向端点在现有 Schema 中只有一个关系类型"
                if selected is not None:
                    planned.append(
                        {
                            "graph_index": graph_index,
                            "relation_index": relation_index,
                            "source_schema": source_schema,
                            "target_schema": target_schema,
                            "selected_relation": selected.relation_en,
                            "relation_zh": selected.relation_zh,
                            "method": method,
                            "confidence": round(confidence, 6),
                            "reason": reason,
                        }
                    )
                    stats[method] += 1
                    continue
                ambiguous_requests.append((relation, source_schema, target_schema, candidates))
                ambiguous_locations.append((graph_index, relation_index))
                ambiguous_candidates.append(candidates)

        if ambiguous_requests:
            selected_rows = self.selector.select_relations(ambiguous_requests)
            if len(selected_rows) != len(ambiguous_requests):
                raise RuntimeError("postSchema 关系 LLM 判定数量与请求数量不一致")
            for location, candidates, selected_row in zip(
                ambiguous_locations,
                ambiguous_candidates,
                selected_rows,
            ):
                selected_name, confidence, reason = selected_row
                candidate = next(
                    (item for item in candidates if item.relation_en == selected_name),
                    None,
                )
                if candidate is None:
                    stats["ambiguous_preserved"] += 1
                    continue
                graph_index, relation_index = location
                relation = graphs[graph_index]["relations"][relation_index]
                planned.append(
                    {
                        "graph_index": graph_index,
                        "relation_index": relation_index,
                        "source_schema": endpoint_schemas[graph_index].get(
                            str(relation.get("source_id") or "")
                        ),
                        "target_schema": endpoint_schemas[graph_index].get(
                            str(relation.get("target_id") or "")
                        ),
                        "selected_relation": candidate.relation_en,
                        "relation_zh": candidate.relation_zh,
                        "method": "post_schema_relation_llm",
                        "confidence": round(float(confidence), 6),
                        "reason": reason,
                    }
                )
                stats["post_schema_relation_llm"] += 1
        planned.sort(key=lambda item: (item["graph_index"], item["relation_index"]))
        return planned, dict(sorted(stats.items()))

    def _apply_decision_overrides(
        self,
        decisions: Sequence[PostSchemaDecision],
        overrides: Mapping[str, Any] | None,
        unmapped_report: Mapping[str, Any],
    ) -> tuple[list[PostSchemaDecision], int, list[dict[str, Any]]]:
        """应用与当前报告指纹绑定的人工审查结果，避免特例污染未来输入。"""

        if not overrides:
            return list(decisions), 0, []
        expected_hash = str(overrides.get("unmapped_entity_report_sha256") or "").strip()
        actual_hash = payload_sha256(unmapped_report)
        if expected_hash and expected_hash != actual_hash:
            raise ValueError("实体判定覆写表不属于当前未映射报告，已拒绝应用")
        raw_overrides = _require_list(overrides.get("decisions"), "overrides.decisions")
        parsed = [
            PostSchemaDecision.from_mapping(item)
            for item in raw_overrides
            if isinstance(item, Mapping)
        ]
        if len(parsed) != len(raw_overrides):
            raise TypeError("overrides.decisions 包含非对象条目")
        override_map: dict[str, PostSchemaDecision] = {}
        for item in parsed:
            if item.raw_type in override_map:
                raise ValueError(f"覆写表包含重复 raw_type：{item.raw_type}")
            override_map[item.raw_type] = item
        base_raw_types = {item.raw_type for item in decisions}
        unknown = sorted(set(override_map) - base_raw_types)
        if unknown:
            raise ValueError(f"覆写表包含当前报告不存在的 raw_type：{unknown}")
        result = [override_map.get(item.raw_type, item) for item in decisions]
        report_locations = {
            (int(item["graph_index"]), str(item.get("entity_id") or ""))
            for item in _require_list(unmapped_report.get("entities"), "entities")
            if isinstance(item, Mapping)
        }
        available_schemas = set(self.snapshot.concepts_by_schema)
        available_schemas.update(item.effective_schema for item in result)
        entity_overrides: list[dict[str, Any]] = []
        seen_locations: set[tuple[int, str]] = set()
        for item in _require_list(overrides.get("entity_overrides", []), "overrides.entity_overrides"):
            if not isinstance(item, Mapping):
                raise TypeError("overrides.entity_overrides 包含非对象条目")
            location = (int(item.get("graph_index")), str(item.get("entity_id") or "").strip())
            schema = str(item.get("schema") or "").strip()
            if location not in report_locations:
                raise ValueError(f"单实体覆写不属于当前未映射报告：{location}")
            if location in seen_locations:
                raise ValueError(f"单实体覆写位置重复：{location}")
            if schema not in available_schemas:
                raise ValueError(f"单实体覆写引用未知 Schema：{schema}")
            seen_locations.add(location)
            entity_overrides.append(
                {
                    "graph_index": location[0],
                    "entity_id": location[1],
                    "schema": schema,
                    "confidence": round(float(item.get("confidence") or 1.0), 6),
                    "reason": str(item.get("reason") or "").strip() or "人工单实体审查覆写",
                }
            )
        return result, len(override_map), entity_overrides

    def _assemble_plan(
        self,
        decisions: Sequence[PostSchemaDecision],
        unmapped_report: Mapping[str, Any],
        graph_payload: Mapping[str, Any],
        *,
        reviewed_override_count: int = 0,
        entity_overrides: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """从已确定的实体决策重新计算关系替换并装配完整计划。"""

        groups, rows, graphs = self._validate_inputs(unmapped_report, graph_payload)
        decision_map = self._validate_entity_decisions(groups, decisions)
        endpoint_schemas = self._prospective_entity_schemas(
            graphs,
            rows,
            decision_map,
            entity_overrides,
        )
        relation_decisions, relation_stats = self._plan_relations(graphs, endpoint_schemas)
        new_concepts: dict[str, NewConceptProposal] = {}
        for decision in decisions:
            if decision.new_concept is not None:
                new_concepts[decision.new_concept.schema] = decision.new_concept
        action_counts = Counter(item.action for item in decisions)
        return {
            "_stage": "stage_05_post_schema_plan",
            "_description": "Stage 05 未映射实体的两级补映射与关系类型替换计划",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_fingerprints": {
                "unmapped_entity_report_sha256": payload_sha256(unmapped_report),
                "schema_aligned_graph_sha256": payload_sha256(graph_payload),
            },
            "schema": {
                "source": self.snapshot.source,
                "concept_count": len(self.snapshot.concepts),
                "relation_count": len(self.snapshot.relations),
            },
            "statistics": {
                "target_entity_count": len(rows),
                "unique_raw_type_count": len(groups),
                "entity_action_counts": dict(sorted(action_counts.items())),
                "new_concept_count": len(new_concepts),
                "reviewed_override_count": reviewed_override_count,
                "entity_override_count": len(entity_overrides),
                "planned_relation_replacement_count": len(relation_decisions),
                "relation_planning_counts": relation_stats,
            },
            "entity_decisions": [item.to_dict() for item in decisions],
            "new_concepts": [new_concepts[key].to_dict() for key in sorted(new_concepts)],
            "entity_overrides": [dict(item) for item in entity_overrides],
            "relation_decisions": relation_decisions,
        }

    def build_plan(
        self,
        unmapped_report: Mapping[str, Any],
        graph_payload: Mapping[str, Any],
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """调用两级选择器并生成包含实体、新概念和关系决策的完整计划。"""

        groups, _rows, _graphs = self._validate_inputs(unmapped_report, graph_payload)
        decisions = self.selector.select_entities(groups, self.snapshot)
        decisions, override_count, entity_overrides = self._apply_decision_overrides(
            decisions,
            overrides,
            unmapped_report,
        )
        return self._assemble_plan(
            decisions,
            unmapped_report,
            graph_payload,
            reviewed_override_count=override_count,
            entity_overrides=entity_overrides,
        )

    def revise_plan(
        self,
        plan: Mapping[str, Any],
        overrides: Mapping[str, Any],
        unmapped_report: Mapping[str, Any],
        graph_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """在不重复调用实体 LLM 的情况下应用审查覆写并重算关系计划。"""

        decisions = self.validate_plan(plan, unmapped_report, graph_payload)
        decisions, override_count, entity_overrides = self._apply_decision_overrides(
            decisions,
            overrides,
            unmapped_report,
        )
        return self._assemble_plan(
            decisions,
            unmapped_report,
            graph_payload,
            reviewed_override_count=override_count,
            entity_overrides=entity_overrides,
        )

    def validate_plan(
        self,
        plan: Mapping[str, Any],
        unmapped_report: Mapping[str, Any],
        graph_payload: Mapping[str, Any],
    ) -> list[PostSchemaDecision]:
        """校验计划输入指纹、实体决策和关系定位仍与当前文件一致。"""

        groups, _rows, graphs = self._validate_inputs(unmapped_report, graph_payload)
        fingerprints = plan.get("input_fingerprints") if isinstance(plan.get("input_fingerprints"), Mapping) else {}
        expected_report_hash = payload_sha256(unmapped_report)
        expected_graph_hash = payload_sha256(graph_payload)
        if fingerprints.get("unmapped_entity_report_sha256") != expected_report_hash:
            raise ValueError("postSchema 计划对应的未映射实体报告已变化")
        if fingerprints.get("schema_aligned_graph_sha256") != expected_graph_hash:
            raise ValueError("postSchema 计划对应的主图已变化")
        raw_decisions = _require_list(plan.get("entity_decisions"), "plan.entity_decisions")
        decisions = [PostSchemaDecision.from_mapping(item) for item in raw_decisions if isinstance(item, Mapping)]
        self._validate_entity_decisions(groups, decisions)
        available_schemas = set(self.snapshot.concepts_by_schema)
        available_schemas.update(item.effective_schema for item in decisions)
        report_locations = {
            (int(item["graph_index"]), str(item.get("entity_id") or ""))
            for item in _require_list(unmapped_report.get("entities"), "entities")
            if isinstance(item, Mapping)
        }
        for item in _require_list(plan.get("entity_overrides", []), "plan.entity_overrides"):
            if not isinstance(item, Mapping):
                raise TypeError("plan.entity_overrides 包含非对象条目")
            location = (int(item.get("graph_index")), str(item.get("entity_id") or ""))
            if location not in report_locations or str(item.get("schema") or "") not in available_schemas:
                raise ValueError(f"计划包含无效的单实体覆写：{location}")
        for item in _require_list(plan.get("relation_decisions"), "plan.relation_decisions"):
            if not isinstance(item, Mapping):
                raise TypeError("plan.relation_decisions 包含非对象条目")
            graph_index = int(item.get("graph_index"))
            relation_index = int(item.get("relation_index"))
            if graph_index < 0 or graph_index >= len(graphs):
                raise ValueError(f"计划关系 graph_index 越界：{graph_index}")
            relations = _require_list(graphs[graph_index].get("relations"), f"graphs[{graph_index}].relations")
            if relation_index < 0 or relation_index >= len(relations):
                raise ValueError(f"计划关系 relation_index 越界：{graph_index}:{relation_index}")
        return decisions

    def apply_plan(
        self,
        plan: Mapping[str, Any],
        unmapped_report: Mapping[str, Any],
        graph_payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """应用已校验计划，替换类型但保留所有原始类型和溯源字段。"""

        decisions = self.validate_plan(plan, unmapped_report, graph_payload)
        decision_map = {item.raw_type: item for item in decisions}
        _groups, rows, _graphs = self._validate_inputs(unmapped_report, graph_payload)
        result = deepcopy(dict(graph_payload))
        graphs = result["graphs"]
        concept_lookup: dict[str, SchemaConcept | NewConceptProposal] = dict(
            self.snapshot.concepts_by_schema
        )
        for decision in decisions:
            if decision.new_concept is not None:
                concept_lookup[decision.new_concept.schema] = decision.new_concept

        target_rows = {
            (int(row["graph_index"]), str(row.get("entity_id") or "")): row for row in rows
        }
        entity_override_map = {
            (int(item["graph_index"]), str(item["entity_id"])): item
            for item in plan.get("entity_overrides", [])
        }
        entity_replacement_count = 0
        target_entity_count = 0
        for graph_index, graph in enumerate(graphs):
            for entity in graph["entities"]:
                entity_id = str(entity.get("id") or "")
                target_row = target_rows.get((graph_index, entity_id))
                if target_row is not None:
                    raw_type = str(target_row.get("raw_type") or "").strip()
                    decision = decision_map[raw_type]
                    entity_override = entity_override_map.get((graph_index, entity_id))
                    effective_schema = (
                        str(entity_override["schema"])
                        if entity_override is not None
                        else decision.effective_schema
                    )
                    concept = concept_lookup[effective_schema]
                    zh_name = concept.zh_name if isinstance(concept, SchemaConcept) else concept.zh_name
                    entity["schema_type"] = effective_schema
                    entity["schema_type_zh"] = zh_name
                    entity["schema_alignment"] = {
                        "status": "POST_SCHEMA_MAPPED",
                        "method": (
                            "post_schema_entity_override"
                            if entity_override is not None
                            else (
                                "post_schema_existing_llm"
                                if decision.action == "MAP_EXISTING"
                                else "post_schema_new_concept"
                            )
                        ),
                        "confidence": round(
                            float(entity_override.get("confidence") or 1.0)
                            if entity_override is not None
                            else decision.confidence,
                            6,
                        ),
                        "reason": (
                            entity_override.get("reason")
                            if entity_override is not None
                            else decision.reason
                        ),
                        "category": concept.category,
                        "post_schema_added": not isinstance(concept, SchemaConcept),
                    }
                    target_entity_count += 1

                schema_type = str(entity.get("schema_type") or "").strip()
                if schema_type:
                    previous_type = entity.get("type")
                    entity["raw_type"] = entity.get("raw_type", previous_type)
                    if "raw_type_zh" not in entity:
                        entity["raw_type_zh"] = entity.get("type_zh")
                    entity["type"] = schema_type
                    if entity.get("schema_type_zh"):
                        entity["type_zh"] = entity["schema_type_zh"]
                    if previous_type != schema_type:
                        entity_replacement_count += 1

        relation_plan = {
            (int(item["graph_index"]), int(item["relation_index"])): item
            for item in plan.get("relation_decisions", [])
        }
        relation_replacement_count = 0
        newly_mapped_relation_count = 0
        for graph_index, graph in enumerate(graphs):
            endpoint_map = {str(item.get("id") or ""): item for item in graph["entities"]}
            for relation_index, relation in enumerate(graph["relations"]):
                source = endpoint_map.get(str(relation.get("source_id") or ""))
                target = endpoint_map.get(str(relation.get("target_id") or ""))
                relation["source_schema"] = source.get("schema_type") if source else None
                relation["target_schema"] = target.get("schema_type") if target else None
                planned = relation_plan.get((graph_index, relation_index))
                selected_relation = (
                    str(planned.get("selected_relation") or "").strip()
                    if planned is not None
                    else str(relation.get("schema_relation") or "").strip()
                )
                if not selected_relation:
                    continue
                previous_type = relation.get("type")
                previous_alignment = deepcopy(relation.get("schema_alignment") or {})
                relation["raw_relation"] = relation.get("raw_relation", previous_type)
                if "raw_type_zh" not in relation:
                    relation["raw_type_zh"] = relation.get("type_zh")
                relation["type"] = selected_relation
                relation["schema_relation"] = selected_relation
                if planned is not None:
                    if planned.get("relation_zh"):
                        relation["type_zh"] = planned["relation_zh"]
                    if planned.get("method") == "existing_schema_relation":
                        relation["schema_alignment"] = previous_alignment
                        relation["schema_alignment"]["post_schema_type_materialized"] = True
                    else:
                        relation["schema_alignment"] = {
                            "status": "POST_SCHEMA_MAPPED",
                            "method": planned.get("method"),
                            "confidence": planned.get("confidence"),
                            "reason": planned.get("reason"),
                            "relation_zh": planned.get("relation_zh"),
                            "previous_status": previous_alignment.get("status"),
                        }
                        if previous_alignment.get("status") == "UNMAPPED":
                            newly_mapped_relation_count += 1
                if previous_type != selected_relation:
                    relation_replacement_count += 1

            metadata = deepcopy(graph.get("metadata") or {})
            metadata["schema_alignment"] = {
                "entity_status_counts": _entity_status_counts(graph["entities"]),
                "relation_status_counts": _relation_status_counts(graph["relations"]),
            }
            graph["metadata"] = metadata

        all_entities = [entity for graph in graphs for entity in graph["entities"]]
        all_relations = [relation for graph in graphs for relation in graph["relations"]]
        apply_stats = {
            "target_entity_count": target_entity_count,
            "entity_type_replacement_count": entity_replacement_count,
            "relation_type_replacement_count": relation_replacement_count,
            "newly_mapped_relation_count": newly_mapped_relation_count,
            "entity_status_counts": _entity_status_counts(all_entities),
            "relation_status_counts": _relation_status_counts(all_relations),
        }
        result["_stage"] = "stage_05_post_schema"
        result["_post_schema"] = {
            "plan_stage": plan.get("_stage"),
            "applied_at": datetime.now(timezone.utc).isoformat(),
            **apply_stats,
        }
        report = deepcopy(dict(plan))
        report["_stage"] = "stage_05_post_schema_report"
        report["apply_statistics"] = apply_stats
        return result, report


__all__ = ["PostSchemaProcessor", "payload_sha256"]
