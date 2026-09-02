"""让 LLM 在向量/端点约束候选与 ``NONE`` 之间做保守选择。"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from src.utils.llm_client import LLMClient

from .models import SchemaRelation, SelectionDecision, SemanticCandidate, clamp_confidence
from .semantic_retriever import build_entity_embedding_text


class CandidateSelector(Protocol):
    """实体、关系候选判定的可替换接口。"""

    def select_entities(
        self,
        requests: Sequence[tuple[Mapping[str, Any], Sequence[SemanticCandidate]]],
    ) -> list[SelectionDecision]: ...

    def select_relations(
        self,
        requests: Sequence[
            tuple[Mapping[str, Any], str, str, Sequence[SchemaRelation]]
        ],
    ) -> list[SelectionDecision]: ...


class SchemaLLMSelector:
    """批量请求 LLM，减少大规模图对齐时的网络往返。"""

    def __init__(self, llm_client: LLMClient | None = None, batch_size: int = 8) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.llm_client = llm_client or LLMClient()
        self.batch_size = batch_size

    @staticmethod
    def _parse_decisions(
        payload: Any,
        allowed: Sequence[set[str]],
    ) -> list[SelectionDecision]:
        rows = payload.get("decisions", []) if isinstance(payload, Mapping) else []
        by_index: dict[int, Mapping[str, Any]] = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, Mapping):
                continue
            try:
                by_index[int(row.get("index"))] = row
            except (TypeError, ValueError):
                continue

        decisions: list[SelectionDecision] = []
        for index, allowed_values in enumerate(allowed):
            row = by_index.get(index)
            if row is None:
                decisions.append(SelectionDecision(None, 0.0, "LLM 未返回该条目的有效判定"))
                continue
            raw_selected = str(row.get("selected") or "NONE").strip()
            selected = None if raw_selected.upper() == "NONE" else raw_selected
            if selected is not None and selected not in allowed_values:
                decisions.append(
                    SelectionDecision(None, 0.0, f"LLM 返回了候选集合外的值：{selected}")
                )
                continue
            decisions.append(
                SelectionDecision(
                    selected=selected,
                    confidence=clamp_confidence(row.get("confidence")),
                    reason=str(row.get("reason") or "").strip() or "LLM 未提供理由",
                )
            )
        return decisions

    def _call_batches(
        self,
        items: Sequence[dict[str, Any]],
        allowed: Sequence[set[str]],
        system_prompt: str,
        task_name: str,
    ) -> list[SelectionDecision]:
        result: list[SelectionDecision] = []
        for start in range(0, len(items), self.batch_size):
            batch_items = list(items[start : start + self.batch_size])
            batch_allowed = list(allowed[start : start + self.batch_size])
            # 每个批次内部重新编号，避免模型输出稀疏或全局索引混淆。
            for local_index, item in enumerate(batch_items):
                item["index"] = local_index
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "逐项判定以下 JSON 数据。只返回 JSON 对象："
                        '{"decisions":[{"index":0,"selected":"候选值或NONE",'
                        '"confidence":0.0,"reason":"简短理由"}]}。\n'
                        f"items={items_to_json(batch_items)}"
                    ),
                },
            ]
            payload = self.llm_client.chat_json(messages)
            result.extend(self._parse_decisions(payload, batch_allowed))
        return result

    def select_entities(
        self,
        requests: Sequence[tuple[Mapping[str, Any], Sequence[SemanticCandidate]]],
    ) -> list[SelectionDecision]:
        items: list[dict[str, Any]] = []
        allowed: list[set[str]] = []
        for entity, candidates in requests:
            items.append(
                {
                    "entity": build_entity_embedding_text(entity),
                    "candidates": [
                        {
                            "schema": candidate.concept.schema,
                            "zhName": candidate.concept.zh_name,
                            "category": candidate.concept.category,
                            "description": candidate.concept.description,
                            "examples": list(candidate.concept.examples),
                            "vector_similarity": round(candidate.similarity, 6),
                        }
                        for candidate in candidates
                    ],
                    "options": [candidate.concept.schema for candidate in candidates] + ["NONE"],
                }
            )
            allowed.append({candidate.concept.schema for candidate in candidates})
        return self._call_batches(
            items,
            allowed,
            system_prompt=(
                "你是石油地质概念 Schema 对齐器。只能从每项 options 中选一个值。"
                "实体与所有候选都不充分吻合时必须选 NONE，不得为了提高覆盖率强制归类。"
                "结合名称、原始类型、中文类型、属性和溯源判断；向量相似度只用于召回，不是最终证据。"
            ),
            task_name="实体 Schema 对齐",
        )

    def select_relations(
        self,
        requests: Sequence[
            tuple[Mapping[str, Any], str, str, Sequence[SchemaRelation]]
        ],
    ) -> list[SelectionDecision]:
        items: list[dict[str, Any]] = []
        allowed: list[set[str]] = []
        for relation, source_schema, target_schema, candidates in requests:
            raw_relation = relation.get("raw_relation", relation.get("type"))
            items.append(
                {
                    "source_schema": source_schema,
                    "target_schema": target_schema,
                    "raw_relation": raw_relation,
                    "relation_name": relation.get("relation_name"),
                    "type_zh": relation.get("type_zh"),
                    "attributes": relation.get("attributes") or {},
                    "provenance": relation.get("provenance"),
                    "candidates": [
                        {"relationEn": item.relation_en, "relationZh": item.relation_zh}
                        for item in candidates
                    ],
                    "options": [item.relation_en for item in candidates] + ["NONE"],
                }
            )
            allowed.append({item.relation_en for item in candidates})
        return self._call_batches(
            items,
            allowed,
            system_prompt=(
                "你是石油地质关系 Schema 对齐器。候选已由有向 source/target Schema 严格过滤。"
                "必须根据原始 relation type、relation_name、type_zh、attributes 和 provenance 区分真实语义；"
                "不能只看端点类型。比如生成与排出必须区分。没有合适关系就选 NONE。"
                "除非原始语义本身只是‘相关’，否则不得选择 RELATED_TO。"
            ),
            task_name="关系 Schema 对齐",
        )


def items_to_json(value: Any) -> str:
    """延迟导入 JSON，统一确保中文原样进入提示词。"""

    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


__all__ = ["CandidateSelector", "SchemaLLMSelector"]
