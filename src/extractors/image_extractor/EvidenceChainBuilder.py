from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from .GeoOntologyRules import GeoOntologyRules


class EvidenceChainBuilder:
    """Ask VLM type-specific questions and normalize answers into evidence chains."""

    def __init__(
        self,
        vlm_json_fn: Callable[[str, List[Dict[str, str]]], Any],
        ontology: GeoOntologyRules | None = None,
    ):
        self.vlm_json_fn = vlm_json_fn
        self.ontology = ontology or GeoOntologyRules()

    def build(
        self,
        image_inputs: List[Dict[str, str]],
        questions: List[Dict[str, str]],
        caption: str = "",
        references: str = "",
        context: str = "",
    ) -> List[Dict[str, Any]]:
        prompt = f"""
你是石油地质图像知识抽取专家。请严格围绕下列视觉问题逐项回答，并为每个回答绑定图像证据、图题证据或 references 证据。

图题 Caption:
{caption}

References:
{references}

Context:
{context}

领域本体约束:
{self.ontology.compact_schema_text()}

问题列表:
{json.dumps(questions, ensure_ascii=False, indent=2)}

输出要求:
1. 只能输出 JSON，不要输出 Markdown。
2. 每个 related_entities 的 type 必须使用领域本体中的实体类型。
3. answer 和 visual_evidence 不能凭空补充图中或文本中没有的内容。
4. 如果图中无法确认，answer 写“无法从图中确认”，confidence 低于 0.45。

JSON 格式:
{{
  "evidence_chain": [
    {{
      "question_id": "q1",
      "question": "原问题",
      "answer": "结构化回答",
      "visual_evidence": "图中可见证据",
      "text_support": "图题/references/context 中的支持证据，没有则为空字符串",
      "related_entities": [
        {{"name": "实体名", "type": "Stratum|Lithology|Reservoir|SourceRock|Fault|Structure|Facies|Curve|Axis|Parameter|Trend|Process|Hydrocarbon|EvidenceRegion"}}
      ],
      "confidence": 0.0
    }}
  ]
}}
"""
        try:
            result = self.vlm_json_fn(prompt, image_inputs)
        except Exception as exc:
            print(f"构建视觉证据链失败，使用文本回退: {exc}")
            result = {}

        raw_chain: Any = result.get("evidence_chain") if isinstance(result, dict) else result
        if not isinstance(raw_chain, list):
            return self._fallback_chain(questions, caption, references, context)

        normalized: List[Dict[str, Any]] = []
        question_map = {q.get("id"): q for q in questions if isinstance(q, dict)}
        for idx, item in enumerate(raw_chain, start=1):
            if not isinstance(item, dict):
                continue
            qid = str(item.get("question_id") or item.get("id") or f"q{idx}")
            question = str(item.get("question") or question_map.get(qid, {}).get("question", "")).strip()
            answer = str(item.get("answer") or "").strip()
            visual_evidence = str(item.get("visual_evidence") or item.get("evidence_span") or "").strip()
            text_support = str(item.get("text_support") or "").strip()
            related_entities = self._normalize_entities(item.get("related_entities"), f"{answer} {visual_evidence}")
            confidence = self._safe_confidence(item.get("confidence"), default=0.68 if answer else 0.35)
            if not question and not answer and not visual_evidence:
                continue
            normalized.append({
                "question_id": qid,
                "question": question,
                "answer": answer,
                "visual_evidence": visual_evidence,
                "text_support": text_support,
                "related_entities": related_entities,
                "confidence": confidence,
            })

        return normalized or self._fallback_chain(questions, caption, references, context)

    def _normalize_entities(self, value: Any, context: str = "") -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                    if not name:
                        continue
                    entity_type = self.ontology.normalize_entity_type(
                        str(item.get("type") or self.ontology.infer_entity_type(name, context))
                    )
                    entities.append({"name": name, "type": entity_type})
                elif item:
                    name = str(item).strip()
                    entities.append({"name": name, "type": self.ontology.infer_entity_type(name, context)})
        return self._dedupe_entities(entities)

    @staticmethod
    def _dedupe_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        result: List[Dict[str, Any]] = []
        for entity in entities:
            name = str(entity.get("name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            result.append(entity)
        return result

    @staticmethod
    def _safe_confidence(value: Any, default: float) -> float:
        try:
            number = float(value)
        except Exception:
            number = default
        return max(0.0, min(1.0, round(number, 3)))

    def _fallback_chain(
        self,
        questions: List[Dict[str, str]],
        caption: str,
        references: str,
        context: str,
    ) -> List[Dict[str, Any]]:
        text = "\n".join(part for part in (caption, references, context) if part)
        entities = []
        for token in (
            "山西组下段", "山3", "山2", "山1", "分流河道", "水下分流河道", "决口扇",
            "分流间洼地", "分流间湾", "泥炭沼泽", "陆相三角洲", "三角洲平原", "三角洲前缘",
            "河口坝", "含煤旋回", "含煤建造", "煤层", "砂体", "北岔沟砂岩", "湖岸线",
            "湖盆", "物源区", "基准面", "可容纳空间", "太原组", "东大窑灰岩",
        ):
            if token in text:
                entities.append({"name": token, "type": self.ontology.infer_entity_type(token, text)})
        first_question = questions[0].get("question", "文本证据回退") if questions else "文本证据回退"
        return [{
            "question_id": "fallback_q1",
            "question": first_question,
            "answer": "视觉模型未返回可解析证据链，当前结果基于图题、references 和上下文进行保守抽取。",
            "visual_evidence": caption or "缺少可解析视觉证据",
            "text_support": text[:500],
            "related_entities": self._dedupe_entities(entities),
            "confidence": 0.42,
        }]
