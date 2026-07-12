from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Tuple

from .CrossModalAligner import CrossModalAligner
from .EvidenceChainBuilder import EvidenceChainBuilder
from .GeoOntologyRules import GeoOntologyRules, unique_by_name
from .GeoQuestionPlanner import GeoQuestionPlanner
from .TripleVerifier import TripleVerifier


JsonFn = Callable[[str], Any]
VlmJsonFn = Callable[[str, List[Dict[str, str]]], Any]


class GeoVlmEcaExtractor:
    """GeoVLM-ECA orchestration layer.

    The public output intentionally keeps the legacy image modal schema while
    adding evidence-chain fields that downstream code can ignore safely.
    """

    def __init__(self, llm_json_fn: JsonFn, vlm_json_fn: VlmJsonFn):
        self.llm_json_fn = llm_json_fn
        self.vlm_json_fn = vlm_json_fn
        self.ontology = GeoOntologyRules()
        self.question_planner = GeoQuestionPlanner(self.ontology)
        self.aligner = CrossModalAligner()
        self.verifier = TripleVerifier(self.ontology)

    def extract(
        self,
        image_paths: List[str],
        image_inputs: List[Dict[str, str]],
        caption: str,
        references: str,
        context: str,
    ) -> Dict[str, Any]:
        classification = self._classify_multi_view(image_inputs, caption, references, context)
        image_type = classification.get("type", "Other Geological Image")

        questions = self.question_planner.plan(image_type, caption, references, context)
        evidence_chain = EvidenceChainBuilder(self.vlm_json_fn, self.ontology).build(
            image_inputs=image_inputs,
            questions=questions,
            caption=caption,
            references=references,
            context=context,
        )

        visual_entities, text_entities = self._extract_entities(evidence_chain, caption, references, context)
        alignments = self.aligner.align(visual_entities, text_entities, evidence_chain)
        candidates = self._generate_candidate_triples(
            evidence_chain=evidence_chain,
            visual_entities=visual_entities,
            text_entities=text_entities,
            alignments=alignments,
            caption=caption,
            references=references,
            context=context,
        )
        triples = self.verifier.verify(candidates, visual_entities, text_entities, alignments, evidence_chain)
        description = self._build_description(image_type, evidence_chain, visual_entities)

        return {
            "image_paths": image_paths,
            "classification": classification,
            "description": description,
            "extraction": {
                "entities": visual_entities,
                "text_entities": text_entities,
                "alignments": alignments,
                "evidence_chain": evidence_chain,
                "candidate_triples": candidates,
                "triples": triples,
                "algorithm": "GeoVLM-ECA",
            },
        }

    def _classify_multi_view(
        self,
        image_inputs: List[Dict[str, str]],
        caption: str,
        references: str,
        context: str,
    ) -> Dict[str, Any]:
        prompt = f"""
你是石油地质多模态知识抽取专家。请从三个视角对图片组进行分类，并进行一致性投票。

分类集合:
- Curve Chart
- Geological Map
- Core/Thin Section
- Schematic Diagram
- Other Geological Image

Caption:
{caption}

References:
{references}

Context:
{context}

请仅输出 JSON:
{{
  "views": {{
    "visual_structure": {{"type": "Curve Chart|Geological Map|Core/Thin Section|Schematic Diagram|Other Geological Image", "reasoning": "..."}},
    "geological_semantics": {{"type": "...", "reasoning": "..."}},
    "extraction_task": {{"type": "...", "reasoning": "..."}}
  }},
  "type": "最终投票类型",
  "reasoning": "说明三个视角如何融合；若有冲突说明裁决原因"
}}
"""
        result = self._run_vlm_json(prompt, image_inputs)
        if isinstance(result, dict) and result.get("type"):
            result["type"] = self._normalize_image_type(str(result.get("type", "")))
            result.setdefault("reasoning", "")
            return result

        image_type = self._heuristic_image_type(caption, references, context)
        return {
            "views": {
                "visual_structure": {"type": image_type, "reasoning": "VLM 分类不可用，使用图题和上下文关键词回退。"},
                "geological_semantics": {"type": image_type, "reasoning": "基于石油地质语义关键词判定。"},
                "extraction_task": {"type": image_type, "reasoning": "根据后续关系抽取任务类型判定。"},
            },
            "type": image_type,
            "reasoning": "视觉模型未返回可解析分类，使用图题、references 和上下文进行保守分类。",
        }

    def _extract_entities(
        self,
        evidence_chain: List[Dict[str, Any]],
        caption: str,
        references: str,
        context: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        prompt = f"""
你是石油地质实体抽取专家。请基于证据链、图题、references 和上下文分别抽取视觉实体与文本实体。

领域本体:
{self.ontology.compact_schema_text()}

证据链:
{json.dumps(evidence_chain, ensure_ascii=False, indent=2)}

Caption:
{caption}

References:
{references}

Context:
{context}

要求:
1. visual_entities 必须来自 evidence_chain 的 answer、visual_evidence 或 related_entities。
2. text_entities 必须来自 Caption、References 或 Context。
3. 每个实体必须给出 type、evidence_span、confidence、source_modality。
4. 仅输出 JSON。

JSON 格式:
{{
  "visual_entities": [
    {{"name": "实体名", "type": "实体类型", "attributes": {{}}, "evidence_span": "证据片段", "confidence": 0.0, "source_modality": "figure"}}
  ],
  "text_entities": [
    {{"name": "实体名", "type": "实体类型", "attributes": {{}}, "evidence_span": "证据片段", "confidence": 0.0, "source_modality": "text"}}
  ]
}}
"""
        result = {}
        if not self._is_fallback_chain(evidence_chain):
            result = self._run_llm_json(prompt)
        visual = []
        text = []
        if isinstance(result, dict):
            visual = self._normalize_entities(result.get("visual_entities"), "figure", evidence_chain)
            text = self._normalize_entities(result.get("text_entities"), "text", evidence_chain)

        if not visual:
            visual = self._entities_from_evidence(evidence_chain)
        if not text:
            text = self._entities_from_text(caption, references, context)

        return unique_by_name(visual), unique_by_name(text)

    def _generate_candidate_triples(
        self,
        evidence_chain: List[Dict[str, Any]],
        visual_entities: List[Dict[str, Any]],
        text_entities: List[Dict[str, Any]],
        alignments: List[Dict[str, Any]],
        caption: str,
        references: str,
        context: str,
    ) -> List[Dict[str, Any]]:
        prompt = f"""
你是石油地质关系抽取专家。请基于证据链、视觉实体、文本实体、图文对齐结果和领域本体生成候选三元组。

领域本体:
{self.ontology.compact_schema_text()}

证据链:
{json.dumps(evidence_chain, ensure_ascii=False, indent=2)}

视觉实体:
{json.dumps(visual_entities, ensure_ascii=False, indent=2)}

文本实体:
{json.dumps(text_entities, ensure_ascii=False, indent=2)}

图文对齐:
{json.dumps(alignments, ensure_ascii=False, indent=2)}

Caption:
{caption}

References:
{references}

Context:
{context}

生成策略:
1. 以视觉证据为主生成一组候选。
2. 以图题和 references 为主生成一组候选。
3. 以领域本体关系模板约束生成一组候选。
4. 只保留 head 或 tail 至少一个来自视觉实体的三元组。
5. relation_type 必须优先使用本体中的关系类型。
6. 每条三元组必须给出 evidence_span、confidence、source_modality。

仅输出 JSON:
{{
  "candidate_triples": [
    {{"head": "实体1", "relation": "中文关系名", "relation_type": "schema_relation", "tail": "实体2", "evidence_span": "证据片段", "confidence": 0.0, "source_modality": "figure"}}
  ]
}}
"""
        result = {}
        if not self._is_fallback_chain(evidence_chain):
            result = self._run_llm_json(prompt)
        raw = []
        if isinstance(result, dict):
            raw = result.get("candidate_triples") or result.get("triples") or []
        candidates = self._normalize_triples(raw)
        if not candidates:
            candidates = self._fallback_triples(visual_entities, text_entities, alignments, evidence_chain)
        return candidates[:60]

    def _build_description(
        self,
        image_type: str,
        evidence_chain: List[Dict[str, Any]],
        visual_entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        answers = [str(item.get("answer", "")).strip() for item in evidence_chain if isinstance(item, dict) and item.get("answer")]
        observations = []
        for item in evidence_chain:
            if not isinstance(item, dict):
                continue
            statement = str(item.get("answer", "")).strip()
            if not statement:
                continue
            observations.append({
                "statement": statement,
                "evidence_span": str(item.get("visual_evidence") or item.get("text_support") or "").strip(),
                "confidence": item.get("confidence", 0.7),
            })
        return {
            "detailed_description": " ".join(answers),
            "image_type": image_type,
            "visual_entities": visual_entities,
            "key_observations": observations,
            "evidence_chain": evidence_chain,
        }

    def _run_llm_json(self, prompt: str) -> Any:
        try:
            return self.llm_json_fn(prompt)
        except Exception as exc:
            print(f"LLM JSON 调用失败: {exc}")
            return {}

    def _run_vlm_json(self, prompt: str, image_inputs: List[Dict[str, str]]) -> Any:
        try:
            return self.vlm_json_fn(prompt, image_inputs)
        except Exception as exc:
            print(f"VLM JSON 调用失败: {exc}")
            return {}

    def _normalize_entities(
        self,
        raw: Any,
        modality: str,
        evidence_chain: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        if not isinstance(raw, list):
            return result
        evidence_text = "\n".join(
            f"{item.get('answer', '')} {item.get('visual_evidence', '')} {item.get('text_support', '')}"
            for item in evidence_chain
            if isinstance(item, dict)
        )
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            entity_type = self.ontology.normalize_entity_type(
                str(item.get("type") or self.ontology.infer_entity_type(name, evidence_text))
            )
            result.append({
                "name": name,
                "type": entity_type,
                "attributes": item.get("attributes") if isinstance(item.get("attributes"), dict) else {},
                "evidence_span": str(item.get("evidence_span") or self._find_evidence(name, evidence_chain)).strip(),
                "confidence": self._safe_confidence(item.get("confidence"), 0.72),
                "source_modality": item.get("source_modality") or modality,
            })
        return result

    def _entities_from_evidence(self, evidence_chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []
        for item in evidence_chain:
            if not isinstance(item, dict):
                continue
            evidence_span = f"{item.get('answer', '')} {item.get('visual_evidence', '')}".strip()
            for entity in item.get("related_entities", []) or []:
                if not isinstance(entity, dict):
                    continue
                name = str(entity.get("name", "")).strip()
                if not name:
                    continue
                entities.append({
                    "name": name,
                    "type": self.ontology.normalize_entity_type(str(entity.get("type") or self.ontology.infer_entity_type(name, evidence_span))),
                    "attributes": {},
                    "evidence_span": evidence_span[:360],
                    "confidence": item.get("confidence", 0.68),
                    "source_modality": "figure",
                })
        return unique_by_name(entities)

    def _entities_from_text(self, caption: str, references: str, context: str) -> List[Dict[str, Any]]:
        text = "\n".join(part for part in (caption, references, context) if part)
        if not text:
            return []
        keywords = []
        domain_terms = (
            "鄂尔多斯盆地", "召井", "山西组下段", "山3", "山2", "山1", "太原组",
            "东大窑灰岩", "北岔沟砂岩", "伊盟古陆", "兴蒙海槽", "湖盆", "近海湖泊",
            "物源区", "北部物源区", "浅变质岩结晶基底", "基准面", "可容纳空间",
            "沉积相", "沉积微相", "微相", "陆相三角洲", "三角洲平原", "三角洲前缘",
            "河流相沉积体系", "分流河道", "水下分流河道", "河口坝", "决口扇",
            "分流间洼地", "分流间湾", "泥炭沼泽", "含煤建造", "含煤旋回", "厚煤层",
            "煤层", "砂体", "储集砂体", "湖岸线", "测井曲线", "岩石相", "测井相",
            "储层", "露头剖面", "取心井段",
        )
        for term in domain_terms:
            if term in text:
                keywords.append(term)

        for match in re.finditer(r"(山\s*\$?_?\s*[123]\s*\^?\s*\{?\s*[123]?\s*\}?)", text):
            cleaned = re.sub(r"[\s$_^\{\}-]+", "", match.group(1))
            if cleaned:
                keywords.append(cleaned)

        result = []
        for name in sorted(set(keywords), key=len, reverse=True)[:40]:
            if not name:
                continue
            result.append({
                "name": name,
                "type": self.ontology.infer_entity_type(name, text),
                "attributes": {},
                "evidence_span": self._text_window(text, name),
                "confidence": 0.62,
                "source_modality": "text",
            })
        return unique_by_name(result)

    def _normalize_triples(self, raw: Any) -> List[Dict[str, Any]]:
        triples: List[Dict[str, Any]] = []
        if not isinstance(raw, list):
            return triples
        for item in raw:
            if not isinstance(item, dict):
                continue
            head = str(item.get("head") or item.get("source") or "").strip()
            tail = str(item.get("tail") or item.get("target") or "").strip()
            relation = str(item.get("relation") or "").strip()
            relation_type = self.ontology.normalize_relation_type(relation, str(item.get("relation_type") or item.get("type") or ""))
            if not head or not tail:
                continue
            triples.append({
                "head": head,
                "relation": relation or relation_type,
                "relation_type": relation_type,
                "tail": tail,
                "evidence_span": str(item.get("evidence_span") or item.get("evidence") or "").strip(),
                "confidence": self._safe_confidence(item.get("confidence"), 0.68),
                "source_modality": item.get("source_modality") or "figure",
            })
        return triples

    @staticmethod
    def _is_fallback_chain(evidence_chain: List[Dict[str, Any]]) -> bool:
        if not evidence_chain:
            return False
        return all(
            isinstance(item, dict) and str(item.get("question_id", "")).startswith("fallback")
            for item in evidence_chain
        )

    def _fallback_triples(
        self,
        visual_entities: List[Dict[str, Any]],
        text_entities: List[Dict[str, Any]],
        alignments: List[Dict[str, Any]],
        evidence_chain: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        triples: List[Dict[str, Any]] = []
        visual_names = {str(item.get("name", "")).strip() for item in visual_entities if isinstance(item, dict)}

        all_entities = unique_by_name([*visual_entities, *text_entities])
        for head in all_entities:
            for tail in all_entities:
                if head is tail:
                    continue
                if head.get("name") not in visual_names and tail.get("name") not in visual_names:
                    continue
                relation_types = self.ontology.allowed_relation_hint(head.get("type", ""), tail.get("type", ""))
                if not relation_types:
                    continue
                triples.append({
                    "head": head.get("name", ""),
                    "relation": self._relation_zh(relation_types[0]),
                    "relation_type": relation_types[0],
                    "tail": tail.get("name", ""),
                    "evidence_span": self._find_pair_evidence(head.get("name", ""), tail.get("name", ""), evidence_chain),
                    "confidence": 0.58,
                    "source_modality": "figure",
                })
                if len(triples) >= 50:
                    return triples
        return triples

    @staticmethod
    def _normalize_image_type(value: str) -> str:
        lowered = value.lower()
        if "curve" in lowered or "chart" in lowered:
            return "Curve Chart"
        if "map" in lowered or "geological" in lowered:
            return "Geological Map"
        if "core" in lowered or "thin" in lowered or "section" in lowered:
            return "Core/Thin Section"
        if "schematic" in lowered or "diagram" in lowered:
            return "Schematic Diagram"
        return "Other Geological Image"

    def _heuristic_image_type(self, caption: str, references: str, context: str) -> str:
        caption_text = str(caption or "").lower()
        if any(k in caption_text for k in ("平面展布", "地质图", "沉积相平面", "展布图")):
            return "Geological Map"
        if any(k in caption_text for k in ("地层划分", "沉积相分析", "剖面", "综合柱状", "分析图")):
            return "Schematic Diagram"

        text = f"{caption} {references} {context}".lower()
        if any(k in text for k in ("曲线", "测井", "chart", "curve")):
            return "Curve Chart"
        if any(k in text for k in ("平面展布", "地质图", "沉积相", "map", "盆地")):
            return "Geological Map"
        if any(k in text for k in ("岩心", "薄片", "显微", "core", "thin")):
            return "Core/Thin Section"
        if any(k in text for k in ("模式", "示意", "流程", "diagram", "成藏")):
            return "Schematic Diagram"
        return "Other Geological Image"

    @staticmethod
    def _safe_confidence(value: Any, default: float) -> float:
        try:
            number = float(value)
        except Exception:
            number = default
        return max(0.0, min(1.0, round(number, 3)))

    @staticmethod
    def _text_window(text: str, name: str, radius: int = 80) -> str:
        idx = text.find(name)
        if idx < 0:
            return text[:160]
        start = max(0, idx - radius)
        end = min(len(text), idx + len(name) + radius)
        return text[start:end]

    @staticmethod
    def _relation_zh(relation_type: str) -> str:
        mapping = {
            "contains": "包含",
            "has_lithology": "主要岩性",
            "has_facies": "发育沉积相",
            "located_in": "位于",
            "adjacent_to": "邻接",
            "precedes": "先于",
            "causes": "导致",
            "controls": "控制",
            "controlled_by": "受控于",
            "indicates": "指示",
            "quantifies": "量化",
            "has_trend": "具有趋势",
            "aligns_with": "图文对齐",
            "supports": "支持",
        }
        return mapping.get(relation_type, relation_type)

    @staticmethod
    def _find_evidence(name: str, evidence_chain: List[Dict[str, Any]]) -> str:
        for item in evidence_chain:
            if not isinstance(item, dict):
                continue
            evidence = f"{item.get('answer', '')} {item.get('visual_evidence', '')} {item.get('text_support', '')}".strip()
            if name and name in evidence:
                return evidence[:360]
        return ""

    @staticmethod
    def _find_pair_evidence(head: str, tail: str, evidence_chain: List[Dict[str, Any]]) -> str:
        fallback = ""
        for item in evidence_chain:
            if not isinstance(item, dict):
                continue
            evidence = f"{item.get('answer', '')} {item.get('visual_evidence', '')} {item.get('text_support', '')}".strip()
            if not fallback and evidence:
                fallback = evidence[:360]
            if head in evidence and tail in evidence:
                return evidence[:360]
        return fallback or "关系由证据链和领域本体模板联合生成。"
