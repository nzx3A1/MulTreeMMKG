from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .GeoOntologyRules import GeoOntologyRules


class TripleVerifier:
    """Ontology and evidence based verification for candidate image triples."""

    def __init__(self, ontology: GeoOntologyRules | None = None, threshold: float = 0.56):
        self.ontology = ontology or GeoOntologyRules()
        self.threshold = threshold

    def verify(
        self,
        triples: Iterable[Dict[str, Any]],
        visual_entities: List[Dict[str, Any]],
        text_entities: List[Dict[str, Any]],
        alignments: List[Dict[str, Any]],
        evidence_chain: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        entity_types = self._entity_type_index(visual_entities, text_entities)
        evidence_text = "\n".join(
            f"{item.get('answer', '')} {item.get('visual_evidence', '')} {item.get('text_support', '')}"
            for item in evidence_chain
            if isinstance(item, dict)
        )
        alignment_pairs = {
            (item.get("visual_entity"), item.get("text_entity")): float(item.get("score", 0))
            for item in alignments
            if isinstance(item, dict)
        }

        verified: List[Dict[str, Any]] = []
        seen = set()
        for raw in triples or []:
            if not isinstance(raw, dict):
                continue
            triple = self._normalize_triple(raw)
            head = triple.get("head", "")
            tail = triple.get("tail", "")
            if not head or not tail or head == tail:
                continue
            key = (head, triple.get("relation_type", ""), tail)
            if key in seen:
                continue
            seen.add(key)

            head_type = entity_types.get(head, "")
            tail_type = entity_types.get(tail, "")
            relation_type = triple.get("relation_type", "")
            ontology_ok = self.ontology.best_effort_allowed(head_type, relation_type, tail_type)
            evidence_score = self._evidence_score(head, tail, triple.get("evidence_span", ""), evidence_text)
            alignment_score = self._alignment_score(head, tail, alignment_pairs)
            model_conf = self._safe_float(triple.get("confidence"), 0.68)
            ontology_score = 1.0 if ontology_ok else 0.35
            score = (
                0.32 * evidence_score
                + 0.2 * alignment_score
                + 0.24 * ontology_score
                + 0.18 * model_conf
                + 0.06 * self._consistency_score(triple, triples)
            )
            score = round(max(0.0, min(1.0, score)), 3)

            if score < self.threshold and not ontology_ok:
                continue

            triple["confidence"] = max(score, round(model_conf, 3))
            if not triple.get("evidence_span"):
                triple["evidence_span"] = self._default_evidence(head, tail, evidence_chain)
            triple["source_modality"] = triple.get("source_modality") or "figure"
            triple["verification"] = {
                "ontology_valid": ontology_ok,
                "evidence_score": round(evidence_score, 3),
                "alignment_score": round(alignment_score, 3),
                "score": score,
            }
            verified.append(triple)

        return sorted(verified, key=lambda item: item.get("confidence", 0), reverse=True)

    def _normalize_triple(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        head = str(raw.get("head") or raw.get("source") or "").strip()
        tail = str(raw.get("tail") or raw.get("target") or "").strip()
        relation = str(raw.get("relation") or raw.get("predicate") or raw.get("type") or "").strip()
        relation_type = self.ontology.normalize_relation_type(relation, str(raw.get("relation_type") or raw.get("type") or ""))
        return {
            "head": head,
            "relation": relation or relation_type,
            "relation_type": relation_type,
            "tail": tail,
            "evidence_span": str(raw.get("evidence_span") or raw.get("evidence") or "").strip(),
            "confidence": self._safe_float(raw.get("confidence"), 0.68),
            "source_modality": str(raw.get("source_modality") or "figure").strip(),
        }

    @staticmethod
    def _entity_type_index(*entity_lists: List[Dict[str, Any]]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for entities in entity_lists:
            for entity in entities or []:
                if not isinstance(entity, dict):
                    continue
                name = str(entity.get("name", "")).strip()
                entity_type = str(entity.get("type", "")).strip()
                if name and entity_type:
                    result.setdefault(name, entity_type)
        return result

    @staticmethod
    def _evidence_score(head: str, tail: str, evidence_span: str, evidence_text: str) -> float:
        haystack = f"{evidence_span}\n{evidence_text}"
        score = 0.25
        if evidence_span:
            score += 0.25
        if head and head in haystack:
            score += 0.25
        if tail and tail in haystack:
            score += 0.25
        return min(1.0, score)

    @staticmethod
    def _alignment_score(head: str, tail: str, pairs: Dict[tuple[str, str], float]) -> float:
        score = 0.0
        for (visual, text), value in pairs.items():
            if head in (visual, text) or tail in (visual, text):
                score = max(score, value)
        return score or 0.45

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return default

    @staticmethod
    def _consistency_score(triple: Dict[str, Any], triples: Iterable[Dict[str, Any]]) -> float:
        key = (triple.get("head"), triple.get("relation_type"), triple.get("tail"))
        count = 0
        for item in triples or []:
            if not isinstance(item, dict):
                continue
            if (item.get("head") or item.get("source"), item.get("relation_type") or item.get("type"), item.get("tail") or item.get("target")) == key:
                count += 1
        return min(1.0, 0.5 + count * 0.25)

    @staticmethod
    def _default_evidence(head: str, tail: str, evidence_chain: List[Dict[str, Any]]) -> str:
        for item in evidence_chain:
            if not isinstance(item, dict):
                continue
            evidence = f"{item.get('answer', '')} {item.get('visual_evidence', '')} {item.get('text_support', '')}".strip()
            if head in evidence or tail in evidence:
                return evidence[:360]
        return "候选关系来源于视觉证据链、图题或 references 的联合约束。"

