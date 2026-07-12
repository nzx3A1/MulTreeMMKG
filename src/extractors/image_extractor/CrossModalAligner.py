from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List


class CrossModalAligner:
    """Bidirectional alignment between visual entities and text entities."""

    def __init__(self, threshold: float = 0.58):
        self.threshold = threshold

    def align(
        self,
        visual_entities: List[Dict[str, Any]],
        text_entities: List[Dict[str, Any]],
        evidence_chain: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        visual_entities = self._dedupe(visual_entities)
        text_entities = self._dedupe(text_entities)
        if not visual_entities or not text_entities:
            return []

        scores: Dict[tuple[int, int], float] = {}
        for vi, visual in enumerate(visual_entities):
            for ti, text in enumerate(text_entities):
                scores[(vi, ti)] = self._score_pair(visual, text, evidence_chain)

        visual_top = {
            vi: max(range(len(text_entities)), key=lambda ti: scores[(vi, ti)])
            for vi in range(len(visual_entities))
        }
        text_top = {
            ti: max(range(len(visual_entities)), key=lambda vi: scores[(vi, ti)])
            for ti in range(len(text_entities))
        }

        alignments: List[Dict[str, Any]] = []
        for vi, visual in enumerate(visual_entities):
            for ti, text in enumerate(text_entities):
                score = scores[(vi, ti)]
                mutual_top1 = visual_top.get(vi) == ti and text_top.get(ti) == vi
                if not mutual_top1 and score < self.threshold:
                    continue
                alignments.append({
                    "visual_entity": visual.get("name", ""),
                    "visual_type": visual.get("type", ""),
                    "text_entity": text.get("name", ""),
                    "text_type": text.get("type", ""),
                    "score": round(score, 3),
                    "mutual_top1": mutual_top1,
                    "evidence_span": self._alignment_evidence(visual, text, evidence_chain),
                    "source_modality": "figure+text",
                })

        return sorted(alignments, key=lambda item: item.get("score", 0), reverse=True)

    def _score_pair(
        self,
        visual: Dict[str, Any],
        text: Dict[str, Any],
        evidence_chain: List[Dict[str, Any]],
    ) -> float:
        visual_name = str(visual.get("name", "")).strip()
        text_name = str(text.get("name", "")).strip()
        name_sim = self._name_similarity(visual_name, text_name)
        type_sim = 1.0 if str(visual.get("type", "")).strip() == str(text.get("type", "")).strip() else 0.35
        evidence_support = self._evidence_support(visual_name, text_name, evidence_chain)
        context_sim = 0.75 if evidence_support > 0 else 0.45
        return min(1.0, 0.38 * name_sim + 0.2 * type_sim + 0.2 * context_sim + 0.22 * evidence_support)

    @staticmethod
    def _name_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        if a in b or b in a:
            return 0.82
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _evidence_support(a: str, b: str, evidence_chain: List[Dict[str, Any]]) -> float:
        joined = "\n".join(
            f"{item.get('answer', '')} {item.get('visual_evidence', '')} {item.get('text_support', '')}"
            for item in evidence_chain
            if isinstance(item, dict)
        )
        if a and b and a in joined and b in joined:
            return 1.0
        if a and a in joined:
            return 0.72
        if b and b in joined:
            return 0.55
        return 0.0

    @staticmethod
    def _alignment_evidence(
        visual: Dict[str, Any],
        text: Dict[str, Any],
        evidence_chain: List[Dict[str, Any]],
    ) -> str:
        visual_name = str(visual.get("name", "")).strip()
        text_name = str(text.get("name", "")).strip()
        for item in evidence_chain:
            if not isinstance(item, dict):
                continue
            evidence = f"{item.get('answer', '')} {item.get('visual_evidence', '')} {item.get('text_support', '')}".strip()
            if visual_name and visual_name in evidence:
                return evidence[:360]
            if text_name and text_name in evidence:
                return evidence[:360]
        return f"视觉实体“{visual_name}”与文本实体“{text_name}”在同一图题或引用上下文中出现。"

    @staticmethod
    def _dedupe(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        result: List[Dict[str, Any]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            result.append(item)
        return result

