from __future__ import annotations

import os
import re
import sys
import uuid
from typing import Any, Dict, List, Optional, Set

# Add project root to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

from modalprocessor.BaseModel import BaseModel
from modalprocessor.embedding_utils import attach_embedding_to_node
from config.top_schema_config import NodeLabel, RelationType


class FormulaModalProcessor(BaseModel):
    ALLOWED_RELATION_TYPES = {
        "spatial",
        "temporal",
        "containment",
        "causal",
        "control",
        "evolution",
        "synonym",
        "related",
        "defines",
    }

    RELATION_TYPE_MAP = {
        "defines": RelationType.HAS_SYMBOL,
        "related": "related",
        "containment": RelationType.CONTAINS,
    }

    COMMAND_IGNORE = {
        "left",
        "right",
        "mathrm",
        "mathbf",
        "mathit",
        "text",
        "frac",
        "sum",
        "int",
        "sqrt",
        "times",
        "cdot",
        "tag",
        "begin",
        "end",
    }

    GREEK_SYMBOLS = {
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "theta",
        "lambda",
        "mu",
        "nu",
        "pi",
        "rho",
        "sigma",
        "tau",
        "phi",
        "omega",
    }

    CONSTANT_SYMBOLS = {"g", "R", "pi", "e"}

    def __init__(self, config_path: str = None, rate_limit_qps: Optional[float] = None):
        """初始化公式处理器，并应用统一的 LLM 请求速率上限。"""
        super().__init__(config_path, rate_limit_qps=rate_limit_qps)

    @classmethod
    def _normalize_relation_type(cls, value: str) -> str:
        rel_type = (value or "").strip().lower()
        alias_map = {
            "belongs_to": "containment",
            "belongsto": "containment",
            "define": "defines",
            "define_as": "defines",
        }
        rel_type = alias_map.get(rel_type, rel_type)
        if rel_type not in cls.ALLOWED_RELATION_TYPES:
            return "related"
        return rel_type

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        s = (symbol or "").strip()
        if not s:
            return ""

        # Normalize common latex wrappers
        s = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", s)
        s = re.sub(r"\\mathbf\{([^{}]+)\}", r"\1", s)
        s = re.sub(r"\\mathit\{([^{}]+)\}", r"\1", s)
        s = s.replace(" ", "")
        s = re.sub(r"_\{([^{}]+)\}", r"_\1", s)
        s = s.replace("{", "").replace("}", "")

        # trim surrounding braces
        if s.startswith("{") and s.endswith("}") and len(s) > 2:
            s = s[1:-1]

        # Concatenation artifact like MH_s -> keep H_s (M will be captured separately)
        if re.fullmatch(r"[A-Z]{2,}_[A-Za-z0-9]+", s):
            prefix, suffix = s.split("_", 1)
            s = f"{prefix[-1]}_{suffix}"

        # Concatenation artifact like Zq_e -> keep q_e
        m = re.search(r"([a-z]+_[A-Za-z0-9]+)$", s)
        if m and re.match(r"^[A-Z][a-z]+_[A-Za-z0-9]+$", s):
            s = m.group(1)

        # Concatenation artifact like sigmaMH_s -> keep H_s (sigma handled separately)
        m = re.search(r"([A-Z]+_[A-Za-z0-9]+)$", s)
        if m and re.match(r"^[a-z]+[A-Z]+_[A-Za-z0-9]+$", s):
            s = m.group(1)
            if re.fullmatch(r"[A-Z]{2,}_[A-Za-z0-9]+", s):
                prefix, suffix = s.split("_", 1)
                s = f"{prefix[-1]}_{suffix}"

        return s.strip()

    @staticmethod
    def _extract_equation_tag(latex: str) -> str:
        if not latex:
            return ""
        m = re.search(r"\\tag\s*\{([^{}]+)\}", latex)
        if m:
            return m.group(1).strip()
        return ""

    def _extract_symbol_candidates(self, latex: str) -> List[str]:
        if not latex:
            return []

        candidates: Set[str] = set()

        normalized_latex = latex
        normalized_latex = re.sub(r"\\mathrm\s*\{\s*([^{}]+)\s*\}", r"\1", normalized_latex)
        normalized_latex = re.sub(r"\\mathbf\s*\{\s*([^{}]+)\s*\}", r"\1", normalized_latex)
        normalized_latex = re.sub(r"\\mathit\s*\{\s*([^{}]+)\s*\}", r"\1", normalized_latex)
        normalized_latex = re.sub(r"\s+", "", normalized_latex)

        # 1) variable with subscript style: q_{e}, H_s, rho_{gl}
        complex_vars = re.findall(r"([A-Za-z]+(?:_\{?[A-Za-z0-9]+\}?)+)", normalized_latex)
        for token in complex_vars:
            normalized = self._normalize_symbol(token)
            if normalized:
                candidates.add(normalized)

        # 2) latex command symbols: keep greek symbols only
        cmd_vars = re.findall(r"\\([A-Za-z]+)", latex)
        for cmd in cmd_vars:
            if cmd in self.GREEK_SYMBOLS:
                candidates.add(cmd)

        # Remove all latex commands to avoid command fragment noise in plain regex
        plain_text = re.sub(r"\\[A-Za-z]+", " ", normalized_latex)
        plain_text = re.sub(r"[{}]", "", plain_text)

        # 3) standalone uppercase symbols (M, R, T, Z, H)
        upper_vars = re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", plain_text)
        for token in upper_vars:
            token = token.strip()
            if not token:
                continue
            candidates.add(token)

        # 4) standalone lowercase one-letter symbols around operators (q, r, g, h)
        lower_vars = re.findall(r"(?<=[=+\-*/(])([a-z])(?=[=+\-*/)])", plain_text)
        for token in lower_vars:
            if token in {"i", "j", "k"}:
                continue
            candidates.add(token)

        # 5) remove obvious numeric/unit noise
        cleaned: List[str] = []
        for token in candidates:
            token = self._normalize_symbol(token)
            if not token:
                continue
            compact = token.replace("_", "")
            if re.fullmatch(r"[0-9\.]+", compact):
                continue
            if token.lower() in {"left", "right", "tag", "frac", "sum", "int", "cos", "sin", "tan"}:
                continue
            if len(token) > 30:
                continue
            cleaned.append(token)

        # stable order
        return sorted(set(cleaned), key=lambda x: (len(x), x))

    def _infer_symbol_type(self, symbol: str) -> str:
        s = (symbol or "").strip()
        if not s:
            return "Variable"
        if s in self.CONSTANT_SYMBOLS:
            return "Constant"
        if any(p in s for p in ["rho", "sigma", "phi", "theta", "mu", "lambda", "omega"]):
            return "Parameter"
        if "_" in s:
            return "Variable"
        if len(s) == 1 and s.isupper():
            return "Parameter"
        return "Variable"

    def _build_formula_node(self, formula_data: Dict[str, Any], chapter_title: str) -> Dict[str, Any]:
        latex = (formula_data.get("latex") or "").strip()
        formula_id = formula_data.get("id") or str(uuid.uuid4())
        eq_tag = self._extract_equation_tag(latex)

        if eq_tag:
            name = f"公式[{eq_tag}]::{formula_id}"
        else:
            short = latex.replace("\n", " ")[:48]
            name = f"公式::{short}::{formula_id}"

        formula_node = {
            "id": formula_id,
            "name": name,
            "type": NodeLabel.FORMULA,
            "label": NodeLabel.FORMULA,
            "label_zh": "公式",
            "latex": latex,
            "display_type": formula_data.get("display_type", "block"),
            "equation_tag": eq_tag,
            "context": formula_data.get("context", ""),
            "chapter_title": chapter_title,
            "title": formula_data.get("title") or chapter_title,
            "source_modality": "formula",
            "evidence_source": chapter_title,
            "evidence_span": latex,
            "confidence": 0.86,
        }
        attach_embedding_to_node(formula_node)
        return formula_node

    def _llm_extract_formula_semantics(
        self,
        latex: str,
        context: str,
        chapter_title: str,
        references: str,
        symbol_candidates: List[str],
        symbol_notes: str = "",
    ) -> Dict[str, Any]:
        prompt = f"""
你是石油地质领域的公式知识抽取专家。请基于公式、上下文和引用信息，抽取结构化知识。

章节：{chapter_title}
公式：{latex}
符号候选：{symbol_candidates}
符号注释（全局）：{symbol_notes}
公式上下文：{context}
引用信息：{references}

任务：
1) 识别公式符号，给出语义解释、单位（若未知可为空）、角色类型。
2) 不要抽取符号间计算、约束、依赖、推导、证据或跨模态对齐关系。
3) 只返回公式含义和符号列表；公式到符号的结构关系由程序统一生成。

输出 JSON：
{{
  "formula_meaning": "一句话概括公式用途",
  "symbols": [
    {{
      "symbol": "q_e",
      "name": "临界充注强度",
      "symbol_type": "Variable|Parameter|Constant|Unit",
      "unit": "",
      "description": "",
      "confidence": 0.0
    }}
  ]
}}

只输出 JSON，不要附加解释。
"""
        result = self.call_openai_json(prompt)
        if isinstance(result, dict):
            return result
        return {}

    def _enrich_symbols_with_notes(
        self,
        symbols: List[Dict[str, Any]],
        symbol_notes_map: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(symbols, list):
            return []

        notes_map = symbol_notes_map or {}
        enriched: List[Dict[str, Any]] = []

        for item in symbols:
            if not isinstance(item, dict):
                continue
            symbol = self._normalize_symbol(item.get("symbol", ""))
            if not symbol:
                continue

            note = notes_map.get(symbol, {}) if isinstance(notes_map, dict) else {}
            name = (item.get("name") or "").strip() or (note.get("name") or "")
            unit = (item.get("unit") or "").strip() or (note.get("unit") or "")
            description = (item.get("description") or "").strip() or (note.get("description") or "")
            symbol_type = (item.get("symbol_type") or "").strip() or self._infer_symbol_type(symbol)

            enriched.append(
                {
                    "symbol": symbol,
                    "name": name or symbol,
                    "symbol_type": symbol_type,
                    "unit": unit,
                    "description": description,
                    "confidence": item.get("confidence", 0.82),
                }
            )

        # 去重（按 symbol 保留信息更完整的）
        merged: Dict[str, Dict[str, Any]] = {}
        for item in enriched:
            key = item["symbol"]
            if key not in merged:
                merged[key] = item
                continue

            existing = merged[key]
            for field in ["name", "symbol_type", "unit", "description", "confidence"]:
                if (not existing.get(field)) and item.get(field):
                    existing[field] = item[field]

        merged_list = list(merged.values())

        # 降噪：若存在 base_x 形式，且 base 本身无注释语义，则移除 base
        all_symbols = [item.get("symbol", "") for item in merged_list]
        underscored_prefix = {s.split("_", 1)[0] for s in all_symbols if "_" in s}
        notes_map = symbol_notes_map or {}

        filtered: List[Dict[str, Any]] = []
        for item in merged_list:
            symbol = item.get("symbol", "")
            if "_" not in symbol and symbol in underscored_prefix:
                has_note = symbol in notes_map
                if not has_note:
                    continue
            filtered.append(item)

        return filtered

    def process_formula(
        self,
        formula_data: Dict[str, Any],
        chapter_title: str = "",
        references: str = "",
        symbol_notes: str = "",
        symbol_notes_map: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        latex = (formula_data.get("latex") or "").strip()
        if not latex:
            return {
                "status": "failed",
                "error": "empty_latex",
                "formula_id": formula_data.get("id", ""),
            }

        formula_node = self._build_formula_node(formula_data, chapter_title)
        context = (formula_data.get("context") or "").strip()
        symbol_candidates = self._extract_symbol_candidates(latex)

        llm_result = self._llm_extract_formula_semantics(
            latex=latex,
            context=context,
            chapter_title=chapter_title,
            references=references,
            symbol_candidates=symbol_candidates,
            symbol_notes=symbol_notes,
        )

        formula_meaning = (llm_result.get("formula_meaning") or "").strip()
        if formula_meaning:
            formula_node["formula_meaning"] = formula_meaning

        symbol_items = llm_result.get("symbols", []) if isinstance(llm_result, dict) else []
        symbol_items = self._enrich_symbols_with_notes(symbol_items, symbol_notes_map)

        # 兜底：LLM漏掉但在候选中且存在符号注释的，也补充为符号节点
        if isinstance(symbol_notes_map, dict):
            existing_symbols = {s.get("symbol") for s in symbol_items if isinstance(s, dict)}
            for cand in symbol_candidates:
                normalized = self._normalize_symbol(cand)
                if not normalized or normalized in existing_symbols:
                    continue
                if normalized not in symbol_notes_map:
                    continue
                note = symbol_notes_map[normalized]
                symbol_items.append(
                    {
                        "symbol": normalized,
                        "name": note.get("name") or normalized,
                        "symbol_type": self._infer_symbol_type(normalized),
                        "unit": note.get("unit") or "",
                        "description": note.get("description") or "",
                        "confidence": 0.76,
                    }
                )
        if not isinstance(symbol_items, list):
            symbol_items = []

        # fallback when llm did not return enough symbols
        llm_symbols = set()
        for item in symbol_items:
            if isinstance(item, dict):
                symbol = self._normalize_symbol(item.get("symbol", ""))
                if symbol:
                    llm_symbols.add(symbol)

        candidate_set = {self._normalize_symbol(s) for s in symbol_candidates if self._normalize_symbol(s)}
        if isinstance(symbol_notes_map, dict) and symbol_notes_map:
            underscored_prefix = {s.split("_", 1)[0] for s in candidate_set if "_" in s}
            filtered_candidates: Set[str] = set()
            for s in candidate_set:
                if s in symbol_notes_map:
                    filtered_candidates.add(s)
                    continue
                if s in self.CONSTANT_SYMBOLS:
                    filtered_candidates.add(s)
                    continue
                if len(s) == 1 and s.isupper():
                    filtered_candidates.add(s)
                    continue
                if "_" not in s and s in underscored_prefix:
                    continue
            candidate_set = filtered_candidates

        merged_symbols = sorted(candidate_set | llm_symbols)

        symbol_nodes: List[Dict[str, Any]] = []
        symbol_name_to_id: Dict[str, str] = {}

        symbol_desc_map: Dict[str, Dict[str, Any]] = {}
        for item in symbol_items:
            if not isinstance(item, dict):
                continue
            symbol = self._normalize_symbol(item.get("symbol", ""))
            if not symbol:
                continue
            symbol_desc_map[symbol] = item

        for symbol in merged_symbols:
            item = symbol_desc_map.get(symbol, {})
            symbol_id = str(uuid.uuid4())
            symbol_type = (item.get("symbol_type") or self._infer_symbol_type(symbol)).strip()
            if symbol_type not in {"Variable", "Parameter", "Constant", "Unit"}:
                symbol_type = self._infer_symbol_type(symbol)

            symbol_node = {
                "id": symbol_id,
                "name": symbol,
                "type": NodeLabel.SYMBOL,
                "label": NodeLabel.SYMBOL,
                "label_zh": "公式符号",
                "symbol_type": symbol_type,
                "description": (item.get("description") or "").strip(),
                "full_name": (item.get("name") or "").strip(),
                "unit": (item.get("unit") or "").strip(),
                "source_formula_id": formula_node["id"],
                "source_modality": "formula",
                "evidence_source": chapter_title or formula_node["name"],
                "evidence_span": latex,
                "confidence": item.get("confidence", 0.82),
            }
            attach_embedding_to_node(symbol_node)
            symbol_nodes.append(symbol_node)
            symbol_name_to_id[symbol] = symbol_id

        relations: List[Dict[str, Any]] = []

        # Required formula -> symbol containment/definition edges
        for symbol_node in symbol_nodes:
            relations.append(
                {
                    "source": formula_node["name"],
                    "source_id": formula_node["id"],
                    "target": symbol_node["name"],
                    "target_id": symbol_node["id"],
                    "relation": RelationType.HAS_SYMBOL,
                    "type": RelationType.HAS_SYMBOL,
                    "description": "Formula defines or uses the symbol",
                    "confidence": symbol_node.get("confidence", 0.82),
                    "evidence_source": chapter_title or formula_node["name"],
                    "evidence_span": latex,
                }
            )

        # simple relation dedup
        uniq = set()
        deduped_relations: List[Dict[str, Any]] = []
        for rel in relations:
            key = (
                rel.get("source_id") or rel.get("source"),
                rel.get("target_id") or rel.get("target"),
                rel.get("relation"),
                rel.get("type"),
            )
            if key in uniq:
                continue
            uniq.add(key)
            deduped_relations.append(rel)

        return {
            "status": "success",
            "formula_node": formula_node,
            "symbol_nodes": symbol_nodes,
            "relations": deduped_relations,
            "formula_meaning": formula_node.get("formula_meaning", ""),
            "raw_llm": llm_result,
        }
