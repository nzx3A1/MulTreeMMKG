import json
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from modalprocessor.formula.FormulaModalProcessor import FormulaModalProcessor
from modalprocessor.embedding_utils import attach_embedding_to_node, attach_embeddings_to_nodes
from config import get_section
from config.top_schema_config import NodeLabel, RelationType


MAX_WORKERS = int(get_section("APIConfig", {}).get("MAX_CONCURRENT_REQUESTS", 4))
STRUCTURAL_FORMULA_RELATIONS = {RelationType.HAS_SYMBOL, "DEFINES"}


def _normalize_formula_latex(formula_item: dict) -> str:
    if not isinstance(formula_item, dict):
        return ""

    latex = (formula_item.get("latex") or "").strip()
    if latex:
        return latex

    raw = (formula_item.get("raw") or "").strip()
    if raw:
        text = raw
        if text.startswith("$$"):
            text = text[2:]
        if text.endswith("$$"):
            text = text[:-2]
        return text.strip()

    content = (formula_item.get("content") or "").strip()
    return content


def _parse_symbol_notes_map(symbol_notes: str) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    if not symbol_notes:
        return result

    normalized = symbol_notes.replace("\n", "；")
    segments = [seg.strip() for seg in re.split(r"[；;]", normalized) if seg.strip()]

    for seg in segments:
        match = re.match(r"^\s*(.+?)\s*[—–－-]{1,2}\s*(.+)$", seg)
        if not match:
            continue

        raw_symbol = match.group(1).strip()
        raw_desc = match.group(2).strip()
        if not raw_symbol or not raw_desc:
            continue

        symbol = raw_symbol
        symbol = re.sub(r"\$", "", symbol)
        symbol = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", symbol)
        symbol = re.sub(r"_\{([^{}]+)\}", r"_\1", symbol)
        symbol = symbol.replace("{", "").replace("}", "")
        symbol = symbol.replace(" ", "")

        if not symbol:
            continue

        unit_match = re.search(r"[，,]\s*([^，,。]+)$", raw_desc)
        unit = ""
        if unit_match:
            maybe_unit = unit_match.group(1).strip()
            if len(maybe_unit) <= 30:
                unit = maybe_unit

        result[symbol] = {
            "name": raw_desc,
            "description": raw_desc,
            "unit": unit,
        }

    return result


def _collect_node_references(node: dict) -> str:
    refs: List[str] = []

    formulas = node.get("formulas", [])
    if isinstance(formulas, list):
        for f in formulas:
            if isinstance(f, dict):
                context = (f.get("context") or "").strip()
                if context:
                    refs.append(context)

    images = node.get("images", [])
    if isinstance(images, list):
        for img in images:
            if not isinstance(img, dict):
                continue
            caption = (img.get("caption") or "").strip()
            if caption:
                refs.append(caption)
            image_refs = img.get("references", [])
            if isinstance(image_refs, list):
                refs.extend([str(r).strip() for r in image_refs if str(r).strip()])

    tables = node.get("table", [])
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue
            caption = (table.get("caption") or "").strip()
            content = (table.get("content") or "").strip()
            context = (table.get("context") or "").strip()
            if caption:
                refs.append(f"表格标题：{caption}")
            if content:
                refs.append(f"表格内容：{content[:1200]}")
            if context:
                refs.append(f"表格上下文：{context[:600]}")

    # remove duplicates, keep order
    seen = set()
    uniq = []
    for r in refs:
        if r in seen:
            continue
        seen.add(r)
        uniq.append(r)

    return "\n".join(uniq[:20])


def build_formula_record(raw_result: dict, metadata: dict) -> dict:
    """
    Normalize formula extraction output to:
    {
      "chunk": {...},
      "entities": [...],
      "relations": [...]
    }
    """
    if not isinstance(raw_result, dict):
        raw_result = {}
    if not isinstance(metadata, dict):
        metadata = {}

    formula_node = raw_result.get("formula_node", {})
    symbol_nodes = raw_result.get("symbol_nodes", [])
    raw_relations = raw_result.get("relations", [])

    # Keep current formula node's base info and original extraction payload.
    chunk_id = str(uuid.uuid4())
    chunk_name = (metadata.get("latex") or "").strip()
    if not chunk_name:
        chunk_name = metadata.get("formula_id") or metadata.get("source_chapter_title") or f"formula_chunk_{chunk_id[:8]}"
    formula_context = ""
    if isinstance(formula_node, dict):
        formula_context = (formula_node.get("context") or "").strip()
    chunk = {
        "id": chunk_id,
        "name": str(chunk_name),
        "type": NodeLabel.FORMULA,
        "label_zh": "公式",
        "status": raw_result.get("status", metadata.get("status", "success")),
        "error": raw_result.get("error", ""),
        "source_chapter_id": metadata.get("source_chapter_id"),
        "source_chapter_title": metadata.get("source_chapter_title"),
        "source_chapter_path": metadata.get("source_chapter_path", []),
        "formula_index_in_node": metadata.get("formula_index_in_node"),
        "formula_id": metadata.get("formula_id"),
        "latex": metadata.get("latex"),
        "display_type": metadata.get("display_type", "block"),
        "references": metadata.get("references", ""),
        "context": formula_context,
        "formula_meaning": (formula_node or {}).get("formula_meaning", "") if isinstance(formula_node, dict) else "",
        "source_modality": "formula",
        "evidence_source": metadata.get("source_chapter_title") or str(chunk_name),
        "evidence_span": metadata.get("latex") or formula_context,
        "confidence": 0.86,
    }
    attach_embedding_to_node(chunk)

    entities_map = {}

    def add_entity(entity: dict):
        if not isinstance(entity, dict):
            return
        key = str(entity.get("id") or entity.get("name") or "").strip()
        if not key:
            return
        existing = entities_map.get(key)
        if not existing:
            entities_map[key] = dict(entity)
            return
        # Merge missing fields only.
        for k, v in entity.items():
            if (existing.get(k) is None or existing.get(k) == "") and v not in (None, ""):
                existing[k] = v

    if isinstance(symbol_nodes, list):
        for symbol_node in symbol_nodes:
            add_entity(symbol_node)

    def ensure_uuid(entity: dict) -> str:
        raw_id = str(entity.get("id", "")).strip()
        try:
            if raw_id:
                return str(uuid.UUID(raw_id))
        except Exception:
            pass
        return str(uuid.uuid4())

    entities = list(entities_map.values())
    old_id_to_new = {}
    name_to_new = {}
    for entity in entities:
        old_id = str(entity.get("id", "")).strip()
        new_id = ensure_uuid(entity)
        entity["id"] = new_id
        if old_id:
            old_id_to_new[old_id] = new_id
        entity_name = str(entity.get("name", "")).strip()
        if entity_name:
            name_to_new[entity_name] = new_id
    attach_embeddings_to_nodes(entities)

    entity_ids = {str(e.get("id", "")).strip() for e in entities if str(e.get("id", "")).strip()}
    entity_names = {str(e.get("name", "")).strip() for e in entities if str(e.get("name", "")).strip()}

    formula_raw_id = ""
    formula_raw_name = ""
    if isinstance(formula_node, dict):
        formula_raw_id = str(formula_node.get("id") or "").strip()
        formula_raw_name = str(formula_node.get("name") or "").strip()

    def remap_formula_endpoint(name: str, endpoint_id: str) -> tuple[str, str, bool]:
        is_formula = (
            (formula_raw_id and endpoint_id == formula_raw_id)
            or (formula_raw_name and name == formula_raw_name)
        )
        if is_formula:
            return chunk["name"], chunk_id, True
        return name, endpoint_id, False

    relations = []
    if isinstance(raw_relations, list):
        for rel in raw_relations:
            if not isinstance(rel, dict):
                continue
            source = (rel.get("source") or "").strip()
            target = (rel.get("target") or "").strip()
            rel_type = (rel.get("type") or rel.get("relation") or "related").strip() or "related"
            if rel_type.upper() not in STRUCTURAL_FORMULA_RELATIONS:
                continue
            rel_type = RelationType.HAS_SYMBOL
            src_old = str(rel.get("source_id") or "").strip()
            tgt_old = str(rel.get("target_id") or "").strip()
            source_id = old_id_to_new.get(src_old) or name_to_new.get(source) or src_old
            target_id = old_id_to_new.get(tgt_old) or name_to_new.get(target) or tgt_old

            source, source_id, source_is_formula = remap_formula_endpoint(source, source_id)
            target, target_id, target_is_formula = remap_formula_endpoint(target, target_id)

            relation_obj = {
                "source": source,
                "source_id": source_id,
                "relation": RelationType.HAS_SYMBOL,
                "target": target,
                "target_id": target_id,
                "type": rel_type,
                "confidence": rel.get("confidence", 0.8),
                "evidence_source": chunk.get("evidence_source", chunk["name"]),
                "evidence_span": rel.get("evidence_span", ""),
            }
            if rel.get("description"):
                relation_obj["description"] = rel.get("description")
            source_in_entities = (
                (source in entity_names) or
                (relation_obj.get("source_id") in entity_ids) or
                source_is_formula
            )
            target_in_entities = (
                (target in entity_names) or
                (relation_obj.get("target_id") in entity_ids) or
                target_is_formula
            )
            if source and target and source_in_entities and target_in_entities:
                relations.append(relation_obj)

    source_relations = []
    for entity in entities:
        source_relations.append({
            "source": entity.get("name", ""),
            "source_id": entity.get("id", ""),
            "relation": RelationType.SOURCE,
            "target": chunk["name"],
            "target_id": chunk_id,
            "type": RelationType.SOURCE,
            "description": f"实体 '{entity.get('name', '')}' 来源于 chunk '{chunk['name']}'",
            "confidence": entity.get("confidence", 0.82),
            "evidence_source": chunk.get("evidence_source", chunk["name"]),
            "source_modality": "formula",
        })

    relations.extend(source_relations)

    return {
        "chunk": chunk,
        "entities": entities,
        "relations": relations,
    }


def main():
    base_dir = project_root
    input_file = base_dir / "output" / "toc_structure.json"
    output_file = base_dir / "output" / "modal" / "formula_extraction_results.json"

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    print(f"Loading TOC from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    basic_info = data.get("basicInformation", {}) if isinstance(data, dict) else {}
    symbol_notes = (basic_info.get("symbol_notes") or "").strip() if isinstance(basic_info, dict) else ""
    symbol_notes_map = _parse_symbol_notes_map(symbol_notes)

    try:
        processor = FormulaModalProcessor()
        print("FormulaModalProcessor initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize formula processor: {e}")
        return

    formula_tasks = []

    def process_toc_nodes(nodes: List[dict], parent_titles=None):
        if parent_titles is None:
            parent_titles = []

        for node in nodes:
            current_titles = parent_titles + [node.get("title", "")]
            chapter_title = node.get("title", "")
            chapter_id = node.get("id")

            node_references = _collect_node_references(node)
            formulas = node.get("formulas", [])

            if isinstance(formulas, list):
                for index, formula in enumerate(formulas, start=1):
                    if not isinstance(formula, dict):
                        continue
                    latex = (formula.get("latex") or "").strip()
                    if not latex:
                        latex = _normalize_formula_latex(formula)
                    if not latex:
                        continue

                    formula_tasks.append({
                        "formula_data": {
                            "id": formula.get("id") or f"{chapter_id}-{index}",
                            "latex": latex,
                            "display_type": formula.get("display_type", "block"),
                            "context": formula.get("context") or node.get("content", ""),
                        },
                        "chapter_title": chapter_title,
                        "references": node_references,
                        "metadata": {
                            "source_chapter_id": chapter_id,
                            "source_chapter_title": chapter_title,
                            "source_chapter_path": [t for t in current_titles if t],
                            "formula_index_in_node": index,
                            "formula_id": formula.get("id") or f"{chapter_id}-{index}",
                            "latex": latex,
                            "display_type": formula.get("display_type", "block"),
                            "references": node_references,
                        },
                    })

            children = node.get("children", [])
            if isinstance(children, list) and children:
                process_toc_nodes(children, current_titles)

    toc_nodes = data.get("toc", [])
    if isinstance(toc_nodes, list):
        process_toc_nodes(toc_nodes)

    def process_one_formula(task: dict) -> dict:
        metadata = dict(task["metadata"])
        print(f"Processing formula #{metadata['formula_index_in_node']} in chapter {task['chapter_title']}")
        try:
            result = processor.process_formula(
                formula_data=task["formula_data"],
                chapter_title=task["chapter_title"],
                references=task["references"],
                symbol_notes=symbol_notes,
                symbol_notes_map=symbol_notes_map,
            )
            metadata["status"] = result.get("status", "success")
            normalized_result = build_formula_record(result, metadata)
            print(f"  Done. status={result.get('status')}")
            return normalized_result
        except Exception as e:
            metadata["status"] = "failed"
            err_item = {
                "status": "failed",
                "error": str(e),
            }
            print(f"  Error processing formula: {e}")
            return build_formula_record(err_item, metadata)

    all_formula_results = []
    print(f"Processing {len(formula_tasks)} formulas with {MAX_WORKERS} worker threads.")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_one_formula, task): task for task in formula_tasks}
        for future in as_completed(futures):
            all_formula_results.append(future.result())

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_formula_results, f, ensure_ascii=False, indent=2)

    success_count = len([r for r in all_formula_results if r.get("chunk", {}).get("status") == "success"])
    failed_count = len([r for r in all_formula_results if r.get("chunk", {}).get("status") != "success"])
    print(f"Formula extraction finished: total={len(all_formula_results)}, success={success_count}, failed={failed_count}")
    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()
