from __future__ import annotations

import json
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

current_file = Path(__file__).resolve()
explicitkg_root = current_file.parent.parent.parent
if str(explicitkg_root) not in sys.path:
    sys.path.append(str(explicitkg_root))

from config import get_section
from config.top_schema_config import NodeLabel, RelationType
from modalprocessor.embedding_utils import attach_embedding_to_node, attach_embeddings_to_nodes
from modalprocessor.images.ImageModalProcessor import ImageModalProcessor


MAX_WORKERS = int(get_section("APIConfig", {}).get("MAX_CONCURRENT_REQUESTS", 2))
ATTACH_IMAGE_EMBEDDINGS = bool(get_section("VectorAPIConfig", {}).get("ENABLE_IMAGE_EMBEDDINGS", False))


def derive_caption_from_content(content: str, fallback: str = "") -> str:
    text = str(content or "")
    if not text:
        return fallback

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if not line.startswith("图"):
            continue
        if "Fig" in line:
            line = line.split("Fig", 1)[0].strip()
        if re.match(r"^图\s", line):
            fig_no = ""
            for next_line in lines[idx + 1: idx + 3]:
                match = re.search(r"Fig[．.\s]*(\d+)", next_line, flags=re.IGNORECASE)
                if match:
                    fig_no = match.group(1)
                    break
            if fig_no:
                return f"图{fig_no} {line[1:].strip()}"
        return line
    return fallback


def build_image_group_record(
    node: Dict[str, Any],
    current_titles: List[str],
    image_group: Dict[str, Any],
    raw_result: Dict[str, Any],
    references_text: str,
    context_text: str,
) -> Dict[str, Any]:
    extraction = raw_result.get("extraction", {}) if isinstance(raw_result, dict) else {}
    triples = extraction.get("triples", []) if isinstance(extraction, dict) else []
    extracted_entities = extraction.get("entities", []) if isinstance(extraction, dict) else []
    description = raw_result.get("description", {}) if isinstance(raw_result, dict) else {}

    entities_map: Dict[str, Dict[str, Any]] = {}

    def add_entity(name: str, entity_type: str = NodeLabel.FIGURE_NODE, extra: Dict[str, Any] | None = None) -> None:
        key = str(name or "").strip()
        if not key:
            return
        if key not in entities_map:
            entities_map[key] = {
                "id": str(uuid.uuid4()),
                "name": key,
                "type": entity_type,
                "label_zh": "图像节点",
                "source_modality": "figure",
                "evidence_source": image_group.get("caption", "") or node.get("title", ""),
                "confidence": 0.78,
            }
        if isinstance(extra, dict):
            for k, v in extra.items():
                if v not in (None, "", [], {}):
                    entities_map[key][k] = v

    relations: List[Dict[str, Any]] = []

    if isinstance(extracted_entities, list):
        for entity in extracted_entities:
            if not isinstance(entity, dict):
                continue
            entity_name = str(entity.get("name", "")).strip()
            if not entity_name:
                continue
            add_entity(entity_name, NodeLabel.FIGURE_NODE, {
                "properties": entity.get("attributes") if isinstance(entity.get("attributes"), dict) else {},
                "visual_type": entity.get("type", ""),
                "evidence_span": entity.get("evidence_span", ""),
                "confidence": entity.get("confidence", 0.78),
                "source_modality": entity.get("source_modality", "figure"),
            })

    if isinstance(triples, list):
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            head = str(triple.get("head") or triple.get("source") or "").strip()
            tail = str(triple.get("tail") or triple.get("target") or "").strip()
            relation = str(triple.get("relation") or "").strip()
            relation_type = str(triple.get("relation_type") or triple.get("type") or "related").strip() or "related"
            if not head or not tail:
                continue

            add_entity(head)
            add_entity(tail)
            relation_obj = {
                "source": head,
                "relation": relation or relation_type,
                "target": tail,
                "type": relation_type,
                "source_id": entities_map.get(head, {}).get("id"),
                "target_id": entities_map.get(tail, {}).get("id"),
                "confidence": triple.get("confidence", 0.78),
                "evidence_span": triple.get("evidence_span", ""),
                "evidence_source": image_group.get("caption", "") or node.get("title", ""),
                "source_modality": triple.get("source_modality", "figure"),
            }
            if triple.get("verification"):
                relation_obj["verification"] = triple.get("verification")
            relations.append(relation_obj)

    visual_entities = description.get("visual_entities", []) if isinstance(description, dict) else []
    if isinstance(visual_entities, list):
        for item in visual_entities:
            if isinstance(item, dict):
                entity_name = str(item.get("name", "")).strip()
                add_entity(entity_name, NodeLabel.FIGURE_NODE, {
                    "properties": item.get("attributes") if isinstance(item.get("attributes"), dict) else {},
                    "visual_type": item.get("type", ""),
                    "evidence_span": item.get("evidence_span", ""),
                    "confidence": item.get("confidence", 0.78),
                })
            else:
                add_entity(str(item).strip())

    chunk_id = str(uuid.uuid4())
    chunk_name = (
        image_group.get("caption", "")
        or node.get("title", "")
        or f"image_chunk_{chunk_id[:8]}"
    )
    chunk = {
        "id": chunk_id,
        "name": chunk_name,
        "type": NodeLabel.FIGURE,
        "label_zh": "图片",
        "source_chapter_id": node.get("id"),
        "source_chapter_title": node.get("title"),
        "chapter_path": current_titles,
        "image_group_caption": image_group.get("caption", ""),
        "image_paths": raw_result.get("image_paths", image_group.get("path", [])),
        "references": image_group.get("references", []),
        "references_text": references_text,
        "context": context_text,
        "classification": raw_result.get("classification", {}).get("type", ""),
        "reasoning": raw_result.get("classification", {}).get("reasoning", ""),
        "description": raw_result.get("description", {}).get("detailed_description", ""),
        "evidence_chain": extraction.get("evidence_chain", []),
        "source_modality": "figure",
        "evidence_source": node.get("title") or chunk_name,
        "confidence": 0.8,
    }
    if ATTACH_IMAGE_EMBEDDINGS:
        attach_embedding_to_node(chunk)

    entities = list(entities_map.values())
    if ATTACH_IMAGE_EMBEDDINGS:
        attach_embeddings_to_nodes(entities)

    for entity in entities:
        relations.append({
            "source": entity.get("name", ""),
            "source_id": entity.get("id", ""),
            "relation": RelationType.HAS_FIGURE_NODE,
            "target": chunk_name,
            "target_id": chunk_id,
            "type": RelationType.HAS_FIGURE_NODE,
            "description": f"实体 '{entity.get('name', '')}' 来源于 chunk '{chunk_name}'",
            "confidence": entity.get("confidence", 0.78),
            "evidence_source": chunk.get("evidence_source", chunk_name),
            "source_modality": "figure",
        })

    return {
        "chunk": chunk,
        "entities": entities,
        "relations": relations,
    }


def collect_image_tasks(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_tasks: List[Dict[str, Any]] = []

    def process_toc_nodes(nodes: List[Dict[str, Any]], parent_titles: List[str] | None = None) -> None:
        if parent_titles is None:
            parent_titles = []
        for node in nodes:
            current_titles = parent_titles + [node.get("title", "")]
            images = node.get("images", [])
            if isinstance(images, list):
                for image_group in images:
                    if not isinstance(image_group, dict):
                        continue
                    paths = image_group.get("path", [])
                    if isinstance(paths, str):
                        paths = [paths]
                    if not paths:
                        continue
                    refs = image_group.get("references", [])
                    if isinstance(refs, str):
                        refs = [refs]
                    node_content = str(node.get("content", "") or "")
                    references_text = "\n".join(str(item) for item in refs if item)
                    if not references_text and node_content:
                        references_text = node_content[:1600]
                    caption = str(image_group.get("caption", "") or "").strip()
                    if not caption:
                        caption = derive_caption_from_content(node_content, node.get("title", ""))
                    image_group_for_task = dict(image_group)
                    image_group_for_task["caption"] = caption
                    context_text = (
                        f"Chapter: {' > '.join(current_titles)}\n"
                        f"Section content:\n{node_content[:2400]}\n"
                        f"Context references:\n{references_text}"
                    )
                    image_tasks.append({
                        "node": node,
                        "current_titles": current_titles,
                        "image_group": image_group_for_task,
                        "paths": paths,
                        "caption": caption,
                        "references_text": references_text,
                        "context_text": context_text,
                    })
            children = node.get("children", [])
            if isinstance(children, list):
                process_toc_nodes(children, current_titles)

    if isinstance(data.get("toc"), list):
        process_toc_nodes(data["toc"])
    return image_tasks


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent.parent
    input_file = base_dir / "output" / "toc_structure.json"
    output_file = base_dir / "output" / "modal" / "image_extraction_results.json"

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        processor = ImageModalProcessor()
        print("ImageModalProcessor initialized successfully.")
    except Exception as exc:
        print(f"Failed to initialize processor: {exc}")
        return

    image_tasks = collect_image_tasks(data)

    def process_one_image_group(task: Dict[str, Any]) -> Dict[str, Any]:
        caption = task["caption"] or task["node"].get("title", "")
        print(f"Processing image group: {caption[:50]}... ({len(task['paths'])} images)")
        result = processor.process_image(
            image_paths=task["paths"],
            caption=caption,
            references=task["references_text"],
            context=task["context_text"],
        )
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result["error"])
        return build_image_group_record(
            node=task["node"],
            current_titles=task["current_titles"],
            image_group=task["image_group"],
            raw_result=result,
            references_text=task["references_text"],
            context_text=task["context_text"],
        )

    all_image_results: List[Dict[str, Any]] = []
    print(f"Processing {len(image_tasks)} image groups with {MAX_WORKERS} worker threads.")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_one_image_group, task): idx
            for idx, task in enumerate(image_tasks)
        }
        indexed_results: Dict[int, Dict[str, Any]] = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                indexed_results[idx] = future.result()
                print(f"Successfully processed image group #{idx + 1}.")
            except Exception as exc:
                task = image_tasks[idx]
                print(f"Error processing image group {str(task.get('caption', ''))[:30]}: {exc}")
        for idx in sorted(indexed_results):
            all_image_results.append(indexed_results[idx])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_image_results, f, ensure_ascii=False, indent=2)

    print(f"Processing complete. Processed {len(all_image_results)} image groups.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
