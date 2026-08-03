"""二维平面地层—测井图批处理的目标筛选、任务构建和质量校验。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from model import Graph
from src.utils.json_io import read_json

from ...schema_models import ImageExtractionTask, as_string_tuple
from ..subclassifier import (
    StratigraphicProfileSubtype,
    StratigraphicSubtypeClassification,
)


TARGET_SUBTYPE = StratigraphicProfileSubtype.TWO_DIMENSIONAL_STRATIGRAPHIC_LOG


def load_two_dimensional_target_chunks(source_path: Path) -> tuple[int, list[dict[str, Any]]]:
    """中文说明：从用户指定根数组中只取二维子类型 Chunk，并校验唯一 ID、唯一图片和图片存在性。"""

    payload = read_json(source_path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"二维抽取源文件必须是非空根数组：{source_path}")
    targets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for manifest_index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise TypeError(f"第 {manifest_index} 条 Chunk 不是 JSON 对象")
        subtype = raw.get("stratigraphic_subtype")
        if not isinstance(subtype, Mapping):
            raise ValueError(f"第 {manifest_index} 条 Chunk 缺少 stratigraphic_subtype")
        if str(subtype.get("subtype") or "") != TARGET_SUBTYPE.value:
            continue
        chunk_id = str(raw.get("id") or raw.get("chunk_id") or "").strip()
        image_paths = as_string_tuple(raw.get("image_path"))
        if not chunk_id or len(image_paths) != 1:
            raise ValueError(f"目标 Chunk {manifest_index} 必须具有 id 和唯一 image_path")
        image_path = str(Path(image_paths[0]).resolve())
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"目标图片不存在：{image_path}")
        if chunk_id in seen_ids or image_path.casefold() in seen_paths:
            raise ValueError(f"二维目标清单出现重复 Chunk 或图片：{chunk_id}")
        seen_ids.add(chunk_id)
        seen_paths.add(image_path.casefold())
        chunk = dict(raw)
        chunk["image_path"] = [image_path]
        chunk["manifest_index"] = manifest_index
        chunk["target_index"] = len(targets)
        targets.append(chunk)
    if not targets:
        raise ValueError(f"源文件中没有 {TARGET_SUBTYPE.value} 目标 Chunk")
    return len(payload), targets


def build_two_dimensional_task(chunk: Mapping[str, Any]) -> ImageExtractionTask:
    """中文说明：把单图片二维目标 Chunk 转换为统一图片抽取任务。"""

    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
    image_paths = as_string_tuple(chunk.get("image_path"))
    if not chunk_id or len(image_paths) != 1:
        raise ValueError("二维目标 Chunk 必须具有 id 和唯一 image_path")
    return ImageExtractionTask(
        document_id=str(chunk.get("document_id") or chunk_id.split(":section:", 1)[0]),
        chunk_id=chunk_id,
        image_id=f"{chunk_id}:image:0",
        image_index=0,
        image_path=image_paths[0],
        caption=str(chunk.get("caption") or ""),
        references=as_string_tuple(chunk.get("references")),
        classification_code=str(chunk.get("classification_code") or "A03"),
        classification_type=str(chunk.get("classification_type") or "地层—剖面—测井"),
        section_id=str(chunk.get("section_id") or ""),
        section_title=str(chunk.get("section_title") or ""),
        raw_chunk=chunk,
    )


def classification_from_chunk(chunk: Mapping[str, Any]) -> StratigraphicSubtypeClassification:
    """中文说明：复用源 JSON 的人工子分类及证据，不再额外调用模型重复分类。"""

    raw = chunk.get("stratigraphic_subtype")
    if not isinstance(raw, Mapping):
        raise ValueError("二维目标 Chunk 缺少 stratigraphic_subtype")
    subtype = StratigraphicProfileSubtype(str(raw.get("subtype") or ""))
    if subtype is not TARGET_SUBTYPE:
        raise ValueError(f"Chunk 子类型不是二维目标：{subtype.value}")
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = str(raw.get("evidence") or "").strip()
    if not evidence:
        raise ValueError("二维目标 Chunk 的子分类缺少视觉证据")
    return StratigraphicSubtypeClassification(
        subtype=subtype,
        confidence=max(0.0, min(1.0, confidence)),
        evidence=evidence,
        source=str(raw.get("source") or "manual_visual_mock"),
    )


def validate_two_dimensional_graph(graph: Graph, task: ImageExtractionTask) -> dict[str, Any]:
    """中文说明：逐实体和关系检查来源，并确认存在上下位置关系且事件保持为空。"""

    provenance_errors: list[str] = []
    expected = {
        "source_modality": "image",
        "source_image_path": task.image_path,
        "source_image_id": task.image_id,
        "source_chunk_id": task.chunk_id,
        "image_index": task.image_index,
    }
    for kind, records in (("实体", graph.entities), ("关系", graph.relations)):
        for record in records:
            if not str(record.provenance or "").strip():
                provenance_errors.append(f"{kind} {record.id} 缺少 provenance")
            for key, value in expected.items():
                if record.metadata.get(key) != value:
                    provenance_errors.append(
                        f"{kind} {record.id} 的 {key}={record.metadata.get(key)!r}，应为 {value!r}"
                    )
            if not str(record.metadata.get("visual_evidence") or "").strip():
                provenance_errors.append(f"{kind} {record.id} 缺少 visual_evidence")

    relation_types = [relation.type for relation in graph.relations]
    vertical_count = sum(
        relation_type in {"directly_overlies", "directly_underlies"}
        for relation_type in relation_types
    )
    forbidden_position_relations = [
        relation_type
        for relation_type in relation_types
        if relation_type in {"above", "below", "left_of", "right_of", "adjacent_to"}
    ]
    semantic_position_count = sum(
        relation_type
        in {
            "intersects",
            "cuts_through",
            "offsets",
            "correlates_with",
            "tracks_along",
            "located_in",
            "bounded_by",
        }
        for relation_type in relation_types
    )
    stratigraphic_entity_count = sum(
        entity.type
        in {
            "stratigraphic_unit",
            "horizon",
            "reservoir_interval",
            "shale_interval",
            "seal",
            "source_rock",
        }
        for entity in graph.entities
    )
    quality_errors: list[str] = []
    quality_warnings: list[str] = []
    if len(graph.entities) <= 1:
        quality_errors.append("除剖面根节点外没有抽取实体")
    if vertical_count == 0 and stratigraphic_entity_count >= 2:
        quality_errors.append("没有抽取或推导地层上下位置关系")
    elif vertical_count == 0 and semantic_position_count == 0:
        quality_errors.append("单层位图片没有可验收的轨迹、交切或定位关系")
    elif vertical_count == 0:
        # 中文说明：单层位地震剖面不能虚构第二个命名层，仅以轨迹—层位等可见空间边验收。
        quality_warnings.append("图片仅有一个可辨认地层实体，未强制生成上下层关系")
    if forbidden_position_relations:
        quality_errors.append(
            f"仍包含禁止的位置关系：{sorted(set(forbidden_position_relations))}"
        )
    root_ids = {
        entity.id
        for entity in graph.entities
        if entity.type == "two_dimensional_stratigraphic_log"
    }
    domain_degree = {
        entity.id: 0 for entity in graph.entities if entity.id not in root_ids
    }
    for relation in graph.relations:
        if relation.source_id in root_ids or relation.target_id in root_ids:
            continue
        if relation.source_id in domain_degree:
            domain_degree[relation.source_id] += 1
        if relation.target_id in domain_degree:
            domain_degree[relation.target_id] += 1
    isolated_entity_ids = [
        entity_id for entity_id, degree in domain_degree.items() if degree == 0
    ]
    if isolated_entity_ids:
        quality_errors.append(f"存在无业务关系的非 Chunk 节点：{isolated_entity_ids}")
    if len(graph.events) != 0:
        quality_errors.append("二维实体关系抽取不应生成事件")
    return {
        "reference_errors": graph.validate_references(),
        "provenance_errors": provenance_errors,
        "quality_errors": quality_errors,
        "quality_warnings": quality_warnings,
        "event_count": len(graph.events),
        "stratigraphic_entity_count": stratigraphic_entity_count,
        "vertical_relation_count": vertical_count,
        "horizontal_relation_count": 0,
        "semantic_position_relation_count": semantic_position_count,
        "forbidden_position_relations": forbidden_position_relations,
        "isolated_entity_ids": isolated_entity_ids,
    }
