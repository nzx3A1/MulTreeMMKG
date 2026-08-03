"""表格嵌入混合图批量运行使用的逐图来源校验。"""
from __future__ import annotations

from typing import Any

from model import Graph

from ...schema_models import ImageExtractionTask


def _validate_provenance(graph: Graph, task: ImageExtractionTask) -> list[str]:
    """中文说明：检查每个实体和关系都精确追溯到当前图片、Chunk 与可见视觉证据。"""

    errors: list[str] = []
    expected: dict[str, Any] = {
        "source_modality": "image",
        "source_image_path": task.image_path,
        "source_image_id": task.image_id,
        "source_chunk_id": task.chunk_id,
        "image_index": task.image_index,
    }
    for kind, records in (("实体", graph.entities), ("关系", graph.relations)):
        for record in records:
            if not str(record.provenance or "").strip():
                errors.append(f"{kind} {record.id} 缺少 provenance")
            for key, expected_value in expected.items():
                if record.metadata.get(key) != expected_value:
                    errors.append(
                        f"{kind} {record.id} 的 {key}={record.metadata.get(key)!r}，"
                        f"应为 {expected_value!r}"
                    )
            if not str(record.metadata.get("visual_evidence") or "").strip():
                errors.append(f"{kind} {record.id} 缺少 visual_evidence")
    return errors

