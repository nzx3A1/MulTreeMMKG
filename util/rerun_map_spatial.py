"""仅重新抽取 stage_04 结果中的地图空间图片并原位更新结果文件。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "output" / "stage_03_document_summary.json"
OUTPUT_PATH = PROJECT_ROOT / "output" / "stage_04_image_extraction.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _enable_pydantic_v1_compatibility() -> None:
    """为当前 Pydantic v1 环境补充仓库使用的 v2 前向引用接口。"""

    from pydantic import BaseModel

    if not hasattr(BaseModel, "model_rebuild"):
        BaseModel.model_rebuild = classmethod(  # type: ignore[attr-defined, method-assign]
            lambda cls, *args, **kwargs: cls.update_forward_refs()
        )


def _map_classification_records(payload: Mapping[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    """从旧结果提取地图任务的稳定分类记录与父 Chunk 标识。"""

    target_chunk_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    for graph in payload.get("graphs", []):
        if not isinstance(graph, Mapping):
            continue
        metadata = graph.get("metadata")
        extra = metadata.get("extra") if isinstance(metadata, Mapping) else None
        chunk_id = str(metadata.get("chunk_id") or "") if isinstance(metadata, Mapping) else ""
        routes = extra.get("routes") if isinstance(extra, Mapping) else None
        if not isinstance(routes, list):
            continue
        map_routes = [route for route in routes if isinstance(route, Mapping) and route.get("extractor_kind") == "map_spatial"]
        if not chunk_id or not map_routes:
            continue
        target_chunk_ids.add(chunk_id)
        for route in map_routes:
            records.append(
                {
                    "parent_chunk_id": chunk_id,
                    "image_path": route.get("image_path"),
                    "image_index": int(route.get("image_index") or str(route.get("image_id") or "").rsplit(":", 1)[-1]),
                    "primary_code": route.get("classification_code"),
                    "primary_type": route.get("classification_type"),
                }
            )
    return target_chunk_ids, records


def _replace_graphs(payload: dict[str, Any], refreshed_graphs: list[Any]) -> None:
    """按 Chunk 标识替换地图 Graph，并重新统计整个 stage_04 文件。"""

    refreshed_by_chunk = {
        str(graph.metadata.chunk_id): graph.to_dict(exclude={"metadata": {"raw_response"}})
        for graph in refreshed_graphs
        if graph.metadata.chunk_id
    }
    graphs = [
        refreshed_by_chunk.get(str((raw.get("metadata") or {}).get("chunk_id") or ""), raw)
        if isinstance(raw, Mapping)
        else raw
        for raw in payload.get("graphs", [])
    ]
    payload["graphs"] = graphs
    payload["_status"] = "completed"
    payload["statistics"] = {
        "graph_count": len(graphs),
        "completed_image_chunk_count": len(graphs),
        "entity_count": sum(len(graph.get("entities") or []) for graph in graphs if isinstance(graph, Mapping)),
        "relation_count": sum(len(graph.get("relations") or []) for graph in graphs if isinstance(graph, Mapping)),
        "event_count": sum(len(graph.get("events") or []) for graph in graphs if isinstance(graph, Mapping)),
    }


def _build_cli() -> argparse.ArgumentParser:
    """构建地图重抽取工具的命令行参数。"""

    parser = argparse.ArgumentParser(description="仅重抽 stage_04 中的地图空间图片")
    parser.add_argument("--first-only", action="store_true", help="只重抽结果文件中的第一张地图，用于小范围验证")
    return parser


def main() -> int:
    """执行地图图片重抽取，并将成功或失败状态写回既有 stage_04 文件。"""

    args = _build_cli().parse_args()
    _enable_pydantic_v1_compatibility()
    from src.extractors.extractor_init import collect_chunks
    from src.extractors.image_extractor.classification import RecordImageClassificationProvider
    from src.extractors.image_extractor.pipeline import extract_from_images
    from src.utils.json_io import read_json, write_json
    from src.utils.vlm_client import VLMClient

    existing = read_json(OUTPUT_PATH)
    if not isinstance(existing, dict):
        raise TypeError(f"结果文件必须是 JSON 对象：{OUTPUT_PATH}")
    target_chunk_ids, records = _map_classification_records(existing)
    if args.first_only:
        records = records[:1]
        target_chunk_ids = {str(records[0]["parent_chunk_id"])} if records else set()
    source = read_json(INPUT_PATH)
    if not isinstance(source, Mapping):
        raise TypeError(f"输入文件必须是 JSON 对象：{INPUT_PATH}")
    chunks = [
        chunk for chunk in collect_chunks(source)
        if str(chunk.get("id") or chunk.get("chunk_id") or "") in target_chunk_ids
    ]
    if len(chunks) != len(target_chunk_ids):
        found_ids = {str(chunk.get("id") or chunk.get("chunk_id") or "") for chunk in chunks}
        raise RuntimeError(f"地图 Chunk 与第三阶段输入不一致：缺少={sorted(target_chunk_ids - found_ids)}")

    refreshed = extract_from_images(
        chunks,
        vlm_client=VLMClient(),
        classification_provider=RecordImageClassificationProvider(records),
        context_options={"allowed_extractor_kinds": ["map_spatial"]},
        show_progress=True,
    )
    _replace_graphs(existing, refreshed)
    write_json(OUTPUT_PATH, existing)
    print(
        f"地图空间重抽取完成：Chunk={len(refreshed)}，"
        f"实体={sum(len(graph.entities) for graph in refreshed)}，"
        f"关系={sum(len(graph.relations) for graph in refreshed)}，输出={OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
