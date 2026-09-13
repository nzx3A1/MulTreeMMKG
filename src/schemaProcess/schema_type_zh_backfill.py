"""回填已生成 Stage 05 主图中实体 ``schema_type_zh`` 的命令行工具。"""
from __future__ import annotations

import argparse
import os
import uuid
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.schemaProcess.mapping_filter import resolve_schema_type, resolve_schema_type_zh
from src.schemaProcess.pipeline import DEFAULT_ALIGNED_OUTPUT_PATH, PROJECT_ROOT
from src.utils.json_io import read_json, write_json


DEFAULT_REPORT_PATH = PROJECT_ROOT / "output" / "stage_05_schema_type_zh_backfill_report.json"


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    """校验 JSON 字段为数组，避免静默跳过格式异常的主图。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} 必须是 JSON 数组")
    return value


def backfill_schema_type_fields(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """回填空 Schema 类型字段，保留主图其余结构、实体和关系不变。"""

    result = deepcopy(dict(payload))
    graphs = _require_sequence(result.get("graphs"), "stage_05.graphs")
    raw_type_counts: Counter[str] = Counter()
    updated_examples: list[dict[str, str]] = []
    entity_count = 0
    schema_type_updated_count = 0
    updated_count = 0

    for graph_index, graph in enumerate(graphs):
        if not isinstance(graph, Mapping):
            raise TypeError(f"graphs[{graph_index}] 必须是 JSON 对象")
        entities = _require_sequence(graph.get("entities", []), f"graphs[{graph_index}].entities")
        for entity_index, entity in enumerate(entities):
            if not isinstance(entity, Mapping):
                raise TypeError(f"graphs[{graph_index}].entities[{entity_index}] 必须是 JSON 对象")
            entity_count += 1
            current_schema_type = str(entity.get("schema_type") or "").strip()
            current_schema_type_zh = str(entity.get("schema_type_zh") or "").strip()
            if current_schema_type and current_schema_type_zh:
                continue

            raw_type = entity.get("raw_type", entity.get("type"))
            updated_entity = dict(entity)
            if not current_schema_type:
                updated_entity["schema_type"] = resolve_schema_type(
                    entity.get("type"),
                    entity.get("schema_type"),
                )
                schema_type_updated_count += 1
            if not current_schema_type_zh:
                updated_entity["schema_type_zh"] = resolve_schema_type_zh(
                    raw_type,
                    entity.get("type_zh"),
                    entity.get("schema_type_zh"),
                )
                updated_count += 1
            graph["entities"][entity_index] = updated_entity
            raw_type_counts[str(raw_type or "<empty>")] += 1
            if len(updated_examples) < 20:
                updated_examples.append(
                    {
                        "id": str(entity.get("id") or ""),
                        "name": str(entity.get("name") or ""),
                        "raw_type": str(raw_type or ""),
                        "schema_type": str(updated_entity.get("schema_type") or ""),
                        "schema_type_zh": str(updated_entity.get("schema_type_zh") or ""),
                    }
                )

    report = {
        "statistics": {
            "entity_count": entity_count,
            "updated_schema_type_count": schema_type_updated_count,
            "updated_schema_type_zh_count": updated_count,
            "remaining_empty_schema_type_count": sum(
                1
                for graph in graphs
                for entity in _require_sequence(graph.get("entities", []), "graph.entities")
                if not str(entity.get("schema_type") or "").strip()
            ),
            "remaining_empty_schema_type_zh_count": sum(
                1
                for graph in graphs
                for entity in _require_sequence(graph.get("entities", []), "graph.entities")
                if not str(entity.get("schema_type_zh") or "").strip()
            ),
            "updated_raw_type_counts": dict(sorted(raw_type_counts.items())),
        },
        "updated_examples": updated_examples,
    }
    return result, report


def backfill_schema_type_zh(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """兼容旧调用名称，并同时回填 ``schema_type`` 与中文类型字段。"""

    return backfill_schema_type_fields(payload)


def write_json_atomic(path: str | Path, data: Mapping[str, Any]) -> None:
    """通过同目录临时文件原子替换 JSON，避免中途写入损坏主图。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_json(temporary_path, data)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    """构造中文类型回填工具的命令行参数。"""

    parser = argparse.ArgumentParser(description="回填 Stage 05 主图实体的 Schema 类型字段")
    parser.add_argument("--input", type=Path, default=DEFAULT_ALIGNED_OUTPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ALIGNED_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """读取主图、回填 Schema 类型字段并直接写出主图和报告。"""

    payload = read_json(args.input)
    if not isinstance(payload, Mapping):
        raise TypeError("Stage 05 主图顶层必须是 JSON 对象")
    updated_payload, report = backfill_schema_type_fields(payload)
    write_json_atomic(args.output, updated_payload)
    report["execution"] = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
    }
    write_json_atomic(args.report, report)
    return report


def main() -> int:
    """解析命令行参数并打印可供批处理读取的回填结果。"""

    report = run(build_parser().parse_args())
    statistics = report["statistics"]
    print(
        "Schema 类型回填完成："
        f"实体={statistics['entity_count']}，"
        f"schema_type 回填={statistics['updated_schema_type_count']}，"
        f"schema_type_zh 回填={statistics['updated_schema_type_zh_count']}，"
        f"剩余空值={statistics['remaining_empty_schema_type_count'] + statistics['remaining_empty_schema_type_zh_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_REPORT_PATH",
    "backfill_schema_type_fields",
    "backfill_schema_type_zh",
    "build_parser",
    "main",
    "run",
    "write_json_atomic",
]
