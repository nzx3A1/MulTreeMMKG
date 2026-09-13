"""postSchema 命令行入口：生成计划、补写 Schema，并安全替换 Stage 05 主图。"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_config import settings as model_settings  # noqa: E402
from src.schemaProcess.postSchema.llm_mapper import LLMPostSchemaSelector  # noqa: E402
from src.schemaProcess.postSchema.models import PostSchemaDecision  # noqa: E402
from src.schemaProcess.postSchema.processor import PostSchemaProcessor  # noqa: E402
from src.schemaProcess.postSchema.schema_writer import Neo4jPostSchemaWriter  # noqa: E402
from src.schemaProcess.schema_repository import JsonSchemaRepository, Neo4jSchemaRepository  # noqa: E402
from src.utils.json_io import read_json  # noqa: E402


LOGGER = logging.getLogger(__name__)
DEFAULT_UNMAPPED_PATH = PROJECT_ROOT / "output" / "stage_05_unmapped_entity_types.json"
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "output" / "stage_05_schema_aligned_graph.json"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "output" / "stage_05_post_schema_plan.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "output" / "stage_05_post_schema_report.json"
DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().with_name("mapping_overrides.json")


def write_json_atomic(path: str | Path, value: Any) -> None:
    """先写同目录临时文件再原子替换，避免中断留下半截 JSON。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.post_schema_tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(target)


def build_parser() -> argparse.ArgumentParser:
    """构造支持 dry-run、计划复用和离线 Schema 的命令行参数。"""

    parser = argparse.ArgumentParser(
        description="处理 Stage 05 未映射实体，补充 Schema 并替换主图中的实体/关系类型"
    )
    parser.add_argument("--unmapped-entities", type=Path, default=DEFAULT_UNMAPPED_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--plan-input", type=Path, default=None, help="复用已审查计划，跳过 LLM")
    parser.add_argument(
        "--overrides",
        type=Path,
        default=DEFAULT_OVERRIDES_PATH,
        help="与未映射报告指纹绑定的人工审查覆写表",
    )
    parser.add_argument("--no-overrides", action="store_true", help="不应用人工审查覆写表")
    parser.add_argument("--schema-json", type=Path, default=None, help="只读离线 Schema；默认读取 Neo4j")
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="生成计划和报告，但不写 Neo4j、不覆盖主图")
    parser.add_argument(
        "--skip-schema-write",
        action="store_true",
        help="应用主图但不把 CREATE_NEW 概念写入 Neo4j，通常仅用于离线调试",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def _load_object(path: Path, label: str) -> Mapping[str, Any]:
    """读取 JSON 并保证顶层是对象。"""

    value = read_json(path)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 顶层必须是 JSON 对象：{path}")
    return value


def _new_concepts_from_plan(plan: Mapping[str, Any]) -> list[Any]:
    """从已校验的实体决策提取去重后的新增概念。"""

    decisions = [
        PostSchemaDecision.from_mapping(item)
        for item in plan.get("entity_decisions", [])
        if isinstance(item, Mapping)
    ]
    unique = {
        item.new_concept.schema: item.new_concept
        for item in decisions
        if item.new_concept is not None
    }
    return [unique[key] for key in sorted(unique)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """执行一次计划生成或复用、图重写以及可选 Neo4j 写入。"""

    unmapped_report = _load_object(args.unmapped_entities, "未映射实体报告")
    graph_payload = _load_object(args.graph, "Stage 05 主图")
    repository = JsonSchemaRepository(args.schema_json) if args.schema_json else Neo4jSchemaRepository()
    snapshot = repository.load()
    selector = LLMPostSchemaSelector(batch_size=args.llm_batch_size)
    processor = PostSchemaProcessor(snapshot, selector)
    overrides = None
    if not args.no_overrides and args.overrides and args.overrides.exists():
        overrides = _load_object(args.overrides, "实体判定覆写表")

    if args.plan_input:
        plan = _load_object(args.plan_input, "postSchema 计划")
        LOGGER.info("复用计划：%s", args.plan_input.resolve())
        if overrides:
            plan = processor.revise_plan(plan, overrides, unmapped_report, graph_payload)
            LOGGER.info("已应用人工审查覆写并重新计算关系计划：%s", args.overrides.resolve())
    else:
        LOGGER.info(
            "开始生成 postSchema 两级计划：未映射实体=%s，类型=%s",
            (unmapped_report.get("statistics") or {}).get("unmapped_entity_count"),
            (unmapped_report.get("statistics") or {}).get("unique_raw_type_count"),
        )
        plan = processor.build_plan(unmapped_report, graph_payload, overrides=overrides)
    write_json_atomic(args.plan_output, plan)
    updated_graph, report = processor.apply_plan(plan, unmapped_report, graph_payload)

    write_result = {
        "created_count": 0,
        "reused_count": 0,
        "ensured_count": 0,
        "embedding_dimensions": 0,
    }
    if not args.dry_run:
        new_concepts = _new_concepts_from_plan(plan)
        if new_concepts and args.schema_json and not args.skip_schema_write:
            raise ValueError("使用 --schema-json 时不能写 Neo4j；请增加 --skip-schema-write")
        if new_concepts and not args.skip_schema_write:
            write_result = Neo4jPostSchemaWriter().write(
                new_concepts,
                llm_model=model_settings.llm.model,
            )
        write_json_atomic(args.output, updated_graph)

    report["execution"] = {
        "mode": "dry-run" if args.dry_run else "write",
        "graph_output": None if args.dry_run else str(args.output.resolve()),
        "schema_write_skipped": bool(args.skip_schema_write or args.dry_run),
        "schema_write_result": write_result,
        "llm_model": model_settings.llm.model,
    }
    write_json_atomic(args.report_output, report)
    LOGGER.info(
        "postSchema %s 完成：实体类型替换=%s，关系类型替换=%s，新概念=%s",
        "dry-run" if args.dry_run else "写入",
        report["apply_statistics"]["entity_type_replacement_count"],
        report["apply_statistics"]["relation_type_replacement_count"],
        plan["statistics"]["new_concept_count"],
    )
    return report


def main() -> int:
    """解析参数、配置日志并以退出码报告执行结果。"""

    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )
    try:
        report = run(args)
    except Exception:
        LOGGER.exception("postSchema 处理失败")
        return 1
    print(
        "postSchema 完成："
        f"实体替换={report['apply_statistics']['entity_type_replacement_count']}，"
        f"关系替换={report['apply_statistics']['relation_type_replacement_count']}，"
        f"新增关系映射={report['apply_statistics']['newly_mapped_relation_count']}"
    )
    print(f"计划：{args.plan_output.resolve()}")
    print(f"报告：{args.report_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run", "write_json_atomic"]
