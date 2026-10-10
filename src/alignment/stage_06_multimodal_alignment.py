"""Stage 06 多模态实体与关系对齐命令行入口。

默认启用本地 Embedding 和 LLM 灰区裁决；服务不可用时流程会保留未决候选，
也可通过命令行显式关闭外部模型以运行确定性离线基线。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.pipeline import (
    DEFAULT_CONFLICT_PATH,
    DEFAULT_ENTITY_REPORT_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_UNRESOLVED_PATH,
    MultimodalAlignmentPipeline,
)
from src.alignment.scope_rules import DEFAULT_RULES_PATH


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构造 Stage 06 命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="在同一文档内融合多模态实体，并将全部关系汇集到全局节点"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--entity-report", type=Path, default=DEFAULT_ENTITY_REPORT_PATH)
    parser.add_argument("--unresolved", type=Path, default=DEFAULT_UNRESOLVED_PATH)
    parser.add_argument("--conflicts", type=Path, default=DEFAULT_CONFLICT_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--disable-embedding", action="store_true", help="关闭实体向量召回")
    parser.add_argument("--disable-llm", action="store_true", help="关闭灰区实体对 LLM 裁决")
    parser.add_argument("--min-name-similarity", type=float, default=0.58)
    parser.add_argument("--min-embedding-similarity", type=float, default=0.84)
    parser.add_argument("--auto-merge-threshold", type=float, default=0.78)
    parser.add_argument("--uncertain-threshold", type=float, default=0.58)
    parser.add_argument("--llm-batch-size", type=int, default=12)
    parser.add_argument("--min-llm-same-confidence", type=float, default=0.82)
    parser.add_argument(
        "--max-llm-candidates",
        type=int,
        default=200,
        help="最多提交给 LLM 的灰区候选数；-1 表示处理全部候选",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> int:
    """解析参数、执行 Stage 06 并打印核心守恒统计。"""

    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )
    pipeline = MultimodalAlignmentPipeline(
        rules_path=args.rules,
        enable_embedding=not args.disable_embedding,
        enable_llm=not args.disable_llm,
        min_name_similarity=args.min_name_similarity,
        min_embedding_similarity=args.min_embedding_similarity,
        auto_merge_threshold=args.auto_merge_threshold,
        uncertain_threshold=args.uncertain_threshold,
        llm_batch_size=args.llm_batch_size,
        min_llm_same_confidence=args.min_llm_same_confidence,
        max_llm_candidates=args.max_llm_candidates,
    )
    aligned, _, unresolved, conflicts = pipeline.run(
        input_path=args.input,
        output_path=args.output,
        entity_report_path=args.entity_report,
        unresolved_path=args.unresolved,
        conflict_path=args.conflicts,
    )
    stats = aligned["statistics"]
    print(
        "Stage 06 完成："
        f"实体 {stats['source_entity_count']} -> {stats['global_entity_count']}，"
        f"关系 {stats['source_relation_count']} -> {stats['global_relation_count']}，"
        f"未决候选={unresolved['statistics']['unresolved_candidate_count']}，"
        f"悬空关系={stats['dangling_relation_count']}"
    )
    print(f"守恒校验：{aligned['validation']}")
    print(f"主输出：{args.output.resolve()}")
    print(f"冲突报告：{args.conflicts.resolve()}（{conflicts['statistics']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
