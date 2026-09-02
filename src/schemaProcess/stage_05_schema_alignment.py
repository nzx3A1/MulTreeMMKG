"""Stage 05 Schema 对齐命令行入口。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemaProcess.entity_rules import DEFAULT_RULES_PATH
from src.schemaProcess.pipeline import (
    DEFAULT_ALIGNED_OUTPUT_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_UNMAPPED_ENTITY_PATH,
    DEFAULT_UNMAPPED_RELATION_PATH,
    SchemaAlignmentPipeline,
)
from src.schemaProcess.schema_repository import JsonSchemaRepository, Neo4jSchemaRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 Stage 04 开放抽取实体/关系对齐到石油地质概念 Schema"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ALIGNED_OUTPUT_PATH)
    parser.add_argument("--unmapped-entities", type=Path, default=DEFAULT_UNMAPPED_ENTITY_PATH)
    parser.add_argument("--unmapped-relations", type=Path, default=DEFAULT_UNMAPPED_RELATION_PATH)
    parser.add_argument("--schema-json", type=Path, default=None, help="可选离线 Schema 快照；默认读 Neo4j")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument("--min-llm-confidence", type=float, default=0.72)
    parser.add_argument("--min-vector-similarity", type=float, default=0.25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repository = JsonSchemaRepository(args.schema_json) if args.schema_json else Neo4jSchemaRepository()
    pipeline = SchemaAlignmentPipeline(
        repository=repository,
        rules_path=args.rules,
        top_k=args.top_k,
        embedding_batch_size=args.embedding_batch_size,
        llm_batch_size=args.llm_batch_size,
        min_llm_confidence=args.min_llm_confidence,
        min_vector_similarity=args.min_vector_similarity,
    )
    aligned, entity_report, relation_report = pipeline.run(
        input_path=args.input,
        aligned_output_path=args.output,
        unmapped_entity_path=args.unmapped_entities,
        unmapped_relation_path=args.unmapped_relations,
    )
    stats = aligned["_schema_alignment"]
    print(
        "Stage 05 完成："
        f"实体={stats['entity_status_counts']}，关系={stats['relation_status_counts']}；"
        f"未映射实体={entity_report['statistics']['unmapped_entity_count']}，"
        f"未映射关系={relation_report['statistics']['unmapped_relation_count']}"
    )
    print(f"主输出：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
