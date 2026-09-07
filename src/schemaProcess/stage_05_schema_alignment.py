"""Stage 05 Schema 对齐命令行入口。"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemaProcess.entity_rules import DEFAULT_RULES_PATH
from src.schemaProcess.pipeline import (
    DEFAULT_ALIGNED_OUTPUT_PATH,
    DEFAULT_ENTITY_MAPPING_CACHE_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_SCHEMA_EMBEDDING_CACHE_PATH,
    DEFAULT_UNMAPPED_ENTITY_PATH,
    DEFAULT_UNMAPPED_RELATION_PATH,
    SchemaAlignmentPipeline,
)
from src.schemaProcess.schema_repository import JsonSchemaRepository, Neo4jSchemaRepository
from config.model_config import settings as model_settings


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构造 Stage 05 命令行参数解析器。"""

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
    parser.add_argument("--vector-auto-accept-threshold", type=float, default=0.92)
    parser.add_argument("--vector-auto-accept-margin", type=float, default=0.12)
    parser.add_argument(
        "--disable-category-routing",
        action="store_true",
        help="关闭 Schema 一级分类，所有未命中规则的实体都在全 Schema 中召回",
    )
    parser.add_argument("--category-route-min-score", type=int, default=2)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="控制台日志级别，默认 INFO",
    )
    parser.add_argument(
        "--entity-mapping-cache",
        type=Path,
        default=DEFAULT_ENTITY_MAPPING_CACHE_PATH,
        help="实体映射 JSON 缓存；传空路径无法通过 argparse 禁用，可在 API 中传 None",
    )
    parser.add_argument(
        "--schema-embedding-cache",
        type=Path,
        default=DEFAULT_SCHEMA_EMBEDDING_CACHE_PATH,
        help="Schema embedding 持久化 JSON 缓存",
    )
    return parser


def main() -> int:
    """读取命令行参数并执行一次完整的 Schema 对齐。"""

    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger.info("启动 Stage 05 Schema 对齐，Python=%s", sys.executable)
    logger.info(
        "运行参数：input=%s，top_k=%s，embedding_batch=%s，llm_batch=%s，"
        "vector_auto=(top1>=%.2f, margin>=%.2f)",
        args.input.resolve(),
        args.top_k,
        args.embedding_batch_size,
        args.llm_batch_size,
        args.vector_auto_accept_threshold,
        args.vector_auto_accept_margin,
    )
    logger.info(
        "Embedding 服务：%s，模型=%s，超时=%s 秒；LLM 服务：%s，模型=%s，"
        "单次超时=%s 秒，重试次数=%s",
        model_settings.embedding.base_url,
        model_settings.embedding.model,
        model_settings.embedding.timeout_secs,
        model_settings.llm.base_url,
        model_settings.llm.model,
        model_settings.llm.timeout_secs,
        os.getenv("LLM_RETRY_ATTEMPTS", "3"),
    )
    repository = JsonSchemaRepository(args.schema_json) if args.schema_json else Neo4jSchemaRepository()
    if isinstance(repository, Neo4jSchemaRepository):
        logger.info(
            "Schema 数据源：Neo4j %s，database=%s",
            repository.config.uri,
            repository.config.database,
        )
    else:
        logger.info("Schema 数据源：离线 JSON %s", args.schema_json.resolve())
    pipeline = SchemaAlignmentPipeline(
        repository=repository,
        rules_path=args.rules,
        top_k=args.top_k,
        embedding_batch_size=args.embedding_batch_size,
        llm_batch_size=args.llm_batch_size,
        min_llm_confidence=args.min_llm_confidence,
        min_vector_similarity=args.min_vector_similarity,
        vector_auto_accept_threshold=args.vector_auto_accept_threshold,
        vector_auto_accept_margin=args.vector_auto_accept_margin,
        entity_mapping_cache_path=args.entity_mapping_cache,
        schema_embedding_cache_path=args.schema_embedding_cache,
        enable_category_routing=not args.disable_category_routing,
        category_route_min_score=args.category_route_min_score,
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
