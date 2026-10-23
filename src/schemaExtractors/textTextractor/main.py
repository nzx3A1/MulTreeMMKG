"""命令行入口：读取第三步文本并执行 Schema 约束抽取或仅检查输入与本体。"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


# 中文说明：兼容直接执行本文件和 python -m 两种启动方式。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemaExtractors.schema_loader import DEFAULT_SCHEMA_PATH, load_schema
from src.schemaExtractors.textTextractor.pipeline import (
    DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH, collect_text_chunks, extract_text_from_summary_file,
)
from src.utils.json_io import read_json


def main() -> int:
    """解析运行参数，输出统计；失败或仍需复核时返回非零退出码。"""
    parser = argparse.ArgumentParser(description="依据 schema.md 对第三步文本 Chunk 抽取实体、属性和关系")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="第三步文档总结 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="抽取结果及检查点 JSON")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH, help="本目录 Markdown Schema")
    parser.add_argument("--workers", type=int, default=4, help="Chunk 并发数，每段内部实体、关系、属性三阶段串行")
    parser.add_argument("--validation-retries", type=int, default=1, help="结构/Schema 错误反馈重抽次数")
    parser.add_argument("--max-tokens", type=int, default=8192, help="每次模型调用的最大输出 token 数")
    parser.add_argument("--limit", type=int, help="只抽取前 N 个文本 Chunk，建议使用独立输出文件")
    parser.add_argument("--chunk-id", action="append", help="仅处理指定 Chunk，可重复传入")
    parser.add_argument("--retry-chunk-id", action="append", help="强制重抽指定 Chunk，其余仍从检查点恢复，可重复传入")
    parser.add_argument("--no-resume", action="store_true", help="忽略检查点，重新调用模型")
    parser.add_argument("--check-input", action="store_true", help="仅检查 Schema 和输入，不调用模型或写结果")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.check_input:
        schema = load_schema(args.schema)
        chunks = collect_text_chunks(read_json(args.input))
        print(json.dumps({"entity_types": len(schema.entities), "relation_types": len(schema.relations),
                          "text_chunks": len(chunks), "chunk_ids": [chunk["id"] for chunk in chunks]}, ensure_ascii=False, indent=2))
        return 0
    result = extract_text_from_summary_file(
        args.input, args.output, schema_path=args.schema, max_workers=args.workers,
        validation_retries=args.validation_retries, resume=not args.no_resume,
        limit=args.limit, chunk_ids=args.chunk_id, max_tokens=args.max_tokens,
        force_chunk_ids=args.retry_chunk_id,
    )
    print(json.dumps({"status": result["_status"], "output": str(args.output.resolve()),
                      "statistics": result["statistics"], "validation": result["validation_summary"]}, ensure_ascii=False, indent=2))
    return 1 if result["_status"] in {"partial_failure", "needs_review"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
