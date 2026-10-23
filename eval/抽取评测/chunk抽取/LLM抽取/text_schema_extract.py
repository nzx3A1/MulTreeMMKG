"""调用已有 Schema 约束文本抽取管线，抽取开发集 Chunk 并保存可视化对比结果。

默认读取 eval/dataset/dev/chunk/text_chunk.json，结果写入本目录下的
text_schema_extrction.json；以完整 sample_id 对齐纯 LLM 结果，支持断点恢复。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemaExtractors.schema_loader import DEFAULT_SCHEMA_PATH, load_schema
from src.schemaExtractors.textTextractor.pipeline import collect_text_chunks, extract_text_from_summary_file
from src.utils.json_io import read_json


DEFAULT_INPUT_PATH = PROJECT_ROOT / "eval" / "dataset" / "dev" / "chunk" / "text_chunk.json"
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("text_schema_extrction.json")


def extract_text_schema(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    """复用项目三阶段 Schema 抽取器，完成实体、关系、属性补充及验证汇总。"""
    return extract_text_from_summary_file(input_path, output_path, **kwargs)


def main() -> int:
    """读取命令行参数并执行开发集抽取，未解决的失败或验证错误返回非零退出码。"""
    parser = argparse.ArgumentParser(description="开发集文本 Chunk 的 Schema 约束实体与关系抽取")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="开发集文本 Chunk JSON 数组")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="抽取图谱与验证汇总 JSON")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH, help="Markdown Schema 定义")
    parser.add_argument("--workers", type=int, default=4, help="并发 Chunk 数量")
    parser.add_argument("--validation-retries", type=int, default=1, help="验证错误的模型纠正次数")
    parser.add_argument("--max-tokens", type=int, default=8192, help="单次模型输出 token 上限")
    parser.add_argument("--retry-chunk-id", action="append", help="强制重抽指定完整 sample_id，其余结果继续恢复")
    parser.add_argument("--no-resume", action="store_true", help="忽略已有检查点，重新抽取全部输入")
    parser.add_argument("--check-input", action="store_true", help="仅校验输入与 Schema，不调用模型或写结果")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.check_input:
        schema = load_schema(args.schema)
        chunks = collect_text_chunks(read_json(args.input))
        print(json.dumps({"input": str(args.input.resolve()), "output": str(args.output.resolve()),
                          "text_chunk_count": len(chunks), "entity_types": len(schema.entities),
                          "relation_types": len(schema.relations)}, ensure_ascii=False, indent=2))
        return 0
    result = extract_text_schema(
        args.input, args.output, schema_path=args.schema, max_workers=args.workers,
        validation_retries=args.validation_retries, max_tokens=args.max_tokens,
        resume=not args.no_resume, force_chunk_ids=args.retry_chunk_id,
    )
    print(json.dumps({"status": result["_status"], "output": str(args.output.resolve()),
                      "statistics": result["statistics"], "validation": result["validation_summary"]},
                     ensure_ascii=False, indent=2))
    return 1 if result["_status"] in {"partial_failure", "needs_review"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
