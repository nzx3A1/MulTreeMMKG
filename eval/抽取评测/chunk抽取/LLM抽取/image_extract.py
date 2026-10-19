"""逐图像 Chunk 通过提示词和项目视觉模型抽取实体、关系并保存统一 Graph。"""
from __future__ import annotations

import argparse
from pathlib import Path

from text_extract import DATA_DIR, run


def main() -> int:
    """解析图像 Chunk 的输入输出路径并执行视觉模型抽取。"""
    parser = argparse.ArgumentParser(description="逐图像 Chunk 仅用提示词和项目视觉模型抽取")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "image_chunk.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "image_extraction.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(args.input, args.output, "image", dry_run=args.dry_run)
    print(f"状态={result['_status']}；{args.output.resolve()}")
    return 0 if result["_status"] in {"completed", "validated_no_extraction"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
