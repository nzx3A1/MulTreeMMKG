"""离线运行地层—剖面—测井图片的三类子分类 mock。"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys


# 中文说明：支持直接执行 util 脚本，并复用项目内的分类器与 JSON 工具。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.image_extractor.stratigraphic_profile import StratigraphicProfileSubtypeClassifier
from src.utils.json_io import read_json, write_json


TEST_IMAGE_DIR = PROJECT_ROOT / "src" / "extractors" / "image_extractor" / "stratigraphic_profile" / "testImage"
DEFAULT_SOURCE = TEST_IMAGE_DIR / "image_chunks_stratigraphic_profile.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "stage_03_stratigraphic_profile_subclassification_mock.json"


def main() -> None:
    """中文说明：逐图执行人工视觉 mock 分类、校验路径，并输出统计和独立 JSON 结果。"""

    parser = argparse.ArgumentParser(description="无 API 地层—剖面—测井图片子分类 mock")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="图片 Chunk JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="逐图子分类输出 JSON")
    args = parser.parse_args()

    chunks = read_json(args.source)
    if not isinstance(chunks, list):
        raise TypeError("图片 Chunk 输入必须是 JSON 列表")

    results = StratigraphicProfileSubtypeClassifier().classify_chunks(chunks)
    missing = [item["image_path"] for item in results if not Path(item["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"以下图片不存在：{missing}")

    counts = Counter(item["subtype"] for item in results)
    write_json(
        args.output,
        {
            "mode": "manual_visual_mock_without_api",
            "source_path": str(args.source),
            "image_count": len(results),
            "counts": dict(counts),
            "results": results,
        },
    )

    for index, item in enumerate(results, start=1):
        print(
            f"[{index:02d}/{len(results)}] {Path(item['image_path']).name} | "
            f"{item['subtype_name']} | confidence={item['confidence']:.2f}"
        )
    print(f"完成：图片={len(results)}，分类统计={dict(counts)}，API调用=0")
    print(f"输出文件：{args.output}")


if __name__ == "__main__":
    main()
