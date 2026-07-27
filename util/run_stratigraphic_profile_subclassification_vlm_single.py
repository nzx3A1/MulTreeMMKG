"""使用项目 VLM 仅测试一张地层—剖面—测井图片的子分类。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Mapping


# 中文说明：支持直接运行 util 脚本，并复用项目现有配置、VLM 客户端和 JSON 工具。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_config import settings
from src.extractors.image_extractor.schema_models import ImageExtractionTask, as_string_tuple
from src.extractors.image_extractor.stratigraphic_profile import VLMStratigraphicProfileSubtypeClassifier
from src.utils.json_io import read_json, write_json
from src.utils.vlm_client import VLMClient


TEST_IMAGE_DIR = PROJECT_ROOT / "src" / "extractors" / "image_extractor" / "stratigraphic_profile" / "testImage"
DEFAULT_SOURCE = TEST_IMAGE_DIR / "image_chunks_stratigraphic_profile.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "stage_03_stratigraphic_profile_subclassification_vlm_single.json"


def classify_one_image(
    source_path: Path = DEFAULT_SOURCE,
    image_index: int = 0,
) -> dict[str, Any]:
    """中文说明：只取指定序号的一个图片 Chunk，调用一次 VLM 子分类并返回 Chunk 形状结果。"""

    chunks = read_json(source_path)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"图片 Chunk 文件必须是非空 JSON 数组：{source_path}")
    if image_index < 0 or image_index >= len(chunks):
        raise IndexError(f"图片序号超出范围：{image_index}，有效范围为 0 到 {len(chunks) - 1}")

    raw_chunk = chunks[image_index]
    if not isinstance(raw_chunk, Mapping):
        raise TypeError(f"第 {image_index} 项不是图片 Chunk 对象")
    paths = as_string_tuple(raw_chunk.get("image_path"))
    if len(paths) != 1:
        raise ValueError("单图测试要求所选 Chunk 恰好包含一个 image_path")
    image_path = Path(paths[0])
    if not image_path.is_file():
        raise FileNotFoundError(f"测试图片不存在：{image_path}")

    chunk_id = str(raw_chunk.get("id") or raw_chunk.get("chunk_id") or "")
    task = ImageExtractionTask(
        document_id=str(raw_chunk.get("document_id") or chunk_id.split(":section:", 1)[0]),
        chunk_id=chunk_id,
        image_id=f"{chunk_id}:image:0",
        image_index=0,
        image_path=str(image_path),
        caption=str(raw_chunk.get("caption") or ""),
        references=as_string_tuple(raw_chunk.get("references")),
        classification_code="A03",
        classification_type="地层—剖面—测井",
        raw_chunk=raw_chunk,
    )
    result = VLMStratigraphicProfileSubtypeClassifier().classify(
        task,
        VLMClient(config=settings.vlm),
    )
    output_chunk = deepcopy(dict(raw_chunk))
    output_chunk["stratigraphic_subtype"] = result.to_dict()
    return output_chunk


def main() -> None:
    """中文说明：解析单图序号、执行一次 VLM 分类、写出结果并打印具体类型。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="仅调用一张图片测试地层 VLM 子分类器")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="图片 Chunk JSON")
    parser.add_argument("--index", type=int, default=0, help="只测试的 Chunk 序号，默认第一张（0）")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="单图分类结果 JSON")
    args = parser.parse_args()

    output_chunk = classify_one_image(args.source, args.index)
    write_json(args.output, output_chunk)
    classification = output_chunk["stratigraphic_subtype"]
    print(f"图片：{output_chunk['image_path'][0]}")
    print(f"具体类型：{classification['subtype_name']}（{classification['subtype']}）")
    print(f"置信度：{classification['confidence']}")
    print(f"视觉证据：{classification['evidence']}")
    print(f"输出文件：{args.output}")


if __name__ == "__main__":
    main()
