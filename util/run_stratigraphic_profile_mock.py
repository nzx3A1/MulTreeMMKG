"""用人工逐图构造的模型响应运行全部地层—剖面—测井图片。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Mapping


# 中文说明：支持直接点击或执行 util 脚本，将项目根目录加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.image_extractor import extract_from_images
from src.utils.json_io import read_json, write_json


TEST_DATA_DIR = PROJECT_ROOT / "src" / "extractors" / "image_extractor" / "stratigraphic_profile" / "test_data"
DEFAULT_SOURCE = TEST_DATA_DIR / "image_chunks.json"
DEFAULT_MOCK = TEST_DATA_DIR / "mock_responses.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "stage_04_stratigraphic_profile_mock.json"


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    """中文说明：读取并校验按图片文件名索引的人工模型响应。"""

    payload = read_json(path)
    records = payload.get("responses") if isinstance(payload, Mapping) else None
    if not isinstance(records, dict) or not records:
        raise ValueError(f"mock 文件缺少非空 responses：{path}")
    return records


def prepare_chunks(path: Path, records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """中文说明：保持原 Chunk 内容不变，仅注入人工确认的单图分类与文档 ID。"""

    chunks = deepcopy(read_json(path))
    if not isinstance(chunks, list):
        raise TypeError("图片 Chunk 输入必须是 JSON 列表")
    for chunk in chunks:
        image_paths = chunk.get("image_path", [])
        if not isinstance(image_paths, list) or len(image_paths) != 1:
            raise ValueError(f"本脚本要求每个 Chunk 恰好一张图：{chunk.get('id')}")
        filename = Path(image_paths[0]).name
        if filename not in records:
            raise KeyError(f"图片缺少人工 mock 响应：{filename}")
        chunk["classification"] = dict(records[filename]["classification"])
        chunk.setdefault("document_id", str(chunk.get("id", "")).split(":section:", 1)[0])
    return chunks


class FileMockVLM:
    """从 JSON 文件返回当前来源图片的人工视觉模型响应。"""

    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        """中文说明：保存响应索引和调用计数。"""

        self.records = records
        self.call_count = 0

    def describe_image(self, image_path: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """中文说明：按图片文件名返回 visual 字段，等价模拟一次 VLM JSON 调用。"""

        _ = prompt, kwargs
        filename = Path(image_path).name
        self.call_count += 1
        return deepcopy(self.records[filename]["visual"])


class FileMockLLM:
    """从 JSON 文件返回当前来源图片的人工关系审查响应。"""

    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        """中文说明：保存响应索引和调用计数。"""

        self.records = records
        self.call_count = 0

    def call_openai_json(self, prompt: str, task_name: str = "") -> dict[str, Any]:
        """中文说明：从 Prompt 中定位图片文件名并返回 audit 字段。"""

        for filename, record in self.records.items():
            if filename in prompt:
                self.call_count += 1
                return deepcopy(record["audit"])
        raise KeyError(f"无法从关系审查 Prompt 定位图片：{task_name}")


def main() -> None:
    """中文说明：运行 14 张图片、打印逐图实体关系计数，并写出可直接检查的 Graph JSON。"""

    parser = argparse.ArgumentParser(description="使用人工模型响应离线验证地层—剖面—测井图片抽取器")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="原始图片 Chunk JSON")
    parser.add_argument("--mock", type=Path, default=DEFAULT_MOCK, help="人工 VLM/LLM 响应 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Graph 输出 JSON")
    args = parser.parse_args()

    records = load_records(args.mock)
    chunks = prepare_chunks(args.source, records)
    vlm = FileMockVLM(records)
    llm = FileMockLLM(records)
    graphs = extract_from_images(chunks, llm, vlm, show_progress=False)
    write_json(args.output, [graph.to_dict() for graph in graphs])

    total_entities = 0
    total_relations = 0
    for index, graph in enumerate(graphs, start=1):
        filename = Path(graph.metadata.extra["routes"][0]["image_path"]).name
        total_entities += len(graph.entities)
        total_relations += len(graph.relations)
        print(
            f"[{index:02d}/{len(graphs)}] {filename} | "
            f"实体={len(graph.entities)} 关系={len(graph.relations)} 事件={len(graph.events)} "
            f"状态={graph.metadata.extra.get('status')}"
        )
    print(
        f"完成：图片={len(graphs)}，VLM mock调用={vlm.call_count}，LLM mock调用={llm.call_count}，"
        f"实体总数={total_entities}，关系总数={total_relations}，事件总数={sum(len(graph.events) for graph in graphs)}"
    )
    print(f"输出文件：{args.output}")


if __name__ == "__main__":
    main()
