"""运行单个地质过程与成因模式图片 Chunk 的抽取示例。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Mapping


# 中文说明：支持直接运行 util 脚本，将项目根目录加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.image_extractor import build_image_tasks
from src.extractors.image_extractor.geological_process import (
    GeologicalProcessExtractor,
    apply_geological_review_overlay,
    build_geological_process_graph,
    normalize_geological_visual_result,
)
from src.extractors.image_extractor.schema_models import ImageExtractionContext
from src.utils.json_io import read_json, write_json
from src.utils.llm_client import LLMClient
from src.utils.vlm_client import VLMClient


DEFAULT_SOURCE = PROJECT_ROOT / "output" / "stage_05_image_chunks_classified.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "stage_04_geological_process_sample.json"
DEFAULT_CHUNK_ID = "鄂尔多斯盆地奥陶系马家沟组白云岩储层特征及成因机制_吴东旭:section:14:image:3"


def find_chunk(payload: Any, chunk_id: str) -> Mapping[str, Any]:
    """中文说明：从根列表或带 chunks 字段的 JSON 中查找指定图片 Chunk。"""

    chunks = payload.get("chunks", []) if isinstance(payload, Mapping) else payload
    if not isinstance(chunks, list):
        raise TypeError("输入 JSON 必须是 Chunk 列表或包含 chunks 列表")
    for item in chunks:
        if isinstance(item, Mapping) and str(item.get("id") or item.get("chunk_id") or "") == chunk_id:
            return item
    raise KeyError(f"未找到图片 Chunk：{chunk_id}")


def main() -> None:
    """中文说明：读取单个 Chunk，调用专用抽取器并把 Graph 写成 UTF-8 JSON。"""

    parser = argparse.ArgumentParser(description="抽取单张地质过程与成因模式图的时空知识图谱")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="包含图片 Chunk 的 JSON")
    parser.add_argument("--chunk-id", default=DEFAULT_CHUNK_ID, help="待抽取图片 Chunk ID")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="单图 Graph 输出路径")
    parser.add_argument("--no-relation-audit", action="store_true", help="只调用 VLM，不调用文本 LLM 审查关系")
    parser.add_argument("--rebuild-from", type=Path, help="复用已有真实模型 raw_response，仅执行最新本地校正和 Graph 重建")
    parser.add_argument("--review-overlay", type=Path, help="可选的人工/外部证据复核 JSON，在本地校正后叠加")
    args = parser.parse_args()

    chunk = find_chunk(read_json(args.source), args.chunk_id)
    tasks = build_image_tasks(chunk)
    if len(tasks) != 1:
        raise ValueError(f"该示例要求 Chunk 只包含一张图片，实际为 {len(tasks)} 张")

    extractor = GeologicalProcessExtractor()
    if args.rebuild_from:
        # 中文说明：复用已完成的真实模型响应，便于快速验证确定性校正规则而不重复付费调用模型。
        saved = read_json(args.rebuild_from)
        metadata = saved.get("metadata", {}) if isinstance(saved, Mapping) else {}
        raw_response = metadata.get("raw_response", {}) if isinstance(metadata, Mapping) else {}
        visual = deepcopy(raw_response.get("visual", {})) if isinstance(raw_response, Mapping) else {}
        audit = deepcopy(raw_response.get("relation_audit", {})) if isinstance(raw_response, Mapping) else {}
        if not isinstance(visual, dict) or not visual:
            raise ValueError("--rebuild-from 文件缺少 metadata.raw_response.visual")
        normalize_geological_visual_result(tasks[0], visual)
        if args.review_overlay:
            overlay = read_json(args.review_overlay)
            if not isinstance(overlay, Mapping):
                raise TypeError("--review-overlay 必须是 JSON 对象")
            apply_geological_review_overlay(visual, overlay, audit if isinstance(audit, dict) else None)
        graph = build_geological_process_graph(tasks[0], visual, audit if isinstance(audit, Mapping) else {})
        previous_extra = metadata.get("extra", {}) if isinstance(metadata, Mapping) else {}
        graph.metadata.extra.update(
            {
                "model_errors": [],
                "vlm_called": True,
                "vlm_call_count": previous_extra.get("vlm_call_count", 2) if isinstance(previous_extra, Mapping) else 2,
                "lithology_audit": previous_extra.get("lithology_audit", {}) if isinstance(previous_extra, Mapping) else {},
                "llm_audit_called": bool(audit),
                "live_model_called_this_run": False,
                "rebuild_from": str(args.rebuild_from),
                "review_overlay_path": str(args.review_overlay) if args.review_overlay else "",
            }
        )
    else:
        graph = extractor.extract(
            tasks[0],
            ImageExtractionContext(
                llm_client=LLMClient(),
                vlm_client=VLMClient(),
                options={"enable_relation_audit": not args.no_relation_audit},
            ),
        )
    write_json(args.output, graph.to_dict())
    print(
        f"抽取状态={graph.metadata.extra.get('status')}，"
        f"实体={len(graph.entities)}，关系={len(graph.relations)}，事件={len(graph.events)}"
    )
    if graph.metadata.extra.get("model_errors"):
        print("模型错误：" + "；".join(graph.metadata.extra["model_errors"]))
    print(f"输出文件：{args.output}")


if __name__ == "__main__":
    main()
