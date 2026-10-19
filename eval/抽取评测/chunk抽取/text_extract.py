"""读取 dev 文本 Chunk，调用现有文本抽取算法并逐 Chunk 保存图谱。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.text_extractor.text_extractor import extract_from_text
from src.utils.llm_client import LLMClient

EVAL_DIR = Path(__file__).resolve().parent


class AuditedLLMClient(LLMClient):
    """记录抽取器发起的请求及异常，防止模型失败被误记为成功抽取。"""

    def __init__(self) -> None:
        """初始化项目模型客户端和线程安全的调用计数。"""
        super().__init__()
        self._audit_lock = Lock()
        self.request_count = 0
        self.errors: list[str] = []
        self.invalid_responses: list[str] = []

    def chat_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict | list:
        """转发抽取器请求，并检查实体或关系响应的基本结构。"""
        try:
            prompt = json.loads(messages[0]["content"])
            expected = "entities" if "entities" in prompt.get("Output", {}) else "relations"
        except (IndexError, KeyError, TypeError, ValueError):
            expected = "unknown"
        with self._audit_lock:
            self.request_count += 1
        try:
            result = super().chat_json(messages, **kwargs)
        except Exception as exc:
            with self._audit_lock:
                self.errors.append(f"{expected}: {type(exc).__name__}")
            raise
        if not isinstance(result, dict) or not isinstance(result.get(expected), list):
            with self._audit_lock:
                self.invalid_responses.append(expected)
        return result


def load_chunks(path: Path) -> dict[str, list[dict]]:
    """校验评测样本并按文档转换成文本抽取器需要的 Chunk。"""
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("文本样本文件必须是 JSON 数组")
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("文本样本必须是 JSON 对象")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or "::" not in sample_id or sample_id in seen:
            raise ValueError(f"样本 ID 无效或重复：{sample_id}")
        seen.add(sample_id)
        document_id, _ = sample_id.split("::", 1)
        section = sample.get("section")
        content = sample.get("content")
        if not isinstance(section, dict) or not isinstance(content, dict) or not isinstance(content.get("text"), str):
            raise ValueError(f"文本样本结构无效：{sample_id}")
        grouped[document_id].append({
            "id": sample_id,
            "document_id": document_id,
            "modality": "text",
            "order": sample["order"],
            "section_id": section["section_id"],
            "section_title": section["section_title"],
            "text": content["text"],
        })
    return dict(grouped)


def run(input_path: Path, output_path: Path, *, dry_run: bool = False, max_workers: int = 1) -> dict:
    """调用项目文本抽取器并逐 Chunk 保存，模型响应异常时停止后续文档。"""
    grouped = load_chunks(input_path)
    if max_workers < 1:
        raise ValueError("max_workers 必须大于或等于 1")
    all_ids = [chunk["id"] for chunks in grouped.values() for chunk in chunks]
    records: list[dict] = []
    client = None if dry_run else AuditedLLMClient()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_result(status: str) -> dict:
        """原子写入本轮已完成图谱和仍待处理的样本列表。"""
        completed_ids = {graph["metadata"]["chunk_id"] for graph in records}
        result = {
            "_status": status,
            "input": str(input_path.resolve()),
            "statistics": {
                "chunk_count": len(all_ids),
                "graph_count": len(records),
                "entity_count": None if dry_run else sum(len(graph["entities"]) for graph in records),
                "relation_count": None if dry_run else sum(len(graph["relations"]) for graph in records),
                "model_request_count": 0 if client is None else client.request_count,
                "model_error_count": 0 if client is None else len(client.errors),
                "invalid_response_count": 0 if client is None else len(client.invalid_responses),
            },
            "model_errors": [] if client is None else list(client.errors),
            "invalid_response_stages": [] if client is None else list(client.invalid_responses),
            "pending_sample_ids": [sample_id for sample_id in all_ids if sample_id not in completed_ids],
            "graphs": records,
        }
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_path)
        return result

    if dry_run:
        return write_result("validated_no_extraction")
    result = write_result("running")
    for document_id, chunks in grouped.items():
        def persist(graph: Any, index: int, total: int) -> None:
            """在每个 Chunk 完成后更新检查点。"""
            records.append(graph.to_dict(exclude={"metadata": {"raw_response"}}))
            write_result("running")
            print(f"{document_id}: {index}/{total} 个文本 Chunk 已保存", flush=True)

        # 中文说明：这里只调用项目抽取算法，具体模型请求由 src/extractors 发起。
        extract_from_text(chunks, client, max_workers=max_workers, on_graph_completed=persist, show_progress=True)
        if client.errors or client.invalid_responses:
            return write_result("partial_model_error")
    return write_result("completed" if len(records) == len(all_ids) else "partial_missing_graph")


def main() -> int:
    """解析参数并运行文本抽取或输入检查。"""
    parser = argparse.ArgumentParser(description="用项目文本抽取器处理 dev 文本 Chunk")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "eval/dataset/dev/chunk/text_chunk.json")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "text_extraction.json")
    parser.add_argument("--dry-run", action="store_true", help="只校验输入，不执行抽取或请求模型")
    parser.add_argument("--max-workers", type=int, default=1, help="同一文档的文本抽取并发数")
    args = parser.parse_args()
    result = run(args.input, args.output, dry_run=args.dry_run, max_workers=args.max_workers)
    print(f"文本结果：{result['statistics']['graph_count']}/{result['statistics']['chunk_count']} 个 Graph，状态={result['_status']}，{args.output.resolve()}")
    return 0 if result["_status"] in {"completed", "validated_no_extraction"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
