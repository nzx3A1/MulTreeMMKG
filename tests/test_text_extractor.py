"""无 Schema 文本实体与关系抽取流程测试。"""
from __future__ import annotations

import json
import threading
from typing import Any

from model import Graph, SourceModality
from src.extractors.text_extractor import extract_from_text
from src.extractors.text_extractor import text_extractor as text_extractor_module
from src.extractors.text_extractor.pipeline import extract_text_chunks_to_file


class _FakeLLM:
    """按实体、关系顺序返回自由类型结果，并记录模型提示词。"""

    def __init__(self) -> None:
        """初始化调用计数和消息记录，供断言请求次数与提示词内容。"""

        self.calls = 0
        self.messages: list[Any] = []

    def chat_json(self, messages: Any, **_: Any) -> dict[str, Any]:
        """为每次非空 Chunk 返回两段可直接解析的模型响应。"""

        self.calls += 1
        self.messages.append(messages)
        if self.calls % 2:
            return {
                "entities": [
                    {
                        "id": "entity_1",
                        "name": "延长组",
                        "official_name": "延长组",
                        "type": "formation",
                        "type_zh": "组",
                        "attributes": {},
                        "provenance": "延长组位于鄂尔多斯盆地",
                    },
                    {
                        "id": "entity_2",
                        "name": "鄂尔多斯盆地",
                        "official_name": "鄂尔多斯盆地",
                        "type": "basin",
                        "type_zh": "盆地",
                        "attributes": {},
                        "provenance": "鄂尔多斯盆地",
                    },
                ]
            }
        return {
            "relations": [
                {
                    "id": "relation_1",
                    "type": "LOCATED_IN",
                    "relation_name": "位于",
                    "type_zh": "位于",
                    "source_id": "entity_1",
                    "source_name": "延长组",
                    "source_type": "formation",
                    "target_id": "entity_2",
                    "target_name": "鄂尔多斯盆地",
                    "target_type": "basin",
                    "attributes": {},
                    "provenance": "延长组位于鄂尔多斯盆地",
                }
            ]
        }


def _chunks() -> list[dict[str, Any]]:
    """构造乱序的同文档文本 Chunk，其中包含一个空正文窗口。"""

    return [
        {
            "id": "c2",
            "order": 2,
            "document_id": "doc-1",
            "section_id": "s1",
            "section_title": "地层位置",
            "modality": "text",
            "text": "",
        },
        {
            "id": "c1",
            "order": 1,
            "document_id": "doc-1",
            "section_id": "s1",
            "section_title": "地层位置",
            "modality": "text",
            "text": "延长组位于鄂尔多斯盆地。",
        },
    ]


def test_extract_from_text_directly_extracts_without_schema() -> None:
    """非空 Chunk 只进行实体和关系两次调用，且不产生 Schema 元数据。"""

    llm = _FakeLLM()
    graphs = extract_from_text(_chunks(), llm, max_workers=4, show_progress=False)

    assert llm.calls == 2
    assert [graph.metadata.chunk_id for graph in graphs] == ["c1", "c2"]
    assert all(isinstance(graph, Graph) for graph in graphs)
    first = graphs[0]
    assert [entity.type for entity in first.entities] == ["formation", "basin"]
    assert first.relations[0].type == "LOCATED_IN"
    assert (
        first.relations[0].source_name,
        first.relations[0].source_type,
        first.relations[0].target_name,
        first.relations[0].target_type,
    ) == ("延长组", "formation", "鄂尔多斯盆地", "basin")
    assert first.validate_references() == []
    assert "schema_selection" not in first.metadata.extra
    assert first.metadata.raw_response is None
    assert graphs[1].metadata.extra["empty_reason"] == "empty_text"
    prompt_texts = [json.loads(messages[0]["content"]) for messages in llm.messages]
    assert all(
        not ({"entity_schema_whitelist", "relation_schema_whitelist", "concepts", "relation_constraints"} & set(prompt))
        for prompt in prompt_texts
    )


def test_extract_from_text_generates_stable_ids() -> None:
    """同文档的相同自由类型候选在重复抽取时仍应生成稳定 ID。"""

    first = extract_from_text(_chunks(), _FakeLLM(), max_workers=4, show_progress=False)[0]
    second = extract_from_text(_chunks(), _FakeLLM(), max_workers=4, show_progress=False)[0]

    assert [item.id for item in first.entities] == [item.id for item in second.entities]
    assert [item.id for item in first.relations] == [item.id for item in second.relations]


def test_extract_from_text_calls_persistence_callback_after_each_chunk() -> None:
    """每完成一个排序后的 Chunk 都要触发一次持久化回调。"""

    completed: list[tuple[str, int, int]] = []

    def record(graph: Graph, index: int, total: int) -> None:
        """记录回调参数，用于验证抽取和持久化的顺序一致。"""

        completed.append((str(graph.metadata.chunk_id), index, total))

    extract_from_text(
        _chunks(), _FakeLLM(), max_workers=4, on_graph_completed=record, show_progress=False,
    )

    assert completed == [("c1", 1, 2), ("c2", 2, 2)]


def test_extract_from_text_uses_four_worker_threads(monkeypatch: Any) -> None:
    """Chunk 抽取应最多允许四个工作线程同时调用 LLM 抽取流程。"""

    lock = threading.Lock()
    four_started = threading.Event()
    active = 0
    peak_active = 0

    def fake_extract(chunk: Any, document: Any, llm_client: Any) -> Graph:
        nonlocal active, peak_active
        _ = document, llm_client
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            if active == 4:
                four_started.set()
        assert four_started.wait(timeout=1)
        with lock:
            active -= 1
        return Graph.from_chunk("doc-1", str(chunk["id"]), SourceModality.TEXT, stage="test")

    monkeypatch.setattr(text_extractor_module, "extract_text_chunk_graph", fake_extract)
    chunks = [
        {"id": f"c{index}", "order": index, "document_id": "doc-1", "modality": "text", "text": "正文"}
        for index in range(6)
    ]

    graphs = extract_from_text(chunks, object(), max_workers=4, show_progress=False)

    assert peak_active == 4
    assert [graph.metadata.chunk_id for graph in graphs] == [f"c{index}" for index in range(6)]


def test_text_pipeline_writes_jsonl_and_resumes_without_reprocessing(tmp_path: Any, capsys: Any) -> None:
    """检查点续跑应跳过已完成 Chunk，并保留最终聚合 JSON。"""

    output_path = tmp_path / "stage_04_text_extraction.json"
    first_llm = _FakeLLM()
    graphs = extract_text_chunks_to_file(
        _chunks(), output_path, llm_client=first_llm, max_workers=4, show_progress=False,
    )

    journal_path = output_path.with_suffix(".jsonl")
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert first_llm.calls == 2
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 2
    assert len(graphs) == 2
    assert result["_status"] == "completed"
    assert all(
        "raw_response" not in graph["metadata"]
        and "schema_selection" not in graph["metadata"].get("extra", {})
        for graph in result["graphs"]
    )
    assert "[1/2] Chunk c1 已写入" in capsys.readouterr().out

    resumed_llm = _FakeLLM()
    resumed = extract_text_chunks_to_file(
        _chunks(), output_path, llm_client=resumed_llm, max_workers=4, show_progress=False,
    )
    assert resumed_llm.calls == 0
    assert len(resumed) == 2
