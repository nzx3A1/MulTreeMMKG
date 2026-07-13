"""基于两级 Schema 结果的文本 Graph 抽取流程测试。"""
from __future__ import annotations

import json
from typing import Any

from model import Graph
from src.extractors.extractor_init import InitExtractor
from src.extractors.text_extractor import build_document_context, extract_from_text
from src.extractors.text_extractor.pipeline import extract_text_chunks_to_file
from src.extractors.text_extractor.schema_models import (
    DocumentSchemaContext,
    RelevantSchema,
    SchemaConcept,
    SchemaRelation,
)


LOCAL_SCHEMA = RelevantSchema(
    concepts=(
        SchemaConcept(schema="Formation", zh_name="组", final_score=0.9),
        SchemaConcept(schema="Basin", zh_name="盆地", final_score=0.9),
    ),
    relations=(
        SchemaRelation(
            source_schema="Formation",
            relation_en="LOCATED_IN",
            relation_zh="位于",
            target_schema="Basin",
            edge_score=0.9,
        ),
    ),
    selection_confidence=0.9,
)


class _FakeSelector:
    """记录整篇 Schema 选择次数，并为每个 Chunk 返回同一测试子图。"""

    def __init__(self) -> None:
        """初始化调用计数器。"""

        self.calls = 0

    def prepare_document(self, chunks: Any) -> DocumentSchemaContext:
        """构造与真实选择器接口一致的文档级 Schema 上下文。"""

        self.calls += 1
        document = build_document_context(chunks)
        return DocumentSchemaContext(
            document=document,
            document_schema_pool=LOCAL_SCHEMA,
            chunk_schemas={str(chunk["id"]): LOCAL_SCHEMA for chunk in document.chunks},
        )


class _FakeLLM:
    """返回包含合法候选和越界候选的结构化模型响应。"""

    def __init__(self) -> None:
        """保存调用次数和模型收到的消息。"""

        self.calls = 0
        self.messages: list[Any] = []

    def chat_json(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        """返回用于验证实体、关系、事件和拒绝逻辑的固定 JSON。"""

        self.calls += 1
        self.messages.append(messages)
        if self.calls == 2:
            return {
                "entities": [
                    {
                        "name": "延长组",
                        "official_name": "延长组",
                        "type": "Formation",
                        "type_zh": "组",
                    },
                    {
                        "name": "鄂尔多斯盆地",
                        "official_name": "鄂尔多斯盆地",
                        "type": "Basin",
                        "type_zh": "盆地",
                    },
                    {
                        "name": "相邻段落实体",
                        "official_name": "相邻段落实体",
                        "type": "Basin",
                        "type_zh": "盆地",
                    },
                ]
            }
        if self.calls == 3:
            return {
                "relations": [
                    {
                        "temp_id": "r1",
                        "type": "located_in",
                        "source_id": "entity_1",
                        "target_id": "entity_2",
                        "attributes": {},
                        "provenance": "延长组位于鄂尔多斯盆地",
                        "metadata": {},
                    },
                    {
                        "temp_id": "r2",
                        "type": "LOCATED_IN",
                        "source_id": "entity_2",
                        "target_id": "entity_1",
                        "provenance": "延长组位于鄂尔多斯盆地",
                    },
                ]
            }
        if self.calls == 4:
            return {
                "relations": [
                    {
                        "source_id": "entity_1",
                        "target_id": "entity_2",
                        "relation_name": "位于",
                        "type": "LOCATED_IN",
                        "type_zh": "位于",
                    },
                    {
                        "source_id": "entity_2",
                        "target_id": "entity_1",
                        "relation_name": "位于",
                        "type": "LOCATED_IN",
                        "type_zh": "位于",
                    },
                ]
            }
        return {
            "entities": [
                {
                    "temp_id": "e1",
                    "name": "延长组",
                    "type": "Formation",
                    "type_zh": "错误中文名会被规范化",
                    "aliases": [],
                    "attributes": {},
                    "provenance": "延长组位于鄂尔多斯盆地",
                    "metadata": {"confidence": 0.95},
                },
                {
                    "temp_id": "e2",
                    "name": "鄂尔多斯盆地",
                    "type": "Basin",
                    "aliases": [],
                    "attributes": {},
                    "provenance": "鄂尔多斯盆地",
                    "metadata": {},
                },
                {
                    "temp_id": "e3",
                    "name": "相邻段落实体",
                    "type": "Basin",
                    "provenance": "该证据只存在于相邻上下文",
                },
            ],
            "relations": [
                {
                    "temp_id": "r1",
                    "type": "located_in",
                    "source_id": "e1",
                    "target_id": "e2",
                    "attributes": {},
                    "provenance": "延长组位于鄂尔多斯盆地",
                    "metadata": {},
                },
                {
                    "temp_id": "r2",
                    "type": "LOCATED_IN",
                    "source_id": "e2",
                    "target_id": "e1",
                    "provenance": "延长组位于鄂尔多斯盆地",
                },
            ],
            "events": [
                {
                    "temp_id": "v1",
                    "type": "observation",
                    "name": "延长组位置观测",
                    "participants": ["e1", "e2", "missing"],
                    "time": None,
                    "location": "鄂尔多斯盆地",
                    "attributes": {},
                    "provenance": "延长组位于鄂尔多斯盆地",
                    "metadata": {},
                }
            ],
        }


class _StreamingFakeSelector:
    """模拟文档池只选一次、局部 Schema 在逐 Chunk 循环中即时选择的新接口。"""

    def __init__(self) -> None:
        """初始化文档池和局部选择调用记录。"""

        self.document_calls = 0
        self.chunk_calls: list[str] = []

    def select_document_schema_pool(self, document: Any) -> RelevantSchema:
        """为整篇文档返回一次固定候选池。"""

        self.document_calls += 1
        return LOCAL_SCHEMA

    def select_chunk_schema(self, chunk: Any, document: Any, pool: Any) -> RelevantSchema:
        """记录局部 Schema 按 Chunk 顺序即时选择。"""

        self.chunk_calls.append(str(chunk["id"]))
        return LOCAL_SCHEMA


def _chunks() -> list[dict[str, Any]]:
    """构造顺序乱序且含一个空正文的同文档文本 Chunk。"""

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


def test_extract_from_text_builds_valid_graphs_with_one_document_selection() -> None:
    """整篇只选择一次 Schema，并将合法候选映射为引用完整的 Graph。"""

    selector = _FakeSelector()
    llm = _FakeLLM()

    graphs = extract_from_text(_chunks(), llm, selector)

    assert selector.calls == 1
    assert llm.calls == 4
    prompt_tasks = [json.loads(messages[0]["content"])["task"] for messages in llm.messages]
    assert "抽取实体" in prompt_tasks[0]
    assert "选择最合适的已定义类型" in prompt_tasks[1]
    assert "判断关系" in prompt_tasks[2]
    assert "方向及类型均匹配" in prompt_tasks[3]
    assert [graph.metadata.chunk_id for graph in graphs] == ["c1", "c2"]
    assert all(isinstance(graph, Graph) for graph in graphs)
    first = graphs[0]
    assert len(first.entities) == 3
    assert len(first.relations) == 2
    assert len(first.events) == 0
    assert first.entities[0].type_zh == "组"
    assert first.relations[0].type == "LOCATED_IN"
    assert first.relations[0].relation_name == "位于"
    assert first.entities[2].metadata["validation"]["passed"] is False
    assert "evidence_not_in_text" in first.entities[2].metadata["validation"]["errors"]
    assert first.relations[1].metadata["validation"]["passed"] is False
    assert "relation_not_in_schema" in first.relations[1].metadata["validation"]["errors"]
    assert first.validate_references() == []
    assert first.metadata.extra["validation"]["rejected_count"] == 2
    assert first.metadata.extra["validation"]["retained_invalid_count"] == 2
    assert graphs[1].metadata.extra["empty_reason"] == "empty_text"


def test_extract_from_text_generates_stable_ids() -> None:
    """相同文档与模型候选重复抽取时应生成相同实体、关系和事件 ID。"""

    first = extract_from_text(_chunks()[:1] + _chunks()[1:], _FakeLLM(), _FakeSelector())[0]
    second = extract_from_text(_chunks(), _FakeLLM(), _FakeSelector())[0]

    assert [item.id for item in first.entities] == [item.id for item in second.entities]
    assert [item.id for item in first.relations] == [item.id for item in second.relations]
    assert [item.id for item in first.events] == [item.id for item in second.events]


def test_extract_from_text_calls_persistence_callback_after_each_chunk() -> None:
    """每个 Chunk 完成后应立即回调持久化逻辑，再继续处理下一段。"""

    completed: list[tuple[str, int, int]] = []

    def record(graph: Graph, index: int, total: int) -> None:
        """记录逐 Chunk 回调的来源 ID、当前序号和总数。"""

        completed.append((str(graph.metadata.chunk_id), index, total))

    extract_from_text(
        _chunks(),
        _FakeLLM(),
        _FakeSelector(),
        on_graph_completed=record,
    )

    assert completed == [("c1", 1, 2), ("c2", 2, 2)]


def test_text_pipeline_writes_jsonl_and_checkpoint_per_chunk(tmp_path: Any, capsys: Any) -> None:
    """文本管线应逐行保存 Graph，并输出每个 Chunk 的完成日志。"""

    output_path = tmp_path / "stage_04_text_extraction.json"

    graphs = extract_text_chunks_to_file(
        _chunks(),
        output_path,
        llm_client=_FakeLLM(),
        schema_selector=_FakeSelector(),
    )

    journal_path = output_path.with_suffix(".jsonl")
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(lines) == 2
    assert len(graphs) == 2
    assert result["_status"] == "completed"
    assert result["statistics"]["completed_text_chunk_count"] == 2
    assert "[1/2] Chunk c1 已写入" in capsys.readouterr().out


def test_streaming_selector_resolves_schema_inside_chunk_loop() -> None:
    """新版选择器应只建一次文档池，并在每段抽取前即时选择局部 Schema。"""

    selector = _StreamingFakeSelector()
    completed: list[str] = []

    def record(graph: Graph, index: int, total: int) -> None:
        """记录 Graph 完成顺序以对比局部 Schema 选择顺序。"""

        completed.append(str(graph.metadata.chunk_id))

    extract_from_text(
        _chunks(),
        _FakeLLM(),
        selector,
        on_graph_completed=record,
    )

    assert selector.document_calls == 1
    assert selector.chunk_calls == ["c1", "c2"]
    assert completed == ["c1", "c2"]


def test_text_pipeline_resumes_without_reprocessing_completed_chunks(tmp_path: Any) -> None:
    """再次启动文本管线时应读取 JSONL，并跳过已经持久化的 Chunk。"""

    output_path = tmp_path / "stage_04_text_extraction.json"
    extract_text_chunks_to_file(
        _chunks(),
        output_path,
        llm_client=_FakeLLM(),
        schema_selector=_FakeSelector(),
    )
    resumed_llm = _FakeLLM()

    graphs = extract_text_chunks_to_file(
        _chunks(),
        output_path,
        llm_client=resumed_llm,
        schema_selector=_FakeSelector(),
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert resumed_llm.calls == 0
    assert len(graphs) == 2
    assert result["statistics"]["graph_count"] == 2
    assert len(output_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_init_extractor_dispatches_all_text_chunks_to_extract_from_text(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    """统一入口应初始化后通过 extract 将全部文本 Chunk 交给 extract_from_text。"""

    recorded: dict[str, Any] = {}

    def fake_extract_from_text(chunks: Any, llm_client: Any, **kwargs: Any) -> list[Graph]:
        """记录统一调度器传递的 Chunk、客户端和文本输出路径。"""

        recorded["chunk_ids"] = [str(chunk["id"]) for chunk in chunks]
        recorded["llm_client"] = llm_client
        recorded["output_path"] = kwargs.get("output_path")
        return [Graph() for _ in chunks]

    monkeypatch.setattr("src.extractors.text_extractor.extract_from_text", fake_extract_from_text)
    llm_client = object()
    extractor = InitExtractor(llm_client=llm_client, vlm_client=object(), show_progress=False)

    results = extractor.extract(_chunks(), output_dir=tmp_path)

    assert recorded["chunk_ids"] == ["c2", "c1"]
    assert recorded["llm_client"] is llm_client
    assert recorded["output_path"] == tmp_path / "stage_04_text_extraction.json"
    assert len(results["text"]) == 2
