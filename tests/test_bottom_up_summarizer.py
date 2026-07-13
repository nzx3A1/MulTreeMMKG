"""自底向上摘要器测试。"""
from __future__ import annotations

from threading import Lock
from time import sleep
from typing import Any

from src.summarizer import bottom_up_summarizer as summarizer_module
from src.summarizer.bottom_up_summarizer import SummaryModelRouter, summarize_document_tree


TEST_CONCEPT_CATEGORIES = ("构造地质", "沉积学", "油气地质", "地球化学")
"""单元测试使用的 ConceptCategory 候选，避免连接真实 Neo4j。"""


def test_load_concept_categories_queries_neo4j_once_and_deduplicates(monkeypatch: Any) -> None:
    """总结任务应使用指定 Cypher 读取类别，并清理重复名称。"""

    class FakeSession:
        """模拟 Neo4j 会话并记录只读查询。"""

        def __enter__(self) -> "FakeSession":
            """返回当前测试会话。"""

            return self

        def __exit__(self, *args: Any) -> None:
            """结束测试会话，不执行额外操作。"""

        def run(self, query: str) -> list[dict[str, str]]:
            """校验查询文本并返回带重复项的候选名称。"""

            assert "MATCH (n:ConceptCategory)" in query
            assert "WHERE n.name IS NOT NULL" in query
            assert "RETURN n.name AS name" in query
            return [{"name": "构造地质"}, {"name": "沉积学"}, {"name": "构造地质"}]

    class FakeDriver:
        """模拟 Neo4j 驱动的连接检查、会话创建与关闭。"""

        def verify_connectivity(self) -> None:
            """模拟成功连接数据库。"""

        def session(self, *, database: str) -> FakeSession:
            """确认使用 Schema 数据库并返回测试会话。"""

            assert database == summarizer_module.neo4j_settings.schema_db.database
            return FakeSession()

        def close(self) -> None:
            """模拟关闭驱动。"""

    monkeypatch.setattr(summarizer_module.GraphDatabase, "driver", lambda *args, **kwargs: FakeDriver())

    assert summarizer_module._load_concept_categories() == ("构造地质", "沉积学")


class FakeSummaryClient:
    """记录模型请求并按调用顺序返回可预测摘要的测试替身。"""

    def __init__(self) -> None:
        """初始化请求记录，便于断言父章节使用了子章节摘要。"""

        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """保存请求并返回不同编号的非空摘要。"""

        self.calls.append(messages)
        return (
            f'{{"summary":"summary-{len(self.calls)}","schemaKeys":[],"ConceptCategories":'
            '["构造地质","沉积学","油气地质"]}'
        )


def test_summarize_document_tree_adds_section_and_document_summaries() -> None:
    """叶子先使用多模态 Chunk，父章节与全文随后聚合已生成的子摘要。"""

    stage02 = {
        "_stage": 2,
        "document": {
            "title": "Paper",
            "sections": [
                {
                    "id": "1",
                    "title": "Parent",
                    "order": 0,
                    "chunks": [],
                    "children": [
                        {
                            "id": "1.1",
                            "title": "Leaf",
                            "order": 0,
                            "chunks": [
                                {"id": "text-1", "order": 0, "modality": "text", "text": "source text"},
                                {"id": "image-1", "order": 1, "modality": "image", "caption": "figure caption", "image_path": ["data:image/png;base64,AA=="]},
                            ],
                            "children": [],
                        }
                    ],
                }
            ],
        },
    }

    client = FakeSummaryClient()
    result = summarize_document_tree(
        stage02,
        llm_client=client,
        show_progress=False,
        concept_categories=TEST_CONCEPT_CATEGORIES,
    )
    leaf = result["document"]["sections"][0]["children"][0]
    parent = result["document"]["sections"][0]

    assert "summary" not in stage02["document"]
    assert leaf["summary"] == "summary-1"
    assert parent["summary"] == "summary-2"
    assert result["document"]["summary"] == "summary-3"
    assert any(part["type"] == "image_url" for part in client.calls[0][0]["content"])
    assert "石油地质" in client.calls[0][0]["content"][0]["text"]
    assert "summary-1" in client.calls[1][0]["content"][0]["text"]
    assert "summary-2" in client.calls[2][0]["content"][0]["text"]
    assert "一级章节摘要" in client.calls[2][0]["content"][0]["text"]
    assert '"构造地质"' in client.calls[0][0]["content"][0]["text"]


def test_summarize_document_tree_marks_empty_section_without_calling_model() -> None:
    """没有 Chunk 和子章节的空章节也应获得摘要字段，但不浪费模型调用。"""

    client = FakeSummaryClient()
    result = summarize_document_tree({
        "document": {"title": "Paper", "sections": [{"id": "1", "title": "Empty", "chunks": [], "children": []}]}
    }, llm_client=client, show_progress=False, concept_categories=TEST_CONCEPT_CATEGORIES)

    assert result["document"]["sections"][0]["summary"] == "本章节未包含可用于生成摘要的内容。"
    assert result["document"]["sections"][0]["ConceptCategories"] == []
    assert result["document"]["summary"] == "summary-1"
    assert len(client.calls) == 1


def test_summarize_document_tree_adds_only_whitelisted_schema_keys() -> None:
    """结构化响应中的 schemaKeys 应去重，并过滤模型生成的非白名单词。"""

    class StructuredClient:
        """返回包含合法、重复和非法 schemaKeys 的结构化测试响应。"""

        def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            """模拟模型按约定返回 JSON 对象。"""

            return '{"summary":"研究储层裂缝特征。","schemaKeys":["储层/储集层","裂缝","裂缝","非法词"],"ConceptCategories":["构造地质","沉积学","构造地质","非法类别","油气地质","地球化学"]}'

    result = summarize_document_tree({
        "document": {
            "title": "Paper",
            "sections": [{"id": "1", "title": "储层", "chunks": [{"id": "t", "modality": "text", "text": "裂缝"}], "children": []}],
        }
    }, llm_client=StructuredClient(), show_progress=False, concept_categories=TEST_CONCEPT_CATEGORIES)

    section = result["document"]["sections"][0]
    assert section["summary"] == "研究储层裂缝特征。"
    assert section["schemaKeys"] == ["储层/储集层", "裂缝"]
    assert section["ConceptCategories"] == ["构造地质", "沉积学", "油气地质"]


def test_summarize_document_tree_runs_sections_serially() -> None:
    """同一层级的章节应逐个执行，全文摘要必须等待它们全部结束。"""

    class ConcurrentClient:
        """通过短暂阻塞记录调用峰值，用于验证线程池的实际并发度。"""

        def __init__(self) -> None:
            """初始化线程安全的并发计数器。"""

            self._lock = Lock()
            self.active = 0
            self.max_active = 0

        def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            """模拟模型耗时调用，并返回非空摘要。"""

            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            sleep(0.03)
            with self._lock:
                self.active -= 1
            return '{"summary":"summary","schemaKeys":[],"ConceptCategories":["构造地质","沉积学","油气地质"]}'

    client = ConcurrentClient()
    result = summarize_document_tree({
        "document": {
            "title": "Paper",
            "sections": [
                {"id": str(index), "title": f"Section {index}", "chunks": [{"id": f"t{index}", "modality": "text", "text": "content"}], "children": []}
                for index in range(3)
            ],
        }
    }, llm_client=client, show_progress=False, concept_categories=TEST_CONCEPT_CATEGORIES)

    assert client.max_active == 1
    assert result["document"]["summary"] == "summary"


def test_summary_model_router_disables_thinking_only_for_m3_text_requests() -> None:
    """M3 文字请求应关闭 thinking，含图片请求仍交由视觉客户端处理。"""

    class RecordingClient:
        """记录路由器转发的参数，避免发起真实模型调用。"""

        def __init__(self) -> None:
            """初始化调用记录。"""

            self.calls: list[dict[str, Any]] = []

        def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            """保存请求参数并返回测试文本。"""

            self.calls.append(kwargs)
            return "summary"

    text_client = RecordingClient()
    vision_client = RecordingClient()
    router = SummaryModelRouter(text_client, vision_client, text_model="MiniMax-M3")

    router.chat([{"role": "user", "content": [{"type": "text", "text": "content"}]}])
    router.chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}])

    assert text_client.calls[0]["extra_body"]["thinking"] == {"type": "disabled"}
    assert vision_client.calls == [{}]


def test_summarize_document_tree_retries_when_model_returns_only_thinking() -> None:
    """仅含 think 标签的响应不应中断任务，应自动重试并使用重试结果。"""

    class RetryingClient:
        """首个调用模拟只返回思考内容，后续调用返回有效摘要。"""

        def __init__(self) -> None:
            """初始化调用计数。"""

            self.call_count = 0

        def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            """按调用次数构造模型的异常空响应与正常响应。"""

            self.call_count += 1
            return (
                "<think>internal reasoning</think>"
                if self.call_count == 1
                else '{"summary":"valid summary","schemaKeys":[],"ConceptCategories":["构造地质","沉积学","油气地质"]}'
            )

    client = RetryingClient()
    result = summarize_document_tree({
        "document": {
            "title": "Paper",
            "sections": [{"id": "1", "title": "Section", "chunks": [{"id": "text", "modality": "text", "text": "content"}], "children": []}],
        }
    }, llm_client=client, show_progress=False, concept_categories=TEST_CONCEPT_CATEGORIES)

    assert client.call_count == 3
    assert result["document"]["sections"][0]["summary"] == "valid summary"
