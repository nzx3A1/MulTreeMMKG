"""阶段一基础设施冒烟测试。

覆盖数据模型、配置单例、JSON IO、日志和三类模型客户端的离线可用性。
这些测试不访问外部网络，便于在未配置真实 API Key 的环境中验证基础层。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from config.app_config import settings as app_settings
from config.model_config import settings as model_settings
from config.neo4j_config import settings as neo4j_settings
from config.schema_config import load_json_schema, settings as schema_settings
from model import (
    Entity,
    FormulaChunk,
    GraphEdge,
    GraphNode,
    ImageChunk,
    KnowledgeGraph,
    Paper,
    PipelineStage,
    Relation,
    Section,
    SourceModality,
    StageStatus,
    TableChunk,
    TextChunk,
)
from src.utils.embedding_client import EmbeddingClient
from src.utils.json_io import read_json, write_json
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger
from src.utils.vlm_client import VLMClient


class _FakeChatCompletions:
    """模拟 OpenAI chat.completions.create 返回结构。"""

    def create(self, **kwargs):
        content = '{"ok": true}' if kwargs.get("response_format") else "hello"
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAIClient:
    """组合 chat 测试端点。"""

    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


def _mock_embedding_requests():
    """模拟 requests.post 返回向量数据。"""

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "data": [{"embedding": [1.0] * 1024}]
    }
    return patch("src.utils.embedding_client.requests.post", return_value=mock_response)


def test_models_can_validate_and_serialize():
    """核心模型应能创建、校验并序列化为字典。"""

    paper = Paper(id="paper_001", sections=[Section(id="sec_1", title="绪论")])
    chunk = TextChunk(id="chunk_1", document_id=paper.id, section_id="sec_1", text="测试文本")
    entity = Entity(
        id="ent_1",
        name="砂岩储层",
        official_name="Sandstone Reservoir",
        type="Reservoir",
        type_zh="储层",
    )
    relation = Relation(
        id="rel_1",
        type="CONTAINS",
        official_name="contains",
        type_zh="包含",
        source_id=entity.id,
        target_id=chunk.id,
    )
    graph = KnowledgeGraph(
        nodes=[GraphNode(id=entity.id, type="Entity", name=entity.name)],
        edges=[GraphEdge(id="edge_1", type="SOURCE_FROM", source_id=entity.id, target_id=chunk.id)],
    )
    stage = PipelineStage(number=1, name="基础设施", status=StageStatus.SUCCESS)

    assert paper.to_dict()["sections"][0]["title"] == "绪论"
    assert chunk.modality == SourceModality.TEXT
    assert entity.to_dict()["official_name"] == "Sandstone Reservoir"
    assert entity.to_dict()["type_zh"] == "储层"
    assert relation.to_dict()["type_zh"] == "包含"
    assert graph.node_ids() == {"ent_1"}
    assert stage.status == StageStatus.SUCCESS


def test_chunks_are_modeled_by_modality():
    """不同模态应使用不同 Chunk 类承载各自字段。"""

    text_chunk = TextChunk(id="text_1", document_id="paper_001", text="正文内容")
    table_chunk = TableChunk(id="table_1", document_id="paper_001", markdown="| A | B |")
    image_chunk = ImageChunk(id="image_1", document_id="paper_001", image_path="images/fig1.png", caption="图1")
    formula_chunk = FormulaChunk(id="formula_1", document_id="paper_001", latex="E=mc^2", context="公式说明")

    assert text_chunk.modality == SourceModality.TEXT
    assert table_chunk.modality == SourceModality.TABLE
    assert image_chunk.modality == SourceModality.IMAGE
    assert formula_chunk.modality == SourceModality.FORMULA


def test_config_singletons_are_importable():
    """配置单例应可导入，并暴露项目路径、模型、Neo4j 和 schema 参数。"""

    assert app_settings.project_root.exists()
    assert model_settings.llm.model
    assert neo4j_settings.document_db.database == "petrommkg-document"
    entity_schema = load_json_schema(schema_settings.entity_schema_path)
    assert isinstance(entity_schema, dict)


def test_json_io_roundtrip_with_chinese(tmp_path):
    """JSON 工具应自动创建目录并保留中文字符。"""

    target = tmp_path / "nested" / "sample.json"
    write_json(target, {"name": "塔里木盆地", "values": [1, 2, 3]})

    assert read_json(target)["name"] == "塔里木盆地"
    assert "塔里木盆地" in target.read_text(encoding="utf-8")


def test_clients_support_injected_openai_compatible_client(tmp_path):
    """LLM/VLM/Embedding 客户端应支持注入假客户端以便离线测试。"""

    fake_client = _FakeOpenAIClient()
    llm = LLMClient(client=fake_client)

    assert llm.chat([{"role": "user", "content": "hi"}]) == "hello"
    assert llm.chat_json([{"role": "user", "content": "json"}]) == {"ok": True}

    with _mock_embedding_requests():
        embedding = EmbeddingClient()
        vectors = embedding.encode(["a", "b"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 1024

    assert EmbeddingClient.cosine_similarity([1, 0], [1, 0]) == 1.0

    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert VLMClient.image_to_data_url(image_path).startswith("data:image/png;base64,")


def test_logger_can_be_created():
    """日志工具应能返回可调用的 logger 对象。"""

    logger = get_logger("stage01")
    assert hasattr(logger, "info")
