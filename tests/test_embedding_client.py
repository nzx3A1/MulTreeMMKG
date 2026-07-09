"""Embedding 客户端测试。

验证 EmbeddingClient 能正常初始化并调用 encode/encode_one 方法。
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.utils.embedding_client import EmbeddingClient


def test_embedding_client_can_be_created():
    """EmbeddingClient 应能正常初始化。"""

    embedding = EmbeddingClient()
    assert embedding is not None


def test_embedding_client_encode(mock_requests_post):
    """EmbeddingClient.encode 应能正常调用并返回向量列表。"""

    embedding = EmbeddingClient()
    vectors = embedding.encode(["a", "b"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 1024


def test_embedding_client_encode_one(mock_requests_post):
    """EmbeddingClient.encode_one 应能正常调用并返回单向量。"""

    embedding = EmbeddingClient()
    vector = embedding.encode_one("test")
    assert len(vector) == 1024


def test_embedding_client_cosine_similarity():
    """EmbeddingClient.cosine_similarity 应能正确计算余弦相似度。"""

    assert EmbeddingClient.cosine_similarity([1, 0], [1, 0]) == 1.0
    assert EmbeddingClient.cosine_similarity([1, 0], [0, 1]) == 0.0


@pytest.fixture
def mock_requests_post():
    """模拟 requests.post 返回向量数据。"""

    with patch("src.utils.embedding_client.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [{"embedding": [1.0] * 1024}]
        }
        mock_post.return_value = mock_response
        yield mock_post


@pytest.mark.live
def test_embedding_client_live_encode():
    """EmbeddingClient 应能调用真实 API 并返回向量。"""

    embedding = EmbeddingClient()
    vectors = embedding.encode(["测试文本"])
    assert isinstance(vectors, list)
    assert len(vectors) == 1
    assert isinstance(vectors[0], list)
    assert len(vectors[0]) > 0

    