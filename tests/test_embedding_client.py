"""Embedding 客户端测试。

验证 EmbeddingClient 能正常初始化并调用 encode/encode_one 方法。
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 - 直接运行测试文件时注入项目根目录。
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


def _run_with_mock_requests(func) -> None:
    """直接运行测试文件时，为 embedding API 调用提供本地 mock。"""

    with patch("src.utils.embedding_client.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": [{"embedding": [1.0] * 1024}]}
        mock_post.return_value = mock_response
        func(mock_post)


def main() -> None:
    """直接点击运行本文件时，顺序执行测试并打印结果。"""

    print("开始运行 EmbeddingClient 测试...")
    test_embedding_client_can_be_created()
    print("1. 初始化测试通过")
    _run_with_mock_requests(test_embedding_client_encode)
    print("2. 模拟 encode 测试通过，返回 1024 维向量")
    _run_with_mock_requests(test_embedding_client_encode_one)
    print("3. 模拟 encode_one 测试通过，返回 1024 维向量")
    test_embedding_client_cosine_similarity()
    print("4. 余弦相似度测试通过")
    print("5. 开始真实 Embedding API 调用测试...")
    test_embedding_client_live_encode()
    print("6. 真实 Embedding API 调用测试通过")
    print("EmbeddingClient 测试全部完成")


if __name__ == "__main__":
    main()

    
