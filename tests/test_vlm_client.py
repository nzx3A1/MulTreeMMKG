"""VLM 客户端测试。

验证 VLMClient 能正常初始化并调用 describe_image 方法。
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace
from PIL import Image

from src.utils.vlm_client import VLMClient


class _FakeChatCompletions:
    """模拟 OpenAI chat.completions.create 返回结构。"""

    def create(self, **kwargs):
        message = SimpleNamespace(content="这是一张图片")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAIClient:
    """模拟 OpenAI 客户端。"""

    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


def test_vlm_client_can_be_created():
    """VLMClient 应能正常初始化。"""

    fake_client = _FakeOpenAIClient()
    vlm = VLMClient(client=fake_client)
    assert vlm is not None


def test_vlm_client_image_to_data_url(tmp_path):
    """VLMClient.image_to_data_url 应能将图片转换为 data URL。"""

    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    data_url = VLMClient.image_to_data_url(image_path)
    assert data_url.startswith("data:image/png;base64,")


def test_vlm_client_normalize_image_ref(tmp_path):
    """VLMClient.normalize_image_ref 应能规范化图片引用。"""

    assert VLMClient.normalize_image_ref("http://example.com/img.png").startswith("http://")
    assert VLMClient.normalize_image_ref("data:image/png;base64,xxx").startswith("data:image/")

    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    normalized = VLMClient.normalize_image_ref(image_path)
    assert normalized.startswith("data:image/png;base64,")


def test_vlm_client_describe_image(tmp_path):
    """VLMClient.describe_image 应能正常调用并返回描述。"""

    fake_client = _FakeOpenAIClient()
    vlm = VLMClient(client=fake_client)

    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    response = vlm.describe_image(image_path, "描述这张图片")
    assert response == "这是一张图片"


@pytest.mark.live
def test_vlm_client_live_describe_image(tmp_path):
    """VLMClient 应能调用真实 API 并返回图片描述。"""

    vlm = VLMClient()

    image_path = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(image_path)

    response = vlm.describe_image(image_path, "描述这张图片")
    assert isinstance(response, str)
    assert len(response) > 0
