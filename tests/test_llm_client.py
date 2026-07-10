"""LLM 客户端测试。

验证 LLMClient 能正常初始化并调用 chat/chat_json 方法。
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 - 直接运行测试文件时注入项目根目录。
import pytest
from types import SimpleNamespace

from src.utils.llm_client import LLMClient


class _FakeChatCompletions:
    """模拟 OpenAI chat.completions.create 返回结构。"""

    def __init__(self):
        """保存最后一次请求参数，供默认思考开关测试使用。"""

        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        content = '{"ok": true}' if kwargs.get("response_format") else "hello"
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAIClient:
    """模拟 OpenAI 客户端。"""

    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


def test_llm_client_can_be_created():
    """LLMClient 应能正常初始化。"""

    fake_client = _FakeOpenAIClient()
    llm = LLMClient(client=fake_client)
    assert llm is not None


def test_llm_client_chat():
    """LLMClient.chat 应能正常调用并返回文本。"""

    fake_client = _FakeOpenAIClient()
    llm = LLMClient(client=fake_client)

    response = llm.chat([{"role": "user", "content": "hi"}])
    assert response == "hello"
    request = fake_client.chat.completions.last_kwargs
    assert request["extra_body"]["thinking"] == {"type": "disabled"}


def test_llm_client_chat_json():
    """LLMClient.chat_json 应能正常调用并返回解析后的 JSON。"""

    fake_client = _FakeOpenAIClient()
    llm = LLMClient(client=fake_client)

    response = llm.chat_json([{"role": "user", "content": "json"}])
    assert response == {"ok": True}


@pytest.mark.live
def test_llm_client_live_chat():
    """LLMClient 应能调用真实 API 并返回响应。"""

    llm = LLMClient()
    response = llm.chat([{"role": "user", "content": "你好"}])
    assert isinstance(response, str)
    assert len(response) > 0


def main() -> None:
    """直接点击运行本文件时，顺序执行测试并打印结果。"""

    print("开始运行 LLMClient 测试...")
    test_llm_client_can_be_created()
    print("1. 初始化测试通过")
    test_llm_client_chat()
    print("2. 模拟 chat 调用测试通过，返回 hello")
    test_llm_client_chat_json()
    print("3. 模拟 chat_json 调用测试通过，返回 {'ok': True}")
    print("4. 开始真实 API 调用测试...")
    test_llm_client_live_chat()
    print("5. 真实 API 调用测试通过")
    print("LLMClient 测试全部完成")


if __name__ == "__main__":
    main()
