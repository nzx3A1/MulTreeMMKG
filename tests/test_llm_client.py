"""LLM 客户端测试。

验证 LLMClient 能正常初始化并调用 chat/chat_json 方法。
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from src.utils.llm_client import LLMClient


class _FakeChatCompletions:
    """模拟 OpenAI chat.completions.create 返回结构。"""

    def create(self, **kwargs):
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
