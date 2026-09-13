"""验证 LLM 客户端针对 CUP DeepSeek 的配置隔离和请求参数。"""
from __future__ import annotations

from config.model_config import OpenAICompatibleConfig
from src.utils.llm_client import LLMClient


def test_cup_deepseek_is_llm_only_default() -> None:
    """CUP DeepSeek 默认配置应使用独立端点与环境变量注入密钥。"""

    config = OpenAICompatibleConfig()

    assert config.base_url == "https://assistant.cup.edu.cn/api/v1"
    assert config.model == "deepseek-r1-cup"
    assert isinstance(config.api_key, str)
    assert config.enable_thinking is True


def test_cup_deepseek_uses_chat_template_thinking_flag() -> None:
    """CUP DeepSeek 的思考参数必须放入 chat_template_kwargs。"""

    client = LLMClient(
        config=OpenAICompatibleConfig(api_key="test-key", enable_thinking=True),
        client=object(),
    )

    assert client._thinking_extra_body("deepseek-r1-cup") == {
        "chat_template_kwargs": {"thinking": True}
    }


def test_other_text_models_keep_existing_thinking_contract() -> None:
    """非 CUP DeepSeek 模型应继续使用既有顶层开关格式。"""

    client = LLMClient(
        config=OpenAICompatibleConfig(api_key="test-key", enable_thinking=False),
        client=object(),
    )

    assert client._thinking_extra_body("qwen3.8-flash") == {"enable_thinking": False}
