"""LLM 客户端封装。

客户端面向 OpenAI 兼容接口，提供普通 chat 和 JSON 输出两种调用方式。构造时可
注入测试用 client，因此基础测试不需要真实网络或 API Key。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from config.model_config import OpenAICompatibleConfig, settings


class LLMClient:
    """轻量 LLM 客户端，封装 OpenAI 兼容 chat/completions 调用。"""

    def __init__(self, config: Optional[OpenAICompatibleConfig] = None, client: Any = None) -> None:
        """创建客户端实例，client 参数用于测试或自定义 SDK 注入。"""

        self.config = config or settings.llm
        self._client = client

    def _get_client(self) -> Any:
        """延迟创建 OpenAI SDK 客户端，避免导入阶段触发网络依赖。"""

        if self._client is not None:
            return self._client
        if not self.config.api_key:
            raise ValueError("LLM_API_KEY or OPENAI_API_KEY is required to call LLM service")
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_secs,
        )
        return self._client

    @staticmethod
    def _thinking_disabled_extra_body(model: str) -> Dict[str, Any]:
        """为支持混合思考的兼容模型生成默认关闭思考的请求参数。"""

        normalized_model = model.lower()
        if normalized_model.startswith("minimax"):
            # MiniMax-M3 可真正关闭；M2.x 会接受但由服务端决定是否保留思考。
            return {"thinking": {"type": "disabled"}}
        if normalized_model.startswith("qwen"):
            # Qwen OpenAI 兼容接口要求把非标准参数放在 extra_body 中。
            return {"enable_thinking": False}
        return {}

    def chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """发送对话消息并返回模型文本输出，默认关闭支持该能力的模型思考。"""

        model = kwargs.pop("model", self.config.model)
        extra_body = self._thinking_disabled_extra_body(model)
        # 调用方显式传参优先，可按需重新开启思考或补充厂商特有参数。
        extra_body.update(kwargs.pop("extra_body", {}))
        request = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
            **kwargs,
        }
        if extra_body:
            request["extra_body"] = extra_body
        response = self._get_client().chat.completions.create(**request)
        return response.choices[0].message.content or ""

    def chat_json(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any] | List[Any]:
        """请求 JSON 格式输出并解析为 Python 对象。"""

        kwargs.setdefault("response_format", {"type": "json_object"})
        content = self.chat(messages, **kwargs)
        return json.loads(content)

