"""文本大模型客户端。

实现沿用提供脚本的重试、限流、思考标签清理与宽容 JSON 解析策略，并兼容工程原有的
``chat(messages)``、``chat_json(messages)`` 调用方式。
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional, Sequence

from config.model_config import OpenAICompatibleConfig, settings
from .llm_rate_limiter import GLOBAL_LLM_RATE_LIMITER, qps_to_min_interval


def prompt_to_text(prompt: Any) -> str:
    """将字符串或结构化 Prompt 统一转换为模型可读的文本。"""

    return prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False, indent=2)


def _strip_code_fences(content: str) -> str:
    """去除模型返回值最外层的 Markdown 代码围栏。"""

    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_fragment(content: str) -> str:
    """从含有说明文字的模型输出中截取最可能的 JSON 对象或数组。"""

    text = _strip_code_fences(content)
    if not text or text[0] in "[{":
        return text
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return text
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    return text[start : end + 1].strip() if end > start else text


def safe_json_loads(content: str) -> Any:
    """安全解析模型 JSON；无法解析时返回空对象供上游按无结果处理。"""

    try:
        return json.loads(_extract_json_fragment(content))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


class LLMClient:
    """OpenAI 兼容的文本模型客户端，提供串行、限流和自动重试请求。"""

    def __init__(
        self,
        config: Optional[OpenAICompatibleConfig] = None,
        client: Any = None,
        rate_limit_qps: Optional[float] = None,
    ) -> None:
        """保存模型配置和可选 SDK 注入客户端，SDK 在首次调用时才创建。"""

        self.config = config or settings.llm
        self._client = client
        configured_qps = float(os.getenv("LLM_RATE_LIMIT_QPS", "0"))
        effective_qps = configured_qps if rate_limit_qps is None else rate_limit_qps
        self.min_request_interval = qps_to_min_interval(float(effective_qps or 0))
        self.retry_attempts = max(1, int(os.getenv("LLM_RETRY_ATTEMPTS", "3")))
        self.retry_backoff_base = float(os.getenv("LLM_RETRY_BACKOFF_BASE", "2"))
        self.extra_throttle_sec = float(os.getenv("LLM_EXTRA_THROTTLE_SEC", "0"))

    def _get_client(self) -> Any:
        """延迟创建 OpenAI SDK 客户端，便于测试注入假客户端。"""

        if self._client is None:
            if not self.config.api_key:
                raise ValueError("LLM_API_KEY or OPENAI_API_KEY is required to call LLM service")
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_secs,
            )
        return self._client

    def _rate_limit(self) -> None:
        """在每次请求开始前执行全局 QPS 限流。"""

        GLOBAL_LLM_RATE_LIMITER.wait(self.min_request_interval)

    @staticmethod
    def _thinking_disabled_extra_body(model: str) -> dict[str, Any]:
        """生成与提供脚本一致的默认关闭思考参数。"""

        _ = model
        return {"enable_thinking": False}

    @staticmethod
    def _normalize_messages(prompt_or_messages: Any) -> list[dict[str, Any]]:
        """兼容原有 messages 调用与提供脚本的单个 Prompt 调用。"""

        if isinstance(prompt_or_messages, Sequence) and not isinstance(prompt_or_messages, (str, bytes)):
            return [dict(message) for message in prompt_or_messages]
        return [{"role": "user", "content": prompt_to_text(prompt_or_messages)}]

    def chat_with_retry(self, prompt: Any, task_name: str = "", **kwargs: Any) -> str:
        """按指数退避重试模型请求，并清理返回中的 ``<think>`` 内容。"""

        messages = self._normalize_messages(prompt)
        model = kwargs.pop("model", self.config.model)
        extra_body = self._thinking_disabled_extra_body(model)
        extra_body.update(kwargs.pop("extra_body", {}))
        request = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
            "timeout": kwargs.pop("timeout", self.config.timeout_secs),
            **kwargs,
        }
        if extra_body:
            request["extra_body"] = extra_body

        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                self._rate_limit()
                if self.extra_throttle_sec > 0:
                    time.sleep(self.extra_throttle_sec)
                response = self._get_client().chat.completions.create(**request)
                text = (response.choices[0].message.content or "").strip()
                return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
            except Exception as exc:  # 网络或服务端异常均按同一策略重试。
                last_error = exc
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_backoff_base**attempt)

        label = f"（{task_name}）" if task_name else ""
        raise RuntimeError(f"LLM 请求失败{label}，已重试 {self.retry_attempts} 次：{last_error}") from last_error

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """发送 OpenAI 兼容消息并返回清理后的纯文本响应。"""

        return self.chat_with_retry(messages, **kwargs)

    def call_json(self, prompt: Any, task_name: str = "") -> Any:
        """调用模型并以提供脚本的宽容方式解析 JSON 响应。"""

        return safe_json_loads(self.chat_with_retry(prompt, task_name=task_name))

    def call_openai_json(self, prompt: Any, task_name: str = "") -> Any:
        """保留 BaseModel 风格方法名，方便旧脚本迁移到本工程。"""

        return self.call_json(prompt, task_name=task_name)

    def chat_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any] | list[Any]:
        """请求并安全解析 JSON，模型未遵循格式时返回空对象。"""

        kwargs.setdefault("response_format", {"type": "json_object"})
        return safe_json_loads(self.chat(messages, **kwargs))


# 中文说明：旧抽取脚本可继续使用 LLMJsonClient 名称，无需改动调用点。
LLMJsonClient = LLMClient


class DisabledLLMClient:
    """用于只校验流程的空实现，避免触发真实模型请求。"""

    def chat(self, messages: Any, **kwargs: Any) -> str:
        """返回空 JSON 字符串以兼容文本调用接口。"""

        _ = messages, kwargs
        return "{}"

    def chat_json(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        """返回空对象以兼容结构化抽取接口。"""

        _ = messages, kwargs
        return {}
