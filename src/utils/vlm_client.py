"""多模态大模型客户端。

该模块以指定的 ``BaseModel.py`` 为蓝本，保留其 OpenAI 配置、限流、重试和 JSON 清洗
逻辑，并补齐本工程需要的本地图片编码与 OpenAI 兼容视觉消息构造。
"""
from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from config.model_config import OpenAIVLMCompatibleConfig, settings
from .llm_client import safe_json_loads
from .llm_rate_limiter import GLOBAL_LLM_RATE_LIMITER, qps_to_min_interval


class BaseModel:
    """指定脚本的多模态基类，封装视觉模型的 OpenAI 兼容请求。"""

    def __init__(
        self,
        config: Optional[OpenAIVLMCompatibleConfig] = None,
        client: Any = None,
        rate_limit_qps: Optional[float] = None,
    ) -> None:
        """加载视觉模型配置并保存可选的 SDK 注入客户端。"""

        self.config = config or settings.vlm
        self.client = client
        self.model_name_vl = self.config.model
        self.timeout = self.config.timeout_secs
        self.temperature = self.config.temperature
        configured_qps = float(os.getenv("VLM_RATE_LIMIT_QPS", os.getenv("LLM_RATE_LIMIT_QPS", "0")))
        effective_qps = configured_qps if rate_limit_qps is None else rate_limit_qps
        self.min_request_interval = qps_to_min_interval(float(effective_qps or 0))
        self.retry_attempts = max(1, int(os.getenv("VLM_RETRY_ATTEMPTS", os.getenv("LLM_RETRY_ATTEMPTS", "3"))))
        self.retry_backoff_base = float(os.getenv("VLM_RETRY_BACKOFF_BASE", os.getenv("LLM_RETRY_BACKOFF_BASE", "2")))

    def _get_client(self) -> Any:
        """延迟创建 OpenAI SDK 客户端，避免导入阶段触发服务依赖。"""

        if self.client is None:
            if not self.config.api_key:
                raise ValueError("VLM_API_KEY is required to call VLM service")
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.timeout,
            )
        return self.client

    def _rate_limit(self) -> None:
        """调用视觉模型前执行与文本模型共享的 QPS 限流。"""

        GLOBAL_LLM_RATE_LIMITER.wait(self.min_request_interval)

    def _strip_code_fences(self, content: str) -> str:
        """移除视觉模型 JSON 回答可能附带的 Markdown 代码围栏。"""

        text = str(content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def safe_json_loads(self, content: str) -> Any:
        """按指定 BaseModel 的容错策略解析结构化模型结果。"""

        return safe_json_loads(self._strip_code_fences(content))

    def chat_with_retry(self, messages: list[dict[str, Any]], task_name: str = "", **kwargs: Any) -> str:
        """发送视觉消息并在请求异常时按指数退避重试。"""

        model = kwargs.pop("model", self.model_name_vl)
        extra_body = {"enable_thinking": False}
        extra_body.update(kwargs.pop("extra_body", {}))
        request = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
            "timeout": kwargs.pop("timeout", self.timeout),
            "extra_body": extra_body,
            **kwargs,
        }
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                self._rate_limit()
                response = self._get_client().chat.completions.create(**request)
                text = (response.choices[0].message.content or "").strip()
                return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
            except Exception as exc:  # 服务端限流和临时网络错误均应重试。
                last_error = exc
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_backoff_base**attempt)
        label = f"（{task_name}）" if task_name else ""
        raise RuntimeError(f"VLM 请求失败{label}，已重试 {self.retry_attempts} 次：{last_error}") from last_error

    def call_openai_json(self, prompt: str, task_name: str = "") -> Any:
        """兼容指定 BaseModel 的文本 JSON 调用入口。"""

        content = self.chat_with_retry([{"role": "user", "content": prompt}], task_name=task_name)
        return self.safe_json_loads(content)


class VLMClient(BaseModel):
    """工程视觉客户端，在 BaseModel 请求逻辑上增加图片输入辅助方法。"""

    @staticmethod
    def image_to_data_url(image_path: str | Path) -> str:
        """将本地图片转为 OpenAI 兼容的 base64 data URL。"""

        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def normalize_image_ref(image: str | Path) -> str:
        """保持 URL 或 data URL 不变，并将本地路径编码为 data URL。"""

        value = str(image)
        return value if value.startswith(("http://", "https://", "data:image/")) else VLMClient.image_to_data_url(value)

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """发送任意 OpenAI 兼容视觉消息并返回文本响应。"""

        return self.chat_with_retry(messages, **kwargs)

    def describe_image(self, image_path: str | Path, prompt: str, **kwargs: Any) -> str:
        """携带一张图片和提示词调用视觉模型，返回原始文本结果。"""

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self.normalize_image_ref(image_path)}},
            ],
        }]
        return self.chat(messages, **kwargs)

    def chat_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any] | list[Any]:
        """请求并容错解析视觉模型的 JSON 响应。"""

        kwargs.setdefault("response_format", {"type": "json_object"})
        return self.safe_json_loads(self.chat(messages, **kwargs))
