"""VLM 客户端封装。

支持把本地图片、URL 或已编码 data URL 传入 OpenAI 兼容多模态接口，返回模型
对图片的描述或结构化抽取文本。
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Optional

from config.model_config import OpenAIVLMCompatibleConfig, settings
from .llm_client import LLMClient


class VLMClient:
    """视觉语言模型客户端。"""

    def __init__(self, config: Optional[OpenAIVLMCompatibleConfig] = None, client: Any = None) -> None:
        """创建 VLM 客户端，底层复用 LLMClient 的 OpenAI 兼容调用。"""

        self.config = config or settings.vlm
        self._llm = LLMClient(config=self.config, client=client)

    @staticmethod
    def image_to_data_url(image_path: str | Path) -> str:
        """将本地图片编码为 data URL。"""

        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def normalize_image_ref(image: str | Path) -> str:
        """规范化图片引用，URL/data URL 原样返回，本地路径转 data URL。"""

        value = str(image)
        if value.startswith(("http://", "https://", "data:image/")):
            return value
        return VLMClient.image_to_data_url(value)

    def describe_image(self, image_path: str | Path, prompt: str, **kwargs: Any) -> str:
        """给定图像和提示词，返回模型描述或抽取结果。"""

        image_url = self.normalize_image_ref(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        return self._llm.chat(messages, **kwargs)
