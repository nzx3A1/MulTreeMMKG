"""Embedding 客户端封装。

为实体对齐、文本相似度计算等模块提供统一向量接口，并附带余弦相似度工具。
真实服务调用采用 requests 直接调用 Ollama ``/api/embed`` API。
"""
from __future__ import annotations

import logging
import math
from typing import Any, Iterable, List, Optional

import requests

from config.model_config import EmbeddingConfig, settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Embedding 模型客户端。"""

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        """创建 embedding 客户端。"""

        self.config = config or settings.embedding

    def encode(self, texts: List[str]) -> List[List[float]]:
        """使用 Ollama 批量接口将文本列表编码为向量列表。"""

        if not texts:
            return []
        valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
        if not valid_texts:
            return []
        return self._get_embeddings(valid_texts)

    def encode_one(self, text: str) -> List[float]:
        """编码单条文本。"""

        embeddings = self.encode([text])
        return embeddings[0] if embeddings else []

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """调用 Ollama API 获取一批文本的嵌入向量。"""

        payload = {"model": self.config.model, "input": texts}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                self.config.base_url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout_secs,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                logger.error(
                    "Ollama 向量接口返回数量不匹配：期望 %s，实际 %s",
                    len(texts),
                    len(embeddings) if isinstance(embeddings, list) else "无效格式",
                )
                return []
            if any(not isinstance(embedding, list) for embedding in embeddings):
                logger.error("Ollama 向量接口响应中包含无效 embedding")
                return []

            dimensions = {len(embedding) for embedding in embeddings}
            if len(dimensions) != 1:
                logger.error("Ollama 同一批次返回了不同维度的向量：%s", dimensions)
                return []
            actual_dimensions = dimensions.pop()
            if self.config.dimensions and actual_dimensions != self.config.dimensions:
                logger.error(
                    "向量维度不匹配：期望 %s，实际 %s",
                    self.config.dimensions,
                    actual_dimensions,
                )
                return []
            return embeddings
        except requests.HTTPError as exc:
            response_text = exc.response.text[:1000] if exc.response is not None else ""
            logger.error("Ollama 获取嵌入向量失败: %s；服务端响应: %s", exc, response_text)
        except (requests.RequestException, ValueError) as exc:
            logger.error("Ollama 获取嵌入向量失败: %s", exc)

        return []

    @staticmethod
    def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
        """计算两个向量的余弦相似度。"""

        left_values = list(left)
        right_values = list(right)
        if len(left_values) != len(right_values) or not left_values:
            return 0.0
        dot = sum(a * b for a, b in zip(left_values, right_values))
        left_norm = math.sqrt(sum(a * a for a in left_values))
        right_norm = math.sqrt(sum(b * b for b in right_values))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)
