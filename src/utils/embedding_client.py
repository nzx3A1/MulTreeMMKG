"""Embedding 客户端封装。

为实体对齐、文本相似度计算等模块提供统一向量接口，并附带余弦相似度工具。
真实服务调用采用 requests 直接调用 SiliconFlow embeddings API。
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
        """将文本列表编码为向量列表。"""

        if not texts:
            return []

        results: List[List[float]] = []
        for text in texts:
            embedding = self._get_embedding(text)
            if embedding is not None:
                results.append(embedding)
        return results

    def encode_one(self, text: str) -> List[float]:
        """编码单条文本。"""

        embedding = self._get_embedding(text)
        return embedding if embedding is not None else []

    def _get_embedding(self, text: str) -> List[float] | None:
        """调用 SiliconFlow API 获取文本的嵌入向量。"""

        if not text or not isinstance(text, str):
            return None

        payload = {
            "model": self.config.model,
            "input": text,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.config.base_url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout_secs,
            )
            response.raise_for_status()
            data = response.json()
            rows = data.get("data", [])
            if not rows or not isinstance(rows[0].get("embedding"), list):
                logger.error("向量接口响应中缺少有效的 embedding 数据")
                return None

            embedding = rows[0]["embedding"]
            if self.config.dimensions and len(embedding) != self.config.dimensions:
                logger.error(
                    "向量维度不匹配：期望 %s，实际 %s",
                    self.config.dimensions,
                    len(embedding),
                )
                return None
            return embedding
        except requests.HTTPError as exc:
            response_text = exc.response.text[:1000] if exc.response is not None else ""
            logger.error("获取嵌入向量失败: %s；服务端响应: %s", exc, response_text)
        except (requests.RequestException, ValueError) as exc:
            logger.error("获取嵌入向量失败: %s", exc)

        return None

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

