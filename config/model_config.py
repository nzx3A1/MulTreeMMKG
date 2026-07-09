"""模型服务配置。

本文件集中保存 LLM、VLM、Embedding 和 MinerU 的当前开发期配置。现阶段按用户
要求保留明文默认值，同时仍允许通过环境变量或 .env 覆盖这些默认值。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


def _load_env_file(path: Path) -> Dict[str, str]:
    """读取简单的 .env 文件，并将未设置的键注入当前进程环境。"""

    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_load_env_file(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """OpenAI 兼容聊天模型配置。"""

    base_url: str = "https://api.minimaxi.com/v1"
    api_key: str = "sk-cp-bdhae6J8LMsO4lgK9zH4SDcTUkjRPkZqB9waNC6unS--GIKH99HESOSJNjhWZjemIXbs-97DDOotP8O46oV5tfyjru0HYaf6f9372mnZK2ireENH0M1niUs"
    model: str = "MiniMax-M3"
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout_secs: float = 120.0

@dataclass(frozen=True)
class OpenAIVLMCompatibleConfig:
    """OpenAI 兼容视觉模型配置。"""

    base_url: str = "https://ws-qxohtk9wmrvcddu0.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    api_key: str = "sk-ws-H.RPRYPMD.MflL.MEYCIQDva8op7aHb5DZ0cdM1B2ATcEq39cn6eSMA1-gYdIxbagIhAOYtVkeCUlkySXzti__apd4Mpq6N6t06ugZ9sH1UmJmo"
    model: str = "qwen3.6-27b"
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout_secs: float = 120.0

@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding 模型配置。"""

    base_url: str = "https://api.siliconflow.cn/v1/embeddings"
    api_key: str = "sk-lwctfhzpjhwclurfgdtpkwynqkawporxgvrhkjrtbuujayij"
    model: str = "BAAI/bge-large-zh-v1.5"
    dimensions: int | None = 1024
    batch_size: int = 32
    timeout_secs: float = 60.0


@dataclass(frozen=True)
class MinerUConfig:
    """MinerU 云端接口配置。"""

    token: str = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1OTIwMDEwNiIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc4MDMwMDU5MywiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTgzNzMyOTg1NzMiLCJvcGVuSWQiOm51bGwsInV1aWQiOiI0MzU0Y2UwOS0zMzBmLTRlZTYtOWM0NS0zMzkxOGQxNWVmOWIiLCJlbWFpbCI6IiIsImV4cCI6MTc4ODA3NjU5M30.JTbfGAAmQHprsWUJ0Dk1FnlR_93_Cb1uaMe9hjrfUMDR8ou9DfbG9oCWlJN1AoanMaa0TJGqvdyZ0Pnf_0yHZw"
    batch_url: str = "https://mineru.net/api/v4/file-urls/batch"
    timeout_secs: float = 120.0


@dataclass(frozen=True)
class ModelSettings:
    """模型相关配置聚合。"""

    llm: OpenAICompatibleConfig
    vlm: OpenAIVLMCompatibleConfig
    embedding: EmbeddingConfig
    mineru: MinerUConfig


def load_model_settings() -> ModelSettings:
    """从环境变量加载模型配置，未设置时使用文件中的明文默认值。"""

    llm = OpenAICompatibleConfig(
        base_url=os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", OpenAICompatibleConfig.base_url)),
        api_key=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", OpenAICompatibleConfig.api_key)),
        model=os.getenv("LLM_MODEL", OpenAICompatibleConfig.model),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", str(OpenAICompatibleConfig.max_tokens))),
        timeout_secs=float(os.getenv("LLM_TIMEOUT_SECS", "120")),
    )
    vlm = OpenAIVLMCompatibleConfig(
        base_url=os.getenv("VLM_BASE_URL", OpenAIVLMCompatibleConfig.base_url),
        api_key=os.getenv("VLM_API_KEY", OpenAIVLMCompatibleConfig.api_key),
        model=os.getenv("VLM_MODEL", OpenAIVLMCompatibleConfig.model),
        temperature=float(os.getenv("VLM_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("VLM_MAX_TOKENS", str(OpenAIVLMCompatibleConfig.max_tokens))),
        timeout_secs=float(os.getenv("VLM_TIMEOUT_SECS", "120")),
    )
    embedding = EmbeddingConfig(
        base_url=os.getenv("EMBEDDING_BASE_URL", EmbeddingConfig.base_url),
        api_key=os.getenv("EMBEDDING_API_KEY", EmbeddingConfig.api_key),
        model=os.getenv("EMBEDDING_MODEL", EmbeddingConfig.model),
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS")) if os.getenv("EMBEDDING_DIMENSIONS") else EmbeddingConfig.dimensions,
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        timeout_secs=float(os.getenv("EMBEDDING_TIMEOUT_SECS", "60")),
    )
    mineru = MinerUConfig(
        token=os.getenv("MINERU_TOKEN", MinerUConfig.token),
        batch_url=os.getenv("MINERU_BATCH_URL", MinerUConfig.batch_url),
        timeout_secs=float(os.getenv("MINERU_TIMEOUT_SECS", "120")),
    )
    return ModelSettings(llm=llm, vlm=vlm, embedding=embedding, mineru=mineru)


settings = load_model_settings()
