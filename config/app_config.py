"""应用级全局配置。

settings 是项目通用配置单例，集中管理目录、日志、并发、重试和阶段开关等
基础参数。下游模块可通过 `from config.app_config import settings` 引用。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from .model_config import PROJECT_ROOT, _load_env_file
from . import model_config, neo4j_config


_load_env_file(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class AppSettings:
    """应用运行配置。"""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_pdf_dir: Path = PROJECT_ROOT / "data" / "raw_pdf"
    mineru_output_dir: Path = PROJECT_ROOT / "data" / "mineru_output"
    output_dir: Path = PROJECT_ROOT / "output"
    log_dir: Path = PROJECT_ROOT / "logs"
    schema_dir: Path = PROJECT_ROOT / "schema"
    prompt_dir: Path = PROJECT_ROOT / "prompts"
    default_encoding: str = "utf-8"
    log_level: str = "INFO"
    max_workers: int = 4
    retry_times: int = 3
    timeout_secs: float = 120.0
    enabled_stages: Dict[int, bool] = field(default_factory=lambda: {stage: True for stage in range(1, 13)})
    model: model_config.ModelSettings = field(default_factory=model_config.load_model_settings)
    neo4j: neo4j_config.Neo4jSettings = field(default_factory=neo4j_config.load_neo4j_settings)


def load_app_settings() -> AppSettings:
    """从环境变量加载应用配置并创建必要目录。"""

    enabled_stages = {stage: os.getenv(f"STAGE_{stage:02d}_ENABLED", "true").lower() in {"1", "true", "yes"} for stage in range(1, 13)}
    app_settings = AppSettings(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        max_workers=int(os.getenv("MAX_WORKERS", "4")),
        retry_times=int(os.getenv("RETRY_TIMES", "3")),
        timeout_secs=float(os.getenv("APP_TIMEOUT_SECS", "120")),
        enabled_stages=enabled_stages,
    )
    for directory in (
        app_settings.data_dir,
        app_settings.raw_pdf_dir,
        app_settings.mineru_output_dir,
        app_settings.output_dir,
        app_settings.log_dir,
        app_settings.schema_dir,
        app_settings.prompt_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return app_settings


settings = load_app_settings()
