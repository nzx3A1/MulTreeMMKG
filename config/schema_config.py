"""知识图谱 Schema 配置。

本模块负责定位和加载 schema 目录下的实体、关系、事件定义。阶段一只提供
统一入口；具体领域白名单会在后续 Schema 阶段继续补全。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict

from .model_config import PROJECT_ROOT, _load_env_file


_load_env_file(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class SchemaSettings:
    """Schema 加载与约束配置。"""

    schema_dir: Path = PROJECT_ROOT / "schema"
    version: str = "0.1.0"
    allow_open_relations: bool = False
    entity_schema_file: str = "entity_schema.json"
    relation_schema_file: str = "relation_schema.json"
    event_schema_file: str = "event_schema.json"

    @property
    def entity_schema_path(self) -> Path:
        """返回实体 schema 文件路径。"""

        return self.schema_dir / self.entity_schema_file

    @property
    def relation_schema_path(self) -> Path:
        """返回关系 schema 文件路径。"""

        return self.schema_dir / self.relation_schema_file

    @property
    def event_schema_path(self) -> Path:
        """返回事件 schema 文件路径。"""

        return self.schema_dir / self.event_schema_file


def load_json_schema(path: str | Path) -> Dict[str, Any]:
    """加载 JSON schema 文件。

    当前仓库的 schema 文件仍处于占位阶段，若文件不存在或暂时不是合法 JSON，
    返回带错误信息的空 schema，避免阶段一配置导入被后续阶段占位文件阻塞。
    """

    schema_path = Path(path)
    if not schema_path.exists():
        return {"_version": settings.version, "_missing": str(schema_path)}
    try:
        return json.loads(schema_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        return {"_version": settings.version, "_invalid": str(schema_path), "_error": str(exc)}


def load_schema_settings() -> SchemaSettings:
    """从环境变量加载 schema 配置。"""

    return SchemaSettings(
        schema_dir=Path(os.getenv("SCHEMA_DIR", str(PROJECT_ROOT / "schema"))),
        version=os.getenv("SCHEMA_VERSION", "0.1.0"),
        allow_open_relations=os.getenv("ALLOW_OPEN_RELATIONS", "false").lower() in {"1", "true", "yes"},
    )


settings = load_schema_settings()
