"""Stage 05 实体映射缓存：提供进程内去重和跨运行持久化能力。"""
from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Mapping

from src.utils.json_io import read_json, write_json

from .models import SchemaSnapshot
from .semantic_retriever import build_schema_embedding_text


CACHE_VERSION = 1
logger = logging.getLogger(__name__)


def _normalize_component(value: Any) -> str:
    """归一化映射键字段，消除大小写、空白和 Unicode 表示差异。"""

    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return " ".join(text.split())


def build_entity_mapping_key(entity: Mapping[str, Any]) -> str:
    """根据名称、原始类型和中文类型生成稳定的实体映射键。"""

    values = (
        entity.get("name") or entity.get("official_name") or "",
        entity.get("raw_type", entity.get("type")),
        entity.get("type_zh") or "",
    )
    return json.dumps(
        [_normalize_component(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_mapping_cache_namespace(
    snapshot: SchemaSnapshot,
    rules_path: str | Path,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """根据 Schema、规则文件和运行参数生成缓存命名空间。"""

    try:
        rules_content = Path(rules_path).read_bytes()
    except OSError:
        rules_content = b""
    payload = {
        "version": CACHE_VERSION,
        "concepts": [build_schema_embedding_text(concept) for concept in snapshot.concepts],
        "rules_sha256": hashlib.sha256(rules_content).hexdigest(),
        "extra": dict(extra or {}),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class EntityMappingCache:
    """保存实体映射模板，支持当前进程缓存和 JSON 跨运行缓存。"""

    def __init__(self, path: str | Path | None, namespace: str) -> None:
        """加载指定命名空间的缓存；路径为 None 时只保留进程内缓存。"""

        self.path = Path(path) if path is not None else None
        self.namespace = namespace
        self.entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """读取持久化缓存，版本或命名空间不匹配时安全忽略旧数据。"""

        if self.path is None or not self.path.is_file():
            return
        try:
            payload = read_json(self.path)
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(payload, Mapping):
            return
        if payload.get("version") != CACHE_VERSION or payload.get("namespace") != self.namespace:
            return
        entries = payload.get("entries")
        if not isinstance(entries, Mapping):
            return
        self.entries = {
            str(key): dict(value)
            for key, value in entries.items()
            if isinstance(value, Mapping)
        }
        logger.info(
            "[实体缓存] 已加载 %s 条映射：%s",
            len(self.entries),
            self.path,
        )

    def get(self, key: str) -> dict[str, Any] | None:
        """按实体映射键读取缓存模板。"""

        value = self.entries.get(key)
        return dict(value) if value is not None else None

    def set(self, key: str, template: Mapping[str, Any]) -> None:
        """写入一个不包含实体原始字段的映射模板。"""

        self.entries[key] = dict(template)

    def save(self) -> None:
        """将当前缓存安全写入 JSON 文件；禁用持久化时不执行写入。"""

        if self.path is None:
            return
        write_json(
            self.path,
            {
                "version": CACHE_VERSION,
                "namespace": self.namespace,
                "entries": self.entries,
            },
        )
        logger.info("[实体缓存] 已保存 %s 条映射：%s", len(self.entries), self.path)


__all__ = [
    "CACHE_VERSION",
    "EntityMappingCache",
    "build_entity_mapping_key",
    "build_mapping_cache_namespace",
]
