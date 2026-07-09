"""JSON 读写工具。

统一处理 UTF-8、中文保留、父目录创建和 Pydantic 模型序列化，让各阶段产物
写入 output/ 时保持一致格式。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    """为 json.dump 提供 Pydantic 和常见对象的兜底序列化。"""

    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: str | Path) -> dict | list:
    """读取 JSON 文件并返回 dict 或 list。"""

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, data: dict | list | Any, indent: int = 2) -> None:
    """写出 JSON 文件，自动创建父目录并保留中文字符。"""

    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent, default=_json_default)
        file.write("\n")
