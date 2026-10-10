"""多模态实体与关系使用的确定性文本规范化工具。

这里仅执行可解释的 Unicode、空白、标点和配置化领域同义转换；Embedding 和
LLM 只参与后续候选判断，不能改变原始文本。
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence


_SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
    }
)
_REMOVABLE_PATTERN = re.compile(r"[\s\-_—–·•,，。.;；:：/\\|()（）\[\]【】{}<>《》'\"“”‘’]+")


def normalize_name(value: Any, replacements: Mapping[str, str] | None = None) -> str:
    """规范化实体名称，并按配置应用安全的领域同义替换。"""

    text = unicodedata.normalize("NFKC", str(value or "")).translate(_SUPERSCRIPT_TRANSLATION)
    text = text.strip().lower()
    for source, target in sorted(
        (replacements or {}).items(), key=lambda item: (-len(str(item[0])), str(item[0]))
    ):
        if source:
            text = text.replace(str(source).lower(), str(target).lower())
    return _REMOVABLE_PATTERN.sub("", text)


def normalize_relation(value: Any) -> str:
    """把中英文关系标识转换为稳定的比较键。"""

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return "<EMPTY>"
    if re.fullmatch(r"[A-Za-z0-9_\-\s]+", text):
        return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()
    return normalize_name(text)


def name_similarity(left_names: Sequence[str], right_names: Sequence[str]) -> float:
    """计算两组已规范化名称之间的最大字符相似度。"""

    best = 0.0
    for left in left_names:
        for right in right_names:
            if not left or not right:
                continue
            if left == right:
                return 1.0
            ratio = SequenceMatcher(None, left, right).ratio()
            shorter, longer = sorted((left, right), key=len)
            if len(shorter) >= 2 and shorter in longer:
                coverage = len(shorter) / max(1, len(longer))
                ratio = max(ratio, 0.72 + 0.24 * coverage)
            best = max(best, ratio)
    return best


def normalized_value(value: Any) -> str:
    """把属性值转换为可比较且稳定的字符串。"""

    if isinstance(value, str):
        return normalize_name(value)
    return stable_json(value)


def stable_json(value: Any) -> str:
    """以固定键顺序序列化任意 JSON 兼容值。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_id(prefix: str, *parts: Any, length: int = 20) -> str:
    """依据输入内容生成可重复的短哈希 ID。"""

    raw = "\x1f".join(stable_json(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


__all__ = [
    "name_similarity",
    "normalize_name",
    "normalize_relation",
    "normalized_value",
    "stable_id",
    "stable_json",
]
