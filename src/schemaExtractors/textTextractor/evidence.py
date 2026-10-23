"""对当前 Chunk 的原文句子编号，并将模型引用安全回填为未经改写的原文证据。"""
from __future__ import annotations

import re
from typing import Any


def evidence_sentences(text: str) -> list[dict[str, str]]:
    """按中文句末标点划分原文，为含公式或特殊符号的句子提供稳定引用编号。"""
    return [{"id": f"sentence_{index + 1}", "text": match.group(0)}
            for index, match in enumerate(re.finditer(r"[^。！？]+[。！？]*|[。！？]+", text))]


def resolve_provenance(raw: dict[str, Any], text: str) -> Any:
    """只接受已存在且连续升序的句子编号，直接拼接对应原文，禁止模糊改写证据。"""
    if "evidence_sentence_ids" not in raw:
        return raw.get("provenance")
    identifiers = raw["evidence_sentence_ids"]
    if not isinstance(identifiers, list) or not identifiers or not all(isinstance(item, str) for item in identifiers):
        return None
    sentences = evidence_sentences(text)
    indexes = {sentence["id"]: index for index, sentence in enumerate(sentences)}
    if any(identifier not in indexes for identifier in identifiers):
        return None
    selected = [indexes[identifier] for identifier in identifiers]
    if selected != list(range(selected[0], selected[-1] + 1)):
        return None
    return "".join(sentences[index]["text"] for index in selected)
