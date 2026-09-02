"""不依赖 Schema 的文本实体与关系抽取提示词。"""
from __future__ import annotations

from typing import Any


def build_entity_prompt(text: str, context: Any = None) -> dict[str, Any]:
    """构造自由类型实体抽取提示词，实体类型由模型按正文语义直接给出。"""

    return {
        "role": "石油地质领域知识图谱实体抽取专家",
        "task": "根据 current_text 抽取其中明确出现且具有知识图谱价值的实体，输出 JSON 对象。",
        "context_for_disambiguation_only": context or {},
        "rules": [
            "不得使用或猜测任何预定义 Schema 白名单；type 按实体语义自由填写简洁英文类型。",
            "每个实体必须生成本次输出内唯一的 id，格式为 entity_1、entity_2，并从 1 连续编号。",
            "type_zh 填写与 type 对应的中文类型名。",
            "name 保持原文完整实体名称；official_name 无法可靠确定时与 name 相同。",
            "数值、单位、测试条件、方法、层位和位置等信息写入 attributes。",
            "provenance 必须是 current_text 中的原文证据字符串。",
            "只返回形如 {\"entities\": [...]} 的 JSON 对象，不要输出解释、Markdown 或代码块。",
        ],
        "current_text": text,
        "Output": {
            "entities": [
                {
                    "id": "entity_1",
                    "name": "实体名称",
                    "official_name": "实体官方名称",
                    "type": "free_form_english_type",
                    "type_zh": "中文类型名",
                    "aliases": [],
                    "attributes": {},
                    "provenance": "原文证据句",
                }
            ]
        },
    }


def build_relation_prompt(
    entity_list: Any,
    input_text: str,
    context: Any = None,
) -> dict[str, Any]:
    """构造自由关系抽取提示词，只要求关系端点引用已抽取实体。"""

    return {
        "role": "石油地质领域知识图谱关系抽取专家",
        "task": "根据 current_text 和 entity_list 判断有明确证据的关系，输出 JSON 对象。",
        "context_for_disambiguation_only": context or {},
        "rules": [
            "不得使用或猜测任何预定义关系 Schema、方向白名单或候选筛选结果。",
            "type 按关系语义自由填写简洁英文类型，建议使用大写下划线命名；type_zh 填写中文关系名。",
            "source_id 和 target_id 必须直接复制 entity_list 中的实体 id，不得使用实体名称或编造 id。",
            "source_name、source_type、target_name、target_type 必须与对应实体一致。",
            "只抽取正文有明确语义证据的关系，数值和限定条件写入 attributes。",
            "provenance 必须是 current_text 中的原文证据字符串。",
            "只返回形如 {\"relations\": [...]} 的 JSON 对象，不要输出解释、Markdown 或代码块。",
        ],
        "Input": {"current_text": input_text, "entity_list": entity_list},
        "Output": {
            "relations": [
                {
                    "id": "relation_1",
                    "type": "FREE_FORM_RELATION",
                    "relation_name": "关系名称",
                    "type_zh": "中文关系名",
                    "source_id": "entity_1",
                    "source_name": "源实体名称",
                    "source_type": "源实体类型",
                    "target_id": "entity_2",
                    "target_name": "目标实体名称",
                    "target_type": "目标实体类型",
                    "attributes": {},
                    "provenance": "原文证据句",
                }
            ]
        },
    }


__all__ = ["build_entity_prompt", "build_relation_prompt"]
