"""文本模态实体、关系和事件抽取提示词。

所有输出字段严格对应 ``model.graph`` 中的 Entity、Relation 和 Event，原文证据
直接使用 provenance 字符串表示，保证结果可以直接构造 Pydantic 模型。
"""
from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "0.1.0"
ENTITY_TYPE_WHITELIST_TEXT = "由调用方传入的 EntityConcept Schema 英文类型白名单"
RELATION_TYPE_WHITELIST_TEXT = "由调用方提供的关系 Schema 英文类型白名单"
RELATION_CONSTRAINT_TEXT = "关系 source_id 和 target_id 必须引用 entity_list 中已存在的实体 id"
EVENT_TYPE_WHITELIST_TEXT = (
    "geological_process、experiment、observation、charging、migration、accumulation、other"
)


def ENTITY_PROMPT(text: str, concept_nodes: str = "") -> dict[str, Any]:
    """构造输出字段与 Entity 模型完全一致的实体抽取提示词。"""

    schema_text = concept_nodes or ENTITY_TYPE_WHITELIST_TEXT
    evidence = "鄂尔多斯盆地延长组长7段主要发育暗色泥岩，TOC为2.1%~6.5%。"
    return {
        "role": "石油地质领域知识图谱实体抽取专家",
        "schema_version": SCHEMA_VERSION,
        "task": "根据 Input Summary 抽取实体，输出可直接映射到 model.graph.Entity 的 JSON 数组。",
        "entity_schema_whitelist": schema_text,
        "rules": [
            "只抽取原文明确出现且具有知识图谱价值的石油地质对象、参数和专业概念。",
            "每个实体必须生成当前输出内唯一的字符串 id，格式为 entity_1、entity_2，并从 1 连续编号。",
            "type 必须使用实体 Schema 白名单中的英文类型；type_zh 填写对应中文类型名。",
            "name 是原文中的完整实体名称；official_name 是标准名称，无法确定时为 null。",
            "aliases 必须是字符串数组；description 无法从原文可靠概括时为 null。",
            "数值、单位、测试条件、方法、层位和位置等信息写入 attributes。",
            "normalized_id 在本阶段固定为 null，由后续实体对齐阶段填写。",
            "provenance 必须是原文证据字符串，禁止输出数组或对象。",
            "抽取置信度写入 metadata.confidence，范围为 0 到 1。",
            "不要输出 Entity 模型以外的 label_zh、properties、evidence_span、confidence 等顶层字段。",
            "只返回 JSON 数组，不要输出解释、Markdown 或代码块。",
        ],
        "example": {
            "input": evidence,
            "output": [
                {
                    "id": "entity_1",
                    "name": "鄂尔多斯盆地",
                    "official_name": "鄂尔多斯盆地",
                    "type": "Basin",
                    "type_zh": "盆地",
                    "aliases": [],
                    "description": None,
                    "attributes": {},
                    "provenance": evidence,
                    "normalized_id": None,
                    "metadata": {"confidence": 0.95},
                },
                {
                    "id": "entity_2",
                    "name": "TOC",
                    "official_name": "总有机碳含量",
                    "type": "TOC",
                    "type_zh": "总有机碳",
                    "aliases": ["总有机碳含量"],
                    "description": None,
                    "attributes": {"min_value": 2.1, "max_value": 6.5, "unit": "%"},
                    "provenance": "TOC为2.1%~6.5%",
                    "normalized_id": None,
                    "metadata": {"confidence": 0.94},
                },
            ],
        },
        "Input Summary": text,
        "Output": [
            {
                "id": "entity_1",
                "name": "实体名称",
                "official_name": None,
                "type": "Schema英文类型",
                "type_zh": "中文类型名",
                "aliases": [],
                "description": None,
                "attributes": {},
                "provenance": "原文证据句",
                "normalized_id": None,
                "metadata": {"confidence": 0.0},
            }
        ],
    }


def RELATION_PROMPT(entity_list: Any, input_text: str) -> dict[str, Any]:
    """构造引用实体 ID 且字段与 Relation 模型一致的关系抽取提示词。"""

    evidence = "延长组长7段位于鄂尔多斯盆地。"
    return {
        "role": "石油地质领域知识图谱关系抽取专家",
        "schema_version": SCHEMA_VERSION,
        "task": "从 input_text 抽取实体关系，输出可直接映射到 model.graph.Relation 的 JSON 数组。",
        "relation_schema_whitelist": RELATION_TYPE_WHITELIST_TEXT,
        "relation_constraints": RELATION_CONSTRAINT_TEXT,
        "rules": [
            "每条关系必须生成当前输出内唯一的字符串 id，格式为 relation_1、relation_2，并连续编号。",
            "type 必须使用关系 Schema 白名单中的英文类型；official_name 为标准关系名称。",
            "type_zh 填写中文关系名，无法确定时为 null。",
            "source_id 和 target_id 必须直接复制 entity_list 中的实体 id，不得使用实体名称或编造 id。",
            "关系描述、限定条件、时间和数值等非模型顶层字段统一写入 attributes。",
            "provenance 必须是原文证据字符串；置信度写入 metadata.confidence。",
            "禁止输出 source、target、source_type、target_type、relation、description、evidence_span 等额外顶层字段。",
            "不得抽取只有共现而无明确语义证据的关系。",
            "只返回 JSON 数组，不要输出解释、Markdown 或代码块。",
        ],
        "example": {
            "entity_list": [
                {"id": "entity_1", "name": "延长组长7段", "type": "StratigraphicMember"},
                {"id": "entity_2", "name": "鄂尔多斯盆地", "type": "Basin"},
            ],
            "input": evidence,
            "output": [
                {
                    "id": "relation_1",
                    "type": "located_in",
                    "official_name": "located_in",
                    "type_zh": "位于",
                    "source_id": "entity_1",
                    "target_id": "entity_2",
                    "attributes": {"description": evidence},
                    "provenance": evidence,
                    "metadata": {"confidence": 0.95},
                }
            ],
        },
        "Input": {"input_text": input_text, "entity_list": entity_list},
        "Output": [
            {
                "id": "relation_1",
                "type": "Schema英文关系类型",
                "official_name": None,
                "type_zh": None,
                "source_id": "entity_list中源实体的id",
                "target_id": "entity_list中目标实体的id",
                "attributes": {},
                "provenance": "原文证据句",
                "metadata": {"confidence": 0.0},
            }
        ],
    }


def EVENT_PROMPT(input_text: str, entity_list: Any = None) -> dict[str, Any]:
    """构造引用参与实体 ID 且字段与 Event 模型一致的事件抽取提示词。"""

    evidence = "延长组长7段在晚三叠世沉积于鄂尔多斯盆地深湖环境。"
    return {
        "role": "石油地质领域知识图谱事件抽取专家",
        "schema_version": SCHEMA_VERSION,
        "task": "从 Input 文本抽取明确事件，输出可直接映射到 model.graph.Event 的 JSON 数组。",
        "event_schema_whitelist": EVENT_TYPE_WHITELIST_TEXT,
        "rules": [
            "每个事件必须生成当前输出内唯一的字符串 id，格式为 event_1、event_2，并连续编号。",
            "type 优先使用 geological_process、experiment、observation、charging、migration、accumulation；无法归类时使用 other。",
            "name 必须是包含对象和行为的完整事件名称，禁止仅输出“事件”“过程”或“作用”。",
            "participants 必须是实体 id 数组，只能引用 entity_list 中的 id；没有可靠参与实体时输出空数组。",
            "time 和 location 分别使用字符串或 null，不要放入 attributes 重复保存。",
            "行为、对象、环境、条件、结果等信息写入 attributes。",
            "provenance 必须是原文证据字符串；置信度写入 metadata.confidence。",
            "禁止输出 label_zh、properties、evidence_span、confidence 等额外顶层字段。",
            "只返回 JSON 数组，不要输出解释、Markdown 或代码块。",
        ],
        "example": {
            "entity_list": [
                {"id": "entity_1", "name": "延长组长7段"},
                {"id": "entity_2", "name": "鄂尔多斯盆地"},
            ],
            "input": evidence,
            "output": [
                {
                    "id": "event_1",
                    "type": "geological_process",
                    "name": "延长组长7段沉积事件",
                    "participants": ["entity_1", "entity_2"],
                    "time": "晚三叠世",
                    "location": "鄂尔多斯盆地深湖环境",
                    "attributes": {"behavior": "沉积"},
                    "provenance": evidence,
                    "metadata": {"confidence": 0.9},
                }
            ],
        },
        "Input": {"input_text": input_text, "entity_list": entity_list or []},
        "Output": [
            {
                "id": "event_1",
                "type": "EventType枚举值或Schema事件类型",
                "name": "完整事件名称",
                "participants": [],
                "time": None,
                "location": None,
                "attributes": {},
                "provenance": "原文证据句",
                "metadata": {"confidence": 0.0},
            }
        ],
    }


__all__ = ["ENTITY_PROMPT", "RELATION_PROMPT", "EVENT_PROMPT"]
