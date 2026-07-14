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

def ENTITY_PROMPT_noSchema( text: str, context: Any = None,) -> dict[str, Any]:
     """构造输出字段与 Entity 模型完全一致的实体抽取提示词。"""
     return {
        "role": "石油地质领域知识图谱实体抽取专家",
        "task": "根据 text 抽取原文中所有明确出现的石油地质相关的实体。，输出 JSON 对象。",
        # "context_for_disambiguation_only": context or {},
        "rules": [
            "type_zh 填写合适的中文石油地质概念类型名。",
            "name 是原文中的完整实体名称；official_name 是标准名称，无法确定时和 name 相同。",
            "数值、单位、测试条件、方法、层位和位置等属性写入 attributes。",
            "provenance 必须是原文证据字符串，禁止输出数组或对象。",
            # "上下文只能用于类型和指代消歧，provenance 必须来自 current_text。",
            "只返回形如 {\"entities\": [...]} 的 JSON 对象，不要输出解释、Markdown 或代码块。",
        ],
        "example": {
  "input": "鄂尔多斯盆地伊陕斜坡延长组长7段主要发育深湖相黑色页岩，TOC为2.1%~6.5%，Ro为0.8%~1.2%，平均孔隙度为5.6%，属于成熟优质烃源岩。",
  "output": {
    "entities": [
      {
        "name": "鄂尔多斯盆地",
        "official_name": "鄂尔多斯盆地",
        "type_zh": "盆地",
        "attributes": {},
        "provenance": "鄂尔多斯盆地"
      },
      {
        "name": "伊陕斜坡",
        "official_name": "伊陕斜坡",
        "type_zh": "斜坡",
        "attributes": {
          "parent_basin": "鄂尔多斯盆地"
        },
        "provenance": "鄂尔多斯盆地伊陕斜坡"
      },
      {
        "name": "延长组长7段",
        "official_name": "延长组长7段",
        "type_zh": "段",
        "attributes": {
          "formation": "延长组",
          "member": "长7段"
        },
        "provenance": "延长组长7段"
      },
      {
        "name": "深湖相",
        "official_name": "深湖相",
        "type_zh": "沉积相",
        "attributes": {},
        "provenance": "主要发育深湖相黑色页岩"
      },
      {
        "name": "黑色页岩",
        "official_name": "黑色页岩",
        "type_zh": "岩石",
        "attributes": {
          "toc": {
            "min_value": 2.1,
            "max_value": 6.5,
            "unit": "%"
          },
          "ro": {
            "min_value": 0.8,
            "max_value": 1.2,
            "unit": "%"
          },
          "porosity": {
            "value": 5.6,
            "unit": "%",
            "statistic": "平均值"
          },
          "maturity_stage": "成熟优质烃源岩",
        },
        "provenance": "延长组长7段主要发育深湖相黑色页岩，TOC为2.1%~6.5%，Ro为0.8%~1.2%，平均孔隙度为5.6%，属于成熟优质烃源岩"
      }
    ]
  }
},
        "current_text": text,
        "Output": {"entities": [
            {
                "name": "实体名称",
                "official_name": "实体名称的官方名称",
                "type_zh": "中文类型名",
                "attributes": {},
                "provenance": "原文证据句",
            }
        ]},
    }

def RELATION_PROMPT_noSchema(
    entity_list: Any,
    input_text: str,
    context: Any = None,
) -> dict[str, Any]:
    """构造引用实体 ID 且字段与 Relation 模型一致的关系抽取提示词。"""

    evidence = "延长组长7段位于鄂尔多斯盆地。"
    return {
        "role": "石油地质领域知识图谱关系抽取专家",
        "task": "根据 current_text 和已抽取实体判断关系，输出 model.graph.Relation JSON 对象。",
        # "context_for_disambiguation_only": context or {},
        "rules": [
            "type_zh 填写中文关系名，无法确定时为 null。",
            "source_id 和 target_id 必须直接复制 entity_list 中的实体 id，不得使用实体名称或编造 id。",
            "关系描述、限定条件、时间和数值等非模型顶层字段统一写入 attributes。",
            "provenance 必须是原文证据字符串。",
            "禁止输出 source、target、relation、description、evidence_span 等未定义顶层字段。",
            # "上下文只能用于消歧，provenance 必须来自 current_text。",
            "只返回形如 {\"relations\": [...]} 的 JSON 对象，不要输出解释、Markdown 或代码块。",
        ],
        "example": {
            "entity_list": [
                {"id": "entity_1", "name": "延长组长7段", "type": "Formation", "type_zh": "段"},
                {"id": "entity_2", "name": "鄂尔多斯盆地", "type": "Basin", "type_zh": "盆地"},
            ],
            "input": evidence,
            "output": {"relations": [
                {
                    "relation_name": "位于",
                    "source_id": "entity_1",
                    "source_name": "延长组长7段",
                    "source_type": "Formation",
                    "target_id": "entity_2",
                    "target_name": "鄂尔多斯盆地",
                    "target_type": "Basin",
                    "attributes": {},
                    "provenance": evidence,
                }
            ]},
        },
        "Input": {"current_text": input_text, "entity_list": entity_list},
        "Output": {"relations": [
            {
                "relation_name": "原文的关系名称",
                "source_id": "entity_list中源实体的id",
                "source_name": "source_id对应实体的name",
                "source_type": "source_id对应实体的type",
                "target_id": "entity_list中目标实体的id",
                "target_name": "target_id对应实体的name",
                "target_type": "target_id对应实体的type",
                "attributes": {},
                "provenance": "原文证据句",
            }
        ]},
    }



def ENTITYFULL_PROMPT_Schema(entity_list: Any, concepts: Any = None) -> dict[str, Any]:
    """构造无约束实体候选的 Schema 类型归属与名称规范化提示词。"""

    return {
        "role": "石油地质领域知识图谱实体类型归属与名称规范化专家",
        "task": (
            "逐一检查 entity_list 中已经抽取的实体，仅依据实体信息和 concepts 白名单，"
            "为每个实体选择最合适的已定义类型，并补充官方名称。"
        ),
        "entity_list": entity_list or [],
        "concepts": concepts or [],
        "concept_field_meaning": {
            "schema": "必须原样用于输出 type 的英文 Schema 类型名",
            "zh_name": "与 schema 对应的中文类型名，用于输出 type_zh",
            "description": "类型定义，是判断归属的主要依据",
            "example": "该类型的参考实例，只用于辅助判断",
        },
        "rules": [
            "必须处理 entity_list 中的每个实体，不得新增、删除、合并或拆分实体。",
            "输出实体的顺序和 name 必须与 entity_list 完全一致，不得改写 name。",
            "优先依据 concepts 中的 description 判断语义类型，再参考 zh_name 和 example；不能只按字面相似度猜测。",
            "若能归入 concepts，type 必须逐字复制对应 concept.schema，type_zh 必须逐字复制对应 concept.zh_name。",
            "若 concepts 中没有合适类型，type 必须为 other，type_zh 保留该实体原有的 type_zh。",
            "official_name 填写该实体公认、规范的中文全称；无法可靠确定时必须与 name 相同，不得臆造。",
            "只返回 JSON 对象，不要输出解释、Markdown 或代码块。",
        ],
        "output_format": {
            "entities": [
                {
                    "name": "与输入完全一致的实体名称",
                    "official_name": "实体官方名称，未知时等于 name",
                    "type_zh": "命中 concepts 时为对应 zh_name，否则为原 type_zh",
                    "type": "命中 concepts 时为对应 schema，否则为 other",
                }
            ]
        },
    }





def RELATIONFULL_PROMPT_Schema(relation_list: Any, concepts: Any = None) -> dict[str, Any]:
    """构造无约束关系候选的 Schema 类型归属与名称规范化提示词。"""

    return {
        "role": "石油地质领域知识图谱关系类型归属与名称规范化专家",
        "task": (
            "逐一检查 relation_list 中已经抽取的关系，仅依据关系语义和 concepts 关系白名单，"
            "为每条关系选择方向及类型均匹配的已定义 Schema 关系。"
        ),
        "relation_list": relation_list or [],
        "concepts": concepts or [],
        "concept_field_meaning": {
            "source_schema": "关系允许的源实体英文 Schema 类型",
            "relationEn": "必须原样用于输出 type 的英文关系类型",
            "relationZh": "必须原样用于输出 type_zh 和 relation_name 的中文标准关系名",
            "target_schema": "关系允许的目标实体英文 Schema 类型",
        },
        "rules": [
            "必须处理 relation_list 中的每条关系，不得新增、删除、合并、拆分或改变关系顺序。",
            "source_id 和 target_id 必须与输入逐字一致，禁止交换方向、改写 ID 或引用其他实体。",
            "source_name、source_type、target_name、target_type 必须与输入逐字一致，不得删除或改写。",
            "应同时判断关系语义、源实体方向和目标实体方向，只有完全匹配时才能归入 concepts。",
            "若能归入 concepts，type 必须逐字复制 relationEn，type_zh 和 relation_name 必须逐字复制 relationZh。",
            "若 concepts 中没有合适关系，type 必须为 other；type_zh 和 relation_name 保留输入中的原始中文关系名。",
            "不得修改 attributes、provenance、metadata 等首次抽取内容，也不要在输出中补写这些字段。",
            "只返回 JSON 对象，不要输出解释、Markdown 或代码块。",
        ],
        "output_format": {
            "relations": [
                {
                    "source_id": "与输入完全一致的源实体 ID",
                    "source_name": "与输入完全一致的源实体名称",
                    "source_type": "与输入完全一致的源实体类型",
                    "target_id": "与输入完全一致的目标实体 ID",
                    "target_name": "与输入完全一致的目标实体名称",
                    "target_type": "与输入完全一致的目标实体类型",
                    "relation_name": "命中时为 relationZh，否则保留原关系名",
                    "type_zh": "命中时为 relationZh，否则保留原中文关系名",
                    "type": "命中时为 relationEn，否则为 other",
                }
            ]
        },
    }












###################################################严格限制模式################################################
def ENTITY_PROMPT(
    text: str,
    concept_nodes: Any = None,
    context: Any = None,
) -> dict[str, Any]:
    """构造输出字段与 Entity 模型完全一致的实体抽取提示词。"""

    schema_text = concept_nodes or ENTITY_TYPE_WHITELIST_TEXT
    evidence = "鄂尔多斯盆地延长组长7段主要发育暗色泥岩，TOC为2.1%~6.5%。"
    return {
        "role": "石油地质领域知识图谱实体抽取专家",
        "schema_version": SCHEMA_VERSION,
        "task": "只根据 current_text 抽取实体，输出可映射到 model.graph.Entity 的 JSON 对象。",
        "entity_schema_whitelist": schema_text,
        "context_for_disambiguation_only": context or {},
        "rules": [
            "只抽取原文明确出现且具有知识图谱价值的石油地质对象、参数和专业概念。",
            "每个实体必须生成当前输出内唯一的字符串 id，格式为 entity_1、entity_2，并从 1 连续编号。",
            "type 必须使用实体 Schema 白名单中的英文类型；type_zh 填写对应中文类型名。",
            "name 是原文中的完整实体名称；official_name 是标准名称，无法确定时和 name 相同。",
            "aliases 必须是字符串数组；description 无法从原文可靠概括时为 null。",
            "数值、单位、测试条件、方法、层位和位置等信息写入 attributes。",
            "normalized_id 在本阶段固定为 null，由后续实体对齐阶段填写。",
            "provenance 必须是原文证据字符串，禁止输出数组或对象。",
            "不要输出 Entity 模型以外的 label_zh、properties、evidence_span、confidence 等顶层字段。",
            "上下文只能用于类型和指代消歧，provenance 必须来自 current_text。",
            "只返回形如 {\"entities\": [...]} 的 JSON 对象，不要输出解释、Markdown 或代码块。",
        ],
        "example": {
            "input": text,
            "output": {"entities": [
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
                },
            ]},
        },
        "current_text": text,
        "Output": {"entities": [
            {
                "id": "entity_1",
                "name": "实体名称",
                "official_name": "实体名称的官方名称",
                "type": "Schema英文类型",
                "type_zh": "中文类型名",
                "aliases": [],
                "description": None,
                "attributes": {},
                "provenance": "原文证据句",
                "normalized_id": None,
            }
        ]},
    }


def RELATION_PROMPT(
    entity_list: Any,
    input_text: str,
    relation_schema: Any = None,
    context: Any = None,
) -> dict[str, Any]:
    """构造引用实体 ID 且字段与 Relation 模型一致的关系抽取提示词。"""

    evidence = "延长组长7段位于鄂尔多斯盆地。"
    return {
        "role": "石油地质领域知识图谱关系抽取专家",
        "schema_version": SCHEMA_VERSION,
        "task": "只根据 current_text 和已抽取实体判断关系，输出 model.graph.Relation JSON 对象。",
        "relation_schema_whitelist": relation_schema or RELATION_TYPE_WHITELIST_TEXT,
        "relation_constraints": RELATION_CONSTRAINT_TEXT,
        "context_for_disambiguation_only": context or {},
        "rules": [
            "每条关系必须生成当前输出内唯一的字符串 id，格式为 relation_1、relation_2，并连续编号。",
            "type 必须使用关系 Schema 白名单中的英文类型；official_name 为标准关系名称。",
            "type_zh 填写中文关系名，无法确定时为 null。",
            "source_id 和 target_id 必须直接复制 entity_list 中的实体 id，不得使用实体名称或编造 id。",
            "source_name、source_type 必须分别复制 source_id 对应实体的 name、type。",
            "target_name、target_type 必须分别复制 target_id 对应实体的 name、type。",
            "关系描述、限定条件、时间和数值等非模型顶层字段统一写入 attributes。",
            "provenance 必须是原文证据字符串。",
            "禁止输出 source、target、relation、description、evidence_span 等未定义顶层字段。",
            "不得抽取只有共现而无明确语义证据的关系。",
            "上下文只能用于消歧，provenance 必须来自 current_text。",
            "只返回形如 {\"relations\": [...]} 的 JSON 对象，不要输出解释、Markdown 或代码块。",
        ],
        "example": {
            "entity_list": [
                {"id": "entity_1", "name": "延长组长7段", "type": "StratigraphicMember"},
                {"id": "entity_2", "name": "鄂尔多斯盆地", "type": "Basin"},
            ],
            "input": evidence,
            "output": {"relations": [
                {
                    "id": "relation_1",
                    "type": "located_in",
                    "official_name": "located_in",
                    "type_zh": "位于",
                    "source_id": "entity_1",
                    "source_name": "延长组长7段",
                    "source_type": "StratigraphicMember",
                    "target_id": "entity_2",
                    "target_name": "鄂尔多斯盆地",
                    "target_type": "Basin",
                    "attributes": {"description": evidence},
                    "provenance": evidence,
                }
            ]},
        },
        "Input": {"current_text": input_text, "entity_list": entity_list},
        "Output": {"relations": [
            {
                "id": "relation_1",
                "type": "Schema英文关系类型",
                "official_name": None,
                "type_zh": None,
                "source_id": "entity_list中源实体的id",
                "source_name": "source_id对应实体的name",
                "source_type": "source_id对应实体的type",
                "target_id": "entity_list中目标实体的id",
                "target_name": "target_id对应实体的name",
                "target_type": "target_id对应实体的type",
                "attributes": {},
                "provenance": "原文证据句",
            }
        ]},
    }


def EVENT_PROMPT(
    input_text: str,
    entity_list: Any = None,
    context: Any = None,
) -> dict[str, Any]:
    """构造引用参与实体 ID 且字段与 Event 模型一致的事件抽取提示词。"""

    evidence = "延长组长7段在晚三叠世沉积于鄂尔多斯盆地深湖环境。"
    return {
        "role": "石油地质领域知识图谱事件抽取专家",
        "schema_version": SCHEMA_VERSION,
        "task": "只根据 current_text 抽取明确事件，输出 model.graph.Event JSON 对象。",
        "event_schema_whitelist": EVENT_TYPE_WHITELIST_TEXT,
        "context_for_disambiguation_only": context or {},
        "rules": [
            "每个事件必须生成当前输出内唯一的字符串 id，格式为 event_1、event_2，并连续编号。",
            "type 优先使用 geological_process、experiment、observation、charging、migration、accumulation；无法归类时使用 other。",
            "name 必须是包含对象和行为的完整事件名称，禁止仅输出“事件”“过程”或“作用”。",
            "participants 必须是实体 id 数组，只能引用 entity_list 中的 id；没有可靠参与实体时输出空数组。",
            "time 和 location 分别使用字符串或 null，不要放入 attributes 重复保存。",
            "行为、对象、环境、条件、结果等信息写入 attributes。",
            "provenance 必须是原文证据字符串。",
            "禁止输出 label_zh、properties、evidence_span、confidence 等额外顶层字段。",
            "上下文只能用于消歧，provenance 必须来自 current_text。",
            "只返回形如 {\"events\": [...]} 的 JSON 对象，不要输出解释、Markdown 或代码块。",
        ],
        "example": {
            "entity_list": [
                {"id": "entity_1", "name": "延长组长7段"},
                {"id": "entity_2", "name": "鄂尔多斯盆地"},
            ],
            "input": evidence,
            "output": {"events": [
                {
                    "id": "event_1",
                    "type": "geological_process",
                    "name": "延长组长7段沉积事件",
                    "participants": ["entity_1", "entity_2"],
                    "time": "晚三叠世",
                    "location": "鄂尔多斯盆地深湖环境",
                    "attributes": {"behavior": "沉积"},
                    "provenance": evidence,
                }
            ]},
        },
        "Input": {"current_text": input_text, "entity_list": entity_list or []},
        "Output": {"events": [
            {
                "id": "event_1",
                "type": "EventType枚举值或Schema事件类型",
                "name": "完整事件名称",
                "participants": [],
                "time": None,
                "location": None,
                "attributes": {},
                "provenance": "原文证据句",
            }
        ]},
    }


__all__ = [
    "ENTITY_PROMPT", "RELATION_PROMPT", "EVENT_PROMPT", "RELATION_PROMPT_noSchema",
    "ENTITY_PROMPT_noSchema", "ENTITYFULL_PROMPT_Schema", "RELATIONFULL_PROMPT_Schema",
]
