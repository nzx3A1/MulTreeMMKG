"""统一知识图模型。

一个 :class:`Graph` 同时保存实体、关系和事件。单个 Chunk 的抽取结果和跨 Chunk
汇总结果使用完全相同的类型，区别只记录在 ``metadata`` 的来源范围中。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from pydantic import Field

from .base import ID, MMKGBaseModel, SourceModality


class GraphMetadata(MMKGBaseModel):
    """描述 Graph 的来源范围、生成阶段及其他扩展信息。"""

    document_id: Optional[ID] = None  # 所属文档；汇总多文档图时可为空
    chunk_id: Optional[ID] = None  # 单 Chunk 抽取图的来源 Chunk，汇总图为空
    modality: Optional[SourceModality] = None  # 单 Chunk 的来源模态，汇总图通常为 MIXED
    stage: Optional[str] = None  # 生成阶段，例如 stage_04_text_extraction_front
    raw_response: Optional[Any] = None  # 单 Chunk 调试时保留的原始模型响应
    extra: Dict[str, Any] = Field(default_factory=dict)  # 调用参数、统计信息等扩展元数据



class Entity(MMKGBaseModel):
    """知识图谱实体。"""

    id: ID                             # 实体唯一标识
    name: str                          # 实体名称
    official_name: Optional[str] = None  # 实体官方名称或标准名称
    type: str                          # 实体类型（来自 Schema 白名单）
    type_zh: Optional[str] = None       # 实体类型中文名称
    aliases: List[str] = Field(default_factory=list)  # 别名列表
    attributes: Dict[str, Any] = Field(default_factory=dict)  # 属性键值对
    provenance: str = ""                # 原文证据字符串
    normalized_id: Optional[ID] = None # 对齐后的规范化 ID
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


class EventType(str, Enum):
    """常见事件类别，schema 阶段可继续细化。"""

    GEOLOGICAL_PROCESS = "geological_process"  # 地质过程（沉积、成岩等）
    EXPERIMENT = "experiment"                  # 实验事件
    OBSERVATION = "observation"                # 观测事件
    CHARGING = "charging"                      # 充注事件
    MIGRATION = "migration"                    # 运移事件
    ACCUMULATION = "accumulation"              # 聚集事件
    OTHER = "other"                            # 其他事件


class Event(MMKGBaseModel):
    """知识图谱事件。"""

    id: ID                             # 事件唯一标识
    type: EventType | str              # 事件类型
    name: str                          # 事件名称
    participants: List[ID] = Field(default_factory=list)  # 参与实体 ID 列表
    time: Optional[str] = None         # 时间信息
    location: Optional[str] = None     # 位置信息
    attributes: Dict[str, Any] = Field(default_factory=dict)  # 属性键值对
    provenance: str = ""                # 原文证据字符串
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


class Relation(MMKGBaseModel):
    """知识图谱关系。"""

    id: ID                             # 关系唯一标识
    type: str                          # 关系类型（来自 Schema 白名单）
    relation_name: Optional[str] = None  # 关系名称或标准名称，可为空
    type_zh: Optional[str] = None       # 关系类型中文名称
    source_id: ID                      # 源实体 ID
    target_id: ID                      # 目标实体 ID
    attributes: Dict[str, Any] = Field(default_factory=dict)  # 属性键值对
    provenance: str = ""                # 原文证据字符串
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 扩展元数据



class Graph(MMKGBaseModel):
    """实体、关系和事件的统一容器，适用于 Chunk 抽取与全局汇总。"""

    entities: List[Entity] = Field(default_factory=list)  # 图中的领域实体节点
    relations: List[Relation] = Field(default_factory=list)  # 实体之间的语义关系边
    events: List[Event] = Field(default_factory=list)  # 关联多个实体的地质或实验事件
    metadata: GraphMetadata = Field(default_factory=GraphMetadata)  # 图的来源与阶段信息

    def entity_ids(self) -> set[ID]:
        """返回实体 ID 集合，供关系和事件参与者的引用校验使用。"""

        return {entity.id for entity in self.entities}

    def validate_references(self) -> list[str]:
        """检查关系端点和事件参与者是否都引用当前图中的实体。"""

        entity_ids = self.entity_ids()
        errors: list[str] = []
        for relation in self.relations:
            if relation.source_id not in entity_ids:
                errors.append(f"关系 {relation.id} 的源实体不存在：{relation.source_id}")
            if relation.target_id not in entity_ids:
                errors.append(f"关系 {relation.id} 的目标实体不存在：{relation.target_id}")
        for event in self.events:
            for participant_id in event.participants:
                if participant_id not in entity_ids:
                    errors.append(f"事件 {event.id} 的参与实体不存在：{participant_id}")
        return errors

    @classmethod
    def from_chunk(
        cls,
        document_id: ID,
        chunk_id: ID,
        modality: SourceModality,
        entities: Iterable[Entity] = (),
        relations: Iterable[Relation] = (),
        events: Iterable[Event] = (),
        raw_response: Any = None,
        stage: Optional[str] = None,
    ) -> "Graph":
        """创建一个仅表示单个 Chunk 抽取产物的 Graph。"""

        return cls(
            entities=list(entities),
            relations=list(relations),
            events=list(events),
            metadata=GraphMetadata(
                document_id=document_id,
                chunk_id=chunk_id,
                modality=modality,
                stage=stage,
                raw_response=raw_response,
            ),
        )

    @classmethod
    def merge(
        cls,
        graphs: Iterable["Graph"],
        document_id: Optional[ID] = None,
        stage: Optional[str] = None,
    ) -> "Graph":
        """按 ID 去重合并多个 Chunk Graph，并生成同类型的汇总 Graph。"""

        graph_list = list(graphs)
        entities: dict[ID, Entity] = {}
        relations: dict[ID, Relation] = {}
        events: dict[ID, Event] = {}
        for graph in graph_list:
            entities.update({entity.id: entity for entity in graph.entities})
            relations.update({relation.id: relation for relation in graph.relations})
            events.update({event.id: event for event in graph.events})

        resolved_document_id = document_id
        if resolved_document_id is None:
            source_document_ids = {graph.metadata.document_id for graph in graph_list if graph.metadata.document_id}
            resolved_document_id = source_document_ids.pop() if len(source_document_ids) == 1 else None
        return cls(
            entities=list(entities.values()),
            relations=list(relations.values()),
            events=list(events.values()),
            metadata=GraphMetadata(
                document_id=resolved_document_id,
                modality=SourceModality.MIXED,
                stage=stage,
                extra={"merged_graph_count": len(graph_list)},
            ),
        )


# 旧名称保留为别名，已有调用方迁移期间仍可得到同一种 Graph 类型。
KnowledgeGraph = Graph
