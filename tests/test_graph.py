"""统一 Graph 模型测试。"""
from __future__ import annotations

from model import Entity, Event, Graph, Relation, SourceModality


def test_graph_stores_chunk_entities_relations_and_events() -> None:
    """单个 Chunk 的实体、关系和事件应保存于同一个 Graph 中并保留来源信息。"""

    entity_a = Entity(id="entity-a", name="延长组", type="Stratum")
    entity_b = Entity(id="entity-b", name="长7段", type="Reservoir")
    relation = Relation(id="relation-a-b", type="CONTAINS", source_id=entity_a.id, target_id=entity_b.id)
    event = Event(id="event-deposition", type="geological_process", name="沉积作用", participants=[entity_a.id, entity_b.id])

    graph = Graph.from_chunk(
        document_id="document-1",
        chunk_id="chunk-1",
        modality=SourceModality.TEXT,
        entities=[entity_a, entity_b],
        relations=[relation],
        events=[event],
        stage="stage_04_text_extraction",
    )

    assert graph.metadata.chunk_id == "chunk-1"
    assert graph.metadata.modality == SourceModality.TEXT
    assert graph.relations[0].source_id == "entity-a"
    assert graph.events[0].participants == ["entity-a", "entity-b"]
    assert graph.validate_references() == []


def test_graph_merge_returns_the_same_graph_type() -> None:
    """多个 Chunk Graph 汇总后仍应是 Graph，并按 ID 汇总实体、关系和事件。"""

    entity = Entity(id="entity-a", name="延长组", type="Stratum")
    graph_a = Graph.from_chunk("document-1", "chunk-1", SourceModality.TEXT, entities=[entity])
    graph_b = Graph.from_chunk(
        "document-1",
        "chunk-2",
        SourceModality.TABLE,
        relations=[Relation(id="relation-a-a", type="RELATED_TO", source_id="entity-a", target_id="entity-a")],
        events=[Event(id="event-1", type="experiment", name="测试", participants=["entity-a"])],
    )

    merged = Graph.merge([graph_a, graph_b], stage="stage_10_merge")

    assert isinstance(merged, Graph)
    assert merged.metadata.document_id == "document-1"
    assert merged.metadata.chunk_id is None
    assert merged.metadata.modality == SourceModality.MIXED
    assert len(merged.entities) == len(merged.relations) == len(merged.events) == 1
    assert merged.validate_references() == []
