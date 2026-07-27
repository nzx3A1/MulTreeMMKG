"""为表格结构图和领域实体选择 Neo4j 概念 Schema。

选择器优先读取 ``petrommkg-schema``，连接不可用时退回仓库中已经定义的表格
结构 Schema。回退仅保证结构图可生成，不会伪造数据库中不存在的领域关系。
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Mapping, Sequence

from src.extractors.text_extractor.schema_models import RelevantSchema, SchemaConcept, SchemaRelation
from src.extractors.text_extractor.schema_repository import Neo4jSchemaRepository

from .schema_models import RecognizedTable


STRUCTURAL_CONCEPTS: dict[str, tuple[str, str]] = {
    "Table": ("表格", "由行、列、表头和单元格组成的结构化数据载体。"),
    "Caption": ("标题/图题/表题", "说明表格编号、主题和内容范围的文本。"),
    "TableHeader": ("表头", "说明字段、数据维度或分组含义的标题单元。"),
    "TableRow": ("表格行", "表格中的横向记录单元。"),
    "TableColumn": ("表格列", "表格中的纵向字段单元。"),
    "TableCell": ("表格单元格", "行列交叉位置的最小数据单元。"),
    "Parameter": ("参数", "可测量、计算、统计或用于描述对象性质的量。"),
    "Unit": ("单位", "表示参数或数值计量尺度的标准单位。"),
    "DataSeries": ("数据系列", "表格中由一组有序数据组成的数据集合。"),
}

DOMAIN_HINTS: dict[str, tuple[str, str]] = {
    "井": ("Well", "井"),
    "样品": ("Sample", "样品"),
    "样本": ("Sample", "样品"),
    "岩性": ("Lithology", "岩性"),
    "岩石": ("Lithology", "岩性"),
    "矿物": ("Mineral", "矿物"),
    "石英": ("Mineral", "矿物"),
    "长石": ("Mineral", "矿物"),
    "储层": ("Reservoir", "储层/储集层"),
    "层位": ("ReservoirInterval", "目的层/储层段"),
    "地层": ("Formation", "组"),
    "组": ("Formation", "组"),
    "段": ("StratigraphicMember", "段"),
    "孔隙": ("PoreStructure", "孔隙结构"),
    "裂缝": ("Fracture", "裂缝"),
    "实验": ("Experiment", "实验"),
    "分析方法": ("AnalyticalMethod", "分析方法"),
    "岩心": ("Core", "岩心"),
    "区块": ("Block", "区块"),
    "盆地": ("Basin", "盆地"),
}


def _relation(source: str, relation_en: str, relation_zh: str, target: str) -> SchemaRelation:
    """构造一条静态合法关系，统一英文关系为大写。"""

    return SchemaRelation(source, relation_en.upper(), relation_zh, target)


STATIC_RELATIONS: tuple[SchemaRelation, ...] = (
    _relation("Table", "HAS_CAPTION", "具有标题", "Caption"),
    _relation("Table", "HAS_HEADER", "具有表头", "TableHeader"),
    _relation("Table", "HAS_ROW", "具有表格行", "TableRow"),
    _relation("Table", "HAS_COLUMN", "具有表格列", "TableColumn"),
    _relation("Table", "HAS_CELL", "具有单元格", "TableCell"),
    _relation("Table", "HAS_DATA_SERIES", "具有数据系列", "DataSeries"),
    _relation("TableHeader", "PART_OF", "属于", "Table"),
    _relation("TableHeader", "DEFINES_COLUMN", "定义表格列", "TableColumn"),
    _relation("TableHeader", "REPRESENTS", "表示", "Parameter"),
    _relation("TableHeader", "HAS_UNIT", "具有单位", "Unit"),
    _relation("TableRow", "PART_OF", "属于", "Table"),
    _relation("TableRow", "CONTAINS_CELL", "包含单元格", "TableCell"),
    _relation("TableColumn", "PART_OF", "属于", "Table"),
    _relation("TableColumn", "CONTAINS_CELL", "包含单元格", "TableCell"),
    _relation("TableColumn", "REPRESENTS", "表示", "Parameter"),
    _relation("TableColumn", "HAS_UNIT", "具有单位", "Unit"),
    _relation("TableCell", "PART_OF", "属于", "Table"),
    _relation("TableCell", "LOCATED_IN_ROW", "位于表格行", "TableRow"),
    _relation("TableCell", "LOCATED_IN_COLUMN", "位于表格列", "TableColumn"),
    _relation("TableCell", "REPRESENTS", "表示", "Parameter"),
    _relation("TableCell", "HAS_UNIT", "具有单位", "Unit"),
    _relation("TableCell", "MEMBER_OF_SERIES", "属于数据系列", "DataSeries"),
    _relation("Table", "REPORTS", "报告", "Sample"),
    _relation("Table", "REPORTS", "报告", "Experiment"),
    _relation("Table", "REPORTS", "报告", "AnalyticalMethod"),
    _relation("Table", "DESCRIBES", "描述", "Formation"),
    _relation("Table", "DESCRIBES", "描述", "StratigraphicMember"),
    _relation("Table", "DESCRIBES", "描述", "ReservoirInterval"),
    _relation("Table", "DESCRIBES", "描述", "Lithology"),
    _relation("Table", "DESCRIBES", "描述", "Reservoir"),
    _relation("Sample", "RECORDED_IN", "记录于", "Table"),
    _relation("Sample", "COLLECTED_FROM", "采自", "Well"),
    _relation("Sample", "COLLECTED_FROM", "采自", "Lithology"),
    _relation("Sample", "COLLECTED_FROM", "采自", "Formation"),
    _relation("Sample", "COLLECTED_FROM", "采自", "StratigraphicMember"),
    _relation("Sample", "COLLECTED_FROM", "采自", "ReservoirInterval"),
    _relation("Lithology", "COMPOSED_OF", "由……组成", "Mineral"),
    _relation("Parameter", "DESCRIBES", "描述", "Sample"),
    _relation("Parameter", "DESCRIBES", "描述", "Lithology"),
    _relation("Parameter", "DESCRIBES", "描述", "Reservoir"),
    _relation("Parameter", "DESCRIBES", "描述", "PoreStructure"),
)


def build_table_query_text(table: RecognizedTable) -> str:
    """用表题、章节、表头和代表行构造受控长度的 Schema 查询文本。"""

    representative_rows = [" | ".join(row) for row in table.grid.matrix[:8]]
    parts = [
        table.source.section_title,
        table.source.caption,
        "；".join(table.grid.header_paths),
        "\n".join(representative_rows),
    ]
    return "\n".join(part for part in parts if part).strip()[:4000]


def infer_domain_types(text: str) -> dict[str, str]:
    """依据表题和表头关键词推断需要召回的领域概念类型。"""

    result: dict[str, str] = {}
    for keyword, (schema, zh_name) in DOMAIN_HINTS.items():
        if keyword in text:
            result.setdefault(schema, zh_name)
    return result


class TableSchemaSelector:
    """组合固定表格结构 Schema、Neo4j 领域概念和合法关系。"""

    def __init__(self, repository: Neo4jSchemaRepository | None = None) -> None:
        """保存可注入的只读仓储，便于离线测试和连接失败回退。"""

        self.repository = repository or Neo4jSchemaRepository()

    def select(
        self,
        table: RecognizedTable,
        *,
        required_types: Mapping[str, str] | None = None,
    ) -> RelevantSchema:
        """为单表选择结构概念、领域概念及其诱导关系子图，并纳入语义阶段已确认的类型。"""

        query_text = build_table_query_text(table)
        hinted_types = infer_domain_types(query_text)
        # 中文说明：主键语义（如“分析号”→Sample）是创建领域实体的直接依据，必须同步加入局部 Schema。
        hinted_types.update({
            str(schema): str(zh_name or schema)
            for schema, zh_name in (required_types or {}).items()
            if str(schema).strip()
        })
        selected: dict[str, SchemaConcept] = {
            schema: SchemaConcept(
                schema=schema,
                zh_name=zh_name,
                category="通用表格图片与公式",
                description=description,
                lexical_score=1.0,
                final_score=1.0,
                selection_reasons=("表格结构固定概念",),
            )
            for schema, (zh_name, description) in STRUCTURAL_CONCEPTS.items()
        }
        fallback_used = False
        database_concepts: dict[str, SchemaConcept] = {}
        try:
            database_concepts = {item.schema: item for item in self.repository.all_concepts()}
        except Exception:
            fallback_used = True
        if set(STRUCTURAL_CONCEPTS) - set(database_concepts):
            # 中文说明：数据库可连接但尚未执行通用表格 CQL 时，也要明确标记为静态结构 Schema 回退。
            fallback_used = True

        for schema, zh_name in hinted_types.items():
            concept = database_concepts.get(schema)
            if concept is None:
                concept = SchemaConcept(
                    schema=schema,
                    zh_name=zh_name,
                    category="领域概念回退",
                    lexical_score=0.8,
                    final_score=0.8,
                    selection_reasons=("表头关键词命中",),
                )
            else:
                concept = replace(
                    concept,
                    lexical_score=max(concept.lexical_score, 0.8),
                    final_score=max(concept.final_score, 0.8),
                    selection_reasons=tuple(dict.fromkeys((*concept.selection_reasons, "表头关键词命中"))),
                )
            selected[schema] = concept

        schema_names = set(selected)
        relations: dict[tuple[str, str, str], SchemaRelation] = {
            item.key: item
            for item in STATIC_RELATIONS
            if item.source_schema in schema_names and item.target_schema in schema_names
        }
        if database_concepts:
            try:
                for relation in self.repository.induced_relations(sorted(schema_names)):
                    normalized = SchemaRelation(
                        relation.source_schema,
                        relation.relation_en.upper(),
                        relation.relation_zh,
                        relation.target_schema,
                        relation.edge_score,
                    )
                    relations[normalized.key] = normalized
            except Exception:
                fallback_used = True

        terms = tuple(dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z0-9_-]*|[\u3400-\u9fff]{2,}", query_text)))[:80]
        confidence = 0.95 if database_concepts else 0.65
        return RelevantSchema(
            concepts=tuple(sorted(selected.values(), key=lambda item: item.schema)),
            relations=tuple(sorted(relations.values(), key=lambda item: item.key)),
            core_categories=("通用表格图片与公式",),
            query_terms=terms,
            selection_confidence=confidence,
            fallback_used=fallback_used or not database_concepts,
            selector_version="table_structural_neo4j_v1",
        )


__all__ = [
    "STATIC_RELATIONS",
    "STRUCTURAL_CONCEPTS",
    "TableSchemaSelector",
    "build_table_query_text",
    "infer_domain_types",
]
