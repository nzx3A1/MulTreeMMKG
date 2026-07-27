"""表格模态抽取总入口：识别 HTML、解析行列并生成统一 Graph。

该模块同时提供可导入的 ``extract_from_tables`` 和直接运行 CLI。结构图始终由
确定性规则生成；领域实体只在表头或记录键提供明确证据时创建。
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

# 中文说明：允许直接执行本文件，并保持与项目其他阶段相同的仓库根目录导入方式。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Entity, Graph, Relation, SourceModality
from src.graph.graph_validator import validate_graph
from src.utils.json_io import read_json, write_json
from src.utils.llm_client import DisabledLLMClient, LLMClient

from src.extractors.table_extractor.schema_models import (
    RecognizedTable,
    TableCell,
    TableSemanticPlan,
    TableSource,
    TableTaskReport,
)
from src.extractors.table_extractor.schema_selector import TableSchemaSelector
from src.extractors.table_extractor.table_parse import (
    RapidTableRecognizer,
    adapt_table_chunk,
    collect_table_sources,
    recognize_table_source,
)


DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "stage_02_document_tree.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "stage_04_table_extraction.json"
DEFAULT_RECOGNITION_PATH = PROJECT_ROOT / "output" / "stage_04_table_recognition.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "output" / "stage_04_table_tasks.json"
DEFAULT_WORK_DIR = PROJECT_ROOT / "output" / "table_extraction"

SUBJECT_HEADER_TYPES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("样品编号", "样品号", "样品", "样本"), "Sample", "样品"),
    (("分析号", "测点编号", "记录号"), "Sample", "样品"),
    (("井号", "井名"), "Well", "井"),
    (("岩性", "岩石类型", "岩石"), "Lithology", "岩性"),
    (("储层", "储层类型"), "Reservoir", "储层/储集层"),
    (("层位", "目的层", "储层段"), "ReservoirInterval", "目的层/储层段"),
    (("地层组", "组名"), "Formation", "组"),
    (("段名",), "StratigraphicMember", "段"),
    (("编号",), "Sample", "样品"),
)
MINERAL_NAMES = ("石英", "钾长石", "斜长石", "长石", "方解石", "白云石", "铁白云石", "黄铁矿", "黏土矿物", "粘土矿物")
DOMAIN_TYPE_ZH = {
    "Sample": "样品", "Well": "井", "Lithology": "岩性", "Reservoir": "储层/储集层",
    "ReservoirInterval": "目的层/储层段", "Formation": "组", "StratigraphicMember": "段",
    "Mineral": "矿物",
}


def _stable_id(prefix: str, *parts: Any) -> str:
    """生成重跑稳定的实体或关系 ID。"""

    raw = "|".join(str(part or "").strip().lower() for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"


def _clean_name(value: Any) -> str:
    """清理实体名中的连续空白和常见空值。"""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return "" if text.lower() in {"", "-", "–", "—", "无", "n/a", "na"} else text


def _unit_from_header(header: str) -> str:
    """从规范表头路径提取单位。"""

    bracket = re.search(r"[（(]([^()（）]{1,20})[)）]", header)
    if bracket:
        return _clean_name(bracket.group(1))
    slash = re.search(r"/\s*([^/\s]{1,20})(?:\s|$)", header)
    return _clean_name(slash.group(1)) if slash else ""


def _infer_subject(header: str) -> tuple[str, str] | None:
    """根据主键表头推断记录对应的领域实体类型。"""

    normalized = header.replace(" ", "")
    for keywords, schema, zh_name in SUBJECT_HEADER_TYPES:
        if any(keyword in normalized for keyword in keywords):
            return schema, zh_name
    return None


def _subject_candidate(header: str) -> tuple[int, str, str] | None:
    """返回表头领域类型及全局优先级，使样品键优先于位置更靠前的井名。"""

    normalized = header.replace(" ", "")
    for priority, (keywords, schema, zh_name) in enumerate(SUBJECT_HEADER_TYPES):
        if any(keyword in normalized for keyword in keywords):
            return priority, schema, zh_name
    return None


def _refine_context_type(header: str, value: str, inferred_type: str) -> str:
    """依据层位值后缀把泛化层位进一步区分为地层组、地层段或储层段。"""

    cleaned = _clean_name(value).replace(" ", "")
    normalized_header = header.replace(" ", "")
    if inferred_type != "ReservoirInterval" or "层位" not in normalized_header:
        return inferred_type
    if cleaned.endswith("组"):
        return "Formation"
    if cleaned.endswith("段"):
        return "StratigraphicMember"
    return inferred_type


def infer_semantic_plan(table: RecognizedTable, llm_client: Any | None = None) -> TableSemanticPlan:
    """用规则优先确定记录方向、数据起点和领域主键，必要时允许 LLM 补充。"""

    grid = table.grid
    header_end = max(grid.header_rows, default=-1)
    plan = TableSemanticPlan(
        orientation=grid.orientation,
        data_start_row=max(0, header_end + 1),
        data_start_col=max(grid.header_columns, default=-1) + 1,
        confidence=0.45,
        reasons=["使用 HTML 网格方向与表头边界"],
    )

    if grid.orientation == "vertical":
        # 中文说明：转置表从各行左侧标签中收集候选，再按样品、井、岩性等领域优先级选择主键。
        candidates: list[tuple[int, int, int, str, str, str]] = []
        for row_index, row in enumerate(grid.matrix):
            cleaned_row = [_clean_name(value) for value in row]
            nonempty = [value for value in cleaned_row if value]
            if row_index in grid.header_rows or len(set(nonempty)) <= 1:
                continue
            if nonempty and (nonempty[0].startswith("注") or re.match(r"^(?:表|Table)\s*\d", nonempty[0], re.IGNORECASE)):
                continue
            label_width = min(len(row), max(grid.header_columns, default=0) + 1)
            for column_index, value in enumerate(row[:label_width]):
                candidate = _subject_candidate(value)
                if candidate is None:
                    continue
                priority, schema, zh_name = candidate
                candidates.append((priority, row_index, column_index, schema, zh_name, value))
        if candidates:
            _, row_index, column_index, schema, zh_name, value = min(candidates)
            plan.subject_row = row_index
            plan.data_start_col = column_index + 1
            plan.subject_type, plan.subject_type_zh = schema, zh_name
            plan.confidence = 0.9
            plan.reasons.append(f"转置表主键行命中：{value}")
            return plan
    else:
        # 中文说明：横向表在全部列中按领域优先级选择，避免前置井名遮蔽更细粒度的样品号。
        candidates = []
        for column_index, header in enumerate(grid.header_paths):
            candidate = _subject_candidate(header)
            if candidate is None:
                continue
            priority, schema, zh_name = candidate
            candidates.append((priority, column_index, schema, zh_name, header))
        if candidates:
            _, column_index, schema, zh_name, header = min(candidates)
            plan.subject_column = column_index
            plan.subject_type, plan.subject_type_zh = schema, zh_name
            plan.confidence = 0.9
            plan.reasons.append(f"横向表主键列命中：{header}")
            return plan

    if llm_client is not None and not isinstance(llm_client, DisabledLLMClient):
        try:
            prompt = {
                "task": "判断表格记录主键，只输出JSON",
                "caption": table.source.caption,
                "section": table.source.section_title,
                "orientation": grid.orientation,
                "headers": grid.header_paths,
                "sample_rows": grid.matrix[:8],
                "output": {
                    "subject_column": "横向表主键列索引或null",
                    "subject_row": "转置表主键行索引或null",
                    "subject_type": "Neo4j EntityConcept.schema 或 TableRow",
                    "subject_type_zh": "中文类型",
                },
            }
            payload = llm_client.call_json(prompt, task_name="表格主键语义判断")
            if isinstance(payload, Mapping):
                subject_column = payload.get("subject_column")
                subject_row = payload.get("subject_row")
                if subject_column is not None and 0 <= int(subject_column) < grid.column_count:
                    plan.subject_column = int(subject_column)
                if subject_row is not None and 0 <= int(subject_row) < grid.row_count:
                    plan.subject_row = int(subject_row)
                plan.subject_type = str(payload.get("subject_type") or "TableRow")
                plan.subject_type_zh = str(payload.get("subject_type_zh") or "表格行")
                if plan.subject_column is not None or plan.subject_row is not None:
                    plan.confidence = 0.7
                    plan.reasons.append("LLM 补充主键计划")
        except Exception as exc:
            plan.reasons.append(f"LLM 主键判断失败，保留规则计划：{exc}")
    return plan


class _GraphAssembler:
    """集中创建实体和关系，并保证 ID、端点字段及 Schema 方向一致。"""

    def __init__(self, table: RecognizedTable, schema: Any) -> None:
        """初始化单表图谱缓存和合法关系索引。"""

        self.table = table
        self.schema = schema
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}
        self.concept_map = schema.concept_map
        self.allowed_relations = {
            (item.source_schema, item.relation_en.upper(), item.target_schema): item
            for item in schema.relations
        }
        self.skipped_relations: list[dict[str, str]] = []

    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        provenance: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Entity:
        """创建或复用实体，并从局部 Schema 补齐中文类型。"""

        if entity_id in self.entities:
            return self.entities[entity_id]
        concept = self.concept_map.get(entity_type)
        entity = Entity(
            id=entity_id,
            name=name,
            official_name=name,
            type=entity_type,
            type_zh=concept.zh_name if concept else None,
            attributes=dict(attributes or {}),
            provenance=provenance,
            metadata=dict(metadata or {}),
        )
        self.entities[entity_id] = entity
        return entity

    def add_relation(
        self,
        source: Entity,
        relation_type: str,
        target: Entity,
        *,
        provenance: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> Relation | None:
        """仅创建 Schema 中存在的有向关系，非法候选记录到跳过列表。"""

        normalized_type = relation_type.upper()
        rule = self.allowed_relations.get((source.type, normalized_type, target.type))
        if rule is None:
            self.skipped_relations.append({
                "source_type": source.type,
                "relation": normalized_type,
                "target_type": target.type,
            })
            return None
        relation_id = _stable_id("rel", source.id, normalized_type, target.id)
        relation = Relation(
            id=relation_id,
            type=normalized_type,
            relation_name=rule.relation_en,
            type_zh=rule.relation_zh,
            source_id=source.id,
            source_name=source.name,
            source_type=source.type,
            target_id=target.id,
            target_name=target.name,
            target_type=target.type,
            attributes=dict(attributes or {}),
            provenance=provenance,
            metadata={"schema_validated": True},
        )
        self.relations[relation_id] = relation
        return relation


def _cell_provenance(table: RecognizedTable, cell: TableCell | None = None) -> str:
    """生成可定位到图片和逻辑坐标的表格证据字符串。"""

    image = table.source.image_path or table.source.chunk_id
    if cell is None:
        return f"{image} | table={table.source.task_id}"
    return (
        f"{image} | table={table.source.task_id} | "
        f"row={cell.row_start}-{cell.row_end},col={cell.col_start}-{cell.col_end} | {cell.raw_text}"
    )


def _cell_at(table: RecognizedTable, row: int, column: int) -> TableCell | None:
    """查找覆盖指定逻辑坐标的原始单元格。"""

    return next((
        cell for cell in table.grid.cells
        if cell.row_start <= row <= cell.row_end and cell.col_start <= column <= cell.col_end
    ), None)


def _normalized_cell_attributes(cell: TableCell) -> dict[str, Any]:
    """把单元格文本、数值归一化和空间信息装配为图谱属性。"""

    return {
        "raw_value": cell.raw_text,
        "value": cell.text,
        "value_type": cell.value_type,
        "numeric_value": cell.numeric_value,
        "min_value": cell.min_value,
        "max_value": cell.max_value,
        "comparator": cell.comparator,
        "unit": cell.unit,
        "is_missing": cell.is_missing,
        "row_start": cell.row_start,
        "row_end": cell.row_end,
        "col_start": cell.col_start,
        "col_end": cell.col_end,
        "rowspan": cell.rowspan,
        "colspan": cell.colspan,
        "bbox": cell.bbox,
        "ocr_confidence": cell.ocr_confidence,
    }


def _build_structural_graph(
    table: RecognizedTable,
    assembler: _GraphAssembler,
) -> dict[str, Any]:
    """创建 Table/Header/Row/Column/Cell/Parameter/Unit 结构节点与关系。"""

    source = table.source
    table_id = _stable_id("table", source.document_id, source.task_id)
    table_name = source.caption or source.section_title or source.task_id
    table_entity = assembler.add_entity(
        table_id,
        table_name,
        "Table",
        attributes={
            "chunk_id": source.chunk_id,
            "task_id": source.task_id,
            "section_id": source.section_id,
            "section_title": source.section_title,
            "source_path": source.image_path,
            "row_count": table.grid.row_count,
            "column_count": table.grid.column_count,
            "orientation": table.grid.orientation,
            "recognition_engine": table.engine,
            "recognition_score": table.quality.score,
            "canonical_html_path": str(Path(table.artifact_dir) / "canonical.html"),
        },
        provenance=_cell_provenance(table),
        metadata={"source_modality": "table"},
    )

    if source.caption:
        caption = assembler.add_entity(
            _stable_id("caption", table_id, source.caption),
            source.caption,
            "Caption",
            attributes={"text": source.caption},
            provenance=_cell_provenance(table),
        )
        assembler.add_relation(table_entity, "HAS_CAPTION", caption, provenance=_cell_provenance(table))

    rows: dict[int, Entity] = {}
    for row_index in range(table.grid.row_count):
        row_entity = assembler.add_entity(
            _stable_id("table_row", table_id, row_index),
            f"{table_name} 第{row_index + 1}行",
            "TableRow",
            attributes={"row_index": row_index, "values": table.grid.matrix[row_index]},
            provenance=f"{_cell_provenance(table)} | row={row_index}",
        )
        rows[row_index] = row_entity
        assembler.add_relation(table_entity, "HAS_ROW", row_entity, provenance=row_entity.provenance)
        assembler.add_relation(row_entity, "PART_OF", table_entity, provenance=row_entity.provenance)

    columns: dict[int, Entity] = {}
    column_parameters: dict[int, Entity] = {}
    row_parameters: dict[int, Entity] = {}
    units: dict[str, Entity] = {}
    is_vertical = table.grid.orientation == "vertical"
    for column_index in range(table.grid.column_count):
        header = table.grid.header_paths[column_index] if column_index < len(table.grid.header_paths) else f"列{column_index + 1}"
        column_entity = assembler.add_entity(
            _stable_id("table_col", table_id, column_index),
            f"{table_name} / {header}",
            "TableColumn",
            attributes={"column_index": column_index, "header_path": header},
            provenance=f"{_cell_provenance(table)} | column={column_index} | {header}",
        )
        columns[column_index] = column_entity
        assembler.add_relation(table_entity, "HAS_COLUMN", column_entity, provenance=column_entity.provenance)
        assembler.add_relation(column_entity, "PART_OF", table_entity, provenance=column_entity.provenance)

        if not is_vertical and not header.startswith("列"):
            parameter_name = re.sub(r"\s*/\s*[^/]+$", "", header).strip() or header
            parameter = assembler.add_entity(
                _stable_id("parameter", table_id, parameter_name),
                parameter_name,
                "Parameter",
                attributes={"header_path": header},
                provenance=column_entity.provenance,
            )
            column_parameters[column_index] = parameter
            assembler.add_relation(column_entity, "REPRESENTS", parameter, provenance=column_entity.provenance)
            unit_name = _unit_from_header(header)
            if unit_name:
                unit = units.get(unit_name) or assembler.add_entity(
                    _stable_id("unit", unit_name),
                    unit_name,
                    "Unit",
                    attributes={"symbol": unit_name},
                    provenance=column_entity.provenance,
                )
                units[unit_name] = unit
                assembler.add_relation(column_entity, "HAS_UNIT", unit, provenance=column_entity.provenance)

    header_set = set(table.grid.header_rows)
    header_columns = set(table.grid.header_columns)
    if is_vertical:
        label_column = max(header_columns, default=0)
        for row_index in range(table.grid.row_count):
            if row_index in header_set:
                continue
            label_cell = _cell_at(table, row_index, label_column)
            parameter_name = _clean_name(label_cell.text if label_cell else "")
            if (
                not parameter_name
                or (label_cell is not None and label_cell.colspan >= table.grid.column_count)
                or parameter_name.startswith("注")
                or re.match(r"^(?:表|Table)\s*\d", parameter_name, re.IGNORECASE)
            ):
                continue
            parameter = assembler.add_entity(
                _stable_id("parameter", table_id, "row", row_index, parameter_name),
                parameter_name,
                "Parameter",
                attributes={"row_index": row_index, "orientation": "vertical"},
                provenance=_cell_provenance(table, label_cell),
            )
            row_parameters[row_index] = parameter
            unit_name = _unit_from_header(parameter_name)
            if unit_name:
                unit = units.get(unit_name) or assembler.add_entity(
                    _stable_id("unit", unit_name),
                    unit_name,
                    "Unit",
                    attributes={"symbol": unit_name},
                    provenance=_cell_provenance(table, label_cell),
                )
                units[unit_name] = unit
        # 中文说明：转置表的参数沿行分布，因此不把 YC101 等样品列名误建为 Parameter。

    headers: list[Entity] = []
    for cell in table.grid.cells:
        if cell.row_start not in header_set and cell.col_start not in header_columns:
            continue
        if not _clean_name(cell.text):
            continue
        header_entity = assembler.add_entity(
            _stable_id("table_header", table_id, cell.cell_id, cell.text),
            cell.text,
            "TableHeader",
            attributes=_normalized_cell_attributes(cell),
            provenance=_cell_provenance(table, cell),
        )
        headers.append(header_entity)
        assembler.add_relation(table_entity, "HAS_HEADER", header_entity, provenance=header_entity.provenance)
        assembler.add_relation(header_entity, "PART_OF", table_entity, provenance=header_entity.provenance)
        for column_index in range(cell.col_start, cell.col_end + 1):
            if column_index in columns:
                assembler.add_relation(header_entity, "DEFINES_COLUMN", columns[column_index], provenance=header_entity.provenance)
            if column_index in column_parameters:
                assembler.add_relation(header_entity, "REPRESENTS", column_parameters[column_index], provenance=header_entity.provenance)
        if cell.row_start in row_parameters and cell.col_start in header_columns:
            assembler.add_relation(
                header_entity,
                "REPRESENTS",
                row_parameters[cell.row_start],
                provenance=header_entity.provenance,
            )

    cell_entities: dict[str, Entity] = {}
    for cell in table.grid.cells:
        cell_entity = assembler.add_entity(
            _stable_id("table_cell", table_id, cell.cell_id),
            cell.text or f"空单元格 {cell.cell_id}",
            "TableCell",
            attributes=_normalized_cell_attributes(cell),
            provenance=_cell_provenance(table, cell),
        )
        cell_entities[cell.cell_id] = cell_entity
        assembler.add_relation(table_entity, "HAS_CELL", cell_entity, provenance=cell_entity.provenance)
        assembler.add_relation(cell_entity, "PART_OF", table_entity, provenance=cell_entity.provenance)
        for row_index in range(cell.row_start, cell.row_end + 1):
            if row_index in rows:
                assembler.add_relation(cell_entity, "LOCATED_IN_ROW", rows[row_index], provenance=cell_entity.provenance)
                assembler.add_relation(rows[row_index], "CONTAINS_CELL", cell_entity, provenance=cell_entity.provenance)
        for column_index in range(cell.col_start, cell.col_end + 1):
            if column_index in columns:
                assembler.add_relation(cell_entity, "LOCATED_IN_COLUMN", columns[column_index], provenance=cell_entity.provenance)
                assembler.add_relation(columns[column_index], "CONTAINS_CELL", cell_entity, provenance=cell_entity.provenance)
            if column_index in column_parameters and cell.row_start not in header_set:
                assembler.add_relation(cell_entity, "REPRESENTS", column_parameters[column_index], provenance=cell_entity.provenance)
            unit_name = cell.unit or _unit_from_header(table.grid.header_paths[column_index] if column_index < len(table.grid.header_paths) else "")
            if unit_name and unit_name in units:
                assembler.add_relation(cell_entity, "HAS_UNIT", units[unit_name], provenance=cell_entity.provenance)
        if is_vertical and cell.col_start > max(header_columns, default=0):
            for row_index in range(cell.row_start, cell.row_end + 1):
                if row_index in row_parameters:
                    parameter = row_parameters[row_index]
                    assembler.add_relation(cell_entity, "REPRESENTS", parameter, provenance=cell_entity.provenance)
                    unit_name = _unit_from_header(parameter.name)
                    if unit_name and unit_name in units:
                        assembler.add_relation(cell_entity, "HAS_UNIT", units[unit_name], provenance=cell_entity.provenance)

    return {
        "table": table_entity,
        "rows": rows,
        "columns": columns,
        "parameters": {**column_parameters, **{f"row:{key}": value for key, value in row_parameters.items()}},
        "column_parameters": column_parameters,
        "row_parameters": row_parameters,
        "cells": cell_entities,
        "headers": headers,
    }


def _row_attributes(table: RecognizedTable, row_index: int) -> dict[str, Any]:
    """将一行数据按完整表头路径转换为领域实体属性。"""

    attributes: dict[str, Any] = {}
    normalized_values: list[dict[str, Any]] = []
    for column_index, value in enumerate(table.grid.matrix[row_index]):
        header = table.grid.header_paths[column_index] if column_index < len(table.grid.header_paths) else f"列{column_index + 1}"
        attributes[header] = value
        cell = _cell_at(table, row_index, column_index)
        if cell and cell.value_type in {"number", "range"}:
            normalized_values.append({
                "field": header,
                "raw_value": cell.raw_text,
                "numeric_value": cell.numeric_value,
                "min_value": cell.min_value,
                "max_value": cell.max_value,
                "comparator": cell.comparator,
                "unit": cell.unit or _unit_from_header(header),
            })
    attributes["normalized_values"] = normalized_values
    return attributes


def _vertical_attributes(table: RecognizedTable, column_index: int, data_start_col: int) -> dict[str, Any]:
    """将转置表的一列转换为领域实体属性。"""

    attributes: dict[str, Any] = {}
    normalized_values: list[dict[str, Any]] = []
    label_width = max(1, min(data_start_col, table.grid.column_count - 1))
    for row_index, row in enumerate(table.grid.matrix):
        labels = []
        for label_col in range(label_width):
            value = _clean_name(row[label_col])
            if value and (not labels or labels[-1] != value):
                labels.append(value)
        key = " / ".join(labels) or f"行{row_index + 1}"
        value = row[column_index] if column_index < len(row) else ""
        attributes[key] = value
        cell = _cell_at(table, row_index, column_index)
        if cell and cell.value_type in {"number", "range"}:
            normalized_values.append({
                "field": key,
                "raw_value": cell.raw_text,
                "numeric_value": cell.numeric_value,
                "min_value": cell.min_value,
                "max_value": cell.max_value,
                "comparator": cell.comparator,
                "unit": cell.unit or _unit_from_header(key),
            })
    attributes["normalized_values"] = normalized_values
    return attributes


def _add_domain_semantics(
    table: RecognizedTable,
    plan: TableSemanticPlan,
    assembler: _GraphAssembler,
    structural: Mapping[str, Any],
) -> list[Entity]:
    """依据主键计划创建领域实体，并附加确定性属性与 Schema 合法关系。"""

    if plan.subject_type == "TableRow":
        return []
    table_entity: Entity = structural["table"]
    domain_entities: list[Entity] = []
    record_entities_by_row: dict[int, Entity] = {}
    lithology_entities_by_row: dict[int, Entity] = {}
    if plan.orientation == "vertical" and plan.subject_row is not None:
        for column_index in range(plan.data_start_col, table.grid.column_count):
            name = _clean_name(table.grid.matrix[plan.subject_row][column_index])
            if not name:
                continue
            cell = _cell_at(table, plan.subject_row, column_index)
            entity = assembler.add_entity(
                _stable_id("domain", table.source.document_id, plan.subject_type, name),
                name,
                plan.subject_type,
                attributes=_vertical_attributes(table, column_index, plan.data_start_col),
                provenance=_cell_provenance(table, cell),
                metadata={"table_record_axis": "column", "column_index": column_index},
            )
            domain_entities.append(entity)
    elif plan.subject_column is not None:
        for row_index in range(plan.data_start_row, table.grid.row_count):
            name = _clean_name(table.grid.matrix[row_index][plan.subject_column])
            if not name:
                continue
            cell = _cell_at(table, row_index, plan.subject_column)
            entity = assembler.add_entity(
                _stable_id("domain", table.source.document_id, plan.subject_type, name),
                name,
                plan.subject_type,
                attributes=_row_attributes(table, row_index),
                provenance=_cell_provenance(table, cell),
                metadata={"table_record_axis": "row", "row_index": row_index},
            )
            domain_entities.append(entity)
            record_entities_by_row[row_index] = entity
            if plan.subject_type == "Lithology":
                lithology_entities_by_row[row_index] = entity

    table_relation = {
        "Sample": "REPORTS",
        "Experiment": "REPORTS",
        "AnalyticalMethod": "REPORTS",
        "Lithology": "DESCRIBES",
        "Reservoir": "DESCRIBES",
        "ReservoirInterval": "DESCRIBES",
        "Formation": "DESCRIBES",
        "StratigraphicMember": "DESCRIBES",
    }.get(plan.subject_type)
    for entity in domain_entities:
        if table_relation:
            assembler.add_relation(table_entity, table_relation, entity, provenance=entity.provenance)
        if entity.type == "Sample":
            assembler.add_relation(entity, "RECORDED_IN", table_entity, provenance=entity.provenance)
        for parameter in structural["parameters"].values():
            assembler.add_relation(parameter, "DESCRIBES", entity, provenance=entity.provenance)

    if plan.subject_type == "Sample" and plan.subject_column is not None:
        # 中文说明：样品表逐行建立 Sample→井/地层/岩性上下文，属性值仍保留在样品和原始单元格中。
        context_types = {"Well", "Formation", "StratigraphicMember", "ReservoirInterval", "Lithology"}
        for row_index, sample in record_entities_by_row.items():
            for column_index, header in enumerate(table.grid.header_paths):
                if column_index == plan.subject_column:
                    continue
                inferred = _infer_subject(header)
                if inferred is None:
                    continue
                raw_type, _ = inferred
                value = _clean_name(table.grid.matrix[row_index][column_index])
                context_type = _refine_context_type(header, value, raw_type)
                if not value or context_type not in context_types:
                    continue
                cell = _cell_at(table, row_index, column_index)
                context = assembler.add_entity(
                    _stable_id("domain", table.source.document_id, context_type, value),
                    value,
                    context_type,
                    attributes={"source_header": header},
                    provenance=_cell_provenance(table, cell),
                    metadata={"table_context": True},
                )
                if context not in domain_entities:
                    domain_entities.append(context)
                assembler.add_relation(sample, "COLLECTED_FROM", context, provenance=_cell_provenance(table, cell))
                if context_type in {"Formation", "StratigraphicMember", "ReservoirInterval", "Lithology"}:
                    assembler.add_relation(table_entity, "DESCRIBES", context, provenance=_cell_provenance(table, cell))
                if context_type == "Lithology":
                    lithology_entities_by_row[row_index] = context

    # 中文说明：存在明确岩性实体时才建立 Lithology→Mineral；无岩性证据的样品表不虚构岩石类别。
    if lithology_entities_by_row:
        for column_index, header in enumerate(table.grid.header_paths):
            mineral_name = next((name for name in MINERAL_NAMES if name in header), "")
            if not mineral_name:
                continue
            mineral = assembler.add_entity(
                _stable_id("mineral", mineral_name),
                mineral_name,
                "Mineral",
                attributes={"source_header": header},
                provenance=f"{_cell_provenance(table)} | column={column_index}",
            )
            if mineral not in domain_entities:
                domain_entities.append(mineral)
            for row_index, entity in lithology_entities_by_row.items():
                cell = _cell_at(table, row_index, column_index)
                if cell is not None and not cell.is_missing:
                    assembler.add_relation(entity, "COMPOSED_OF", mineral, provenance=_cell_provenance(table, cell))
    return domain_entities


def _required_domain_types(table: RecognizedTable, plan: TableSemanticPlan) -> dict[str, str]:
    """在选择局部 Schema 前收集主键及样品上下文可能创建的全部领域类型。"""

    required = {plan.subject_type: plan.subject_type_zh} if plan.subject_type else {}
    if plan.subject_type != "Sample" or plan.subject_column is None:
        return required
    context_types = {"Well", "Formation", "StratigraphicMember", "ReservoirInterval", "Lithology"}
    for column_index, header in enumerate(table.grid.header_paths):
        inferred = _infer_subject(header)
        if inferred is None or column_index == plan.subject_column:
            continue
        inferred_type, _ = inferred
        values = [
            _clean_name(table.grid.matrix[row_index][column_index])
            for row_index in range(plan.data_start_row, table.grid.row_count)
        ]
        for value in values:
            context_type = _refine_context_type(header, value, inferred_type)
            if value and context_type in context_types:
                required[context_type] = DOMAIN_TYPE_ZH[context_type]
    if any(any(name in header for name in MINERAL_NAMES) for header in table.grid.header_paths):
        required["Mineral"] = DOMAIN_TYPE_ZH["Mineral"]
    return required


def build_table_graph(
    table: RecognizedTable,
    *,
    schema_selector: TableSchemaSelector | None = None,
    llm_client: Any | None = None,
) -> Graph:
    """将一张已通过质量门的标准表格转换为结构与领域统一 Graph。"""

    selector = schema_selector or TableSchemaSelector()
    plan = infer_semantic_plan(table, llm_client=llm_client)
    # 中文说明：先确定主键语义，再把该类型作为局部 Schema 的强制概念，避免实体生成与校验候选不一致。
    required_types = _required_domain_types(table, plan)
    schema = selector.select(table, required_types=required_types)
    assembler = _GraphAssembler(table, schema)
    structural = _build_structural_graph(table, assembler)
    domain_entities = _add_domain_semantics(table, plan, assembler, structural)
    graph = Graph.from_chunk(
        table.source.document_id,
        table.source.task_id,
        SourceModality.TABLE,
        entities=assembler.entities.values(),
        relations=assembler.relations.values(),
        stage="stage_04_table_extraction",
    )
    validation = validate_graph(graph, schema)
    graph.metadata.extra.update({
        "recognition": {
            "engine": table.engine,
            "artifact_dir": table.artifact_dir,
            "quality": table.quality.to_dict(),
            "row_count": table.grid.row_count,
            "column_count": table.grid.column_count,
        },
        "semantic_plan": plan.to_dict(),
        "schema_selection": schema.to_dict(),
        "domain_entity_count": len(domain_entities),
        "skipped_schema_relations": assembler.skipped_relations,
        "validation": validation,
    })
    return graph


def extract_from_tables(
    chunks: Sequence[Mapping[str, Any] | TableSource],
    llm_client: Any | None = None,
    *,
    output_path: str | Path | None = None,
    recognition_output_path: str | Path | None = None,
    report_output_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    engine: str = "auto",
    rapid_model: str = "unitable",
    device: str = "auto",
    ocr_device: str = "cpu",
    ocr_backend: str = "onnxruntime",
    ocr_limit_side_len: int = 1600,
    use_llm_semantic: bool = False,
    minimum_score: float = 0.55,
    show_progress: bool = True,
) -> list[Graph]:
    """批量识别表格并生成 Graph，单表失败不会中断其他任务。"""

    sources: list[TableSource] = []
    for item in chunks:
        if isinstance(item, TableSource):
            sources.append(item)
        elif isinstance(item, Mapping):
            sources.extend(adapt_table_chunk(item, allow_image_chunk=str(item.get("modality") or "").lower() == "image"))
    active_work_dir = Path(work_dir or DEFAULT_WORK_DIR)
    active_work_dir.mkdir(parents=True, exist_ok=True)
    recognizer = RapidTableRecognizer(
        model_name=rapid_model,
        device=device,
        ocr_device=ocr_device,
        ocr_backend=ocr_backend,
        ocr_limit_side_len=ocr_limit_side_len,
    ) if any(str(item.kind) == "image" for item in sources) else None
    selector = TableSchemaSelector()
    semantic_client = llm_client if use_llm_semantic else DisabledLLMClient()
    graphs: list[Graph] = []
    recognized_tables: list[RecognizedTable] = []
    reports: list[TableTaskReport] = []

    for index, source in enumerate(sources, start=1):
        if show_progress:
            print(f"[{index}/{len(sources)}] 表格任务 {source.task_id}，输入={source.image_path or source.kind}")
        try:
            recognized = recognize_table_source(
                source,
                work_dir=active_work_dir,
                recognizer=recognizer,
                engine=engine,
                minimum_score=minimum_score,
            )
            recognized_tables.append(recognized)
            if not recognized.quality.passed:
                reports.append(TableTaskReport(
                    task_id=source.task_id,
                    chunk_id=source.chunk_id,
                    status="manual_review",
                    engine=recognized.engine,
                    artifact_dir=recognized.artifact_dir,
                    errors=recognized.quality.errors,
                    warnings=recognized.quality.warnings,
                ))
                if show_progress:
                    print(f"  质量门未通过：score={recognized.quality.score:.3f}，已保留人工复核产物")
                continue
            graph = build_table_graph(recognized, schema_selector=selector, llm_client=semantic_client)
            graphs.append(graph)
            reports.append(TableTaskReport(
                task_id=source.task_id,
                chunk_id=source.chunk_id,
                status="success",
                engine=recognized.engine,
                artifact_dir=recognized.artifact_dir,
                graph_entity_count=len(graph.entities),
                graph_relation_count=len(graph.relations),
                warnings=recognized.quality.warnings,
            ))
            if show_progress:
                print(
                    f"  完成：{recognized.grid.row_count}行×{recognized.grid.column_count}列，"
                    f"质量={recognized.quality.score:.3f}，实体={len(graph.entities)}，关系={len(graph.relations)}"
                )
        except Exception as exc:
            reports.append(TableTaskReport(
                task_id=source.task_id,
                chunk_id=source.chunk_id,
                status="failed",
                errors=[str(exc)],
            ))
            if show_progress:
                print(f"  失败：{exc}")

    if recognition_output_path is not None:
        write_json(recognition_output_path, [item.to_dict() for item in recognized_tables])
    if output_path is not None:
        write_json(output_path, [graph.to_dict() for graph in graphs])
    if report_output_path is not None:
        write_json(report_output_path, [report.to_dict() for report in reports])
    return graphs


def _build_cli() -> argparse.ArgumentParser:
    """构建表格抽取命令行参数。"""

    parser = argparse.ArgumentParser(description="将 Stage-02/03 表格转换为标准 HTML 和表格知识图谱")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Stage-02/03 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="表格 Graph JSON")
    parser.add_argument("--recognition-output", type=Path, default=DEFAULT_RECOGNITION_PATH, help="HTML与网格识别 JSON")
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH, help="逐表任务报告 JSON")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="逐表 HTML、明细和可视化目录")
    parser.add_argument("--engine", choices=("auto", "rapidtable", "mineru"), default="auto", help="图片表格识别引擎")
    parser.add_argument("--rapid-model", choices=("unitable", "slanetplus", "ppstructure_zh"), default="unitable", help="RapidTable 结构模型")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="RapidTable 推理设备")
    parser.add_argument("--ocr-device", choices=("auto", "cpu", "cuda"), default="cpu", help="RapidOCR 推理设备；默认CPU以降低显存峰值")
    parser.add_argument("--ocr-backend", choices=("onnxruntime", "torch"), default="onnxruntime", help="服务器级OCR执行后端")
    parser.add_argument("--ocr-limit-side-len", type=int, default=1600, help="OCR检测最长边；默认覆盖原图且避免无意义放大")
    parser.add_argument("--minimum-score", type=float, default=0.55, help="HTML质量门最低分")
    parser.add_argument("--include-image-candidates", action="store_true", help="同时恢复被误分到 ImageChunk 的表格")
    parser.add_argument("--use-llm-semantic", action="store_true", help="规则无法确定主键时调用配置中的文本模型")
    parser.add_argument("--quiet", action="store_true", help="关闭逐表进度输出")
    return parser


def main() -> int:
    """读取阶段 JSON，执行表格识别和知识图谱转换并打印汇总。"""

    args = _build_cli().parse_args()
    payload = read_json(args.input)
    if not isinstance(payload, Mapping):
        raise TypeError("输入 JSON 顶层必须是对象")
    sources = collect_table_sources(payload, include_image_candidates=args.include_image_candidates)
    print(
        f"表格抽取环境：Python={sys.executable}，任务={len(sources)}，"
        f"引擎={args.engine}，RapidTable模型={args.rapid_model}，设备={args.device}"
    )
    client = LLMClient() if args.use_llm_semantic else DisabledLLMClient()
    graphs = extract_from_tables(
        sources,
        client,
        output_path=args.output,
        recognition_output_path=args.recognition_output,
        report_output_path=args.report_output,
        work_dir=args.work_dir,
        engine=args.engine,
        rapid_model=args.rapid_model,
        device=args.device,
        ocr_device=args.ocr_device,
        ocr_backend=args.ocr_backend,
        ocr_limit_side_len=args.ocr_limit_side_len,
        use_llm_semantic=args.use_llm_semantic,
        minimum_score=max(0.0, min(1.0, args.minimum_score)),
        show_progress=not args.quiet,
    )
    print(f"表格抽取完成：成功生成 {len(graphs)} 个表格 Graph，输出：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_table_graph",
    "extract_from_tables",
    "infer_semantic_plan",
]
