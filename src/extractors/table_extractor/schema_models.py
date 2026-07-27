"""表格识别、结构解析与知识图谱转换使用的数据模型。

本模块只描述表格抽取阶段的中间数据，不把 OCR、HTML 解析或图谱装配逻辑
混入模型层。所有坐标均采用从 0 开始、结束位置为闭区间的逻辑行列坐标。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field

from model.base import MMKGBaseModel


class TableSourceKind(str, Enum):
    """描述表格原始内容的载体类型。"""

    IMAGE = "image"
    HTML = "html"
    MARKDOWN = "markdown"


class TableSource(MMKGBaseModel):
    """保存从 Stage-02 Chunk 归一化得到的单张表格输入。"""

    document_id: str = ""
    chunk_id: str
    task_id: str
    section_id: str = ""
    section_title: str = ""
    caption: str = ""
    references: list[str] = Field(default_factory=list)
    kind: TableSourceKind
    content: str = ""
    image_path: str | None = None
    source_chunk: dict[str, Any] = Field(default_factory=dict)


class TableCell(MMKGBaseModel):
    """表示 HTML 中一个未展开的原始单元格及其逻辑跨度。"""

    cell_id: str
    row_start: int = Field(ge=0)
    row_end: int = Field(ge=0)
    col_start: int = Field(ge=0)
    col_end: int = Field(ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    raw_text: str = ""
    text: str = ""
    is_header: bool = False
    is_missing: bool = False
    bbox: list[float] = Field(default_factory=list)
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    value_type: Literal["text", "number", "range", "missing"] = "text"
    numeric_value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    comparator: str = ""
    unit: str = ""


class TableGrid(MMKGBaseModel):
    """保存标准 HTML 展开后的矩形网格和表头语义。"""

    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    cells: list[TableCell] = Field(default_factory=list)
    matrix: list[list[str]] = Field(default_factory=list)
    header_rows: list[int] = Field(default_factory=list)
    header_columns: list[int] = Field(default_factory=list)
    header_paths: list[str] = Field(default_factory=list)
    orientation: Literal["horizontal", "vertical", "matrix", "unknown"] = "unknown"


class TableQualityReport(MMKGBaseModel):
    """记录 HTML 是否可进入后续行列解析与图谱生成。"""

    passed: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    nonempty_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    ocr_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    geometry_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RecognizedTable(MMKGBaseModel):
    """保存一次表格识别的标准 HTML、网格和可追溯中间信息。"""

    source: TableSource
    engine: str
    raw_html: str
    canonical_html: str
    grid: TableGrid
    quality: TableQualityReport
    details: dict[str, Any] = Field(default_factory=dict)
    artifact_dir: str = ""


class TableSemanticPlan(MMKGBaseModel):
    """描述从行列网格映射到领域对象时采用的确定性计划。"""

    orientation: Literal["horizontal", "vertical", "matrix", "unknown"] = "unknown"
    data_start_row: int = Field(default=0, ge=0)
    data_start_col: int = Field(default=0, ge=0)
    subject_column: int | None = Field(default=None, ge=0)
    subject_row: int | None = Field(default=None, ge=0)
    subject_type: str = "TableRow"
    subject_type_zh: str = "表格行"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class TableTaskReport(MMKGBaseModel):
    """记录单个表格任务的执行状态，失败任务也会保留原因。"""

    task_id: str
    chunk_id: str
    status: Literal["success", "failed", "manual_review"]
    engine: str = ""
    artifact_dir: str = ""
    graph_entity_count: int = 0
    graph_relation_count: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "RecognizedTable",
    "TableCell",
    "TableGrid",
    "TableQualityReport",
    "TableSemanticPlan",
    "TableSource",
    "TableSourceKind",
    "TableTaskReport",
]
