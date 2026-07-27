"""表格输入适配、图片识别、HTML 规范化和二维网格解析。

图片识别以 RapidOCR + RapidTable 为本地首选，MinerU 作为可选回退。无论原始
输入来自图片、HTML 还是 Markdown，后续阶段只接收通过质量门的标准 HTML。
"""
from __future__ import annotations

from difflib import SequenceMatcher
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import site
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from bs4 import BeautifulSoup, Tag

from config.model_config import settings as model_settings
from src.utils.json_io import write_json

from .schema_models import (
    RecognizedTable,
    TableCell,
    TableGrid,
    TableQualityReport,
    TableSource,
    TableSourceKind,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TABLE_PARSE_VERSION = "html_grid_v8"
MISSING_VALUES = {"", "-", "–", "—", "/", "na", "n/a", "none", "无", "未检出"}
UNIT_RE = re.compile(
    r"(?:%|‰|MPa|GPa|kPa|Pa|mD|μm(?:²|2)?|um(?:²|2)?|mmol/g|mg/g|mg·g[-⁻]?1|"
    r"g/cm(?:³|3)|kg/m(?:³|3)|℃|°C|Ma|m|cm|mm|km)$",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"^\s*(约|≈|>|<|≥|≤|>=|<=|大于|小于)?\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*(?:[~～—–至]\s*([-+]?\d+(?:\.\d+)?))?\s*"
    r"([^\d\s].*)?$"
)
HEADER_KEYWORDS = (
    "井号", "样品", "编号", "层位", "岩性", "类型", "深度", "参数", "指标",
    "孔隙度", "渗透率", "含量", "平均", "最大", "最小", "类别", "组分",
)
PETROLEUM_SUBHEADER_ALIASES: tuple[tuple[str, str], ...] = (
    ("黏土矿物", "黏土矿物"), ("粘土矿物", "黏土矿物"),
    ("伊蒙混层", "伊蒙混层"), ("碳酸盐矿物", "碳酸盐矿物"),
    ("斜长石", "斜长石"), ("钾长石", "钾长石"), ("方解石", "方解石"),
    ("白云石", "白云石"), ("黄铁矿", "黄铁矿"), ("菱铁矿", "菱铁矿"),
    ("硬石膏", "硬石膏"), ("伊利石", "伊利石"), ("蒙脱石", "蒙脱石"),
    ("高岭石", "高岭石"), ("绿泥石", "绿泥石"), ("石英", "石英"),
    ("长石", "长石"), ("最大值", "最大值"), ("最小值", "最小值"),
    ("平均值", "平均值"), ("中位数", "中位数"), ("最大", "最大"),
    ("最小", "最小"), ("平均", "平均"), ("细砂", "细砂"),
    ("中砂", "中砂"), ("粗砂", "粗砂"), ("粉砂", "粉砂"),
)


def _clean_text(value: Any) -> str:
    """规范化单元格空白，同时保留上下标和业务符号。"""

    text = str(value or "").replace("\xa0", " ").replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def _stable_token(*parts: Any, length: int = 16) -> str:
    """根据输入内容生成可复现的短标识。"""

    raw = "|".join(str(part or "").strip().lower() for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def _safe_name(value: str) -> str:
    """将 Chunk ID 转换为可用于 Windows 目录名的稳定名称。"""

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" ._")
    return f"{cleaned[:80]}_{_stable_token(value, length=8)}" if cleaned else _stable_token(value)


def _as_string_list(value: Any) -> list[str]:
    """把字符串或序列统一转换为去空的字符串列表。"""

    if value is None:
        return []
    values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _existing_image_path(value: str, source_file: str = "") -> Path | None:
    """判断字段是否指向现存图片，并兼容相对原 Markdown 的资源路径。"""

    candidate = Path(value.strip())
    if not candidate.is_absolute() and source_file:
        candidate = Path(source_file).parent / candidate
    try:
        resolved = candidate.expanduser().resolve()
    except OSError:
        return None
    return resolved if resolved.suffix.lower() in IMAGE_SUFFIXES and resolved.is_file() else None


def adapt_table_chunk(chunk: Mapping[str, Any], *, allow_image_chunk: bool = False) -> list[TableSource]:
    """把 Stage-02 的 TableChunk 或候选 ImageChunk 转换为单表输入任务。"""

    modality = str(chunk.get("modality") or "").lower()
    if modality not in {"table", "image"} or (modality == "image" and not allow_image_chunk):
        return []

    chunk_id = str(chunk.get("id") or f"table-{_stable_token(chunk)}")
    document_id = str(chunk.get("document_id") or "")
    source_file = str(chunk.get("source_file") or "")
    caption = _clean_text(chunk.get("caption"))
    common = {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "section_id": str(chunk.get("section_id") or ""),
        "section_title": str(chunk.get("section_title") or ""),
        "caption": caption,
        "references": _as_string_list(chunk.get("references")),
        "source_chunk": dict(chunk),
    }

    paths = _as_string_list(chunk.get("table_path") if modality == "table" else chunk.get("image_path"))
    markdown = str(chunk.get("markdown") or "").strip()
    if modality == "table" and not paths:
        markdown_image = _existing_image_path(markdown, source_file)
        if markdown_image is not None:
            paths = [str(markdown_image)]

    sources: list[TableSource] = []
    for index, path_text in enumerate(paths):
        image_path = _existing_image_path(path_text, source_file)
        if image_path is None:
            continue
        task_id = chunk_id if len(paths) == 1 else f"{chunk_id}:part:{index}"
        sources.append(TableSource(
            **common,
            task_id=task_id,
            kind=TableSourceKind.IMAGE,
            image_path=str(image_path),
        ))
    if sources:
        return sources

    if "<table" in markdown.lower():
        return [TableSource(**common, task_id=chunk_id, kind=TableSourceKind.HTML, content=markdown)]
    if markdown:
        return [TableSource(**common, task_id=chunk_id, kind=TableSourceKind.MARKDOWN, content=markdown)]
    return []


def collect_table_sources(
    payload: Mapping[str, Any],
    *,
    include_image_candidates: bool = False,
) -> list[TableSource]:
    """递归收集 Stage-02/03 中的表格，并可恢复被误分到 ImageChunk 的表格候选。"""

    document = payload.get("document", payload)
    if not isinstance(document, Mapping):
        raise TypeError("Stage-02/03 输入缺少 document 对象")
    document_id = str(document.get("id") or document.get("document_id") or "")
    source_file = str(document.get("source_file") or "")
    sources: list[TableSource] = []

    def visit(section: Mapping[str, Any]) -> None:
        """遍历一个章节并补充 Chunk 所需的文档及章节上下文。"""

        section_id = str(section.get("id") or "")
        section_title = str(section.get("title") or "")
        for raw_chunk in section.get("chunks") or []:
            if not isinstance(raw_chunk, Mapping):
                continue
            chunk = dict(raw_chunk)
            chunk.setdefault("document_id", document_id)
            chunk.setdefault("source_file", source_file)
            chunk.setdefault("section_id", section_id)
            chunk.setdefault("section_title", section_title)
            modality = str(chunk.get("modality") or "").lower()
            allow_image = include_image_candidates and modality == "image"
            sources.extend(adapt_table_chunk(chunk, allow_image_chunk=allow_image))
        for child in section.get("children") or []:
            if isinstance(child, Mapping):
                visit(child)

    for section in document.get("sections") or []:
        if isinstance(section, Mapping):
            visit(section)
    return sources


def markdown_table_to_html(markdown: str) -> str:
    """把常见管道 Markdown 表转换为简单 HTML；不满足格式时明确报错。"""

    rows = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if "|" not in stripped:
            continue
        values = [item.strip() for item in stripped.strip("|").split("|")]
        if values and all(re.fullmatch(r":?-{3,}:?", item or "") for item in values):
            continue
        rows.append(values)
    if not rows:
        raise ValueError("Markdown 中没有可解析的管道表格")
    width = max(len(row) for row in rows)
    html_rows = []
    for row_index, row in enumerate(rows):
        tag = "th" if row_index == 0 else "td"
        cells = "".join(f"<{tag}>{_escape_html(item)}</{tag}>" for item in row + [""] * (width - len(row)))
        html_rows.append(f"<tr>{cells}</tr>")
    return "<table><tbody>" + "".join(html_rows) + "</tbody></table>"


def _escape_html(value: str) -> str:
    """转义 Markdown 单元格文本，避免内容被误当作标签。"""

    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def canonicalize_html(value: str) -> str:
    """用 HTML 解析器重建仅含表格语义标签的标准 HTML。"""

    source = value if "<table" in value.lower() else markdown_table_to_html(value)
    soup = BeautifulSoup(source, "lxml")
    table = soup.find("table")
    if table is None:
        raise ValueError("识别结果不包含 table 标签")

    output = BeautifulSoup("<html><body><table><tbody></tbody></table></body></html>", "lxml")
    output_table = output.find("table")
    output_body = output.find("tbody")
    assert output_table is not None and output_body is not None
    for row in _table_rows(table):
        new_row = output.new_tag("tr")
        for cell in row.find_all(["th", "td"], recursive=False):
            tag_name = "th" if cell.name == "th" else "td"
            new_cell = output.new_tag(tag_name)
            for span_name in ("rowspan", "colspan"):
                try:
                    span = max(1, int(cell.get(span_name, 1)))
                except (TypeError, ValueError):
                    span = 1
                if span > 1:
                    new_cell[span_name] = str(span)
            new_cell.string = _clean_text(cell.get_text(" ", strip=True))
            new_row.append(new_cell)
        if new_row.find(["th", "td"]) is not None:
            output_body.append(new_row)
    if not output_body.find("tr"):
        raise ValueError("table 标签内没有有效行")
    return str(output)


def _table_rows(table: Tag) -> list[Tag]:
    """只返回当前表格自己的行，排除嵌套表格内部的行。"""

    return [
        row for row in table.find_all("tr")
        if row.find_parent("table") is table
    ]


def _parse_value(text: str) -> dict[str, Any]:
    """解析单元格中的数值、范围、比较符和单位，同时保留原文本。"""

    normalized = _clean_text(text)
    if normalized.lower() in MISSING_VALUES:
        return {"value_type": "missing", "is_missing": True}
    compact = normalized.replace(",", "").replace("−", "-")
    match = NUMBER_RE.fullmatch(compact)
    if match is None:
        return {"value_type": "text", "is_missing": False, "unit": _unit_from_text(normalized)}
    comparator = match.group(1) or ""
    first = float(match.group(2))
    second = float(match.group(3)) if match.group(3) is not None else None
    unit = _clean_text(match.group(4) or "")
    if second is None:
        return {
            "value_type": "number", "is_missing": False, "numeric_value": first,
            "comparator": comparator, "unit": unit,
        }
    return {
        "value_type": "range", "is_missing": False,
        "min_value": min(first, second), "max_value": max(first, second),
        "comparator": comparator, "unit": unit,
    }


def _unit_from_text(text: str) -> str:
    """从表头或文本末尾提取常见单位。"""

    bracket = re.search(r"[（(]([^()（）]{1,20})[)）]\s*$", text)
    if bracket:
        return _clean_text(bracket.group(1))
    slash = re.search(r"/\s*([^/\s]{1,20})\s*$", text)
    if slash:
        return _clean_text(slash.group(1))
    match = UNIT_RE.search(text)
    return _clean_text(match.group(0)) if match else ""


def parse_html_table(canonical_html: str, *, details: Mapping[str, Any] | None = None) -> TableGrid:
    """展开标准 HTML 的合并单元格，生成矩形网格和行列语义。"""

    soup = BeautifulSoup(canonical_html, "lxml")
    table = soup.find("table")
    if table is None:
        raise ValueError("标准 HTML 不包含 table")
    occupied: dict[tuple[int, int], str] = {}
    cells: list[TableCell] = []
    matrix_values: dict[tuple[int, int], str] = {}
    max_column = 0
    rows = _table_rows(table)
    for row_index, row in enumerate(rows):
        column_index = 0
        for tag in row.find_all(["th", "td"], recursive=False):
            while (row_index, column_index) in occupied:
                column_index += 1
            try:
                rowspan = max(1, int(tag.get("rowspan", 1)))
            except (TypeError, ValueError):
                rowspan = 1
            try:
                colspan = max(1, int(tag.get("colspan", 1)))
            except (TypeError, ValueError):
                colspan = 1
            text = _clean_text(tag.get_text(" ", strip=True))
            cell_id = f"r{row_index}c{column_index}"
            value_fields = _parse_value(text)
            cell = TableCell(
                cell_id=cell_id,
                row_start=row_index,
                row_end=row_index + rowspan - 1,
                col_start=column_index,
                col_end=column_index + colspan - 1,
                rowspan=rowspan,
                colspan=colspan,
                raw_text=text,
                text=text,
                is_header=tag.name == "th",
                **value_fields,
            )
            cells.append(cell)
            for target_row in range(row_index, row_index + rowspan):
                for target_col in range(column_index, column_index + colspan):
                    occupied[(target_row, target_col)] = cell_id
                    matrix_values[(target_row, target_col)] = text
            column_index += colspan
            max_column = max(max_column, column_index)

    row_count = max((row for row, _ in occupied), default=-1) + 1
    matrix = [
        [matrix_values.get((row, column), "") for column in range(max_column)]
        for row in range(row_count)
    ]
    _attach_rapid_geometry(cells, details or {})
    header_rows, header_columns, orientation = _infer_table_structure(cells, matrix)
    header_paths = _build_header_paths(cells, max_column, header_rows)
    return TableGrid(
        row_count=row_count,
        column_count=max_column,
        cells=cells,
        matrix=matrix,
        header_rows=header_rows,
        header_columns=header_columns,
        header_paths=header_paths,
        orientation=orientation,
    )


def _attach_rapid_geometry(cells: list[TableCell], details: Mapping[str, Any]) -> None:
    """按 RapidTable 返回顺序把框和逻辑坐标附加到标准单元格。"""

    logic_points = details.get("logic_points") or []
    bboxes = details.get("cell_bboxes") or []
    for index, cell in enumerate(cells):
        if index < len(bboxes):
            flat = _flatten_numbers(bboxes[index])
            cell.bbox = flat
        if index < len(logic_points):
            logic = [int(item) for item in _flatten_numbers(logic_points[index])[:4]]
            # 中文说明：只有合法四元组才覆盖 HTML 推导坐标，防止第三方返回格式变化污染网格。
            if len(logic) == 4 and logic[0] <= logic[1] and logic[2] <= logic[3]:
                cell.row_start, cell.row_end, cell.col_start, cell.col_end = logic
                cell.rowspan = logic[1] - logic[0] + 1
                cell.colspan = logic[3] - logic[2] + 1


def _flatten_numbers(value: Any) -> list[float]:
    """递归展开坐标数组，转换为普通浮点列表。"""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: list[float] = []
        for item in value:
            result.extend(_flatten_numbers(item))
        return result
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _infer_table_structure(
    cells: Sequence[TableCell],
    matrix: Sequence[Sequence[str]],
) -> tuple[list[int], list[int], str]:
    """依据标题行、数值密度和左侧指标列判断真实表头范围及横竖方向。"""

    if not matrix:
        return [], [], "unknown"
    column_count = max((len(row) for row in matrix), default=0)
    caption_rows = _detect_caption_rows(cells, column_count)
    explicit_headers = sorted({cell.row_start for cell in cells if cell.is_header})
    first_data_row = len(matrix)
    for index, row in enumerate(matrix):
        nonempty = [value for value in row if _clean_text(value)]
        if not nonempty:
            continue
        numeric_count = sum(_parse_value(value)["value_type"] in {"number", "range"} for value in nonempty)
        numeric_threshold = max(1, (2 * len(nonempty) + 4) // 5)
        if numeric_count >= numeric_threshold and (numeric_count >= 2 or len(nonempty) <= 2):
            first_data_row = index
            break
    if explicit_headers:
        header_rows = [row for row in explicit_headers if row not in caption_rows]
    elif 0 < first_data_row < len(matrix):
        header_rows = [row for row in range(first_data_row) if row not in caption_rows]
    elif first_data_row == 0:
        # 中文说明：首行已呈现高数值密度时视为无表头续表，不能把第一条记录错误提升为字段名。
        header_rows = []
    else:
        candidates = [index for index in range(len(matrix)) if index not in caption_rows]
        header_rows = candidates[:1]

    content_indexes = [index for index in range(len(matrix)) if index not in caption_rows]
    label_width = min(2, column_count)
    keyword_rows = sum(
        any(keyword in _clean_text(value) for value in matrix[index][:label_width] for keyword in HEADER_KEYWORDS)
        for index in content_indexes
    )
    first_content_index = content_indexes[0] if content_indexes else 0
    first_row = [_clean_text(value) for value in matrix[first_content_index]]
    row_keyword_count = sum(any(keyword in value for keyword in HEADER_KEYWORDS) for value in first_row)
    vertical = (
        len(matrix) >= 4
        and len(first_row) >= 3
        and keyword_rows >= max(2, len(content_indexes) // 4)
        and keyword_rows > row_keyword_count
        and len(explicit_headers) < 2
    )
    # 中文说明：两行以上显式 th 是可靠的横向多级表头，不能因字段关键字较多而误判为转置表。
    if vertical and keyword_rows > row_keyword_count:
        descriptor_columns = [0]
        if len(first_row) > 1 and any(
            keyword in first_row[1].replace(" ", "")
            for keyword in ("具体指标", "指标名称", "参数名称", "具体参数", "项目")
        ):
            descriptor_columns.append(1)
        return [first_content_index], descriptor_columns, "vertical"
    if len(matrix) > 1 and len(first_row) > 1:
        return header_rows, [], "horizontal"
    return header_rows, [], "matrix"


def _detect_caption_rows(cells: Sequence[TableCell], column_count: int) -> set[int]:
    """识别跨整表的中英文标题行，使标题既不污染字段路径，也不计入数据起点。"""

    caption_rows: set[int] = set()
    for cell in cells:
        value = _clean_text(cell.text)
        if not value:
            continue
        starts_with_numbered_caption = re.match(
            r"^(?:表|table)\s*[．.]?\s*\d",
            value,
            re.IGNORECASE,
        )
        if starts_with_numbered_caption:
            caption_rows.add(cell.row_start)
        elif column_count > 1 and cell.colspan >= column_count and re.search(r"表|table", value, re.IGNORECASE):
            caption_rows.add(cell.row_start)
    return caption_rows


def _build_header_paths(cells: Sequence[TableCell], column_count: int, header_rows: Sequence[int]) -> list[str]:
    """合并多级表头文本，为每个逻辑列生成完整字段路径。"""

    paths: list[str] = []
    header_set = set(header_rows)
    caption_rows = _detect_caption_rows(cells, column_count)
    for column in range(column_count):
        values: list[str] = []
        for cell in cells:
            if (
                cell.row_start not in header_set
                or cell.row_start in caption_rows
                or not (cell.col_start <= column <= cell.col_end)
            ):
                continue
            # 中文说明：跨整表的标题不应污染每一列的字段名称。
            if cell.colspan >= column_count and column_count > 1:
                continue
            value = _clean_text(cell.text)
            if value and (not values or values[-1] != value):
                values.append(value)
        paths.append(" / ".join(values) if values else f"列{column + 1}")
    return paths


def _find_sparse_header_groups(grid: TableGrid) -> list[dict[str, int]]:
    """从重复的“最小/最大/平均”等子表头中定位应由父级分组名覆盖的列段。"""

    header_rows = sorted(grid.header_rows)
    for parent_row, lower_row in zip(header_rows, header_rows[1:]):
        values = [_clean_text(value) for value in grid.matrix[lower_row]]
        for start_col in range(len(values)):
            max_length = min(6, (len(values) - start_col) // 2)
            for length in range(2, max_length + 1):
                pattern = values[start_col:start_col + length]
                if not all(pattern) or len(set(pattern)) < 2:
                    continue
                if values[start_col + length:start_col + 2 * length] != pattern:
                    continue
                groups: list[dict[str, int]] = []
                group_start = start_col
                while values[group_start:group_start + length] == pattern:
                    groups.append({
                        "parent_row": parent_row,
                        "lower_row": lower_row,
                        "start_col": group_start,
                        "length": length,
                    })
                    group_start += length
                return groups
    return []


def _logical_tag_map(table: Tag) -> dict[str, Tag]:
    """按 rowspan/colspan 展开顺序，将规范 HTML 的真实标签映射回稳定 cell_id。"""

    occupied: set[tuple[int, int]] = set()
    result: dict[str, Tag] = {}
    for row_index, row in enumerate(_table_rows(table)):
        column_index = 0
        for tag in row.find_all(["th", "td"], recursive=False):
            while (row_index, column_index) in occupied:
                column_index += 1
            try:
                rowspan = max(1, int(tag.get("rowspan", 1)))
                colspan = max(1, int(tag.get("colspan", 1)))
            except (TypeError, ValueError):
                rowspan = colspan = 1
            result[f"r{row_index}c{column_index}"] = tag
            for target_row in range(row_index, row_index + rowspan):
                for target_col in range(column_index, column_index + colspan):
                    occupied.add((target_row, target_col))
            column_index += colspan
    return result


def _quality_report(
    grid: TableGrid,
    canonical_html: str,
    details: Mapping[str, Any],
    *,
    minimum_score: float,
) -> TableQualityReport:
    """根据非空率、OCR 覆盖率和几何一致性计算 HTML 质量。"""

    total_slots = max(1, grid.row_count * grid.column_count)
    nonempty = sum(bool(_clean_text(value)) for row in grid.matrix for value in row)
    nonempty_ratio = nonempty / total_slots
    ocr_items = details.get("ocr") or []
    ocr_values = list(dict.fromkeys(
        _clean_text(item.get("text")) for item in ocr_items
        if isinstance(item, Mapping) and _clean_text(item.get("text"))
    ))
    html_values = list(dict.fromkeys(_clean_text(cell.text) for cell in grid.cells if _clean_text(cell.text)))
    if ocr_values and html_values:
        weighted_match = 0.0
        total_weight = 0.0
        for value in html_values:
            compact = re.sub(r"\s+", "", value)
            weight = max(1.0, min(12.0, float(len(compact))))
            best = 0.0
            for candidate in ocr_values:
                candidate_compact = re.sub(r"\s+", "", candidate)
                if compact == candidate_compact or compact in candidate_compact or candidate_compact in compact:
                    best = 1.0
                    break
                best = max(best, SequenceMatcher(None, compact, candidate_compact).ratio())
            # 中文说明：按唯一单元格而非整串顺序计算覆盖，兼容纵排合并字段补全和重叠 OCR 的输出次序。
            weighted_match += weight * (best if best >= 0.45 else 0.0)
            total_weight += weight
        ocr_coverage = weighted_match / max(total_weight, 1.0)
    else:
        ocr_coverage = 1.0
    logic_points = details.get("logic_points") or []
    geometry_consistency = min(len(logic_points), len(grid.cells)) / max(len(logic_points), len(grid.cells), 1)
    if not logic_points:
        geometry_consistency = 1.0
    score = max(0.0, min(1.0, 0.35 * nonempty_ratio + 0.35 * ocr_coverage + 0.30 * geometry_consistency))
    errors: list[str] = []
    warnings: list[str] = []
    if grid.row_count == 0 or grid.column_count == 0:
        errors.append("empty_grid")
    if nonempty_ratio < 0.2:
        errors.append("low_nonempty_ratio")
    if ocr_coverage < 0.45:
        warnings.append("low_ocr_coverage")
    if geometry_consistency < 0.7:
        warnings.append("logic_point_count_mismatch")
    if "<table" not in canonical_html.lower():
        errors.append("missing_table_tag")
    data_start = max(grid.header_rows, default=-1) + 1
    data_rows = grid.matrix[data_start:]
    if len(data_rows) >= 4:
        sparse_limit = max(2, grid.column_count // 4)
        tail = data_rows[-2:]
        if all(sum(bool(_clean_text(value)) for value in row) <= sparse_limit for row in tail):
            # 中文说明：密集长表若只剩井名或层位等少量单元格，通常是结构模型提前终止，不能进入知识图谱。
            errors.append("sparse_tail_rows")
    header_rows = set(grid.header_rows)
    for parent in grid.cells:
        if parent.row_start not in header_rows or parent.colspan < 2 or "矿物" not in parent.text:
            continue
        child_row = parent.row_end + 1
        children = [
            cell for cell in grid.cells
            if cell.row_start == child_row
            and parent.col_start <= cell.col_start <= parent.col_end
            and cell.col_start == cell.col_end
        ]
        child_values = [_clean_text(cell.text) for cell in children]
        if len(children) != parent.colspan or any(not value for value in child_values) or any(
            len(RapidTableRecognizer._petroleum_subheader_terms(value)) > 1 for value in child_values
        ):
            # 中文说明：矿物含量分组的每个数据列必须有独立子表头，否则参数与数值会错列。
            errors.append("ambiguous_petroleum_subheaders")
    passed = not errors and score >= minimum_score
    if not passed and score < minimum_score:
        errors.append("quality_score_below_threshold")
    return TableQualityReport(
        passed=passed,
        score=score,
        nonempty_ratio=nonempty_ratio,
        ocr_coverage=ocr_coverage,
        geometry_consistency=geometry_consistency,
        errors=list(dict.fromkeys(errors)),
        warnings=warnings,
    )


class RapidTableRecognizer:
    """缓存 RapidOCR 和 RapidTable 引擎，避免每张图片重复加载模型。"""

    def __init__(
        self,
        *,
        model_name: str = "unitable",
        device: str = "auto",
        ocr_device: str = "cpu",
        ocr_backend: str = "onnxruntime",
        ocr_limit_side_len: int = 1600,
    ) -> None:
        """保存模型选择，真正的权重在首次识别时延迟加载。"""

        self.model_name = model_name
        self.requested_device = device
        self.requested_ocr_device = ocr_device
        self.ocr_backend = ocr_backend
        self.ocr_limit_side_len = max(960, int(ocr_limit_side_len))
        self.module: Any | None = None
        self.device = "cpu"
        self.ocr_device = "cpu"
        self.ocr_engine: Any | None = None
        self.table_engine: Any | None = None
        self.last_ocr_mode = "full"

    def _build_ocr_engine(self) -> Any:
        """按当前质量配置创建 OCR 引擎，供首次加载与内存异常重建共同调用。"""

        assert self.module is not None
        module = self.module
        ocr_config = module.build_ocr_config(self.ocr_device)
        ocr_config["Det.limit_side_len"] = self.ocr_limit_side_len
        if self.ocr_backend == "onnxruntime":
            # 中文说明：保持 PP-OCRv5 Server 模型，只把执行后端切到 ONNX CPU，兼顾精度与 8GB 显存限制。
            ocr_config["Det.engine_type"] = module.EngineType.ONNXRUNTIME
            ocr_config["Rec.engine_type"] = module.EngineType.ONNXRUNTIME
            ocr_config["EngineConfig.onnxruntime.use_cuda"] = False
        elif self.ocr_backend != "torch":
            raise ValueError(f"不支持的 OCR 后端：{self.ocr_backend}")
        return module.RapidOCR(params=ocr_config)

    def _run_full_ocr(self, image: Any) -> Any:
        """执行完整检测与识别；内存不足时先重建会话，再以原分辨率分块识别。"""

        assert self.module is not None and self.ocr_engine is not None
        for attempt in range(2):
            try:
                result = self.ocr_engine(
                    image,
                    use_det=True,
                    use_cls=False,
                    use_rec=True,
                    text_score=0.3,
                    box_thresh=0.3,
                    return_single_char_box=True,
                )
                self.last_ocr_mode = "full" if attempt == 0 else "full_retry"
                return result
            except Exception as exc:
                if "bad allocation" not in str(exc).lower():
                    raise
                if attempt > 0:
                    return self._run_tiled_ocr(image)
                # 中文说明：只处理已实测的 ONNX 临时分配失败；先释放会话，再按相同服务器级模型重试，不降低图像分辨率。
                self.ocr_engine = None
                gc.collect()
                if hasattr(self.module, "torch") and self.module.torch.cuda.is_available():
                    self.module.torch.cuda.empty_cache()
                self.ocr_engine = self._build_ocr_engine()
        raise RuntimeError("RapidOCR 重试流程意外结束")

    def _run_tiled_ocr(
        self,
        image: Any,
        *,
        tile_size: int = 896,
        overlap: int = 96,
        include_header_band: bool = True,
    ) -> Any:
        """将超出 ONNX 单次内存能力的图片重叠分块识别，并合并为全图坐标。"""

        assert self.module is not None and self.ocr_engine is not None
        module = self.module
        height, width = image.shape[:2]
        step = max(256, tile_size - overlap)

        def starts(length: int) -> list[int]:
            """生成覆盖完整边界的分块起点，末块贴齐图片结尾。"""

            if length <= tile_size:
                return [0]
            values = list(range(0, max(1, length - tile_size + 1), step))
            final = length - tile_size
            if not values or values[-1] != final:
                values.append(final)
            return values

        candidates: list[tuple[Any, str, float]] = []

        def append_result(result: Any, x_offset: int = 0, y_offset: int = 0) -> None:
            """把局部 OCR 输出平移到全图坐标并追加到候选集合。"""

            if result.boxes is None or result.txts is None or result.scores is None:
                return
            for box, text, score in zip(result.boxes, result.txts, result.scores):
                shifted = module.np.asarray(box, dtype=module.np.float32).copy()
                shifted[:, 0] += x_offset
                shifted[:, 1] += y_offset
                candidates.append((shifted, str(text), float(score)))

        for y0 in starts(height):
            for x0 in starts(width):
                tile = image[y0:min(height, y0 + tile_size), x0:min(width, x0 + tile_size)]
                result = self.ocr_engine(
                    tile,
                    use_det=True,
                    use_cls=False,
                    use_rec=True,
                    text_score=0.3,
                    box_thresh=0.3,
                    return_single_char_box=True,
                )
                append_result(result, x0, y0)

        if include_header_band and width > tile_size and height > 192:
            try:
                header_height = min(height, max(192, int(height * 0.62)))
                header_result = self.ocr_engine(
                    image[:header_height, :],
                    use_det=True,
                    use_cls=False,
                    use_rec=True,
                    text_score=0.3,
                    box_thresh=0.3,
                    return_single_char_box=True,
                )
                append_result(header_result)
                self.last_ocr_mode = "tiled_with_header_band"
            except Exception:
                # 中文说明：表头整带是额外召回，不影响已成功的分块结果；内存仍不足时保留纯分块输出。
                self.last_ocr_mode = "tiled"
        else:
            self.last_ocr_mode = "tiled"

        def bounds(box: Any) -> tuple[float, float, float, float]:
            """把四点 OCR 框压缩为用于去重的轴对齐边界。"""

            points = module.np.asarray(box)
            return (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )

        def overlap_ratio(left: Any, right: Any) -> float:
            """计算两个文本框的交集占较小框面积比例，用于删除重叠分块重复框。"""

            lx1, ly1, lx2, ly2 = bounds(left)
            rx1, ry1, rx2, ry2 = bounds(right)
            intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
                0.0, min(ly2, ry2) - max(ly1, ry1)
            )
            smaller = min(max(1.0, (lx2 - lx1) * (ly2 - ly1)), max(1.0, (rx2 - rx1) * (ry2 - ry1)))
            return intersection / smaller

        kept: list[tuple[Any, str, float]] = []
        for candidate in sorted(candidates, key=lambda item: item[2], reverse=True):
            if any(overlap_ratio(candidate[0], item[0]) >= 0.65 for item in kept):
                continue
            kept.append(candidate)
        kept.sort(key=lambda item: (bounds(item[0])[1], bounds(item[0])[0]))
        if not kept:
            raise RuntimeError("RapidOCR 分块识别后没有得到文字")
        # 中文说明：SimpleNamespace 只承载后续流程需要的三项字段，接口与 RapidOCR 输出保持一致。
        return SimpleNamespace(
            boxes=module.np.asarray([item[0] for item in kept]),
            txts=[item[1] for item in kept],
            scores=[item[2] for item in kept],
        )

    def _initialize(self) -> None:
        """从现有 RapidTable 脚本复用 OCR 配置和下标修复逻辑。"""

        if self.table_engine is not None:
            return
        # 中文说明：treeSchemeKG 必须优先使用环境自身 Torch，避免用户站点同名包造成 torchvision ABI 冲突。
        if os.environ.get("CONDA_DEFAULT_ENV", "").lower() == "treeschemekg":
            user_site = str(Path(site.getusersitepackages()).resolve()).lower()
            sys.path[:] = [
                item for item in sys.path
                if not item or str(Path(item).resolve()).lower() != user_site
            ]
        module = importlib.import_module("src.extractors.table_extractor.RapidTable识别")
        if self.model_name not in module.MODEL_TYPES:
            raise ValueError(f"未知 RapidTable 模型：{self.model_name}")
        self.module = module
        self.device = module.resolve_device(self.model_name, self.requested_device)
        # 中文说明：8GB 显存同时加载服务器级 OCR 与 Unitable 容易 OOM，默认让 OCR 在 CPU、结构模型在 GPU。
        self.ocr_device = module.resolve_device("unitable", self.requested_ocr_device)
        self.ocr_engine = self._build_ocr_engine()
        model_type = module.MODEL_TYPES[self.model_name]
        self.table_engine = module.RapidTable(
            module.RapidTableInput(
                model_type=model_type,
                engine_cfg=module.build_engine_config(self.model_name, self.device),
            )
        )

    def recognize(self, image_path: str | Path, artifact_dir: Path) -> tuple[str, dict[str, Any]]:
        """识别单张表格图片并保存 HTML、明细和结构可视化。"""

        self._initialize()
        assert self.module is not None and self.ocr_engine is not None and self.table_engine is not None
        module = self.module
        image_file = Path(image_path).expanduser().resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        image = module.cv2.imdecode(module.np.fromfile(image_file, dtype=module.np.uint8), module.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV 无法读取表格图片：{image_file}")

        ocr_started = time.perf_counter()
        ocr_result = self._run_full_ocr(image)
        assert self.ocr_engine is not None
        if ocr_result.boxes is None or ocr_result.txts is None or ocr_result.scores is None:
            raise RuntimeError("RapidOCR 未识别到表格文字")
        boxes, texts, scores = module.merge_detected_subscripts(
            ocr_result.boxes, ocr_result.txts, ocr_result.scores
        )
        boxes, texts, scores = module.recover_joined_subscripts(
            image, self.ocr_engine, boxes, texts, scores
        )
        texts = module.align_subscripts_with_identifiers(boxes, texts)
        ocr_elapsed = time.perf_counter() - ocr_started
        ocr_results = [(module.np.asarray(boxes), texts, scores)]

        table_started = time.perf_counter()
        if self.model_name == "unitable":
            with module.torch.inference_mode():
                result = self.table_engine(image_file, ocr_results=ocr_results)
        else:
            result = self.table_engine(image_file, ocr_results=ocr_results)
        table_elapsed = time.perf_counter() - table_started
        if not result.pred_htmls:
            raise RuntimeError("RapidTable 没有返回 HTML")
        html = str(result.pred_htmls[0])
        details = {
            "image": str(image_file),
            "engine": "rapidtable",
            "model": self.model_name,
            "device": self.device,
            "ocr_device": self.ocr_device,
            "ocr_backend": self.ocr_backend,
            "ocr_limit_side_len": self.ocr_limit_side_len,
            "ocr_mode": self.last_ocr_mode,
            "timing_seconds": {
                "ocr": round(ocr_elapsed, 6),
                "table": round(table_elapsed, 6),
                "total": round(ocr_elapsed + table_elapsed, 6),
            },
            "ocr": [
                {"text": text, "score": float(score), "box": module.to_jsonable(box)}
                for box, text, score in zip(boxes, texts, scores)
            ],
            "logic_points": module.to_jsonable(result.logic_points[0]),
            "cell_bboxes": module.to_jsonable(result.cell_bboxes[0]),
        }
        (artifact_dir / "rapidtable_raw.html").write_text(html, encoding="utf-8")
        write_json(artifact_dir / "rapidtable_details.json", details)
        try:
            visual_images = result.vis(
                save_dir=artifact_dir / "rapidtable_visualization",
                save_name="rapidtable_result",
            )
            module.save_image_unicode(artifact_dir / "structure_visualization.jpg", visual_images[0])
            logic_image = module.draw_logic_points(result.imgs[0], result.cell_bboxes[0], result.logic_points[0])
            module.save_image_unicode(artifact_dir / "logic_visualization.jpg", logic_image)
        except Exception as exc:
            details["visualization_warning"] = str(exc)
        return html, details

    def repair_sparse_group_headers(
        self,
        image_path: str | Path,
        canonical_html: str,
        grid: TableGrid,
    ) -> tuple[str, list[dict[str, Any]]]:
        """局部直识别重复子表头上方的缺失分组名，并同步修复规范 HTML 的 colspan。"""

        candidates = _find_sparse_header_groups(grid)
        if not candidates:
            return canonical_html, []
        self._initialize()
        assert self.module is not None and self.ocr_engine is not None
        module = self.module
        image_file = Path(image_path).expanduser().resolve()
        image = module.cv2.imdecode(module.np.fromfile(image_file, dtype=module.np.uint8), module.cv2.IMREAD_COLOR)
        if image is None:
            return canonical_html, []
        soup = BeautifulSoup(canonical_html, "lxml")
        table = soup.find("table")
        if table is None:
            return canonical_html, []
        tag_map = _logical_tag_map(table)
        repairs: list[dict[str, Any]] = []

        for candidate in candidates:
            parent_row = candidate["parent_row"]
            start_col = candidate["start_col"]
            length = candidate["length"]
            group_cells = [
                cell for cell in grid.cells
                if cell.row_start == parent_row and start_col <= cell.col_start < start_col + length
            ]
            existing_values = [_clean_text(cell.text) for cell in group_cells if _clean_text(cell.text)]
            label = next((value for value in existing_values if not re.fullmatch(r"[\d.]+", value)), "")
            confidence = 1.0 if label else 0.0
            if not label:
                label, confidence = self._recognize_group_header_crop(image, grid, candidate)
            if not label or confidence < 0.55:
                continue
            label = label.replace("$", "")
            label = re.sub(r"10\s*[−–—-]\s*3", "10^-3", label)
            label = re.sub(r"(?:μm|um)\s*2$", "μm²", label, flags=re.IGNORECASE)
            if re.search(r"μm$", label, re.IGNORECASE) and any(value == "2" for value in existing_values):
                label += "²"
            primary = next((cell for cell in group_cells if cell.col_start == start_col), None)
            if primary is None:
                continue
            primary_tag = tag_map.get(primary.cell_id)
            if primary_tag is None:
                continue
            primary_tag.clear()
            primary_tag.append(label)
            primary_tag["colspan"] = str(length)
            for cell in group_cells:
                if cell.cell_id == primary.cell_id:
                    continue
                tag = tag_map.get(cell.cell_id)
                if tag is not None:
                    tag.decompose()
            repairs.append({
                "row": parent_row,
                "start_col": start_col,
                "colspan": length,
                "text": label,
                "confidence": float(confidence),
                "method": "existing_html" if confidence == 1.0 else "local_rec_only",
            })
        if not repairs:
            return canonical_html, []
        return canonicalize_html(str(table)), repairs

    def _recognize_group_header_crop(
        self,
        image: Any,
        grid: TableGrid,
        candidate: Mapping[str, int],
    ) -> tuple[str, float]:
        """按下一层子表头框估算父表头区域，绕过检测器直接识别横线内文字。"""

        assert self.module is not None and self.ocr_engine is not None
        module = self.module
        lower_row = int(candidate["lower_row"])
        start_col = int(candidate["start_col"])
        end_col = start_col + int(candidate["length"]) - 1
        child_cells = [
            cell for cell in grid.cells
            if cell.row_start == lower_row
            and start_col <= cell.col_start <= end_col
            and len(cell.bbox) >= 8
            and max(cell.bbox) > 0
        ]
        if not child_cells:
            return "", 0.0
        x_values = [value for cell in child_cells for value in cell.bbox[0::2]]
        child_y_values = [value for cell in child_cells for value in cell.bbox[1::2]]
        image_height, image_width = image.shape[:2]
        x1 = max(0, int(min(x_values)) - 45)
        x2 = min(image_width, int(max(x_values)) + 45)
        y2 = min(image_height, int(min(child_y_values)) - 5)
        y1 = max(0, y2 - 55)
        if x2 - x1 < 20 or y2 - y1 < 12:
            return "", 0.0
        crop = image[y1:y2, x1:x2]
        crop = module.cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=module.cv2.INTER_CUBIC)
        result = self.ocr_engine(
            crop,
            use_det=False,
            use_cls=False,
            use_rec=True,
            text_score=0.1,
        )
        texts = list(result.txts or [])
        scores = list(result.scores or [])
        if not texts or not scores:
            return "", 0.0
        return _clean_text(texts[0]), float(scores[0])

    @staticmethod
    def _petroleum_subheader_terms(value: str) -> list[str]:
        """从 OCR 短语中提取石油地质常见矿物、粒级和统计子表头。"""

        compact = re.sub(r"\s+", "", value)
        candidates: list[tuple[int, int, str]] = []
        for alias, canonical in PETROLEUM_SUBHEADER_ALIASES:
            for match in re.finditer(re.escape(alias), compact, flags=re.IGNORECASE):
                candidates.append((match.start(), match.end(), canonical))
        selected: list[tuple[int, int, str]] = []
        for start, end, term in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]))):
            if any(not (end <= kept_start or start >= kept_end) for kept_start, kept_end, _ in selected):
                continue
            selected.append((start, end, term))
        return [term for _, _, term in sorted(selected)]

    @staticmethod
    def _normalize_petroleum_header(value: str) -> str:
        """修正常见石油地质参数表头中的 OCR 空格、上下标和单位噪声。"""

        compact = re.sub(r"\s+", "", value)
        if re.fullmatch(r"深度/?m", compact, flags=re.IGNORECASE):
            return "深度/m"
        if compact.upper().startswith("TOC") and "%" in compact:
            return "TOC/%"
        if compact.startswith("孔隙度") and "%" in compact:
            return "孔隙度/%"
        if compact.startswith("成熟度") and "%" in compact:
            return "成熟度/%"
        if compact.startswith("含气量") and ("m3" in compact.lower() or "m^{3" in compact.lower()):
            return "含气量/(m³/t)"
        if compact.startswith("渗透率"):
            if "md" in compact.lower():
                return "渗透率/mD"
            if "10" in compact or "μm" in compact or "um" in compact.lower():
                return "渗透率/(×10^-3 μm²)"
        if "全岩矿物含量" in compact:
            return "全岩矿物含量/%"
        return _clean_text(value)

    def _repair_dense_nested_headers(
        self,
        canonical_html: str,
        grid: TableGrid,
        ocr_items: Sequence[Mapping[str, Any]],
        column_centers: Sequence[float],
    ) -> tuple[str, list[dict[str, Any]]]:
        """按 OCR 框与列中心拆分密集表中被合并或漏掉的石油地质二级表头。"""

        soup = BeautifulSoup(canonical_html, "lxml")
        table = soup.find("table")
        if table is None:
            return canonical_html, []
        tag_map = _logical_tag_map(table)
        repairs: list[dict[str, Any]] = []
        header_rows = set(grid.header_rows)
        for parent in grid.cells:
            if parent.row_start not in header_rows or parent.colspan < 2:
                continue
            child_row = parent.row_end + 1
            if child_row not in header_rows:
                continue
            group_columns = list(range(parent.col_start, parent.col_end + 1))
            child_cells = [
                cell for cell in grid.cells
                if cell.row_start == child_row
                and cell.row_end == child_row
                and cell.col_start in group_columns
                and cell.col_end == cell.col_start
                and len(cell.bbox) >= 8
                and max(cell.bbox) > 0
            ]
            if len(child_cells) != len(group_columns):
                continue
            y1 = min(value for cell in child_cells for value in cell.bbox[1::2]) - 5.0
            y2 = max(value for cell in child_cells for value in cell.bbox[1::2]) + 5.0
            left_gap = (
                column_centers[group_columns[1]] - column_centers[group_columns[0]]
                if len(group_columns) > 1 else 80.0
            )
            right_gap = (
                column_centers[group_columns[-1]] - column_centers[group_columns[-2]]
                if len(group_columns) > 1 else 80.0
            )
            x1 = column_centers[group_columns[0]] - left_gap * 0.55
            x2 = column_centers[group_columns[-1]] + right_gap * 0.55
            assignments: dict[int, tuple[str, float]] = {}
            for item in sorted(ocr_items, key=lambda value: (float(value["y"]), float(value["x"]))):
                x = float(item["x"])
                y = float(item["y"])
                if not (x1 <= x <= x2 and y1 <= y <= y2):
                    continue
                terms = self._petroleum_subheader_terms(str(item["text"]))
                if not terms or len(terms) > len(group_columns):
                    continue
                targets = sorted(
                    sorted(group_columns, key=lambda column: abs(column_centers[column] - x))[:len(terms)]
                )
                for column, term in zip(targets, terms):
                    score = float(item.get("score") or 0.0)
                    if column not in assignments or score > assignments[column][1]:
                        assignments[column] = (term, score)
            if len(assignments) != len(group_columns):
                continue
            values = [assignments[column][0] for column in group_columns]
            if len(set(values)) != len(values):
                continue
            for column, value in zip(group_columns, values):
                tag = tag_map.get(f"r{child_row}c{column}")
                if tag is not None:
                    tag.string = value
            repairs.append({
                "parent": self._normalize_petroleum_header(parent.text),
                "row": child_row,
                "columns": group_columns,
                "subheaders": values,
                "method": "petroleum_ocr_column_alignment",
            })

        # 中文说明：在子表头归列后统一修复深度、TOC、渗透率等常见单位噪声，避免错误参数名进入图谱。
        for cell in grid.cells:
            if cell.row_start not in header_rows:
                continue
            tag = tag_map.get(cell.cell_id)
            if tag is None:
                continue
            normalized = self._normalize_petroleum_header(tag.get_text(" ", strip=True))
            if normalized:
                tag.string = normalized
        return canonicalize_html(str(table)), repairs

    def recover_dense_tall_table(
        self,
        image_path: str | Path,
        canonical_html: str,
        grid: TableGrid,
        details: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        """用重叠 OCR 和字段几何恢复高密度石油地质表中被结构模型截断的数据行。"""

        data_start = max(grid.header_rows, default=-1) + 1
        data_rows = grid.matrix[data_start:]
        if grid.column_count < 6 or len(data_rows) < 6:
            return canonical_html, []
        tail = data_rows[-min(2, len(data_rows)):]
        sparse_limit = max(2, grid.column_count // 4)
        if not tail or not all(sum(bool(_clean_text(value)) for value in row) <= sparse_limit for row in tail):
            return canonical_html, []

        self._initialize()
        assert self.module is not None
        module = self.module
        image_file = Path(image_path).expanduser().resolve()
        image = module.cv2.imdecode(module.np.fromfile(image_file, dtype=module.np.uint8), module.cv2.IMREAD_COLOR)
        if image is None or image.shape[0] < 800:
            return canonical_html, []

        items: list[dict[str, Any]] = []
        cached_ocr = details.get("ocr") or []
        reuse_tiled_ocr = bool(details.get("dense_tall_table_recovery") and cached_ocr)
        if reuse_tiled_ocr:
            ocr_records = [
                (item.get("box"), item.get("text"), item.get("score"))
                for item in cached_ocr
                if isinstance(item, Mapping) and item.get("box") is not None
            ]
        else:
            # 中文说明：长表恢复前重建 OCR 会话，防止上一张表的大图推理碎片影响多个重叠块。
            self.ocr_engine = None
            gc.collect()
            self.ocr_engine = self._build_ocr_engine()
            tiled = self._run_tiled_ocr(image, tile_size=640, overlap=96, include_header_band=False)
            ocr_records = list(zip(tiled.boxes, tiled.txts, tiled.scores))
        for box, text, score in ocr_records:
            points = module.np.asarray(box)
            value = _clean_text(text)
            if not value:
                continue
            items.append({
                "text": value,
                "score": float(score),
                "box": module.to_jsonable(box),
                "x": float(points[:, 0].mean()),
                "y": float(points[:, 1].mean()),
                "x1": float(points[:, 0].min()),
            })

        column_centers: list[float] = []
        for column_index in range(grid.column_count):
            centers = [
                sum(cell.bbox[0::2]) / 4
                for cell in grid.cells
                if cell.col_start == column_index
                and cell.col_end == column_index
                and len(cell.bbox) >= 8
                and max(cell.bbox) > 0
            ]
            if not centers:
                return canonical_html, []
            column_centers.append(float(module.np.median(centers)))

        key_column = self._select_record_key_column(grid.header_paths)
        if key_column is None:
            return canonical_html, []
        key_header = grid.header_paths[key_column]
        key_center = column_centers[key_column]
        neighbor_gaps = [
            abs(key_center - column_centers[index])
            for index in range(len(column_centers))
            if index != key_column
        ]
        x_tolerance = min(neighbor_gaps, default=80.0) * 0.48
        key_items = [
            item for item in items
            if abs(item["x"] - key_center) <= x_tolerance
            and self._is_record_key(item["text"], key_header)
        ]
        key_items.sort(key=lambda item: item["y"])
        deduplicated_keys: list[dict[str, Any]] = []
        for item in key_items:
            normalized = re.sub(r"\s+", "", item["text"])
            if any(
                re.sub(r"\s+", "", existing["text"]) == normalized
                and abs(existing["y"] - item["y"]) < 8
                for existing in deduplicated_keys
            ):
                continue
            deduplicated_keys.append(item)
        key_items = deduplicated_keys
        if len(key_items) < 6:
            return canonical_html, []

        y_values = [item["y"] for item in key_items]
        recovered_rows: list[list[str]] = []
        for index, key_item in enumerate(key_items):
            previous_gap = key_item["y"] - y_values[index - 1] if index else 35.0
            next_gap = y_values[index + 1] - key_item["y"] if index + 1 < len(y_values) else 35.0
            y_tolerance = max(12.0, min(25.0, min(previous_gap, next_gap) * 0.48))
            buckets: list[list[dict[str, Any]]] = [[] for _ in column_centers]
            for item in items:
                if abs(item["y"] - key_item["y"]) > y_tolerance:
                    continue
                column_index = min(
                    range(len(column_centers)),
                    key=lambda target: abs(column_centers[target] - item["x"]),
                )
                buckets[column_index].append(item)
            row = [
                " ".join(item["text"] for item in sorted(bucket, key=lambda value: value["x1"]))
                for bucket in buckets
            ]
            row[key_column] = key_item["text"]
            if sum(bool(_clean_text(value)) for value in row) >= max(4, grid.column_count // 2):
                recovered_rows.append(row)

        complete_original = sum(
            sum(bool(_clean_text(value)) for value in row) >= max(4, grid.column_count // 2)
            for row in data_rows
        )
        if len(recovered_rows) <= complete_original:
            return canonical_html, []

        self._fill_petroleum_group_columns(
            recovered_rows,
            data_rows,
            items,
            grid.header_paths,
            column_centers,
            key_column,
            record_y_range=(min(y_values), max(y_values)),
        )
        repaired_headers_html, header_repairs = self._repair_dense_nested_headers(
            canonical_html,
            grid,
            items,
            column_centers,
        )
        repaired_html = self._replace_html_data_rows(
            repaired_headers_html,
            data_start=data_start,
            rows=recovered_rows,
        )
        audit_rows = [
            {
                "record_key": row[key_column],
                "nonempty_cells": sum(bool(_clean_text(value)) for value in row),
            }
            for row in recovered_rows
        ]
        details["dense_tall_table_recovery"] = {
            "method": "overlap_ocr_column_geometry_v2",
            "key_header": key_header,
            "key_column": key_column,
            "original_complete_rows": complete_original,
            "recovered_rows": len(recovered_rows),
            "tile_size": 640,
            "overlap": 96,
            "ocr_cache_reused": reuse_tiled_ocr,
            "header_repairs": header_repairs,
            "rows": audit_rows,
        }
        details["ocr"] = [
            {"text": item["text"], "score": item["score"], "box": item["box"]}
            for item in items
        ]
        return repaired_html, audit_rows

    @staticmethod
    def _select_record_key_column(headers: Sequence[str]) -> int | None:
        """按石油地质表常见记录键优先级选择用于重建高表行的锚点列。"""

        priorities = (
            ("样品号", "样品编号", "样本号", "岩心编号", "分析号", "测点编号", "记录号"),
            ("井深", "深度", "顶深", "底深"),
            ("井号", "井名"),
        )
        normalized = [re.sub(r"\s+", "", header) for header in headers]
        for keywords in priorities:
            for column_index, header in enumerate(normalized):
                if any(keyword in header for keyword in keywords):
                    return column_index
        return None

    @staticmethod
    def _is_record_key(value: str, header: str) -> bool:
        """判断 OCR 文本能否作为样品、分析记录、测点或深度行锚点。"""

        compact = re.sub(r"\s+", "", value)
        normalized_header = re.sub(r"\s+", "", header)
        if any(keyword in normalized_header for keyword in ("深度", "井深", "顶深", "底深")):
            return bool(re.fullmatch(r"[-+]?\d{2,5}(?:\.\d+)?", compact))
        if len(compact) > 40 or not re.search(r"\d", compact):
            return False
        return bool(
            re.fullmatch(r"[A-Za-z]{1,8}\d{0,6}[-_/]\d+", compact)
            or re.fullmatch(r"[A-Za-z]{1,8}\d+", compact)
            or re.fullmatch(r"[\u3400-\u9fffA-Za-z0-9]{1,16}[-_/]\d+", compact)
        )

    def _fill_petroleum_group_columns(
        self,
        rows: list[list[str]],
        original_rows: Sequence[Sequence[str]],
        ocr_items: Sequence[Mapping[str, Any]],
        headers: Sequence[str],
        column_centers: Sequence[float],
        key_column: int,
        record_y_range: tuple[float, float],
    ) -> None:
        """补全高表中纵向合并的井名和层位/组段字段，保持石油地质记录语义连续。"""

        well_column = next((
            index for index, header in enumerate(headers)
            if index != key_column and any(keyword in header for keyword in ("井名", "井号"))
        ), None)
        key_prefixes: list[str] = []
        for row in rows:
            prefix = re.sub(r"[-_/]\d+$", "", re.sub(r"\s+", "", row[key_column]))
            if prefix and prefix not in key_prefixes:
                key_prefixes.append(prefix)

        if well_column is not None and key_prefixes:
            prefix_to_well: dict[str, str] = {}
            for row in original_rows:
                if key_column >= len(row) or well_column >= len(row):
                    continue
                key = re.sub(r"\s+", "", row[key_column])
                well = re.sub(r"\s+", "", _clean_text(row[well_column]))
                prefix = re.sub(r"[-_/]\d+$", "", key)
                if prefix and well:
                    prefix_to_well.setdefault(prefix, well)
            gaps = [
                abs(column_centers[well_column] - center)
                for index, center in enumerate(column_centers)
                if index != well_column
            ]
            tolerance = min(gaps, default=80.0) * 0.48
            y_margin = max(35.0, (record_y_range[1] - record_y_range[0]) / max(len(rows) - 1, 1))
            anchors = [
                item for item in ocr_items
                if abs(float(item["x"]) - column_centers[well_column]) <= tolerance
                and record_y_range[0] - y_margin <= float(item["y"]) <= record_y_range[1] + y_margin
                and _clean_text(item["text"])
                and _clean_text(item["text"]) not in {"井名", "井号"}
            ]
            anchors.sort(key=lambda item: float(item["y"]))
            join_gap = 70.0
            anchor_labels: list[str] = []
            current: list[Mapping[str, Any]] = []
            for item in anchors:
                if current and float(item["y"]) - float(current[-1]["y"]) > join_gap:
                    label = "".join(_clean_text(value["text"]).replace(" ", "") for value in current)
                    if "井" in label:
                        anchor_labels.append(label)
                    current = []
                current.append(item)
            if current:
                label = "".join(_clean_text(value["text"]).replace(" ", "") for value in current)
                if "井" in label:
                    anchor_labels.append(label)
            for prefix, label in zip(key_prefixes, anchor_labels):
                prefix_to_well.setdefault(prefix, label)
            for row in rows:
                prefix = re.sub(r"[-_/]\d+$", "", re.sub(r"\s+", "", row[key_column]))
                if prefix in prefix_to_well:
                    row[well_column] = prefix_to_well[prefix]
                elif row[well_column]:
                    # 中文说明：纵排井名 OCR 常插入空格，井号/井名语义单元内部统一去空白。
                    row[well_column] = re.sub(r"\s+", "", _clean_text(row[well_column]))

        layer_column = next((
            index for index, header in enumerate(headers)
            if any(keyword in header for keyword in ("层位", "地层", "组段", "层段"))
        ), None)
        if layer_column is None:
            return
        grouped_indexes: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            prefix = re.sub(r"[-_/]\d+$", "", re.sub(r"\s+", "", row[key_column]))
            grouped_indexes.setdefault(prefix, []).append(index)
        for indexes in grouped_indexes.values():
            anchors: list[tuple[int, str]] = []
            position = 0
            while position < len(indexes):
                row_index = indexes[position]
                value = _clean_text(rows[row_index][layer_column])
                if not value:
                    position += 1
                    continue
                tokens = [value]
                end = position + 1
                if len(value.replace(" ", "")) <= 2:
                    while end < len(indexes):
                        next_value = _clean_text(rows[indexes[end]][layer_column])
                        if not next_value or len(next_value.replace(" ", "")) > 2:
                            break
                        tokens.append(next_value)
                        end += 1
                label = self._normalize_petroleum_layer("".join(tokens))
                anchors.append((position, label))
                position = max(end, position + 1)
            if not anchors:
                continue
            anchors[0] = (0, anchors[0][1])
            for anchor_index, (start, label) in enumerate(anchors):
                stop = anchors[anchor_index + 1][0] if anchor_index + 1 < len(anchors) else len(indexes)
                for relative_index in range(start, stop):
                    rows[indexes[relative_index]][layer_column] = label

    @staticmethod
    def _normalize_petroleum_layer(value: str) -> str:
        """规范纵排 OCR 中常见的龙马溪组、五峰组等石油地质层位文本。"""

        compact = re.sub(r"\s+", "", value)
        if "龙马" in compact and "溪" in compact:
            return "龙马溪组"
        if "五峰" in compact:
            return "五峰组"
        return compact

    @staticmethod
    def _replace_html_data_rows(
        canonical_html: str,
        *,
        data_start: int,
        rows: Sequence[Sequence[str]],
    ) -> str:
        """保留原多级表头，使用恢复后的完整记录重建 HTML 数据区。"""

        soup = BeautifulSoup(canonical_html, "lxml")
        table = soup.find("table")
        if table is None:
            return canonical_html
        html_rows = _table_rows(table)
        if not html_rows:
            return canonical_html
        parent = html_rows[-1].parent
        for row in html_rows[data_start:]:
            row.decompose()
        for values in rows:
            row_tag = soup.new_tag("tr")
            for value in values:
                cell_tag = soup.new_tag("td")
                cell_tag.string = _clean_text(value)
                row_tag.append(cell_tag)
            parent.append(row_tag)
        return canonicalize_html(str(table))


def _find_mineru_table_body(value: Any) -> str:
    """递归读取 MinerU content_list 中第一个非空 table_body。"""

    if isinstance(value, Mapping):
        body = value.get("table_body")
        if isinstance(body, str) and body.strip():
            return body.strip()
        for item in value.values():
            found = _find_mineru_table_body(item)
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found = _find_mineru_table_body(item)
            if found:
                return found
    return ""


def recognize_with_mineru(image_path: str | Path) -> tuple[str, dict[str, Any]]:
    """调用现有 MinerU 图片表格工具并提取其 HTML/Markdown 表格正文。"""

    from util.minerU_parseTable import parse_table_image

    content = parse_table_image(
        Path(image_path),
        token=model_settings.mineru.token,
        request_timeout=model_settings.mineru.timeout_secs,
        poll_interval=5.0,
        max_wait=600.0,
    )
    table_body = _find_mineru_table_body(content)
    if not table_body:
        raise RuntimeError("MinerU 结果中没有 table_body")
    return table_body, {"engine": "mineru", "content_list": content}


def recognize_table_source(
    source: TableSource,
    *,
    work_dir: str | Path,
    recognizer: RapidTableRecognizer | None = None,
    engine: str = "auto",
    minimum_score: float = 0.55,
) -> RecognizedTable:
    """将一个表格输入转为标准 HTML，并执行网格与质量校验。"""

    artifact_dir = Path(work_dir) / _safe_name(source.task_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_path = artifact_dir / "recognized_table.json"
    if cache_path.is_file():
        try:
            cached = RecognizedTable(**json.loads(cache_path.read_text(encoding="utf-8")))
            if (
                cached.source.task_id == source.task_id
                and cached.source.image_path == source.image_path
                and (engine == "auto" or cached.engine.startswith(engine))
            ):
                if cached.details.get("parser_version") == TABLE_PARSE_VERSION:
                    return cached
                # 中文说明：解析规则升级时复用耗时的 OCR/Unitable 原始 HTML，仅重建标准网格与质量报告。
                cached_raw_html = cached.raw_html
                cached_details = dict(cached.details)
                cached_engine = cached.engine
            else:
                cached_raw_html = ""
                cached_details = {}
                cached_engine = ""
        except Exception:
            # 中文说明：缓存损坏或模型结构升级时忽略缓存并重新识别，不让旧产物阻塞任务。
            cached_raw_html = ""
            cached_details = {}
            cached_engine = ""
    else:
        cached_raw_html = ""
        cached_details = {}
        cached_engine = ""
    errors: list[str] = []
    details: dict[str, Any] = cached_details
    raw_html = cached_raw_html
    used_engine = cached_engine or "input"
    active_recognizer = recognizer

    if raw_html:
        pass
    elif source.kind == TableSourceKind.IMAGE or str(source.kind) == "image":
        if not source.image_path:
            raise ValueError(f"图片表格任务缺少 image_path：{source.task_id}")
        if engine in {"auto", "rapidtable"}:
            try:
                active = recognizer or RapidTableRecognizer()
                active_recognizer = active
                raw_html, details = active.recognize(source.image_path, artifact_dir)
                used_engine = f"rapidtable:{active.model_name}"
            except Exception as exc:
                errors.append(f"rapidtable_failed:{exc}")
                if engine == "rapidtable":
                    raise
        if not raw_html and engine in {"auto", "mineru"}:
            raw_html, details = recognize_with_mineru(source.image_path)
            used_engine = "mineru"
    elif source.kind == TableSourceKind.HTML or str(source.kind) == "html":
        raw_html = source.content
        used_engine = "provided_html"
    else:
        raw_html = markdown_table_to_html(source.content)
        used_engine = "provided_markdown"

    if not details.get("logic_points") and details.get("pre_dense_recovery_logic_points"):
        # 中文说明：长表解析规则升级时复用恢复前的结构框，支持直接用已缓存的高精度 OCR 重新归列。
        details["logic_points"] = details["pre_dense_recovery_logic_points"]
        details["cell_bboxes"] = details.get("pre_dense_recovery_cell_bboxes") or []
    elif not details.get("logic_points") and details.get("pre_repair_logic_points"):
        # 中文说明：从上一版修复缓存升级时，先恢复模型原始框以定位局部表头，最终仍会重新计算 HTML 逻辑坐标。
        details["logic_points"] = details["pre_repair_logic_points"]
        details["cell_bboxes"] = details.get("pre_repair_cell_bboxes") or []
    details["parser_version"] = TABLE_PARSE_VERSION
    canonical_html = canonicalize_html(raw_html)
    grid = parse_html_table(canonical_html, details=details)
    if (
        source.image_path
        and used_engine.startswith("rapidtable")
        and active_recognizer is not None
    ):
        try:
            repaired_html, repairs = active_recognizer.repair_sparse_group_headers(
                source.image_path,
                canonical_html,
                grid,
            )
            if repairs:
                canonical_html = repaired_html
                details["header_repairs"] = repairs
                details["pre_repair_logic_points"] = details.get("logic_points") or []
                details["pre_repair_cell_bboxes"] = details.get("cell_bboxes") or []
                details["logic_points"] = []
                details["cell_bboxes"] = []
                details["geometry_recomputed_from_html"] = True
                # 中文说明：DOM 合并后旧模型坐标索引已经失效，必须以修复后的 rowspan/colspan 重新展开逻辑网格。
                grid = parse_html_table(canonical_html, details=details)
        except Exception as exc:
            # 中文说明：局部表头修复是精度增强项，失败时保留已通过的原始结构并留下可追踪告警。
            details.setdefault("header_repair_warnings", []).append(str(exc))
    if source.image_path and used_engine.startswith("rapidtable"):
        try:
            recovery_recognizer = active_recognizer or RapidTableRecognizer()
            recovered_html, recovered_rows = recovery_recognizer.recover_dense_tall_table(
                source.image_path,
                canonical_html,
                grid,
                details,
            )
            if recovered_rows:
                canonical_html = recovered_html
                details["pre_dense_recovery_logic_points"] = details.get("logic_points") or []
                details["pre_dense_recovery_cell_bboxes"] = details.get("cell_bboxes") or []
                details["logic_points"] = []
                details["cell_bboxes"] = []
                details["geometry_recomputed_from_html"] = True
                # 中文说明：长表恢复重写了完整数据区，需按新 HTML 重新建立准确的行列和单元格坐标关系。
                grid = parse_html_table(canonical_html, details=details)
        except Exception as exc:
            # 中文说明：恢复失败不吞掉原表；后续稀疏尾部质量门禁会阻止截断结果进入知识图谱。
            details.setdefault("dense_recovery_warnings", []).append(str(exc))
    quality = _quality_report(grid, canonical_html, details, minimum_score=minimum_score)
    quality.warnings.extend(errors)
    (artifact_dir / "canonical.html").write_text(canonical_html, encoding="utf-8")
    write_json(artifact_dir / "cells.json", grid)
    write_json(artifact_dir / "quality.json", quality)
    result = RecognizedTable(
        source=source,
        engine=used_engine,
        raw_html=raw_html,
        canonical_html=canonical_html,
        grid=grid,
        quality=quality,
        details=dict(details),
        artifact_dir=str(artifact_dir.resolve()),
    )
    write_json(cache_path, result)
    return result


__all__ = [
    "RapidTableRecognizer",
    "adapt_table_chunk",
    "canonicalize_html",
    "collect_table_sources",
    "markdown_table_to_html",
    "parse_html_table",
    "recognize_table_source",
    "recognize_with_mineru",
]
