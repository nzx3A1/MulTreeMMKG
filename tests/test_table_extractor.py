"""表格 HTML、行列关系和知识图谱装配的离线回归测试。"""
from __future__ import annotations

from pathlib import Path

from src.extractors.table_extractor.schema_models import (
    RecognizedTable,
    TableQualityReport,
    TableSource,
    TableSourceKind,
)
from src.extractors.table_extractor.schema_selector import TableSchemaSelector
from src.extractors.table_extractor.table_extractor import build_table_graph, infer_semantic_plan
from src.extractors.table_extractor.table_parse import (
    RapidTableRecognizer,
    _quality_report,
    adapt_table_chunk,
    canonicalize_html,
    parse_html_table,
)


class OfflineRepository:
    """模拟 Neo4j 不可用，验证结构 Schema 回退路径。"""

    def all_concepts(self):
        """抛出连接错误，触发离线回退。"""

        raise ConnectionError("offline")

    def induced_relations(self, schemas):
        """离线状态不返回数据库关系。"""

        _ = schemas
        return ()


def _recognized(html: str, caption: str = "岩性矿物含量表") -> RecognizedTable:
    """构造已通过质量门的表格，避免单元测试加载 OCR 模型。"""

    canonical = canonicalize_html(html)
    grid = parse_html_table(canonical)
    return RecognizedTable(
        source=TableSource(
            document_id="doc-1",
            chunk_id="chunk-1",
            task_id="chunk-1",
            section_id="section-1",
            section_title="储层特征",
            caption=caption,
            kind=TableSourceKind.HTML,
            content=html,
        ),
        engine="test",
        raw_html=html,
        canonical_html=canonical,
        grid=grid,
        quality=TableQualityReport(
            passed=True,
            score=1.0,
            nonempty_ratio=1.0,
            ocr_coverage=1.0,
            geometry_consistency=1.0,
        ),
        artifact_dir="output/test-table",
    )


def test_parse_html_expands_rowspan_and_colspan() -> None:
    """合并单元格应展开为矩形网格，同时保留原始跨度。"""

    table = _recognized(
        """
        <table>
          <tr><th rowspan="2">岩性</th><th colspan="2">孔隙度/%</th></tr>
          <tr><th>最小</th><th>最大</th></tr>
          <tr><td>砂岩</td><td>5.1</td><td>12.3</td></tr>
        </table>
        """
    )
    assert table.grid.row_count == 3
    assert table.grid.column_count == 3
    assert table.grid.matrix[1][0] == "岩性"
    assert table.grid.header_paths == ["岩性", "孔隙度/% / 最小", "孔隙度/% / 最大"]
    assert table.grid.cells[0].rowspan == 2
    assert table.grid.cells[1].colspan == 2


def test_caption_rows_do_not_hide_nested_headers() -> None:
    """表内中英文标题行应被剔除，后续两级字段仍需组成真实表头路径。"""

    table = _recognized(
        """
        <table>
          <tr><td colspan="4">表2 典型岩石物性统计</td></tr>
          <tr><td>Table 2</td><td>Statistics of</td><td>rock</td><td>samples</td></tr>
          <tr><td rowspan="2">岩石类型</td><td rowspan="2">样品个数</td><td colspan="2">有效孔隙度/%</td></tr>
          <tr><td>最小</td><td>最大</td></tr>
          <tr><td>砂岩</td><td>10</td><td>5.1</td><td>12.3</td></tr>
        </table>
        """
    )
    assert table.grid.header_rows == [2, 3]
    assert table.grid.header_paths == ["岩石类型", "样品个数", "有效孔隙度/% / 最小", "有效孔隙度/% / 最大"]


def test_numeric_first_row_is_treated_as_headerless_continuation() -> None:
    """无表头续表的首条数值记录不能被误当作列名并从数据中丢失。"""

    table = _recognized(
        """
        <table>
          <tr><td>FY3-1</td><td>4144.57</td><td>2.32</td><td>6.1</td></tr>
          <tr><td>FY3-2</td><td>4147.49</td><td>3.11</td><td>6.2</td></tr>
        </table>
        """
    )
    assert table.grid.header_rows == []
    assert table.grid.header_paths == ["列1", "列2", "列3", "列4"]
    assert infer_semantic_plan(table).data_start_row == 0


def test_petroleum_record_key_detection_is_domain_general() -> None:
    """长表恢复应支持样品编号、测点编号和井深，不绑定某一口井或某个样品前缀。"""

    recognizer = RapidTableRecognizer()
    assert recognizer._select_record_key_column(["井号", "样品编号", "井深/m"]) == 1
    assert recognizer._select_record_key_column(["岩性", "测点编号", "孔隙度/%"]) == 1
    assert recognizer._select_record_key_column(["层位", "井深/m", "含油饱和度/%"]) == 1
    assert recognizer._is_record_key("FY3-12", "样品号")
    assert recognizer._is_record_key("Z301_08", "岩心编号")
    assert recognizer._is_record_key("4187.82", "井深/m")
    assert not recognizer._is_record_key("孔隙度", "样品号")


def test_dense_table_replacement_preserves_nested_headers() -> None:
    """OCR 重建数据区时应保留 rowspan/colspan 多级表头并替换截断行。"""

    original = canonicalize_html(
        """
        <table>
          <tr><th rowspan="2">样品号</th><th colspan="2">孔隙结构</th></tr>
          <tr><th>孔隙度/%</th><th>渗透率/10^-3μm²</th></tr>
          <tr><td>A1-1</td><td>4.2</td><td>0.08</td></tr>
          <tr><td>A1-2</td><td></td><td></td></tr>
        </table>
        """
    )
    repaired = RapidTableRecognizer._replace_html_data_rows(
        original,
        data_start=2,
        rows=[["A1-1", "4.2", "0.08"], ["A1-2", "5.1", "0.12"], ["A1-3", "5.5", "0.15"]],
    )
    grid = parse_html_table(repaired)
    assert grid.header_paths == ["样品号", "孔隙结构 / 孔隙度/%", "孔隙结构 / 渗透率/10^-3μm²"]
    assert grid.row_count == 5
    assert grid.matrix[-1] == ["A1-3", "5.5", "0.15"]


def test_sparse_tail_rows_fail_quality_gate() -> None:
    """密集表尾部连续缺列时必须失败，防止把结构模型截断结果装配为知识图谱。"""

    html = canonicalize_html(
        """
        <table>
          <tr><th>井名</th><th>样品号</th><th>井深/m</th><th>孔隙度/%</th><th>渗透率/mD</th><th>含油饱和度/%</th></tr>
          <tr><td>A井</td><td>A1-1</td><td>3101</td><td>6.1</td><td>0.10</td><td>42</td></tr>
          <tr><td>A井</td><td>A1-2</td><td>3102</td><td>6.2</td><td>0.11</td><td>43</td></tr>
          <tr><td>A井</td><td>A1-3</td><td>3103</td><td>6.3</td><td>0.12</td><td>44</td></tr>
          <tr><td>A井</td><td>A1-4</td><td>3104</td><td>6.4</td><td>0.13</td><td>45</td></tr>
          <tr><td>A井</td><td></td><td></td><td></td><td></td><td></td></tr>
          <tr><td>A井</td><td></td><td></td><td></td><td></td><td></td></tr>
        </table>
        """
    )
    grid = parse_html_table(html)
    quality = _quality_report(grid, html, {}, minimum_score=0.55)
    assert quality.passed is False
    assert "sparse_tail_rows" in quality.errors


def test_ocr_coverage_is_stable_when_detection_order_changes() -> None:
    """OCR 框顺序与 HTML 行序不一致时，单元格覆盖率仍应按内容正确计算。"""

    html = canonicalize_html(
        """
        <table>
          <tr><th>样品号</th><th>孔隙度/%</th><th>层位</th></tr>
          <tr><td>FY1-1</td><td>5.2</td><td>龙马溪组</td></tr>
          <tr><td>FY1-2</td><td>4.8</td><td>五峰组</td></tr>
        </table>
        """
    )
    grid = parse_html_table(html)
    details = {"ocr": [
        {"text": "五峰组"}, {"text": "4.8"}, {"text": "FY1-2"},
        {"text": "层位"}, {"text": "孔隙度"}, {"text": "/%"},
        {"text": "样品号"}, {"text": "龙马溪组"}, {"text": "5.2"}, {"text": "FY1-1"},
    ]}
    quality = _quality_report(grid, html, details, minimum_score=0.55)
    assert quality.ocr_coverage > 0.9
    assert "low_ocr_coverage" not in quality.warnings


def test_petroleum_layer_normalization_handles_vertical_ocr_order() -> None:
    """纵排地层文字即使 OCR 次序颠倒，也应恢复为规范组名。"""

    assert RapidTableRecognizer._normalize_petroleum_layer("溪组 龙马") == "龙马溪组"
    assert RapidTableRecognizer._normalize_petroleum_layer("五 峰") == "五峰组"


def test_dense_petroleum_subheaders_are_split_by_column_geometry() -> None:
    """合并 OCR 短语应按矿物词和列中心拆成独立二级表头，并规范参数单位。"""

    html = canonicalize_html(
        """
        <table>
          <tr><th rowspan="2">渗透率₂ /×10-3$ m</th><th colspan="3">全岩矿物含量 /%</th></tr>
          <tr><th></th><th>石英 斜长石</th><th>方解石</th></tr>
          <tr><td>0.01</td><td>40</td><td>10</td><td>20</td></tr>
        </table>
        """
    )
    grid = parse_html_table(html)
    for cell in grid.cells:
        if cell.row_start == 1:
            center = 100.0 + 60.0 * cell.col_start
            cell.bbox = [center - 25, 40, center + 25, 40, center + 25, 70, center - 25, 70]
    items = [
        {"text": "石英 斜长石", "score": 0.99, "x": 190.0, "y": 55.0},
        {"text": "方解石", "score": 0.99, "x": 280.0, "y": 55.0},
    ]
    repaired, repairs = RapidTableRecognizer()._repair_dense_nested_headers(
        html,
        grid,
        items,
        [100.0, 160.0, 220.0, 280.0],
    )
    repaired_grid = parse_html_table(repaired)
    assert repairs[0]["subheaders"] == ["石英", "斜长石", "方解石"]
    assert repaired_grid.header_paths == [
        "渗透率/(×10^-3 μm²)",
        "全岩矿物含量/% / 石英",
        "全岩矿物含量/% / 斜长石",
        "全岩矿物含量/% / 方解石",
    ]


def test_adapt_table_chunk_recovers_image_path_from_markdown(tmp_path: Path) -> None:
    """当前 Stage-02 把图片路径放入 markdown 时仍应生成图片表格任务。"""

    image = tmp_path / "table.png"
    image.write_bytes(b"not-an-image-but-existing")
    sources = adapt_table_chunk({
        "id": "table-1",
        "modality": "table",
        "markdown": str(image),
        "table_path": [],
    })
    assert len(sources) == 1
    assert sources[0].kind == "image"
    assert sources[0].image_path == str(image.resolve())


def test_build_table_graph_keeps_structure_and_domain_semantics() -> None:
    """横向岩性表应同时生成表格结构节点、岩性节点和矿物关系。"""

    table = _recognized(
        """
        <table>
          <tr><th>深度/m</th><th>岩性</th><th>石英/%</th></tr>
          <tr><td>3116.5</td><td>砂岩</td><td>30.0</td></tr>
          <tr><td>3120.0</td><td>泥岩</td><td>10.0</td></tr>
        </table>
        """
    )
    graph = build_table_graph(
        table,
        schema_selector=TableSchemaSelector(repository=OfflineRepository()),
    )
    types = {entity.type for entity in graph.entities}
    relation_types = {relation.type for relation in graph.relations}
    assert {"Table", "TableRow", "TableColumn", "TableCell", "Parameter", "Lithology", "Mineral"} <= types
    assert {"HAS_ROW", "HAS_COLUMN", "HAS_CELL", "LOCATED_IN_ROW", "LOCATED_IN_COLUMN", "COMPOSED_OF"} <= relation_types
    assert graph.metadata.modality == "table"
    assert graph.metadata.extra["validation"]["ok"] is True


def test_sample_table_prefers_sample_key_and_links_geological_context() -> None:
    """样品分析表应以样品号为主实体，并关联井、地层组、岩性及矿物组成。"""

    table = _recognized(
        """
        <table>
          <tr><th>井名</th><th>样品号</th><th>层位</th><th>岩性</th><th>石英/%</th></tr>
          <tr><td>富页1井</td><td>FY1-1</td><td>龙马溪组</td><td>页岩</td><td>42.1</td></tr>
          <tr><td>富页1井</td><td>FY1-2</td><td>龙马溪组</td><td>页岩</td><td>38.9</td></tr>
        </table>
        """,
        caption="页岩样品矿物组成表",
    )
    plan = infer_semantic_plan(table)
    assert plan.subject_column == 1
    assert plan.subject_type == "Sample"

    graph = build_table_graph(
        table,
        schema_selector=TableSchemaSelector(repository=OfflineRepository()),
    )
    type_counts = {entity_type: sum(entity.type == entity_type for entity in graph.entities) for entity_type in {
        "Sample", "Well", "Formation", "Lithology", "Mineral"
    }}
    relation_types = {relation.type for relation in graph.relations}
    assert type_counts == {"Sample": 2, "Well": 1, "Formation": 1, "Lithology": 1, "Mineral": 1}
    assert {"REPORTS", "RECORDED_IN", "COLLECTED_FROM", "COMPOSED_OF"} <= relation_types
    assert graph.metadata.extra["validation"]["ok"] is True


def test_infer_vertical_table_uses_key_row() -> None:
    """转置表应把井号所在行作为主键并按数据列创建记录。"""

    table = _recognized(
        """
        <table>
          <tr><td>指标类别</td><td>具体指标</td><td>YC101</td><td>YC102</td></tr>
          <tr><td rowspan="3">基础信息</td><td>井号</td><td>YC101</td><td>YC102</td></tr>
          <tr><td>层位</td><td>长6段</td><td>长7段</td></tr>
          <tr><td>埋深/m</td><td>2460</td><td>2658</td></tr>
          <tr><td>储层物性</td><td>孔隙度/%</td><td>8.6</td><td>9.4</td></tr>
        </table>
        """,
        caption="储层参数转置表",
    )
    # 中文说明：该测试表全部使用 td，方向必须由左侧分类/指标密度自动判断，不能依赖 th 标签。
    assert table.grid.orientation == "vertical"
    assert table.grid.header_columns == [0, 1]
    plan = infer_semantic_plan(table)
    assert plan.subject_row == 1
    assert plan.subject_type == "Well"
    assert plan.data_start_col == 2

    graph = build_table_graph(
        table,
        schema_selector=TableSchemaSelector(repository=OfflineRepository()),
    )
    parameter_names = {entity.name for entity in graph.entities if entity.type == "Parameter"}
    assert {"井号", "层位", "埋深/m", "孔隙度/%"} <= parameter_names
    assert "YC101" not in parameter_names
    assert graph.metadata.extra["validation"]["ok"] is True
