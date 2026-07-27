"""AMarkdownParser 图片表格识别的离线回归测试。"""

from src.parser.AMarkdownParser import AMarkdownParser


def _first_section(markdown: str) -> dict:
    """解析最小 Markdown，并返回第一个正文章节。"""

    result = AMarkdownParser().parse_text(markdown)
    return result["toc"][0]


def test_captioned_image_is_classified_as_table() -> None:
    """“表/Table + 数字”的多行标题后首张图片应归入 table。"""

    section = _first_section(
        """# 测试论文

## 1 结果

表 2 中文表格标题
Table 2 English table title
![](images/table-2.jpg)

![](images/figure-3.jpg)
图 3 普通图片标题
"""
    )

    assert len(section["table"]) == 1
    assert section["table"][0]["path"] == "images/table-2.jpg"
    assert section["table"][0]["caption"] == "表 2 中文表格标题\nTable 2 English table title"
    assert section["table"][0]["source_type"] == "image"
    assert section["images"] == [
        {
            "path": ["images/figure-3.jpg"],
            "caption": "图 3 普通图片标题",
            "references": [],
        }
    ]


def test_only_first_image_after_table_caption_is_table() -> None:
    """表题后的第一张图片归为表格，后续图片仍按普通图片处理。"""

    section = _first_section(
        """# 测试论文

## 1 结果

Table 1 L
A-IC
P-MS zircon results
![](images/table-1.jpg)

![](images/figure-a.jpg)

![](images/figure-b.jpg)
图 5 后续普通图片
"""
    )

    assert [table["path"] for table in section["table"]] == ["images/table-1.jpg"]
    assert section["images"][0]["path"] == ["images/figure-a.jpg", "images/figure-b.jpg"]


def test_inline_table_reference_does_not_claim_following_image() -> None:
    """正文中的表编号引用不能把后续普通图片误识别为图片表格。"""

    section = _first_section(
        """# 测试论文

## 1 结果

分析结果见 Table 3，其中包含多组数据。
![](images/ordinary.jpg)
图 1 普通图片
"""
    )

    assert section["table"] == []
    assert section["images"][0]["path"] == ["images/ordinary.jpg"]


def test_chinese_caption_without_space_is_supported() -> None:
    """中文表号与标题之间没有空格时也应识别为图片表格。"""

    section = _first_section(
        """# 测试论文

## 1 结果

表1火成岩样品分析结果
![](images/table-1.jpg)
"""
    )

    assert [table["path"] for table in section["table"]] == ["images/table-1.jpg"]
