"""多模态内容模型。

MinerU 阶段直接产出 markdown 与内容清单，因此这里不再维护 PDF 版面坐标类，
只保存下游解析、切块和抽取真正需要的文本化内容与资源引用。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import ID, MMKGBaseModel, Provenance, SourceModality


class ContentRole(str, Enum):
    """内容块在论文结构中的角色。"""

    BODY = "body"
    TITLE = "title"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    REFERENCE = "reference"


class ContentBlock(MMKGBaseModel):
    """所有多模态内容块的公共字段。"""

    id: ID
    document_id: Optional[ID] = None
    section_id: Optional[ID] = None
    order: int = Field(default=0, ge=0)
    modality: SourceModality
    role: ContentRole = ContentRole.BODY
    provenance: Optional[Provenance] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextBlock(ContentBlock):
    """正文或标题文本块。"""

    modality: SourceModality = SourceModality.TEXT
    text: str = ""


class Table(ContentBlock):
    """表格内容块，保存 markdown 表格和可选结构化单元格。"""

    modality: SourceModality = SourceModality.TABLE
    caption: Optional[str] = None
    markdown: str = ""
    html: Optional[str] = None
    cells: List[List[str]] = Field(default_factory=list)


class Image(ContentBlock):
    """图片内容块，保存 MinerU 导出的图片路径、URL 和图注。"""

    modality: SourceModality = SourceModality.IMAGE
    path: Optional[str] = None
    url: Optional[str] = None
    caption: Optional[str] = None
    alt_text: Optional[str] = None


class Formula(ContentBlock):
    """公式内容块，保存 LaTeX 或 markdown 中的公式文本。"""

    modality: SourceModality = SourceModality.FORMULA
    latex: str = ""
    caption: Optional[str] = None


class Caption(ContentBlock):
    """图、表、公式等对象的说明文字。"""

    modality: SourceModality = SourceModality.TEXT
    text: str = ""
    target_id: Optional[ID] = None
