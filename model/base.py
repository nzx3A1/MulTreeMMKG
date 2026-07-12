"""基础数据模型。

本文件定义所有阶段共享的 Pydantic 基类、通用 ID 和内容模态枚举。
MinerU 已直接产出 markdown，因此模型层不再维护版面模态或页码坐标字段。
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel


ID = str
Timestamp = datetime


class SourceModality(str, Enum):
    """描述内容或证据来自哪一种业务内容模态。"""

    TEXT = "text"           # 文本模态，正文段落
    TABLE = "table"         # 表格模态，结构化数据
    IMAGE = "image"         # 图像模态，地质图/剖面图/岩心照片等
    FORMULA = "formula"     # 公式模态，LaTeX 公式或图片公式
    METADATA = "metadata"   # 元数据模态，文档元信息
    MIXED = "mixed"         # 混合模态，包含多种模态


class MMKGBaseModel(BaseModel):
    """项目内所有 Pydantic 模型的统一基类。

    这里兼容 Pydantic v1/v2：业务代码优先使用 to_dict/to_json，避免直接依赖
    model_dump 这类版本相关 API。
    """

    class Config:
        extra = "forbid"
        validate_assignment = True
        use_enum_values = True
        allow_population_by_field_name = True

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        """将模型转换为普通字典，默认使用 JSON 兼容值。"""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json", **kwargs)  # type: ignore[attr-defined]
        return self.dict(**kwargs)

    def to_json(self, **kwargs: Any) -> str:
        """将模型转换为 JSON 字符串，默认保留中文。"""

        kwargs.setdefault("ensure_ascii", False)
        if hasattr(self, "model_dump_json"):
            return self.model_dump_json(**kwargs)  # type: ignore[attr-defined]
        return self.json(**kwargs)

    def __getitem__(self, key: str) -> Any:
        """提供只读字典式访问，兼容仍使用 ``result[\"field\"]`` 的旧调用方。"""

        return getattr(self, key)
