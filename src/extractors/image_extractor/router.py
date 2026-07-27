"""从图片分类编码路由到专用抽取器。"""
from __future__ import annotations

from .schema_models import ImageExtractionTask, ImageExtractorKind


CODE_TO_KIND: dict[str, ImageExtractorKind] = {
    **{code: ImageExtractorKind.MAP_SPATIAL for code in ("A01", "A02", "A07", "A18")},
    **{code: ImageExtractorKind.STRATIGRAPHIC_PROFILE for code in ("A03", "A04", "A05", "A06")},
    **{code: ImageExtractorKind.ROCK_MICRO for code in ("A10", "A11", "A12", "A13")},
    **{code: ImageExtractorKind.QUANTITATIVE_CHART for code in ("A14", "A15", "A16", "A17")},
    **{code: ImageExtractorKind.GEOLOGICAL_PROCESS for code in ("A08", "A09")},
    **{code: ImageExtractorKind.COMPOSITE_PANEL for code in ("A19", "A20")},
}


class ImageExtractorRouter:
    """只负责选择抽取器类型，不持有模型客户端或业务实现。"""

    def route(self, task: ImageExtractionTask) -> ImageExtractorKind:
        """中文说明：依据 A01-A20 编码路由，未知或未分类图片进入综合图版兜底器。"""

        code = task.classification_code.strip().upper()
        return CODE_TO_KIND.get(code, ImageExtractorKind.COMPOSITE_PANEL)

