"""综合图版与子图路由抽取器骨架。"""
from ..base import BaseImageExtractor
from ..schema_models import ImageExtractorKind


class CompositePanelExtractor(BaseImageExtractor):
    """后续负责多子图切分、二次路由和跨子图、跨尺度证据对齐。"""

    kind = ImageExtractorKind.COMPOSITE_PANEL
    display_name = "综合图版与子图路由抽取器"
    supported_codes = frozenset({"A19", "A20"})
