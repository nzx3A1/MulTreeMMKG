"""岩石与微观储集空间图片抽取器骨架。"""
from ..base import BaseImageExtractor
from ..schema_models import ImageExtractorKind


class RockMicroExtractor(BaseImageExtractor):
    """后续负责岩性、矿物、孔隙、裂缝、充填和孔喉连通关系抽取。"""

    kind = ImageExtractorKind.ROCK_MICRO
    display_name = "岩石与微观储集空间抽取器"
    supported_codes = frozenset({"A10", "A11", "A12", "A13"})
