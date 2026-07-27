"""地图与平面空间图片抽取器骨架。"""
from ..base import BaseImageExtractor
from ..schema_models import ImageExtractorKind


class MapSpatialExtractor(BaseImageExtractor):
    """后续负责地图实体、空间拓扑、方位、距离和参数梯度抽取。"""

    kind = ImageExtractorKind.MAP_SPATIAL
    display_name = "地图与平面空间抽取器"
    supported_codes = frozenset({"A01", "A02", "A07", "A18"})
