"""默认图片抽取器注册表工厂。"""
from .composite_panel import CompositePanelExtractor
from .geological_process import GeologicalProcessExtractor
from .map_spatial import MapSpatialExtractor
from .quantitative_chart import QuantitativeChartExtractor
from .registry import ImageExtractorRegistry
from .rock_micro import RockMicroExtractor
from .stratigraphic_profile import StratigraphicProfileExtractor


def build_default_registry() -> ImageExtractorRegistry:
    """中文说明：组装六类默认抽取器，后续可通过注册表替换任一实现。"""

    return ImageExtractorRegistry(
        [
            MapSpatialExtractor(),
            StratigraphicProfileExtractor(),
            RockMicroExtractor(),
            QuantitativeChartExtractor(),
            GeologicalProcessExtractor(),
            CompositePanelExtractor(),
        ]
    )
