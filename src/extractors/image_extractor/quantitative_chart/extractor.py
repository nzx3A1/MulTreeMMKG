"""定量图表与实验曲线图片抽取器骨架。"""
from ..base import BaseImageExtractor
from ..schema_models import ImageExtractorKind


class QuantitativeChartExtractor(BaseImageExtractor):
    """后续负责坐标轴、数据序列、趋势、相关性、阈值和时间变化抽取。"""

    kind = ImageExtractorKind.QUANTITATIVE_CHART
    display_name = "定量图表与实验曲线抽取器"
    supported_codes = frozenset({"A14", "A15", "A16", "A17"})
