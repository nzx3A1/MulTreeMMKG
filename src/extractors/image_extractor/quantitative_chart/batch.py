"""已知为定量图表的 ImageChunk 批量抽取便捷入口。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from model import Graph

from ..classification import ImageClassification


class QuantitativeChartClassificationProvider:
    """调用方已确认图片类型时，直接将图片送入定量图表抽取器。"""

    def resolve(self, chunk: Mapping[str, Any], image_path: str, image_index: int) -> ImageClassification:
        _ = (chunk, image_path, image_index)
        return ImageClassification(code="A14", type_name="定量图表（调用方已指定）")


def extract_quantitative_chart_chunks(
    image_chunks: Sequence[Mapping[str, Any]],
    vlm_client: Any,
    *,
    output_path: str | Path | None = None,
    show_progress: bool = True,
    max_points_per_series: int = 24,
) -> list[Graph]:
    """不依赖预先分类记录，直接抽取一组已知为定量图表的 chunk。"""

    # 局部导入避免图片管线装配默认注册表时产生循环依赖。
    from ..pipeline import extract_from_images

    return extract_from_images(
        image_chunks,
        llm_client=None,
        vlm_client=vlm_client,
        classification_provider=QuantitativeChartClassificationProvider(),
        context_options={
            "allowed_extractor_kinds": ["quantitative_chart"],
            "quantitative_chart_max_points_per_series": max_points_per_series,
        },
        output_path=output_path,
        show_progress=show_progress,
    )


__all__ = ["QuantitativeChartClassificationProvider", "extract_quantitative_chart_chunks"]
