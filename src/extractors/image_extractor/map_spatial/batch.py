"""已知为地图或平面空间图的 ImageChunk 批量抽取便捷入口。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from model import Graph

from ..classification import ImageClassification


class MapSpatialClassificationProvider:
    """调用方已确认图片类型时，直接将图片送入地图空间抽取器。"""

    def resolve(self, chunk: Mapping[str, Any], image_path: str, image_index: int) -> ImageClassification:
        """为已确认的地图图片返回固定地图分类。"""

        _ = (chunk, image_path, image_index)
        return ImageClassification(code="A01", type_name="地图与平面空间图（调用方已指定）")


def extract_map_spatial_chunks(
    image_chunks: Sequence[Mapping[str, Any]],
    vlm_client: Any,
    *,
    output_path: str | Path | None = None,
    show_progress: bool = True,
) -> list[Graph]:
    """不依赖预先分类记录，直接抽取一组已知为地图的 chunk。"""

    # 局部导入避免默认注册表加载本包时产生循环依赖。
    from ..pipeline import extract_from_images

    return extract_from_images(
        image_chunks,
        llm_client=None,
        vlm_client=vlm_client,
        classification_provider=MapSpatialClassificationProvider(),
        context_options={"allowed_extractor_kinds": ["map_spatial"]},
        output_path=output_path,
        show_progress=show_progress,
    )


__all__ = ["MapSpatialClassificationProvider", "extract_map_spatial_chunks"]
