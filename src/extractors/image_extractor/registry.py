"""图片抽取器注册表，允许后续替换或新增实现。"""
from __future__ import annotations

from collections.abc import Iterable

from .base import BaseImageExtractor
from .schema_models import ImageExtractorKind


class ImageExtractorRegistry:
    """按抽取器类型保存实例，避免路由层依赖具体实现脚本。"""

    def __init__(self, extractors: Iterable[BaseImageExtractor] = ()) -> None:
        """中文说明：初始化注册表，并按顺序注册传入的抽取器实例。"""

        self._extractors: dict[ImageExtractorKind, BaseImageExtractor] = {}
        for extractor in extractors:
            self.register(extractor)

    def register(self, extractor: BaseImageExtractor, *, replace: bool = False) -> None:
        """中文说明：注册抽取器；显式允许 replace 时可替换默认实现。"""

        if extractor.kind in self._extractors and not replace:
            raise ValueError(f"图片抽取器已注册：{extractor.kind.value}")
        self._extractors[extractor.kind] = extractor

    def get(self, kind: ImageExtractorKind) -> BaseImageExtractor:
        """中文说明：按稳定类型标识取得抽取器，缺失时给出明确错误。"""

        try:
            return self._extractors[kind]
        except KeyError as exc:
            raise KeyError(f"未注册图片抽取器：{kind.value}") from exc

    def kinds(self) -> tuple[ImageExtractorKind, ...]:
        """中文说明：返回当前已注册的抽取器类型，供诊断与测试使用。"""

        return tuple(self._extractors)

