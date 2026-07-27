"""图片多模态知识抽取的统一公开入口。"""

from .base import BaseImageExtractor
from .classification import (
    ImageClassification,
    ImageClassificationProvider,
    InlineImageClassificationProvider,
    RecordImageClassificationProvider,
    VLMImageClassificationProvider,
)
from .factory import build_default_registry
from .pipeline import build_image_tasks, extract_from_images, extract_table_embedded_hybrid_only
from .registry import ImageExtractorRegistry
from .router import ImageExtractorRouter
from .schema_models import ImageExtractionContext, ImageExtractionTask, ImageExtractorKind

__all__ = [
    "BaseImageExtractor",
    "ImageExtractionContext",
    "ImageExtractionTask",
    "ImageClassification",
    "ImageClassificationProvider",
    "InlineImageClassificationProvider",
    "RecordImageClassificationProvider",
    "VLMImageClassificationProvider",
    "ImageExtractorKind",
    "ImageExtractorRegistry",
    "ImageExtractorRouter",
    "build_default_registry",
    "build_image_tasks",
    "extract_from_images",
    "extract_table_embedded_hybrid_only",
]
