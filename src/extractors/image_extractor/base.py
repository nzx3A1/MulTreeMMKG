"""图片专用抽取器的统一基类。"""
from __future__ import annotations

from abc import ABC

from model import Graph
from model.base import SourceModality

from .schema_models import ImageExtractionContext, ImageExtractionTask, ImageExtractorKind


class BaseImageExtractor(ABC):
    """所有图片抽取器共享的接口，后续实现只需覆盖 ``extract``。"""

    kind: ImageExtractorKind
    display_name: str
    supported_codes: frozenset[str] = frozenset()

    def extract(self, task: ImageExtractionTask, context: ImageExtractionContext) -> Graph:
        """中文说明：当前仅返回带路由信息的空 Graph，不调用 LLM 或 VLM。"""

        _ = context
        graph = Graph.from_chunk(
            document_id=task.document_id,
            chunk_id=task.chunk_id,
            modality=SourceModality.IMAGE,
            stage="stage_04_image_extraction_scaffold",
        )
        graph.metadata.extra.update(
            {
                "status": "not_implemented",
                "extractor_kind": self.kind.value,
                "extractor_name": self.display_name,
                "image_id": task.image_id,
                "image_index": task.image_index,
                "image_path": task.image_path,
                "classification_code": task.classification_code,
                "classification_type": task.classification_type,
                "model_called": False,
            }
        )
        return graph

