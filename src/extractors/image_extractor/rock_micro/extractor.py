"""使用 VLM 生成岩石与微观储集空间图片的文本描述。"""
from __future__ import annotations

import os
from typing import Any, Mapping

from model import Graph
from model.base import SourceModality

from ..base import BaseImageExtractor
from ..schema_models import ImageExtractionContext, ImageExtractionTask, ImageExtractorKind
from .prompt import build_rock_micro_description_prompt


class RockMicroExtractor(BaseImageExtractor):
    """仅调用视觉模型描述图片，不抽取实体、关系或事件。"""

    kind = ImageExtractorKind.ROCK_MICRO
    display_name = "岩石与微观储集空间抽取器"
    supported_codes = frozenset({"A10", "A11", "A12", "A13"})

    def extract(self, task: ImageExtractionTask, context: ImageExtractionContext) -> Graph:
        """调用一次 VLM，将自由文本描述原样放入空 Graph 的元数据。"""

        try:
            describe_image = self._resolve_describe_image(context.vlm_client)
        except Exception as exc:
            return self._failed_graph(task, str(exc), model_called=False)

        try:
            response = describe_image(
                task.image_path,
                build_rock_micro_description_prompt(task),
                task_name=f"岩石与微观储集空间图片描述:{task.image_id}",
                max_tokens=int(os.getenv("ROCK_MICRO_VLM_MAX_TOKENS", "2048")),
            )
            description = self._normalize_description(response)
        except Exception as exc:
            return self._failed_graph(task, f"VLM 图片描述失败：{exc}", model_called=True)

        graph = Graph.from_chunk(
            document_id=task.document_id,
            chunk_id=task.chunk_id,
            modality=SourceModality.IMAGE,
            raw_response=description,
            stage="stage_04_image_rock_micro_description",
        )
        graph.metadata.extra.update(self._source_metadata(task))
        graph.metadata.extra.update(
            {
                "status": "completed",
                "description": description,
                "description_language": "zh-CN",
                "description_scope": "single_image",
                "model_called": True,
                "vlm_called": True,
                "vlm_call_count": 1,
                "llm_called": False,
                "model_errors": [],
            }
        )
        return graph

    @staticmethod
    def _resolve_describe_image(vlm_client: Any) -> Any:
        """取得客户端方法，并给缺失依赖返回清晰错误。"""

        if vlm_client is None:
            raise TypeError("RockMicroExtractor 需要 VLMClient")
        describe_image = getattr(vlm_client, "describe_image")
        if not callable(describe_image):
            raise TypeError("RockMicroExtractor 需要支持 describe_image 的 VLMClient")
        return describe_image

    @staticmethod
    def _normalize_description(response: Any) -> str:
        """兼容纯文本响应及少数测试客户端返回的 description 对象。"""

        if isinstance(response, Mapping):
            response = response.get("description", "")
        description = str(response or "").strip()
        if not description:
            raise ValueError("VLM 返回了空文本描述")
        return description

    def _failed_graph(self, task: ImageExtractionTask, error: str, *, model_called: bool) -> Graph:
        """将单图失败转换为可聚合结果，避免中断同一批次的其他图片。"""

        graph = Graph.from_chunk(
            document_id=task.document_id,
            chunk_id=task.chunk_id,
            modality=SourceModality.IMAGE,
            stage="stage_04_image_rock_micro_description",
        )
        graph.metadata.extra.update(self._source_metadata(task))
        graph.metadata.extra.update(
            {
                "status": "model_error",
                "description": "",
                "model_called": model_called,
                "vlm_called": model_called,
                "vlm_call_count": int(model_called),
                "llm_called": False,
                "model_errors": [error],
            }
        )
        return graph

    def _source_metadata(self, task: ImageExtractionTask) -> dict[str, Any]:
        """返回成功与失败结果共享的来源和路由字段。"""

        return {
            "extractor_kind": self.kind.value,
            "extractor_name": self.display_name,
            "image_id": task.image_id,
            "image_index": task.image_index,
            "image_path": task.image_path,
            "source_image_path": task.image_path,
            "classification_code": task.classification_code,
            "classification_type": task.classification_type,
        }
