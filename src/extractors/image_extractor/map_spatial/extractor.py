"""地图与平面空间图片抽取器。"""
from __future__ import annotations

import os
from typing import Any, Mapping

from model import Graph
from model.base import SourceModality
from src.utils.llm_client import safe_json_loads

from ..base import BaseImageExtractor
from ..schema_models import ImageExtractionContext, ImageExtractionTask, ImageExtractorKind
from .graph import build_map_spatial_graph, normalize_map_spatial_result
from .prompt import build_map_spatial_prompt


class MapSpatialExtractor(BaseImageExtractor):
    """抽取地图主题、地质对象、属性值及证据锚定的平面空间关系。"""

    kind = ImageExtractorKind.MAP_SPATIAL
    display_name = "地图与平面空间抽取器"
    supported_codes = frozenset({"A01", "A02", "A07", "A18"})

    def extract(self, task: ImageExtractionTask, context: ImageExtractionContext) -> Graph:
        """调用一次 VLM，规范化其 JSON 后确定性装配统一 Graph。"""

        call_attempted = False
        try:
            describe_image = self._resolve_describe_image(context.vlm_client)
            call_attempted = True
            response = describe_image(
                task.image_path,
                build_map_spatial_prompt(task),
                task_name=f"地图与平面空间视觉抽取:{task.image_id}",
                response_format={"type": "json_object"},
                max_tokens=int(os.getenv("MAP_SPATIAL_VLM_MAX_TOKENS", "12288")),
            )
            payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
            if not isinstance(payload, Mapping) or not payload:
                raw = str(response or "")
                tail = raw[-160:].replace("\n", " ")
                raise ValueError(f"模型响应无法解析为非空 JSON 对象：字符数={len(raw)}，末尾={tail!r}")
            expected = {
                "map",
                "stratigraphic_units",
                "entities",
                "geological_attributes",
                "secondary_features",
                "semantic_relations",
                "spatial_relations",
                "georeference",
            }
            if not expected.intersection(payload):
                raise ValueError("模型 JSON 不包含任何地图空间抽取字段")
            normalized = normalize_map_spatial_result(task, payload)
            graph = build_map_spatial_graph(task, normalized)
            graph.metadata.extra.update(
                {
                    "vlm_called": True,
                    "vlm_call_count": 1,
                    "model_errors": [],
                }
            )
            return graph
        except Exception as exc:
            return self._failed_graph(task, f"VLM地图空间抽取失败：{exc}", model_called=call_attempted)

    @staticmethod
    def _resolve_describe_image(vlm_client: Any) -> Any:
        """取得视觉接口，并把依赖缺失转换为明确错误。"""

        if vlm_client is None:
            raise TypeError("MapSpatialExtractor 需要支持 describe_image 的 VLMClient")
        describe_image = getattr(vlm_client, "describe_image")
        if not callable(describe_image):
            raise TypeError("MapSpatialExtractor 需要支持 describe_image 的 VLMClient")
        return describe_image

    def _failed_graph(self, task: ImageExtractionTask, error: str, *, model_called: bool) -> Graph:
        """单图失败时返回可聚合、可诊断的空 Graph。"""

        graph = Graph.from_chunk(
            document_id=task.document_id,
            chunk_id=task.chunk_id,
            modality=SourceModality.IMAGE,
            stage="stage_04_image_map_spatial_extraction",
        )
        graph.metadata.extra.update(
            {
                "status": "model_error",
                "extractor_kind": self.kind.value,
                "extractor_name": self.display_name,
                "image_id": task.image_id,
                "image_index": task.image_index,
                "image_path": task.image_path,
                "source_image_path": task.image_path,
                "classification_code": task.classification_code,
                "classification_type": task.classification_type,
                "model_called": model_called,
                "vlm_called": model_called,
                "vlm_call_count": int(model_called),
                "model_errors": [error],
            }
        )
        return graph
