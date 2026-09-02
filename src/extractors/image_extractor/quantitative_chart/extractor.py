"""定量图表与实验曲线图片抽取器。"""
from __future__ import annotations

import os
from typing import Any, Mapping

from model import Graph
from model.base import SourceModality
from src.utils.llm_client import safe_json_loads

from ..base import BaseImageExtractor
from ..schema_models import ImageExtractionContext, ImageExtractionTask, ImageExtractorKind
from .graph import build_quantitative_chart_graph, normalize_quantitative_chart_result
from .prompt import build_quantitative_chart_prompt


class QuantitativeChartExtractor(BaseImageExtractor):
    """使用 VLM 抽取坐标轴、数据序列及其定量关系。"""

    kind = ImageExtractorKind.QUANTITATIVE_CHART
    display_name = "定量图表与实验曲线抽取器"
    supported_codes = frozenset({"A14", "A15", "A16", "A17"})

    def extract(self, task: ImageExtractionTask, context: ImageExtractionContext) -> Graph:
        """调用一次 VLM，规范化结构化结果并装配统一知识图。"""

        max_points = self._max_points(context)
        try:
            visual = self._extract_visual(task, context.vlm_client, max_points_per_series=max_points)
            normalized = normalize_quantitative_chart_result(
                task,
                visual,
                max_points_per_series=max_points,
            )
            graph = build_quantitative_chart_graph(task, normalized)
            graph.metadata.extra.update(
                {
                    "vlm_called": True,
                    "vlm_call_count": 1,
                    "model_errors": [],
                    "max_points_per_series": max_points,
                }
            )
            return graph
        except Exception as exc:
            return self._failed_graph(task, [f"VLM定量图表抽取失败：{exc}"])

    @staticmethod
    def _max_points(context: ImageExtractionContext) -> int:
        value = context.options.get(
            "quantitative_chart_max_points_per_series",
            os.getenv("QUANTITATIVE_CHART_MAX_POINTS_PER_SERIES", "24"),
        )
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 24

    @staticmethod
    def _extract_visual(
        task: ImageExtractionTask,
        vlm_client: Any,
        *,
        max_points_per_series: int,
    ) -> dict[str, Any]:
        """携带原图与上下文，要求视觉模型返回单个 JSON 对象。"""

        if vlm_client is None or not hasattr(vlm_client, "describe_image"):
            raise TypeError("QuantitativeChartExtractor 需要支持 describe_image 的 VLMClient")
        response = vlm_client.describe_image(
            task.image_path,
            build_quantitative_chart_prompt(task, max_points_per_series=max_points_per_series),
            task_name=f"定量图表视觉抽取:{task.image_id}",
            response_format={"type": "json_object"},
            max_tokens=int(os.getenv("QUANTITATIVE_CHART_VLM_MAX_TOKENS", "12288")),
        )
        payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
        if not isinstance(payload, Mapping) or not payload:
            raw = str(response or "")
            tail = raw[-160:].replace("\n", " ")
            raise ValueError(f"模型响应无法解析为非空 JSON 对象：字符数={len(raw)}，末尾={tail!r}")
        expected_fields = {
            "chart",
            "panels",
            "axes",
            "series",
            "trends",
            "correlations",
            "thresholds",
            "temporal_changes",
            "comparisons",
            "anomalies",
        }
        if not expected_fields.intersection(payload):
            raise ValueError("模型 JSON 不包含任何定量图表字段")
        return dict(payload)

    def _failed_graph(self, task: ImageExtractionTask, errors: list[str]) -> Graph:
        """单图失败时返回可追踪空 Graph，不中断同批其他图片。"""

        call_attempted = _vlm_call_attempted(errors)
        graph = Graph.from_chunk(
            document_id=task.document_id,
            chunk_id=task.chunk_id,
            modality=SourceModality.IMAGE,
            stage="stage_04_image_quantitative_chart_extraction",
        )
        graph.metadata.extra.update(
            {
                "status": "model_error",
                "extractor_kind": self.kind.value,
                "extractor_name": self.display_name,
                "image_id": task.image_id,
                "image_index": task.image_index,
                "image_path": task.image_path,
                "classification_code": task.classification_code,
                "classification_type": task.classification_type,
                "model_called": call_attempted,
                "vlm_called": call_attempted,
                "vlm_call_count": 1 if call_attempted else 0,
                "model_errors": errors,
            }
        )
        return graph


def _vlm_call_attempted(errors: list[str]) -> bool:
    """区分“缺少客户端”和“请求/解析失败”，便于统计真实模型调用。"""

    return not any("需要支持 describe_image" in error for error in errors)
