"""地图与平面空间图片的分阶段抽取器。

抽取器依次调用图例、实体和语义候选三个 VLM 阶段，再交给程序确定性建图。
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from model import Graph
from model.base import SourceModality

from ..base import BaseImageExtractor
from ..schema_models import ImageExtractionContext, ImageExtractionTask, ImageExtractorKind
from .graph import build_map_spatial_graph, normalize_map_spatial_result
from .relation_schema import RelationSchemaGenerator
from .stages import LegendGrounder, LegendParser, MapEntityExtractor, SemanticRelationExtractor


class MapSpatialExtractor(BaseImageExtractor):
    """按 README 的语义/几何职责边界抽取地图知识图。"""

    kind = ImageExtractorKind.MAP_SPATIAL
    display_name = "地图与平面空间抽取器"
    supported_codes = frozenset({"A01", "A02", "A07", "A18"})

    def __init__(self) -> None:
        """初始化六层流程中的四个可独立测试阶段。"""

        self.legend_parser = LegendParser()
        self.entity_extractor = MapEntityExtractor()
        self.legend_grounder = LegendGrounder()
        self.schema_generator = RelationSchemaGenerator()
        self.semantic_relation_extractor = SemanticRelationExtractor()

    def extract(self, task: ImageExtractionTask, context: ImageExtractionContext) -> Graph:
        """执行三次分阶段 VLM 调用，并用程序生成最终关系。"""

        call_count = 0
        stage_name = "初始化"
        stage_audit: list[dict[str, Any]] = []
        try:
            describe_image = self._resolve_describe_image(context.vlm_client)

            stage_name = "legend_parsing"
            call_count += 1
            legend_result = self.legend_parser.run(task, describe_image)
            stage_audit.append({"stage": stage_name, "status": "completed"})

            stage_name = "entity_extraction"
            call_count += 1
            entity_result = self.entity_extractor.run(task, describe_image, legend_result)
            stage_audit.append({"stage": stage_name, "status": "completed"})

            combined = self._merge_stage_results(legend_result, entity_result)
            grounded = self.legend_grounder.ground(normalize_map_spatial_result(task, combined))
            spatial_mapping = grounded.get("spatial_mapping") if isinstance(grounded.get("spatial_mapping"), Mapping) else {}
            spatial_ids = [
                str(item.get("entity_id") if isinstance(item, Mapping) else item)
                for item in spatial_mapping.get("candidate_entities") or []
            ]
            relation_schema = self.schema_generator.generate(
                [grounded["map"], *grounded.get("entities", [])],
                spatial_candidate_ids=spatial_ids,
            )
            stage_audit.extend([
                {"stage": "legend_grounding", "status": "completed"},
                {"stage": "relation_schema_generation", "status": "completed"},
            ])

            stage_name = "semantic_relation_extraction"
            call_count += 1
            relation_result = self.semantic_relation_extractor.run(
                task,
                describe_image,
                grounded,
                relation_schema,
            )
            stage_audit.append({"stage": stage_name, "status": "completed"})

            final_state = self._merge_stage_results(grounded, relation_result)
            graph = build_map_spatial_graph(task, final_state)
            stage_audit.append({"stage": "deterministic_relation_assembly", "status": "completed"})
            graph.metadata.extra.update({
                "model_called": True,
                "vlm_called": True,
                "vlm_call_count": call_count,
                "vlm_stages": stage_audit,
                "model_errors": [],
            })
            return graph
        except Exception as exc:
            stage_audit.append({"stage": stage_name, "status": "failed", "error": str(exc)})
            return self._failed_graph(
                task,
                f"地图空间分阶段抽取失败（{stage_name}）：{exc}",
                model_called=call_count > 0,
                call_count=call_count,
                stage_audit=stage_audit,
            )

    @staticmethod
    def _merge_stage_results(*results: Mapping[str, Any]) -> dict[str, Any]:
        """合并阶段对象，并对不确定性列表做稳定去重。"""

        merged: dict[str, Any] = {}
        uncertainties: list[str] = []
        for result in results:
            for key, value in result.items():
                if key == "uncertainties":
                    uncertainties.extend(str(item) for item in value or [] if str(item).strip())
                else:
                    merged[key] = deepcopy(value)
        merged["uncertainties"] = list(dict.fromkeys(uncertainties))
        return merged

    @staticmethod
    def _resolve_describe_image(vlm_client: Any) -> Any:
        """取得视觉接口，并把依赖缺失转换为明确错误。"""

        if vlm_client is None:
            raise TypeError("MapSpatialExtractor 需要支持 describe_image 的 VLMClient")
        describe_image = getattr(vlm_client, "describe_image", None)
        if not callable(describe_image):
            raise TypeError("MapSpatialExtractor 需要支持 describe_image 的 VLMClient")
        return describe_image

    def _failed_graph(
        self,
        task: ImageExtractionTask,
        error: str,
        *,
        model_called: bool,
        call_count: int,
        stage_audit: list[dict[str, Any]],
    ) -> Graph:
        """单图任一阶段失败时返回可聚合、可定位阶段的空 Graph。"""

        graph = Graph.from_chunk(
            document_id=task.document_id,
            chunk_id=task.chunk_id,
            modality=SourceModality.IMAGE,
            stage="stage_04_image_map_spatial_extraction",
        )
        graph.metadata.extra.update({
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
            "vlm_call_count": call_count,
            "vlm_stages": stage_audit,
            "model_errors": [error],
        })
        return graph


__all__ = ["MapSpatialExtractor"]
