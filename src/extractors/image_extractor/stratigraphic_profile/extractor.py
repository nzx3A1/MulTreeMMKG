"""地层—剖面—测井大类的子分类与抽取调度器。"""
from __future__ import annotations

import os
from typing import Any, Mapping

from model import Graph
from model.base import SourceModality
from src.utils.llm_client import safe_json_loads

from ..base import BaseImageExtractor
from ..schema_models import ImageExtractionContext, ImageExtractionTask, ImageExtractorKind
from .subclassifier import (
    StratigraphicProfileSubtype,
    StratigraphicSubtypeClassification,
    VLMStratigraphicProfileSubtypeClassifier,
)
from .subtype_strategies import get_stratigraphic_subtype_strategy
from .table_embedded_hybrid import (
    TableEmbeddedHybridPipeline,
    apply_ppstructure_geometry,
    build_table_embedded_hybrid_graph,
    build_table_embedded_hybrid_prompt,
    enrich_table_node_names,
    extract_ppstructure_geometry,
    extract_segmented_table_visual,
    is_table_embedded_hybrid_payload,
)
from .three_dimensional_stratigraphic_model import (
    ThreeDimensionalStratigraphicModelPipeline,
    build_three_dimensional_stratigraphic_model_graph,
    build_three_dimensional_stratigraphic_model_prompt,
    extract_multipass_lithology,
    is_three_dimensional_stratigraphic_model_payload,
)
from .two_dimensional_stratigraphic_log import (
    TwoDimensionalStratigraphicLogPipeline,
    build_two_dimensional_stratigraphic_log_graph,
    build_two_dimensional_stratigraphic_log_prompt,
    is_two_dimensional_stratigraphic_log_payload,
)


class StratigraphicProfileExtractor(BaseImageExtractor):
    """只负责地层图片的子分类、分发和统一元数据记录。"""

    kind = ImageExtractorKind.STRATIGRAPHIC_PROFILE
    display_name = "地层—剖面—测井抽取调度器"
    supported_codes = frozenset({"A03", "A04", "A05", "A06"})

    def __init__(self, subtype_classifier: Any | None = None) -> None:
        """中文说明：初始化子分类器，测试时可注入离线分类结果。"""

        self.subtype_classifier = subtype_classifier or VLMStratigraphicProfileSubtypeClassifier()

    def extract(self, task: ImageExtractionTask, context: ImageExtractionContext) -> Graph:
        """中文说明：先完成子分类，再将任务交给已实现的子类型目录。"""

        configured_classifier = context.options.get("stratigraphic_subtype_classifier")
        classifier = configured_classifier or self.subtype_classifier
        subtype_vlm_called = bool(getattr(classifier, "uses_vlm", False))
        try:
            classification = classifier.classify(task, context.vlm_client)
        except Exception as exc:
            return self._result_graph(
                task,
                status="model_error",
                reason="stratigraphic_subclassification_failed",
                errors=[f"地层图片子分类失败：{exc}"],
                model_called=subtype_vlm_called,
            )

        allowed_subtypes = {
            str(value) for value in context.options.get("allowed_stratigraphic_subtypes", [])
        }
        if allowed_subtypes and classification.subtype.value not in allowed_subtypes:
            return self._result_graph(
                task,
                status="skipped_non_target_subtype",
                reason="stratigraphic_subtype_not_allowed",
                classification=classification,
                model_called=subtype_vlm_called,
            )

        # 中文说明：三个子类型均只分发到各自目录，外层不再包含具体图像解析规则。
        if classification.subtype not in {
            StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID,
            StratigraphicProfileSubtype.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL,
            StratigraphicProfileSubtype.TWO_DIMENSIONAL_STRATIGRAPHIC_LOG,
        }:
            return self._result_graph(
                task,
                status="not_implemented",
                reason="subtype_extractor_package_not_available",
                classification=classification,
                model_called=subtype_vlm_called,
            )

        try:
            if classification.subtype is StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID:
                graph, content_vlm_calls = self._dispatch_table_embedded_hybrid(
                    task,
                    context.vlm_client,
                    classification,
                )
            elif (
                classification.subtype
                is StratigraphicProfileSubtype.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL
            ):
                graph, content_vlm_calls = self._dispatch_three_dimensional_model(
                    task,
                    context.vlm_client,
                    classification,
                )
            else:
                graph, content_vlm_calls = self._dispatch_two_dimensional_log(
                    task,
                    context.vlm_client,
                    classification,
                )
        except Exception as exc:
            subtype_name = classification.subtype_name
            return self._result_graph(
                task,
                status="model_error",
                reason="subtype_extractor_failed",
                errors=[f"{subtype_name}抽取失败：{exc}"],
                classification=classification,
                model_called=True,
            )

        route = get_stratigraphic_subtype_strategy(classification.subtype)
        graph.metadata.extra.update(
            {
                "model_errors": [],
                "model_called": True,
                "vlm_called": True,
                "vlm_call_count": content_vlm_calls + int(subtype_vlm_called),
                "subtype_classification_vlm_called": subtype_vlm_called,
                "stratigraphic_subtype": classification.subtype.value,
                "stratigraphic_subtype_name": classification.subtype_name,
                "stratigraphic_subtype_confidence": classification.confidence,
                "stratigraphic_subtype_evidence": classification.evidence,
                "stratigraphic_subtype_source": classification.source,
                "subtype_extraction_algorithm": route.algorithm_name,
                "subtype_extraction_description": route.algorithm_description,
            }
        )
        return graph

    @staticmethod
    def _dispatch_table_embedded_hybrid(
        task: ImageExtractionTask,
        vlm_client: Any,
        classification: StratigraphicSubtypeClassification,
    ) -> tuple[Graph, int]:
        """中文说明：调用表格混合型子目录的公开能力，外层不定义坐标、轨道或图谱规则。"""

        if vlm_client is None or not hasattr(vlm_client, "describe_image"):
            raise TypeError("子类型抽取需要支持 describe_image 的 VLMClient")
        # 中文说明：PP-StructureV3 在任何 VLM 语义调用前只执行一次，三段抽取和审计复用同一几何目录。
        geometry = extract_ppstructure_geometry(
            task.image_path,
            getattr(vlm_client, "ppstructure_geometry_extractor", None),
        )
        if bool(getattr(vlm_client, "supports_segmented_table_extraction", False)):
            visual = extract_segmented_table_visual(
                task,
                vlm_client,
                classification,
                geometry=geometry,
            )
            content_vlm_calls = 4
        else:
            response = vlm_client.describe_image(
                task.image_path,
                build_table_embedded_hybrid_prompt(task, classification, geometry),
                task_name=f"表格嵌入混合抽取:{task.image_id}",
                response_format={"type": "json_object"},
                max_tokens=int(os.getenv("STRATIGRAPHIC_PROFILE_VLM_MAX_TOKENS", "16384")),
            )
            visual = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
            if isinstance(visual, Mapping):
                visual = apply_ppstructure_geometry(visual, geometry)
                visual = enrich_table_node_names(task, vlm_client, visual)
            content_vlm_calls = 2
        if not isinstance(visual, Mapping) or not is_table_embedded_hybrid_payload(visual):
            raise ValueError("子目录未返回 table_embedded_hybrid.v1 结构")
        intermediate = TableEmbeddedHybridPipeline().run(task, visual)
        return build_table_embedded_hybrid_graph(task, intermediate), content_vlm_calls

    @staticmethod
    def _dispatch_three_dimensional_model(
        task: ImageExtractionTask,
        vlm_client: Any,
        classification: StratigraphicSubtypeClassification,
    ) -> tuple[Graph, int]:
        """中文说明：调用三维子目录完成视觉、上下层序和上下文三元组的联合抽取。"""

        if vlm_client is None or not hasattr(vlm_client, "describe_image"):
            raise TypeError("三维地层模型抽取需要支持 describe_image 的 VLMClient")
        response = vlm_client.describe_image(
            task.image_path,
            build_three_dimensional_stratigraphic_model_prompt(task, classification),
            task_name=f"三维地层建模图抽取:{task.image_id}",
            response_format={"type": "json_object"},
            max_tokens=int(
                os.getenv(
                    "THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL_VLM_MAX_TOKENS",
                    os.getenv("STRATIGRAPHIC_PROFILE_VLM_MAX_TOKENS", "16384"),
                )
            ),
        )
        visual = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
        if not isinstance(visual, Mapping) or not is_three_dimensional_stratigraphic_model_payload(visual):
            raise ValueError("子目录未返回 three_dimensional_stratigraphic_model.v1 结构")
        content_vlm_calls = 1
        if bool(
            getattr(
                vlm_client,
                "supports_three_dimensional_lithology_multipass",
                False,
            )
        ):
            # 中文说明：真实客户端在整图结构识别后，按层界、图例和逐层复核多次调用以提高岩性准确度。
            visual, lithology_vlm_calls = extract_multipass_lithology(
                task,
                vlm_client,
                classification,
                visual,
            )
            content_vlm_calls += lithology_vlm_calls
        intermediate = ThreeDimensionalStratigraphicModelPipeline().run(task, visual)
        return build_three_dimensional_stratigraphic_model_graph(task, intermediate), content_vlm_calls

    @staticmethod
    def _dispatch_two_dimensional_log(
        task: ImageExtractionTask,
        vlm_client: Any,
        classification: StratigraphicSubtypeClassification,
    ) -> tuple[Graph, int]:
        """中文说明：调用二维子目录抽取图元、层序和相对位置，并确定性装配关系。"""

        if vlm_client is None or not hasattr(vlm_client, "describe_image"):
            raise TypeError("二维平面地层—测井图抽取需要支持 describe_image 的 VLMClient")
        response = vlm_client.describe_image(
            task.image_path,
            build_two_dimensional_stratigraphic_log_prompt(task, classification),
            task_name=f"二维平面地层测井视觉抽取:{task.image_id}",
            response_format={"type": "json_object"},
            max_tokens=int(
                os.getenv(
                    "TWO_DIMENSIONAL_STRATIGRAPHIC_LOG_VLM_MAX_TOKENS",
                    os.getenv("STRATIGRAPHIC_PROFILE_VLM_MAX_TOKENS", "16384"),
                )
            ),
        )
        visual = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
        if not isinstance(visual, Mapping) or not is_two_dimensional_stratigraphic_log_payload(
            visual
        ):
            raise ValueError("子目录未返回 two_dimensional_stratigraphic_log.v1 结构")
        intermediate = TwoDimensionalStratigraphicLogPipeline().run(task, visual)
        return build_two_dimensional_stratigraphic_log_graph(task, intermediate), 1

    def _result_graph(
        self,
        task: ImageExtractionTask,
        *,
        status: str,
        reason: str,
        classification: StratigraphicSubtypeClassification | None = None,
        errors: list[str] | None = None,
        model_called: bool,
    ) -> Graph:
        """中文说明：统一生成分类失败、路由跳过或子抽取器未实现的可追踪结果。"""

        graph = Graph.from_chunk(
            document_id=task.document_id,
            chunk_id=task.chunk_id,
            modality=SourceModality.IMAGE,
            stage="stage_04_image_stratigraphic_profile_dispatch",
        )
        route = (
            get_stratigraphic_subtype_strategy(classification.subtype)
            if classification is not None
            else None
        )
        graph.metadata.extra.update(
            {
                "status": status,
                "skip_reason": reason,
                "extractor_kind": self.kind.value,
                "extractor_name": self.display_name,
                "image_id": task.image_id,
                "image_index": task.image_index,
                "image_path": task.image_path,
                "source_image_path": task.image_path,
                "classification_code": task.classification_code,
                "classification_type": task.classification_type,
                "model_called": model_called,
                "model_errors": list(errors or []),
                "events_extracted": False,
                "stratigraphic_subtype": classification.subtype.value if classification else "",
                "stratigraphic_subtype_name": classification.subtype_name if classification else "",
                "stratigraphic_subtype_confidence": classification.confidence if classification else 0.0,
                "stratigraphic_subtype_evidence": classification.evidence if classification else "",
                "stratigraphic_subtype_source": classification.source if classification else "",
                "subtype_extraction_algorithm": route.algorithm_name if route else "",
            }
        )
        return graph
