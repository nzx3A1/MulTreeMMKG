"""地层、剖面、地震与测井图片专用抽取器。"""
from __future__ import annotations

import os
from typing import Any, Mapping

from model import Graph
from model.base import SourceModality
from src.utils.llm_client import safe_json_loads

from ..base import BaseImageExtractor
from ..schema_models import ImageExtractionContext, ImageExtractionTask, ImageExtractorKind
from .graph import build_stratigraphic_profile_graph
from .prompt import (
    build_stratigraphic_relation_audit_prompt,
)
from .subclassifier import (
    StratigraphicProfileSubtype,
    StratigraphicSubtypeClassification,
    VLMStratigraphicProfileSubtypeClassifier,
)
from .subtype_strategies import (
    StratigraphicSubtypeExtractionStrategy,
    get_stratigraphic_subtype_strategy,
)
from .table_embedded_hybrid import (
    TableEmbeddedHybridPipeline,
    build_table_embedded_hybrid_graph,
    build_table_embedded_hybrid_prompt,
    extract_segmented_table_visual,
    is_table_embedded_hybrid_payload,
)


class StratigraphicProfileExtractor(BaseImageExtractor):
    """抽取垂向层序、跨井对比、断层切割、测井曲线和深度关系。"""

    kind = ImageExtractorKind.STRATIGRAPHIC_PROFILE
    display_name = "地层—剖面—测井抽取器"
    supported_codes = frozenset({"A03", "A04", "A05", "A06"})

    def __init__(self, subtype_classifier: Any | None = None) -> None:
        """中文说明：默认使用 VLM 子分类器，也允许测试注入人工 mock 分类器。"""

        self.subtype_classifier = subtype_classifier or VLMStratigraphicProfileSubtypeClassifier()

    def extract(self, task: ImageExtractionTask, context: ImageExtractionContext) -> Graph:
        """中文说明：先调用 VLM 判断图片子类型，再进入对应视觉抽取算法并装配 Graph。"""

        errors: list[str] = []
        visual: dict[str, Any] = {}
        audit: dict[str, Any] = {}
        classification: StratigraphicSubtypeClassification | None = None
        strategy: StratigraphicSubtypeExtractionStrategy | None = None
        configured_classifier = context.options.get("stratigraphic_subtype_classifier")
        classifier = configured_classifier or self.subtype_classifier
        subtype_vlm_called = bool(getattr(classifier, "uses_vlm", False))
        try:
            classification = classifier.classify(task, context.vlm_client)
            strategy = get_stratigraphic_subtype_strategy(classification.subtype)
        except Exception as exc:
            errors.append(f"地层图片子分类失败：{exc}")
            return self._failed_graph(
                task,
                errors,
                classification,
                strategy,
                model_called=subtype_vlm_called,
                subtype_vlm_called=subtype_vlm_called,
            )

        allowed_subtypes = {
            str(value)
            for value in context.options.get("allowed_stratigraphic_subtypes", [])
        }
        if allowed_subtypes and classification.subtype.value not in allowed_subtypes:
            return self._skipped_subtype_graph(
                task,
                classification,
                strategy,
                subtype_vlm_called=subtype_vlm_called,
            )

        try:
            visual = self._extract_visual(task, context.vlm_client, classification, strategy)
        except Exception as exc:
            errors.append(f"VLM视觉抽取失败：{exc}")

        uses_deterministic_table_pipeline = bool(
            classification.subtype is StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID
        )
        if uses_deterministic_table_pipeline and not is_table_embedded_hybrid_payload(visual):
            errors.append("表格嵌入混合图未返回 table_embedded_hybrid.v1 专用结构，禁止回退到原通用算法")
            return self._failed_graph(
                task,
                errors,
                classification,
                strategy,
                model_called=True,
                subtype_vlm_called=subtype_vlm_called,
            )
        if (
            visual
            and not uses_deterministic_table_pipeline
            and context.llm_client is not None
            and context.options.get("enable_relation_audit", True)
        ):
            try:
                audit = self._audit_relations(task, visual, context.llm_client)
            except Exception as exc:
                errors.append(f"LLM关系审查失败：{exc}")

        if not visual:
            return self._failed_graph(
                task,
                errors or ["视觉模型未返回有效 JSON"],
                classification,
                strategy,
                model_called=True,
                subtype_vlm_called=subtype_vlm_called,
            )

        try:
            if uses_deterministic_table_pipeline:
                intermediate = TableEmbeddedHybridPipeline().run(task, visual)
                graph = build_table_embedded_hybrid_graph(task, intermediate)
            else:
                # 中文说明：本轮只替换表格混合分支，另外两类在其专用目录完成前继续使用现有装配器。
                graph = build_stratigraphic_profile_graph(task, visual, audit)
        except Exception as exc:
            errors.append(f"结构化结果或图谱装配失败：{exc}")
            return self._failed_graph(
                task,
                errors,
                classification,
                strategy,
                model_called=True,
                subtype_vlm_called=subtype_vlm_called,
            )
        graph.metadata.extra.update(
            {
                "model_errors": errors,
                "model_called": True,
                "vlm_called": True,
                "vlm_call_count": (
                    3 + int(subtype_vlm_called)
                    if uses_deterministic_table_pipeline
                    and bool(getattr(context.vlm_client, "supports_segmented_table_extraction", False))
                    else 1 + int(subtype_vlm_called)
                ),
                "llm_audit_called": bool(audit),
                "llm_audit_skipped_reason": (
                    "deterministic_table_graph_assembly" if uses_deterministic_table_pipeline else ""
                ),
                "subtype_classification_vlm_called": subtype_vlm_called,
                "stratigraphic_subtype": classification.subtype.value,
                "stratigraphic_subtype_name": classification.subtype_name,
                "stratigraphic_subtype_confidence": classification.confidence,
                "stratigraphic_subtype_evidence": classification.evidence,
                "stratigraphic_subtype_source": classification.source,
                "subtype_extraction_algorithm": strategy.algorithm_name,
                "subtype_extraction_description": strategy.algorithm_description,
            }
        )
        return graph

    def _skipped_subtype_graph(
        self,
        task: ImageExtractionTask,
        classification: StratigraphicSubtypeClassification,
        strategy: StratigraphicSubtypeExtractionStrategy,
        *,
        subtype_vlm_called: bool,
    ) -> Graph:
        """中文说明：子分类不是表格嵌入混合型时只保留分类证据，不调用内容抽取模型。"""

        graph = Graph.from_chunk(
            document_id=task.document_id,
            chunk_id=task.chunk_id,
            modality=SourceModality.IMAGE,
            stage="stage_04_image_stratigraphic_subtype_skipped",
        )
        graph.metadata.extra.update(
            {
                "status": "skipped_non_target_subtype",
                "skip_reason": "stratigraphic_subtype_not_table_embedded_hybrid",
                "extractor_kind": self.kind.value,
                "extractor_name": self.display_name,
                "image_id": task.image_id,
                "image_index": task.image_index,
                "image_path": task.image_path,
                "source_image_path": task.image_path,
                "classification_code": task.classification_code,
                "classification_type": task.classification_type,
                "model_called": subtype_vlm_called,
                "vlm_called": subtype_vlm_called,
                "vlm_call_count": int(subtype_vlm_called),
                "events_extracted": False,
                "stratigraphic_subtype": classification.subtype.value,
                "stratigraphic_subtype_name": classification.subtype_name,
                "stratigraphic_subtype_confidence": classification.confidence,
                "stratigraphic_subtype_evidence": classification.evidence,
                "stratigraphic_subtype_source": classification.source,
                "subtype_extraction_algorithm": strategy.algorithm_name,
            }
        )
        return graph

    @staticmethod
    def _extract_visual(
        task: ImageExtractionTask,
        vlm_client: Any,
        classification: StratigraphicSubtypeClassification,
        strategy: StratigraphicSubtypeExtractionStrategy,
    ) -> dict[str, Any]:
        """中文说明：携带原图和已分流的专用算法 Prompt 请求 VLM 返回结构化 JSON。"""

        if vlm_client is None or not hasattr(vlm_client, "describe_image"):
            raise TypeError("StratigraphicProfileExtractor 需要支持 describe_image 的 VLMClient")
        if (
            classification.subtype is StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID
            and bool(getattr(vlm_client, "supports_segmented_table_extraction", False))
        ):
            # 中文说明：真实客户端按轨道组生成三个短 JSON，规避复杂表格单次长响应超时。
            payload = extract_segmented_table_visual(task, vlm_client, classification)
            return strategy.normalize_visual_result(payload)
        prompt = strategy.build_prompt(task, classification)
        if classification.subtype is StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID:
            # 中文说明：混合表格使用严格几何图元协议，关系由后续程序确定性生成。
            prompt = build_table_embedded_hybrid_prompt(task, classification)
        response = vlm_client.describe_image(
            task.image_path,
            prompt,
            task_name=f"地层剖面测井视觉抽取:{task.image_id}",
            response_format={"type": "json_object"},
            max_tokens=int(os.getenv("STRATIGRAPHIC_PROFILE_VLM_MAX_TOKENS", "16384")),
        )
        payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
        if not isinstance(payload, Mapping) or not payload:
            raise ValueError("模型响应无法解析为非空 JSON 对象")
        return strategy.normalize_visual_result(payload)

    @staticmethod
    def _audit_relations(task: ImageExtractionTask, visual: Mapping[str, Any], llm_client: Any) -> dict[str, Any]:
        """中文说明：调用文本模型检查关系方向和端点，不允许脱离视觉证据创造实体。"""

        prompt = build_stratigraphic_relation_audit_prompt(task, dict(visual))
        if hasattr(llm_client, "call_openai_json"):
            response = llm_client.call_openai_json(prompt, task_name=f"地层剖面测井关系审查:{task.image_id}")
        elif hasattr(llm_client, "chat_json"):
            response = llm_client.chat_json([{"role": "user", "content": prompt}])
        else:
            raise TypeError("LLMClient 缺少 call_openai_json/chat_json 接口")
        return dict(response) if isinstance(response, Mapping) else {}

    def _failed_graph(
        self,
        task: ImageExtractionTask,
        errors: list[str],
        classification: StratigraphicSubtypeClassification | None,
        strategy: StratigraphicSubtypeExtractionStrategy | None,
        *,
        model_called: bool,
        subtype_vlm_called: bool,
    ) -> Graph:
        """中文说明：单图模型失败时返回可追踪失败 Graph，避免中断整批图片。"""

        graph = Graph.from_chunk(
            document_id=task.document_id,
            chunk_id=task.chunk_id,
            modality=SourceModality.IMAGE,
            stage="stage_04_image_stratigraphic_profile_extraction",
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
                "model_errors": errors,
                "subtype_classification_vlm_called": subtype_vlm_called,
                "stratigraphic_subtype": classification.subtype.value if classification else "",
                "stratigraphic_subtype_name": classification.subtype_name if classification else "",
                "stratigraphic_subtype_confidence": classification.confidence if classification else 0.0,
                "stratigraphic_subtype_evidence": classification.evidence if classification else "",
                "stratigraphic_subtype_source": classification.source if classification else "",
                "subtype_extraction_algorithm": strategy.algorithm_name if strategy else "",
            }
        )
        return graph
