"""地质过程与成因模式图片抽取器。"""
from __future__ import annotations

import os
from typing import Any, Mapping

from model import Graph
from model.base import SourceModality
from src.utils.llm_client import safe_json_loads

from ..base import BaseImageExtractor
from ..schema_models import ImageExtractionContext, ImageExtractionTask, ImageExtractorKind
from .graph import build_geological_process_graph
from .prompt import (
    build_lithology_audit_prompt,
    build_relation_audit_prompt,
    build_visual_extraction_prompt,
)
from .visual_consistency import normalize_geological_visual_result


class GeologicalProcessExtractor(BaseImageExtractor):
    """抽取层段岩性、剖面空间拓扑、相对时间和地质过程因果链。"""

    kind = ImageExtractorKind.GEOLOGICAL_PROCESS
    display_name = "地质过程与成因模式抽取器"
    supported_codes = frozenset({"A08", "A09"})

    def extract(self, task: ImageExtractionTask, context: ImageExtractionContext) -> Graph:
        """中文说明：先调用 VLM 识别图像证据，再用 LLM 审查关系并装配 Graph。"""

        errors: list[str] = []
        visual: dict[str, Any] = {}
        lithology_audit: dict[str, Any] = {}
        audit: dict[str, Any] = {}
        try:
            visual = self._extract_visual_structure(task, context.vlm_client)
        except Exception as exc:
            errors.append(f"VLM视觉抽取失败：{exc}")

        if visual and context.vlm_client is not None and context.options.get("enable_lithology_audit", True):
            try:
                lithology_audit = self._audit_stratigraphic_lithology(task, visual, context.vlm_client)
                self._merge_lithology_audit(visual, lithology_audit)
            except Exception as exc:
                errors.append(f"VLM层段岩性复核失败：{exc}")

        if visual:
            # 中文说明：在关系审查前执行确定性校正，让 LLM 看到已消除井/断裂混淆和显式岩性冲突的结果。
            normalize_geological_visual_result(task, visual)

        if visual and context.llm_client is not None and context.options.get("enable_relation_audit", True):
            try:
                audit = self._audit_relations(task, visual, context.llm_client)
            except Exception as exc:
                errors.append(f"LLM关系审查失败：{exc}")

        if not visual:
            return self._failed_graph(task, errors or ["视觉模型未返回有效 JSON"])

        graph = build_geological_process_graph(task, visual, audit)
        graph.metadata.extra["model_errors"] = errors
        graph.metadata.extra["vlm_called"] = True
        graph.metadata.extra["vlm_call_count"] = 1 + int(bool(lithology_audit))
        graph.metadata.extra["lithology_audit"] = lithology_audit
        graph.metadata.extra["llm_audit_called"] = bool(audit)
        return graph

    @staticmethod
    def _extract_visual_structure(task: ImageExtractionTask, vlm_client: Any) -> dict[str, Any]:
        """中文说明：携带原图请求 VLM 返回证据锚定的地质过程 JSON。"""

        if vlm_client is None or not hasattr(vlm_client, "describe_image"):
            raise TypeError("GeologicalProcessExtractor 需要支持 describe_image 的 VLMClient")
        response = vlm_client.describe_image(
            task.image_path,
            build_visual_extraction_prompt(task),
            task_name=f"地质过程图视觉抽取:{task.image_id}",
            response_format={"type": "json_object"},
            # 中文说明：复杂成藏模式图的结构化结果可能超过通用 8192 token，允许为本抽取器单独调高上限。
            max_tokens=int(os.getenv("GEO_PROCESS_VLM_MAX_TOKENS", "12288")),
        )
        payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
        if not isinstance(payload, Mapping) or not payload:
            raw = str(response or "")
            tail = raw[-120:].replace("\n", " ")
            raise ValueError(f"模型响应无法解析为JSON：字符数={len(raw)}，末尾={tail!r}")
        return dict(payload) if isinstance(payload, Mapping) else {}

    @staticmethod
    def _audit_relations(task: ImageExtractionTask, visual: Mapping[str, Any], llm_client: Any) -> dict[str, Any]:
        """中文说明：用文本模型审查视觉结果，禁止脱离证据创造关系。"""

        prompt = build_relation_audit_prompt(task, dict(visual))
        if hasattr(llm_client, "call_openai_json"):
            response = llm_client.call_openai_json(prompt, task_name=f"地质过程图关系审查:{task.image_id}")
        elif hasattr(llm_client, "chat_json"):
            response = llm_client.chat_json([{"role": "user", "content": prompt}])
        else:
            raise TypeError("LLMClient 缺少 call_openai_json/chat_json 接口")
        return dict(response) if isinstance(response, Mapping) else {}

    @staticmethod
    def _audit_stratigraphic_lithology(
        task: ImageExtractionTask,
        visual: Mapping[str, Any],
        vlm_client: Any,
    ) -> dict[str, Any]:
        """中文说明：第二次查看原图，只复核图例填充与各层段岩性，降低整体误判。"""

        response = vlm_client.describe_image(
            task.image_path,
            build_lithology_audit_prompt(task, dict(visual)),
            task_name=f"地质过程图层段岩性复核:{task.image_id}",
            response_format={"type": "json_object"},
        )
        payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
        return dict(payload) if isinstance(payload, Mapping) else {}

    @staticmethod
    def _merge_lithology_audit(visual: dict[str, Any], audit: Mapping[str, Any]) -> None:
        """中文说明：仅在复核覆盖全部原层段时替换层段，并同步采用完整的复核图例。"""

        reviewed = audit.get("stratigraphic_units")
        original = visual.get("stratigraphic_units")
        if not isinstance(reviewed, list) or not isinstance(original, list):
            return
        original_ids = {str(item.get("id")) for item in original if isinstance(item, Mapping) and item.get("id")}
        reviewed_ids = {str(item.get("id")) for item in reviewed if isinstance(item, Mapping) and item.get("id")}
        if original_ids and reviewed_ids == original_ids:
            visual["stratigraphic_units"] = reviewed
            reviewed_legend = audit.get("legend_lithologies")
            if isinstance(reviewed_legend, list) and reviewed_legend:
                visual["legend_lithologies"] = reviewed_legend
        uncertainties = audit.get("uncertainties")
        if isinstance(uncertainties, list):
            visual.setdefault("uncertainties", []).extend(str(item) for item in uncertainties if str(item).strip())

    def _failed_graph(self, task: ImageExtractionTask, errors: list[str]) -> Graph:
        """中文说明：模型不可用时保留失败 Graph，避免整批图片任务被单张图中断。"""

        graph = Graph.from_chunk(
            document_id=task.document_id,
            chunk_id=task.chunk_id,
            modality=SourceModality.IMAGE,
            stage="stage_04_image_geological_process_extraction",
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
                "model_called": True,
                "model_errors": errors,
            }
        )
        return graph
