"""地层—剖面—测井大类内部的离线子分类器。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.llm_client import safe_json_loads

from ..schema_models import ImageExtractionTask, as_string_tuple
from .subclassification_prompt import build_stratigraphic_subclassification_prompt


DEFAULT_MOCK_MANIFEST_PATH = Path(__file__).resolve().parent / "testImage" / "stratigraphic_profile_subtype_mock.json"


class StratigraphicProfileSubtype(str, Enum):
    """地层—剖面—测井图片的三个稳定子类型。"""

    TABLE_EMBEDDED_HYBRID = "table_embedded_hybrid"
    THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL = "three_dimensional_stratigraphic_model"
    TWO_DIMENSIONAL_STRATIGRAPHIC_LOG = "two_dimensional_stratigraphic_log"

    @property
    def display_name(self) -> str:
        """中文说明：返回适合日志、测试结果和汇报材料使用的中文类型名。"""

        return {
            self.TABLE_EMBEDDED_HYBRID: "表格—图像嵌入混合型图",
            self.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL: "三维地层建模图",
            self.TWO_DIMENSIONAL_STRATIGRAPHIC_LOG: "二维平面地层—测井图",
        }[self]


@dataclass(frozen=True)
class StratigraphicSubtypeClassification:
    """保存单张图片的子分类、置信度、视觉依据和分类来源。"""

    subtype: StratigraphicProfileSubtype
    confidence: float
    evidence: str
    source: str

    @property
    def subtype_name(self) -> str:
        """中文说明：暴露中文类型名，避免调用方重复维护枚举到名称的映射。"""

        return self.subtype.display_name

    def to_dict(self) -> dict[str, Any]:
        """中文说明：将分类结果转换为可直接写入 JSON 或 Graph metadata 的字典。"""

        return {
            "subtype": self.subtype.value,
            "subtype_name": self.subtype_name,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "source": self.source,
        }


class StratigraphicProfileSubtypeClassifier:
    """人工视觉 mock 优先、图题规则兜底的无模型子分类器。"""

    uses_vlm = False

    def __init__(self, manifest_path: str | Path | None = DEFAULT_MOCK_MANIFEST_PATH) -> None:
        """中文说明：加载逐图人工 mock；未知图片仍可用保守图题规则继续分类。"""

        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self._records = self._load_records(self.manifest_path)

    @staticmethod
    def _load_records(path: Path | None) -> dict[str, dict[str, Any]]:
        """中文说明：读取图片 Chunk 数组，并按其中新增的 stratigraphic_subtype 字段建立文件名索引。"""

        if path is None:
            return {}
        if not path.is_file():
            raise FileNotFoundError(f"地层子分类 mock 文件不存在：{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records: dict[str, Mapping[str, Any]] = {}
            for chunk in payload:
                if not isinstance(chunk, Mapping):
                    raise TypeError("地层子分类 mock 的每一项都必须是图片 Chunk 对象")
                classification = chunk.get("stratigraphic_subtype")
                if not isinstance(classification, Mapping):
                    raise ValueError(f"图片 Chunk 缺少 stratigraphic_subtype：{chunk.get('id')}")
                for image_path in as_string_tuple(chunk.get("image_path")):
                    records[Path(image_path).name] = classification
        else:
            # 中文说明：暂时兼容上一版字典清单，方便旧结果在迁移期间仍可被读取。
            legacy_records = payload.get("classifications") if isinstance(payload, Mapping) else None
            records = dict(legacy_records) if isinstance(legacy_records, Mapping) else {}
        if not records:
            raise ValueError(f"地层子分类 mock 没有可用的图片分类记录：{path}")
        validated: dict[str, dict[str, Any]] = {}
        for filename, raw in records.items():
            if not isinstance(raw, Mapping):
                raise TypeError(f"图片 {filename} 的子分类记录不是 JSON 对象")
            StratigraphicProfileSubtype(str(raw.get("subtype") or ""))
            validated[str(filename)] = dict(raw)
        return validated

    def classify(
        self,
        task: ImageExtractionTask,
        vlm_client: Any | None = None,
    ) -> StratigraphicSubtypeClassification:
        """中文说明：优先返回人工逐图 mock，未命中时只根据图题做透明、低置信度规则判断。"""

        _ = vlm_client
        filename = Path(task.image_path).name
        record = self._records.get(filename)
        if record is not None:
            return self._from_record(record, source="manual_visual_mock")
        return self._classify_caption(task.caption)

    @staticmethod
    def _from_record(record: Mapping[str, Any], *, source: str) -> StratigraphicSubtypeClassification:
        """中文说明：规范化 mock 字段，并将置信度限制到零到一。"""

        try:
            confidence = float(record.get("confidence", 0.95))
        except (TypeError, ValueError):
            confidence = 0.95
        return StratigraphicSubtypeClassification(
            subtype=StratigraphicProfileSubtype(str(record.get("subtype") or "")),
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            evidence=str(record.get("evidence") or "人工逐图读取确认"),
            source=source,
        )

    @staticmethod
    def _classify_caption(caption: str) -> StratigraphicSubtypeClassification:
        """中文说明：对清单外图片执行可解释兜底；剖面优先于“3-D”字样，避免把二维地震切片误判为三维块体。"""

        normalized = str(caption or "").lower().replace(" ", "")
        if any(token in normalized for token in ("剖面", "连井", "地震切片", "profile", "section")):
            return StratigraphicSubtypeClassification(
                StratigraphicProfileSubtype.TWO_DIMENSIONAL_STRATIGRAPHIC_LOG,
                0.68,
                "图题含剖面、连井或二维切片语义",
                "caption_rule_fallback",
            )
        if any(token in normalized for token in ("三维建模", "三维模式", "3dmodel", "3-dmodel", "块体模型")):
            return StratigraphicSubtypeClassification(
                StratigraphicProfileSubtype.THREE_DIMENSIONAL_STRATIGRAPHIC_MODEL,
                0.66,
                "图题明确指向三维建模或块体模型",
                "caption_rule_fallback",
            )
        if any(token in normalized for token in ("简表", "柱状图", "综合柱状", "测井曲线", "测井图", "table", "column")):
            return StratigraphicSubtypeClassification(
                StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID,
                0.64,
                "图题指向表格、柱状图或多轨测井版面",
                "caption_rule_fallback",
            )
        return StratigraphicSubtypeClassification(
            StratigraphicProfileSubtype.TWO_DIMENSIONAL_STRATIGRAPHIC_LOG,
            0.4,
            "缺少可靠图题线索，按本大类最宽泛的二维类型保守兜底",
            "default_rule_fallback",
        )

    def classify_chunks(self, chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """中文说明：展开每个 Chunk 的图片路径，输出不调用 API 的逐图 mock 分类结果。"""

        results: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
            paths = as_string_tuple(chunk.get("image_path"))
            references = as_string_tuple(chunk.get("references"))
            for image_index, image_path in enumerate(paths):
                task = ImageExtractionTask(
                    document_id=str(chunk.get("document_id") or chunk_id.split(":section:", 1)[0]),
                    chunk_id=chunk_id,
                    image_id=f"{chunk_id}:image:{image_index}",
                    image_index=image_index,
                    image_path=image_path,
                    caption=str(chunk.get("caption") or ""),
                    references=references,
                    raw_chunk=chunk,
                )
                result = self.classify(task)
                results.append(
                    {
                        "chunk_id": chunk_id,
                        "image_id": task.image_id,
                        "image_index": image_index,
                        "image_path": image_path,
                        **result.to_dict(),
                    }
                )
        return results


class VLMStratigraphicProfileSubtypeClassifier:
    """调用项目 VLM 读取图片主导视觉结构并完成三选一子分类。"""

    uses_vlm = True

    def classify(
        self,
        task: ImageExtractionTask,
        vlm_client: Any | None = None,
    ) -> StratigraphicSubtypeClassification:
        """中文说明：发送单张图片和专用提示词，严格校验 VLM 返回的 subtype、置信度和视觉证据。"""

        if vlm_client is None or not hasattr(vlm_client, "describe_image"):
            raise TypeError("VLM 子分类器需要支持 describe_image 的 VLMClient")
        response = vlm_client.describe_image(
            task.image_path,
            build_stratigraphic_subclassification_prompt(task),
            task_name=f"地层图片子分类:{task.image_id}",
            response_format={"type": "json_object"},
            max_tokens=int(os.getenv("STRATIGRAPHIC_SUBCLASSIFICATION_VLM_MAX_TOKENS", "2048")),
        )
        payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
        if not isinstance(payload, Mapping):
            raise ValueError("VLM 子分类响应不是 JSON 对象")

        subtype = StratigraphicProfileSubtype(str(payload.get("subtype") or ""))
        try:
            confidence = float(payload.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("VLM 子分类响应缺少有效 confidence") from exc
        raw_evidence = payload.get("evidence")
        if isinstance(raw_evidence, list):
            evidence_items = [str(item).strip() for item in raw_evidence if str(item).strip()]
            evidence = "；".join(evidence_items)
        else:
            evidence = str(raw_evidence or "").strip()
        if not evidence:
            raise ValueError("VLM 子分类响应缺少可核验的视觉 evidence")

        # 中文说明：重复表头和深度列构成可确定复核的列式版面，修正模型把并排成像测井面板误当成单幅二维切片的问题。
        visual_features = payload.get("visual_features")
        if isinstance(visual_features, Mapping):
            try:
                panel_count = int(visual_features.get("embedded_panel_count") or 0)
            except (TypeError, ValueError):
                panel_count = 0
            has_repeated_columns = visual_features.get("has_repeated_headers_or_depth_columns") is True
            has_cross_well_lines = visual_features.get("has_cross_well_correlation_lines") is True
            if has_cross_well_lines:
                # 中文说明：跨井层位线决定了整幅图的二维连井语义，其优先级高于各井内部重复的测井表头。
                subtype = StratigraphicProfileSubtype.TWO_DIMENSIONAL_STRATIGRAPHIC_LOG
                evidence = f"{evidence}；版式复核：存在跨井连续层位对比线，按二维连井剖面处理"
                source = "vlm_visual_classification+deterministic_layout_refinement"
            elif (
                subtype is StratigraphicProfileSubtype.TWO_DIMENSIONAL_STRATIGRAPHIC_LOG
                and panel_count >= 2
                and has_repeated_columns
            ):
                subtype = StratigraphicProfileSubtype.TABLE_EMBEDDED_HYBRID
                evidence = f"{evidence}；版式复核：{panel_count} 个独立面板重复表头或深度列，按列式混合表格处理"
                source = "vlm_visual_classification+deterministic_layout_refinement"
            else:
                source = "vlm_visual_classification"
        else:
            source = "vlm_visual_classification"

        return StratigraphicSubtypeClassification(
            subtype=subtype,
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            evidence=evidence,
            source=source,
        )
