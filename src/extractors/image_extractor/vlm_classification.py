"""使用视觉多模态模型完成石油地质图片 A01-A20 大类分类。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping

from src.utils.llm_client import safe_json_loads

from .schema_models import ImageExtractionTask


# 中文说明：该分类体系与 util/classify_image_chunks.py 保持一致，供真实视觉分类和路由共同使用。
IMAGE_TAXONOMY: tuple[tuple[str, str], ...] = (
    ("A01", "区域地质与构造位置图"),
    ("A02", "沉积相与古地理分布图"),
    ("A03", "地层柱状与综合柱状图"),
    ("A04", "地质剖面与连井对比图"),
    ("A05", "地震与地球物理剖面图"),
    ("A06", "测井曲线与测井综合图"),
    ("A07", "储层参数与厚度平面分布图"),
    ("A08", "油气藏、成藏与富集模式图"),
    ("A09", "沉积、成岩与孔隙演化模式图"),
    ("A10", "岩心、露头与手标本照片"),
    ("A11", "岩石薄片与显微照片"),
    ("A12", "扫描电镜与微观孔隙图"),
    ("A13", "CT、核磁与孔隙结构实验图"),
    ("A14", "地球化学谱图与实验曲线"),
    ("A15", "统计分布与组成图"),
    ("A16", "参数相关性与散点关系图"),
    ("A17", "时间序列与变化趋势图"),
    ("A18", "勘探成果与有利区预测图"),
    ("A19", "储集空间与岩性综合图版"),
    ("A20", "其他石油地质综合图"),
)
TYPE_BY_CODE = dict(IMAGE_TAXONOMY)


@dataclass(frozen=True)
class VLMImageClassification:
    """保存一次可校验的视觉大类分类结果。"""

    primary_code: str
    secondary_codes: tuple[str, ...]
    confidence: float
    reason: str
    visual_evidence: tuple[str, ...]
    source: str = "vlm_visual_classification"

    @property
    def primary_type(self) -> str:
        """中文说明：通过统一分类体系返回主类型中文名。"""

        return TYPE_BY_CODE[self.primary_code]

    def to_dict(self) -> dict[str, Any]:
        """中文说明：输出与旧分类阶段兼容的字段，并补充可追踪的视觉证据。"""

        return {
            "primary_code": self.primary_code,
            "primary_type": self.primary_type,
            "secondary_codes": list(self.secondary_codes),
            "secondary_types": [TYPE_BY_CODE[code] for code in self.secondary_codes],
            "confidence": self.confidence,
            "reason": self.reason,
            "visual_evidence": list(self.visual_evidence),
            "basis": self.source,
            "source": self.source,
        }


def build_image_classification_prompt(task: ImageExtractionTask) -> str:
    """中文说明：要求模型以图片主体结构为第一证据，在 A01-A20 中严格单选。"""

    taxonomy = "\n".join(f"- {code} {name}" for code, name in IMAGE_TAXONOMY)
    return f"""
你是石油地质论文图片分类专家。请读取随消息提供的当前图片，在 A01-A20 中选择唯一主类别。

图片 ID：{task.image_id}
图题：{task.caption or "无"}
正文参考（仅用于消歧，不得覆盖图片视觉主体）：
{json.dumps(list(task.references), ensure_ascii=False)}

分类体系：
{taxonomy}

判定规则：
1. 必须先看图片主体版式、坐标、图像内容和可见文字，再用图题与正文消歧。
2. 地层层级、厚度/深度轴、岩性柱和沉积相等组成的综合柱状表，主类选 A03；即使右侧含井号或一条趋势曲线，也不要误分成 A06、A07 或 A15。
3. A06 的主体应是多条测井曲线轨；A04/A05 的主体应是横向剖面；A07 的主体应是平面等值线或参数分布。
4. secondary_codes 只保留确有独立视觉面板支持的辅助类别，不要把每个嵌入小栏都列为副类。
5. visual_evidence 写 1 至 4 条图片中直接可见的证据；不得只复述图题，不要输出思维过程。
6. confidence 是 0 到 1 的数值。只输出 JSON 对象，不要 Markdown 或额外文字。

输出格式：
{{
  "primary_code": "A01-A20 中的一个值",
  "primary_type": "对应中文名称",
  "secondary_codes": [],
  "confidence": 0.0,
  "reason": "简短判定结论",
  "visual_evidence": ["直接可见证据1", "直接可见证据2"]
}}
""".strip()


class VLMImageClassifier:
    """调用项目现有 VLM 客户端执行真实图片大类分类。"""

    uses_vlm = True

    def classify(self, task: ImageExtractionTask, vlm_client: Any) -> VLMImageClassification:
        """中文说明：发起一次视觉 API 请求，并严格校验编码、置信度和图片证据。"""

        if vlm_client is None or not hasattr(vlm_client, "describe_image"):
            raise TypeError("VLM 图片分类器需要支持 describe_image 的 VLMClient")
        response = vlm_client.describe_image(
            task.image_path,
            build_image_classification_prompt(task),
            task_name=f"石油地质图片大类分类:{task.image_id}",
            response_format={"type": "json_object"},
            max_tokens=int(os.getenv("IMAGE_CLASSIFICATION_VLM_MAX_TOKENS", "1536")),
        )
        payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
        if not isinstance(payload, Mapping):
            raise ValueError("VLM 图片大类分类响应不是 JSON 对象")

        primary_code = str(payload.get("primary_code") or "").strip().upper()
        if primary_code not in TYPE_BY_CODE:
            raise ValueError(f"VLM 返回无效主类别：{primary_code!r}")
        try:
            confidence = float(payload.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("VLM 图片大类分类响应缺少有效 confidence") from exc
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            raise ValueError("VLM 图片大类分类响应缺少 reason")

        evidence_raw = payload.get("visual_evidence")
        if not isinstance(evidence_raw, list):
            raise ValueError("VLM 图片大类分类响应的 visual_evidence 必须是数组")
        evidence = tuple(str(item).strip() for item in evidence_raw if str(item).strip())
        if not evidence:
            raise ValueError("VLM 图片大类分类响应缺少可核验视觉证据")

        secondary_raw = payload.get("secondary_codes")
        secondary_codes: list[str] = []
        if isinstance(secondary_raw, list):
            for raw_code in secondary_raw:
                code = str(raw_code).strip().upper()
                if code in TYPE_BY_CODE and code != primary_code and code not in secondary_codes:
                    secondary_codes.append(code)
        return VLMImageClassification(
            primary_code=primary_code,
            secondary_codes=tuple(secondary_codes),
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            reason=reason,
            visual_evidence=evidence,
        )
