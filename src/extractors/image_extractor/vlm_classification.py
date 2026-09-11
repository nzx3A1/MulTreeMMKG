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
1. 先判断当前这张图片的主版式：照片/显微图、地图、剖面、示意模型或坐标图表。必须以当前图片可见主体为准；父 Chunk 图注可能同时描述多个子图，只能用于消歧，不能把别的子图内容带入当前图片。
2. 图表分流是硬规则：只要主体是带坐标轴、曲线、散点、柱形、饼形或统计符号的图表，不能因为出现“孔隙、储层、岩石”等词就判为 A12、A19 或 A13。
3. A13 只保留“图像型”的孔隙结构实验：CT 二维切片/三维体、核磁成像、三维孔喉或 node-link 节点连线空间网络；A13 不包含普通坐标图表。即使横轴写着孔隙半径，只要主体是曲线和频率轴，就判 A15（统计分布与组成图），不能判 A13。普通孔隙度—渗透率关系图判 A16，碳氧同位素/稀土/色谱等地球化学图判 A14，随时间或阶段变化的曲线判 A17。
4. A12 仅限扫描电镜等电子显微形貌照片：应看到微观纹理和标尺，不能有明显坐标轴、回归线或统计图例。A11 仅限光学薄片/显微照片。图表不是 A11/A12；三维图的 X/Y/Z 方向标不等于统计坐标轴。
5. 箱体、箭头、阶段 I/II/III/IV、流体运移或成岩反应组成的概念示意图判 A09；油气藏/源储盖/运移富集模式判 A08。没有井间横向对比的“模式图”绝不能判 A04；没有真实显微纹理的流程图不能判 A13。
6. A03 必须用于纵向地层柱状/综合柱状表：看到地层系统、层组/段、深度轴、岩性柱、沉积相和储层栏的组合时，优先判 A03，即使图中有一条相对海平面曲线。A04 必须有地质剖面或多井连井对比的横向层位关系；A05 必须有地震反射同相轴或地球物理剖面；A06 必须有多道随深度变化的测井曲线；A07 必须是平面等值线、色斑或参数空间分布。
7. 多面板同时含薄片/显微照片、岩心照片和孔隙标注时，若当前图片仍是完整图版，判 A19；不要仅因其中一个面板是薄片就判 A11。典型判例：孔隙度—渗透率散点/拟合线=A16；碳氧同位素交会图=A14；孔径—频率分布曲线=A15；白云石化/孔隙发育过程框图=A09；SEM 灰度形貌图=A12；CT 切片或 node-link 三维孔网图=A13。
8. secondary_codes 只保留当前图片中有独立视觉面板支持的辅助类别，不要把父图注中的其他子图类别列为副类；visual_evidence 写 1 至 4 条当前图片直接可见的证据，不要输出思维过程。
9. confidence 是 0 到 1 的数值；版式与语义冲突或图像模糊时降低置信度。只输出 JSON 对象，不要 Markdown 或额外文字。

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
