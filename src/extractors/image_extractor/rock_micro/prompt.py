"""岩石与微观储集空间图片的 VLM 文本描述提示词。"""
from __future__ import annotations

import json

from ..schema_models import ImageExtractionTask


def build_rock_micro_description_prompt(task: ImageExtractionTask) -> str:
    """构造只要求客观文本描述、不要求知识结构化的视觉提示词。"""

    return f"""
你是石油地质与储层岩石学图像描述助手。请直接观察随消息提供的图片，生成一段完整、客观、可独立阅读的中文文本描述。

图片信息：
- 图片 ID：{task.image_id}
- 图片类别：{task.classification_code or "未提供"} {task.classification_type or ""}
- 图题：{task.caption or "无"}
- 所属章节：{task.section_title or "无"}
- 正文参考：{json.dumps(list(task.references), ensure_ascii=False)}

描述要求：
1. 先说明图片类型和整体内容；若包含 (a)、(b) 等多个子图，按可见编号依次描述，并在最后概括各子图共同展示的特征。
2. 只依据图片中可见的颜色、形态、颗粒/晶体、孔隙、裂缝、充填物、连通特征、标注文字及比例尺描述；能读清的标注可原样写出。
3. 比例尺只能按图中可见标注表述，不得自行换算或编造尺寸、孔隙度、渗透率等定量值。
4. 图题、章节和正文参考只用于解释图片，不得覆盖视觉证据；若二者冲突，以图片可见内容为准并指出不确定性。
5. 对无法确认的矿物、岩性、孔隙类型或成因使用“可能”“疑似”或“图中标注为”，不要把推测写成事实。
6. 不执行目标检测、OCR 列表、实体关系抽取或 JSON 结构化；只返回连贯的中文描述正文，不要输出标题、Markdown、代码块或额外说明。
""".strip()


__all__ = ["build_rock_micro_description_prompt"]
