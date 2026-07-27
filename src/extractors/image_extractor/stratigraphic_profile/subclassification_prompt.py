"""地层—剖面—测井图片的 VLM 子分类提示词。"""
from __future__ import annotations

import json

from ..schema_models import ImageExtractionTask


def build_stratigraphic_subclassification_prompt(task: ImageExtractionTask) -> str:
    """中文说明：要求 VLM 只依据当前图片主导视觉结构，在三个稳定子类型中完成单选。"""

    return f"""
你是石油地质图片版式分类专家。请查看随消息提供的这一张图片，只判断它在“地层—剖面—测井”大类内部的子类型。

来源图片：{task.image_path}
图题：{task.caption or "无"}
正文参考（仅用于理解缩写，不能替代图片视觉证据）：
{json.dumps(list(task.references), ensure_ascii=False)}

只能从以下三个 subtype 中选择一个：

1. table_embedded_hybrid（表格—图像嵌入混合型图）
   - 显式表格网格、合并单元格或多轨道测井版面占主导。
   - 地层列、深度列、岩性列、曲线轨、储层栏、成像测井或照片面板嵌在同一行列结构中。
   - 单井多轨测井、综合柱状表、地层简表、表格内嵌地层柱均归此类。
   - 两个及以上并排/分栏的成像测井面板，如果各自重复出现表头、深度列、比例尺或方位刻度，视为重复列单元组成的混合表格；即使边框不完整，也归此类。

2. three_dimensional_stratigraphic_model（三维地层建模图）
   - 图片真实表现三维透视块体、顶面与侧壁、地层体、曲面或空间拓扑。
   - 可通过遮挡、透视、多个可见面或三维体边界判断空间关系。
   - 仅图题出现“三维”不够；如果当前展示的是三维地震数据的一张平面剖切图，仍属于二维类型。

3. two_dimensional_stratigraphic_log（二维平面地层—测井图）
   - 二维地质剖面、二维地震切片、连井对比剖面或扁平地层柱占主导。
   - 重点是沿横纵坐标追踪层界、井轨迹、断层、反射轴或跨井对比线。
   - 多口井即使各自带重复表头和多条测井轨，只要井间空白区存在横向/斜向层位对比线，整体仍是连井二维剖面，必须归此类。
   - 少量图例、标注框或表头不改变二维主体判断。
   - 单个连续成像条带、单幅二维剖面或共享一组坐标的连续切片归此类；不要把“多个重复表头和深度列的独立面板”误归到这里。

判定顺序：
1. 先检查是否存在真实三维块体和多个可见面；存在则选 three_dimensional_stratigraphic_model。
2. 再检查是否存在跨多口井的连续层位对比线；存在则选 two_dimensional_stratigraphic_log，不能因每口井内部有测井表头而改选表格型。
3. 否则检查是否由规则网格、多轨道行列对齐，或“至少两个重复表头/深度列的独立成像图像面板”主导；是则选 table_embedded_hybrid。
4. 其余以二维剖面、切片、连井或层界追踪为主的图片选 two_dimensional_stratigraphic_log。

约束：
- 必须读取图片本身，不得只按文件名、图题关键词或正文分类。
- evidence 只写 1 至 3 条可直接看见的版式或几何特征，不要输出思维过程。
- confidence 必须是 0 到 1 之间的数值；图片裁切或证据不足时降低置信度。
- subtype 必须逐字使用上述英文稳定值。
- 只输出一个 JSON 对象，不要 Markdown，不要额外文字。

输出格式：
{{
  "subtype": "table_embedded_hybrid | three_dimensional_stratigraphic_model | two_dimensional_stratigraphic_log",
  "subtype_name": "对应中文名称",
  "confidence": 0.0,
  "evidence": ["可见证据1", "可见证据2"],
  "visual_features": {{
    "has_table_grid": false,
    "has_multi_track_layout": false,
    "embedded_panel_count": 0,
    "has_repeated_headers_or_depth_columns": false,
    "has_cross_well_correlation_lines": false,
    "has_3d_block_or_surfaces": false,
    "has_2d_axes_or_section": false
  }}
}}
""".strip()
