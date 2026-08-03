"""三维地层模型的多轮层界、图例和逐层岩性识别流程。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from src.utils.llm_client import safe_json_loads

from ...schema_models import ImageExtractionTask
from ..subclassifier import StratigraphicSubtypeClassification
from .review import apply_manual_review_corrections


MULTIPASS_SCHEMA_VERSION = "three_dimensional_lithology.segment.v1"
REVIEW_THRESHOLD = float(os.getenv("THREE_DIMENSIONAL_LITHOLOGY_REVIEW_THRESHOLD", "0.86"))
MAX_LAYER_SEGMENTS = int(os.getenv("THREE_DIMENSIONAL_MAX_LAYER_SEGMENTS", "24"))
UNIT_COLOR_BATCH_SIZE = max(
    1,
    int(os.getenv("THREE_DIMENSIONAL_UNIT_COLOR_BATCH_SIZE", "4")),
)
UNIT_COLOR_ACCEPT_THRESHOLD = float(
    os.getenv("THREE_DIMENSIONAL_UNIT_COLOR_ACCEPT_THRESHOLD", "0.82")
)
UNIT_COLOR_DISTANCE_THRESHOLD = int(
    os.getenv("THREE_DIMENSIONAL_UNIT_COLOR_DISTANCE_THRESHOLD", "48")
)


def _confidence(value: Any, default: float = 0.75) -> float:
    """中文说明：把各轮模型置信度限制到零到一，供复核触发和最终质量统计使用。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


def _items(value: Any) -> list[Mapping[str, Any]]:
    """中文说明：只保留模型数组中的对象，避免脏字符串进入确定性合并。"""

    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _image_size(image_path: str) -> tuple[int, int]:
    """中文说明：读取原图真实尺寸，所有分段和图例边界框都使用同一像素坐标。"""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"来源图片不存在：{path}")
    with Image.open(path) as image:
        return image.width, image.height


def _detect_coordinate_space(payload: Mapping[str, Any], width: int, height: int) -> str:
    """中文说明：识别模型是否违背指令返回零到一千归一化坐标，避免矮图纵坐标被截断。"""

    declared = str(payload.get("coordinate_space") or "").strip().lower()
    boxes: list[list[float]] = []

    def collect(value: Any, key: str = "") -> None:
        """中文说明：递归收集响应中的 bbox 数组，仅用于判断同一阶段采用的坐标体系。"""

        if isinstance(value, Mapping):
            for child_key, child in value.items():
                collect(child, str(child_key))
        elif isinstance(value, list):
            if key in {"bbox", "legend_bbox"} and len(value) == 4:
                try:
                    boxes.append([float(item) for item in value])
                except (TypeError, ValueError):
                    return
            else:
                for child in value:
                    collect(child)

    collect(payload)
    if boxes and all(0 <= number <= 1000 for box in boxes for number in box):
        if declared in {"normalized_0_1000", "normalized", "0_1000"}:
            return "normalized_0_1000"
        if any(box[0] > width or box[2] > width or box[1] > height or box[3] > height for box in boxes):
            return "normalized_0_1000"
        # 中文说明：部分模型误报 original_pixels，却在非 1000 像素宽图上反复以 x=1000 收边；该强特征优先于声明。
        right_edges_at_1000 = sum(abs(box[2] - 1000) < 0.5 for box in boxes)
        if width != 1000 and right_edges_at_1000 >= max(2, len(boxes) // 2):
            return "normalized_0_1000"
    return "original_pixels"


def _normalize_bbox(
    value: Any,
    width: int,
    height: int,
    *,
    coordinate_space: str = "original_pixels",
) -> list[int]:
    """中文说明：把像素或零到一千坐标统一成真实像素并裁剪，无效框返回空数组。"""

    if not isinstance(value, list) or len(value) != 4:
        return []
    try:
        raw_x0, raw_y0, raw_x1, raw_y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return []
    if coordinate_space == "normalized_0_1000":
        x0, x1 = round(raw_x0 * width / 1000), round(raw_x1 * width / 1000)
        y0, y1 = round(raw_y0 * height / 1000), round(raw_y1 * height / 1000)
    else:
        x0, y0, x1, y1 = map(round, (raw_x0, raw_y0, raw_x1, raw_y1))
    x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
    y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else []


def _parse_stage(response: Any, expected_stage: str) -> dict[str, Any]:
    """中文说明：解析单轮严格 JSON，并阻止错阶段响应混入其他字段。"""

    payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"多轮岩性阶段 {expected_stage} 没有返回非空 JSON")
    if str(payload.get("schema_version") or "") != MULTIPASS_SCHEMA_VERSION:
        raise ValueError(f"多轮岩性阶段 {expected_stage} 返回了错误 schema_version")
    if str(payload.get("stage") or "") != expected_stage:
        raise ValueError(
            f"多轮岩性响应错位：expected={expected_stage}, actual={payload.get('stage')}"
        )
    return dict(payload)


def _call_stage(
    vlm_client: Any,
    image_path: str,
    prompt: str,
    *,
    stage: str,
    task_name: str,
    call_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """中文说明：执行一次可追踪 VLM 调用，记录 Prompt 指纹和完整结构化响应。"""

    response = vlm_client.describe_image(
        image_path,
        prompt,
        task_name=task_name,
        response_format={"type": "json_object"},
        max_tokens=int(os.getenv("THREE_DIMENSIONAL_LITHOLOGY_VLM_MAX_TOKENS", "8192")),
    )
    payload = _parse_stage(response, stage)
    call_records.append(
        {
            "stage": stage,
            "task_name": task_name,
            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
            "response": payload,
        }
    )
    return payload


def build_layer_inventory_prompt(
    task: ImageExtractionTask,
    classification: StratigraphicSubtypeClassification,
    known_units: list[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> str:
    """中文说明：生成只冻结地层区域和内部可见薄层、不允许识别岩性的第一轮 Prompt。"""

    unit_catalog = [
        {
            "unit_id": str(unit.get("id") or ""),
            "name": str(unit.get("name") or ""),
            "parent_unit_id": str(unit.get("parent_unit_id") or ""),
            "order_bottom_to_top": unit.get("order_bottom_to_top"),
        }
        for unit in known_units
        if str(unit.get("id") or "")
    ]
    return f"""
你正在执行三维地层模型的第一轮“层界冻结”。本轮不得识别或猜测任何岩性。

图片 ID：{task.image_id}
图题：{task.caption}
子类型：{classification.subtype.value}
原图真实尺寸：width={width}, height={height}，坐标原点在左上角。

整图识别已经给出以下命名地层；unit_id、名称和顺序全部冻结，不得改名、合并、拆分或新建命名地层：
{json.dumps(unit_catalog, ensure_ascii=False, indent=2)}

任务：
1. 为每个命名地层记录其在 front、side、top 或 cutaway 面上的真实像素 bbox；同一地层跨两个面时使用同一个 unit_id。
2. 每个命名地层至少建立一个 visual_layer_segment。
3. 正面/侧面内部每个颜色或纹理边界清楚的垂向薄层建立 vertical_layer；顶面若有不同颜色或纹理的横向岩相分区，分别建立 lateral_facies_zone，二者不得混淆。
4. “马五1-4”等编号范围不等于图上必然有四层；只按实际可见边界拆分，但必须覆盖其全部可见纹理类型。
5. bbox 只能覆盖三维模型本体，严禁覆盖模型外部的图例、标题、指北针或说明文字；断裂红线也不是层段。
6. 不得依据图例给薄层命名，不得输出 lithology_id。边界不清时宁可保留一个较粗 segment，并写入 uncertainties。
7. 优先输出真实像素；如果视觉系统只能输出 0-1000 坐标，必须把 coordinate_space 明确写成 normalized_0_1000。
8. 只输出 JSON，不要 Markdown。

输出：
{{
  "schema_version":"{MULTIPASS_SCHEMA_VERSION}",
  "stage":"layer_inventory",
  "coordinate_space":"original_pixels|normalized_0_1000",
  "units":[{{
    "unit_id":"",
    "visible_regions":[{{"surface":"front|side|top|cutaway","bbox":[0,0,0,0],"boundary_evidence":"","confidence":0.0}}]
  }}],
  "visual_layer_segments":[{{
    "segment_id":"segment_1",
    "parent_unit_id":"冻结的 unit_id",
    "segment_role":"vertical_layer|lateral_facies_zone",
    "order_within_parent":0,
    "visible_regions":[{{"surface":"front|side|top|cutaway","bbox":[0,0,0,0],"boundary_evidence":"","confidence":0.0}}],
    "boundary_evidence":"",
    "confidence":0.0
  }}],
  "uncertainties":[]
}}
""".strip()


def build_legend_catalog_prompt(
    task: ImageExtractionTask,
    known_lithologies: list[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> str:
    """中文说明：生成图例和图内显式岩性文字目录，不允许在本轮把岩性关联到地层。"""

    hints = [str(item.get("name") or "") for item in known_lithologies if str(item.get("name") or "")]
    return f"""
你正在执行三维地层模型的第二轮“岩性候选目录识别”。本轮不得判断任何地层属于什么岩性。

图片 ID：{task.image_id}
图题：{task.caption}
原图真实尺寸：width={width}, height={height}。
整图初读曾出现以下候选名称，仅供定位，必须以当前图片图例的可见文字重新核实：
{json.dumps(hints, ensure_ascii=False)}

要求：
1. 逐项读取图例的原始文字、真实 bbox、颜色、砖纹/斜线/点状/横线/竖线等纹理特征。
2. 同时扫描模型本体内明确写出的岩性词，例如“泥质岩”；只有文字本身就是岩性名称时才可加入目录，并标记 catalog_source=explicit_layer_label。O2yj、马五1-4 等地层代号不是岩性词。
3. 文字清楚时以文字为准；文字不清楚时名称写“未知图例岩性N”，不得凭地质常识命名。
4. 每个 lithology_id 唯一稳定；不得关联 unit_id 或 segment_id。
5. legend_bbox 只覆盖完整图例区域；优先输出真实像素，若只能输出 0-1000 坐标则声明 coordinate_space。
6. 只输出 JSON。

输出：
{{
  "schema_version":"{MULTIPASS_SCHEMA_VERSION}",
  "stage":"legend_catalog",
  "coordinate_space":"original_pixels|normalized_0_1000",
  "legend_bbox":[0,0,0,0],
  "legend_items":[{{
    "lithology_id":"lith_1",
    "raw_text":"",
    "normalized_name":"",
    "catalog_source":"legend|explicit_layer_label",
    "bbox":[0,0,0,0],
    "color_features":[],
    "pattern_features":[],
    "text_explicit":true,
    "evidence":"",
    "confidence":0.0
  }}],
  "uncertainties":[]
}}
""".strip()


def build_layer_inventory_review_prompt(
    task: ImageExtractionTask,
    known_units: list[Mapping[str, Any]],
    inventory: Mapping[str, Any],
    legend: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> str:
    """中文说明：生成第二视角层界审查 Prompt，专门纠正漏层、归一化坐标和误框图例。"""

    unit_catalog = [
        {
            "unit_id": unit.get("id"),
            "name": unit.get("name"),
            "order_bottom_to_top": unit.get("order_bottom_to_top"),
        }
        for unit in known_units
    ]
    legend_brief = {
        "legend_bbox": legend.get("legend_bbox", []),
        "patterns": [
            {
                "name": item.get("normalized_name"),
                "color": item.get("color_features", []),
                "pattern": item.get("pattern_features", []),
            }
            for item in _items(legend.get("legend_items"))
        ],
    }
    return f"""
你正在独立复核三维地层模型的层界清单。本轮仍然不得给层段分配 lithology_id，但必须修正第一次清点的漏层和错误区域。

图片 ID：{task.image_id}
原图尺寸：width={width}, height={height}
冻结地层：{json.dumps(unit_catalog, ensure_ascii=False, indent=2)}
第一次层界结果：{json.dumps(dict(inventory), ensure_ascii=False, indent=2)}
已定位的图例及纹理提示：{json.dumps(legend_brief, ensure_ascii=False, indent=2)}

逐项检查：
1. 每个冻结 unit_id 至少有一个位于三维模型本体上的 segment，不得遗漏底部薄层。
2. 外部图例、标题、指北针、井名和说明文字不是层段；segment bbox 不得落在 legend_bbox 内。
3. 正面/侧面上垂向叠置的颜色或纹理带分别标记 segment_role=vertical_layer；同一命名地层内存在多条可见带时必须拆开。
4. 顶面不同颜色/纹理表示横向岩相分区时，分别标记 segment_role=lateral_facies_zone；它们共享 parent_unit_id，但不参与上下层序。
5. 红色断裂、裂缝、井线和箭头不能作为层段。不得因“1-4”编号凭空制造四个层，必须有实际视觉边界。
6. 输出完整替代清单，不是修订片段；优先使用真实像素，必要时声明 normalized_0_1000。
7. 只输出 JSON。

输出：
{{
  "schema_version":"{MULTIPASS_SCHEMA_VERSION}",
  "stage":"layer_inventory_review",
  "coordinate_space":"original_pixels|normalized_0_1000",
  "units":[{{"unit_id":"","visible_regions":[{{"surface":"front|side|top|cutaway","bbox":[0,0,0,0],"boundary_evidence":"","confidence":0.0}}]}}],
  "visual_layer_segments":[{{
    "segment_id":"segment_1",
    "parent_unit_id":"冻结的 unit_id",
    "segment_role":"vertical_layer|lateral_facies_zone",
    "order_within_parent":0,
    "visible_regions":[{{"surface":"front|side|top|cutaway","bbox":[0,0,0,0],"boundary_evidence":"","confidence":0.0}}],
    "boundary_evidence":"",
    "confidence":0.0
  }}],
  "uncertainties":[]
}}
""".strip()


def build_layer_lithology_prompt(
    task: ImageExtractionTask,
    segment: Mapping[str, Any],
    legend_items: list[Mapping[str, Any]],
    *,
    stage: str,
) -> str:
    """中文说明：生成一次只允许判断单个冻结层段的岩性 Prompt，候选严格受图例白名单限制。"""

    if stage not in {"layer_lithology", "independent_review"}:
        raise ValueError(f"不支持的逐层识别阶段：{stage}")
    review_instruction = (
        "这是独立盲审。你没有看到第一次识别结论，必须仅依据当前拼图重新判断。"
        if stage == "independent_review"
        else "这是第一次逐层识别。"
    )
    catalog = [
        {
            "lithology_id": item.get("lithology_id"),
            "name": item.get("normalized_name"),
            "color_features": item.get("color_features", []),
            "pattern_features": item.get("pattern_features", []),
            "text_explicit": item.get("text_explicit", False),
            "catalog_source": item.get("catalog_source", "legend"),
        }
        for item in legend_items
    ]
    return f"""
你正在执行三维地层模型的单层岩性识别。{review_instruction}

拼图说明：A 为完整原图；B/C 等后续面板是目标层段在不同可见面的放大图；最后一个面板是图例区域。
图片 ID：{task.image_id}
与该图直接关联的正文上下文（只可作为 reference_context 证据，不得当作图中显式证据）：
{json.dumps(list(task.references), ensure_ascii=False, indent=2)}
目标层段（ID、父地层和边界均已冻结，不得修改）：
{json.dumps(dict(segment), ensure_ascii=False, indent=2)}

允许选择的岩性仅限以下图例目录，不得创造图例外名称：
{json.dumps(catalog, ensure_ascii=False, indent=2)}

判断步骤：
1. 分别观察每个可见面的颜色、砖纹、斜线、点状、横线、竖线或混合纹理。
2. 与图例逐项比较，记录 matched_features 和 conflicting_features。
3. 同一层段不同面结论一致时提高置信度；遮挡、透视变形或裁剪混入邻层时必须降低置信度。
4. 必须逐一检查混合颜色和混合纹理，不得匹配到第一种岩性后就停止；有独立证据时可输出 primary、secondary、interbed 多种成分。
5. 同一岩性只输出一次，最多一种 primary，其余标为 secondary 或 interbed；可估算时各 visible_fraction 之和应接近 1。
6. 要区分目标层内的混合岩性与裁剪边缘混入的相邻层；只有在多个可见面、层内文字或稳定图例纹理中能独立复核的成分才能入 composition。
7. 地层代号（如 O2yj、O1-2y、马五1-4）不是岩性证据；explicit_layer_text 只用于层内文字直接写出“泥质岩、灰岩、白云岩”等岩性词。
8. 图中纹理不够清楚、但正文明确说出该冻结地层的岩性时，可用 evidence_source=reference_context，必须写出正文依据且 decision_confidence 不得高于 0.85。
9. 若候选存在任何 conflicting_features，needs_independent_review 必须为 true，decision_confidence 不得高于 0.75。
10. 没有可靠匹配时 composition 返回空数组，并在 unknown_patterns 说明；不得使用地质常识猜测。
11. 只输出 JSON。

输出：
{{
  "schema_version":"{MULTIPASS_SCHEMA_VERSION}",
  "stage":"{stage}",
  "segment_id":"{segment.get('segment_id')}",
  "surface_observations":[{{
    "surface":"front|side|top|cutaway",
    "observed_color":"",
    "observed_pattern":"",
    "occlusion":"",
    "candidates":[{{"lithology_id":"","matched_features":[],"conflicting_features":[],"confidence":0.0}}]
  }}],
  "composition":[{{
    "lithology_id":"图例白名单 ID",
    "role":"primary|secondary|interbed",
    "visible_fraction":null,
    "evidence_source":"explicit_layer_text|legend_pattern|cross_face_agreement|reference_context",
    "evidence":"",
    "confidence":0.0
  }}],
  "unknown_patterns":[],
  "decision_confidence":0.0,
  "needs_independent_review":false,
  "uncertainties":[]
}}
""".strip()


def build_unit_color_lithology_prompt(
    task: ImageExtractionTask,
    unit_batch: list[Mapping[str, Any]],
    legend_items: list[Mapping[str, Any]],
    *,
    batch_index: int,
    batch_count: int,
) -> str:
    """中文说明：生成按地层底色分批扫描的 Prompt，要求一次找全每个地层单元内的所有图例纹理。"""

    catalog = [
        {
            "lithology_id": item.get("lithology_id"),
            "name": item.get("normalized_name"),
            "color_features": item.get("color_features", []),
            "pattern_features": item.get("pattern_features", []),
        }
        for item in legend_items
    ]
    return f"""
你正在执行三维地层模型的“同色地层单元岩性全量扫描”，当前是第 {batch_index}/{batch_count} 批。

拼图说明：A 为完整原图；每个目标地层单元随后都有 ORIGINAL 原始区域和 COLOR MASK 同色像素高亮区域；最后是图例。COLOR MASK 只用于划定同一底色的空间范围，黑色线条、砖纹、斜线等岩性纹理仍需回看 ORIGINAL。
图片 ID：{task.image_id}
本批目标地层单元及程序提取的底色签名：
{json.dumps(unit_batch, ensure_ascii=False, indent=2)}

允许选择的岩性仅限以下图例目录：
{json.dumps(catalog, ensure_ascii=False, indent=2)}

判断步骤：
1. 对每个 unit_id 独立处理，先用 COLOR MASK 确认该底色在正面、侧面或顶面的全部连续区域，再回到 ORIGINAL 检查这些区域内的全部纹理。
2. 颜色只用于地层范围归属，不能单独证明岩性；岩性必须由图例纹理、层内显式岩性文字或跨面一致性确认。
3. 必须逐项检查图例，不得识别出第一种岩性后停止。同一底色范围内出现砖纹、斜线叠加、横线、竖线、点状等多种稳定纹理时，应全部写入 composition。
4. 断层红线、裂缝、井轨迹、文字、边框和相邻地层不得当作岩性纹理。仅在裁剪边缘偶然出现的纹理不得关联到当前 unit_id。
5. 每个 composition 项必须填写 matched_patterns；仅颜色相似而 matched_patterns 为空的候选不得输出。
6. 如果某个地层单元的颜色签名不可靠、同色范围跨越多个无法区分的地层，composition 返回空数组并写入 unresolved_reason，不得猜测。
7. 同一岩性在同一 unit_id 下只输出一次；最多一种 primary，其余为 secondary 或 interbed。
8. 只输出 JSON，不要 Markdown。

输出：
{{
  "schema_version":"{MULTIPASS_SCHEMA_VERSION}",
  "stage":"unit_color_lithology",
  "batch_index":{batch_index},
  "unit_results":[{{
    "unit_id":"本批白名单 unit_id",
    "observed_color_signature":["#RRGGBB"],
    "same_color_region_evidence":"",
    "checked_surfaces":["front|side|top|cutaway"],
    "composition":[{{
      "lithology_id":"图例白名单 ID",
      "role":"primary|secondary|interbed",
      "matched_patterns":[],
      "evidence":"",
      "confidence":0.0
    }}],
    "coverage_complete":false,
    "decision_confidence":0.0,
    "unresolved_reason":"",
    "uncertainties":[]
  }}],
  "uncertainties":[]
}}
""".strip()


def build_arbitration_prompt(
    segment: Mapping[str, Any],
    legend_items: list[Mapping[str, Any]],
    primary: Mapping[str, Any],
    review: Mapping[str, Any],
) -> str:
    """中文说明：生成只裁决两个既有候选、不得创造第三种岩性的分歧处理 Prompt。"""

    return f"""
你正在裁决同一冻结层段的两次独立岩性识别分歧。只能从两次答案已经使用的 lithology_id 中选择，或判定 unknown；不得创建新地层、新层段或新岩性。

目标层段：{json.dumps(dict(segment), ensure_ascii=False, indent=2)}
图例白名单：{json.dumps(list(legend_items), ensure_ascii=False, indent=2)}
第一次答案：{json.dumps(dict(primary), ensure_ascii=False, indent=2)}
独立盲审答案：{json.dumps(dict(review), ensure_ascii=False, indent=2)}

以图例纹理匹配、跨面一致性和冲突特征为依据裁决。证据不足时 composition 返回空数组。

输出：
{{
  "schema_version":"{MULTIPASS_SCHEMA_VERSION}",
  "stage":"arbitration",
  "segment_id":"{segment.get('segment_id')}",
  "composition":[{{"lithology_id":"","role":"primary|secondary|interbed","evidence":"","confidence":0.0}}],
  "decision_confidence":0.0,
  "reason":"",
  "uncertainties":[]
}}
""".strip()


def build_global_audit_prompt(
    inventory: Mapping[str, Any],
    legend: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    unit_color_results: list[Mapping[str, Any]],
) -> str:
    """中文说明：生成逐层与同色单元联合审查 Prompt，只输出可追溯修正而不重写完整结果。"""

    return f"""
你正在执行三维地层逐层岩性结果的最终一致性审查。只能引用已有 segment_id 和 lithology_id，不得创建新对象。

层段清单：{json.dumps(dict(inventory), ensure_ascii=False, indent=2)}
图例目录：{json.dumps(dict(legend), ensure_ascii=False, indent=2)}
逐层结果：{json.dumps(list(results), ensure_ascii=False, indent=2)}
同色地层单元扫描结果：{json.dumps(list(unit_color_results), ensure_ascii=False, indent=2)}

检查同一层跨面矛盾、相邻层裁剪串层、同色范围是否越过真实层界、逐层结果与地层单元全量扫描是否冲突、图例外名称和漏掉的低置信度层。多岩性本身不是错误；只有缺少独立图例纹理证据时才应撤回。只在图片证据明确时给 correction 或 unit_correction；否则加入 unresolved_segments 或 unresolved_units。

输出：
{{
  "schema_version":"{MULTIPASS_SCHEMA_VERSION}",
  "stage":"global_audit",
  "corrections":[{{
    "segment_id":"",
    "action":"add|remove|replace|change_role|request_review",
    "old_lithology_id":"",
    "new_lithology_id":"",
    "new_role":"primary|secondary|interbed",
    "reason":"",
    "evidence":"",
    "confidence":0.0
  }}],
  "unit_corrections":[{{
    "unit_id":"",
    "action":"add|remove|replace|change_role",
    "old_lithology_id":"",
    "new_lithology_id":"",
    "new_role":"primary|secondary|interbed",
    "matched_patterns":[],
    "reason":"",
    "evidence":"",
    "confidence":0.0
  }}],
  "conflicts":[],
  "unresolved_segments":[],
  "unresolved_units":[]
}}
""".strip()


def _normalize_inventory(
    raw: Mapping[str, Any],
    known_units: list[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """中文说明：校验单位白名单和层段坐标，并为模型漏掉的命名地层建立保守回退层段。"""

    known_by_id = {str(unit.get("id") or ""): unit for unit in known_units if str(unit.get("id") or "")}
    coordinate_space = _detect_coordinate_space(raw, width, height)
    unit_regions: dict[str, list[dict[str, Any]]] = {unit_id: [] for unit_id in known_by_id}
    uncertainties = [str(item) for item in raw.get("uncertainties", []) if str(item).strip()] if isinstance(raw.get("uncertainties"), list) else []
    for unit in _items(raw.get("units")):
        unit_id = str(unit.get("unit_id") or "")
        if unit_id not in known_by_id:
            uncertainties.append(f"层界清点返回未知 unit_id={unit_id}，已丢弃")
            continue
        for region in _items(unit.get("visible_regions")):
            bbox = _normalize_bbox(
                region.get("bbox"),
                width,
                height,
                coordinate_space=coordinate_space,
            )
            if bbox:
                unit_regions[unit_id].append(
                    {
                        "surface": str(region.get("surface") or "side"),
                        "bbox": bbox,
                        "boundary_evidence": str(region.get("boundary_evidence") or "可见层界"),
                        "confidence": _confidence(region.get("confidence")),
                    }
                )

    segments: list[dict[str, Any]] = []
    seen_segment_ids: set[str] = set()
    for index, segment in enumerate(_items(raw.get("visual_layer_segments"))):
        if len(segments) >= MAX_LAYER_SEGMENTS:
            uncertainties.append(f"可见层段超过上限 {MAX_LAYER_SEGMENTS}，后续层段已截断")
            break
        segment_id = str(segment.get("segment_id") or f"segment_{index + 1}")
        parent_id = str(segment.get("parent_unit_id") or "")
        if not segment_id or segment_id in seen_segment_ids or parent_id not in known_by_id:
            uncertainties.append(f"无效或重复层段 {segment_id}，parent={parent_id}，已丢弃")
            continue
        regions = []
        for region in _items(segment.get("visible_regions")):
            bbox = _normalize_bbox(
                region.get("bbox"),
                width,
                height,
                coordinate_space=coordinate_space,
            )
            if bbox:
                regions.append(
                    {
                        "surface": str(region.get("surface") or "side"),
                        "bbox": bbox,
                        "boundary_evidence": str(region.get("boundary_evidence") or segment.get("boundary_evidence") or "可见纹理边界"),
                        "confidence": _confidence(region.get("confidence"), _confidence(segment.get("confidence"))),
                    }
                )
        if not regions:
            regions = list(unit_regions.get(parent_id, []))
        if not regions:
            uncertainties.append(f"层段 {segment_id} 没有有效 bbox，已丢弃")
            continue
        seen_segment_ids.add(segment_id)
        parent_name = str(known_by_id[parent_id].get("name") or parent_id)
        segments.append(
            {
                "segment_id": segment_id,
                "parent_unit_id": parent_id,
                "parent_unit_name": parent_name,
                "segment_role": str(segment.get("segment_role") or "vertical_layer"),
                "order_within_parent": int(segment.get("order_within_parent") or 0),
                "visible_regions": regions,
                "boundary_evidence": str(segment.get("boundary_evidence") or "；".join(region["boundary_evidence"] for region in regions)),
                "confidence": _confidence(segment.get("confidence")),
            }
        )

    parents_with_segments = {segment["parent_unit_id"] for segment in segments}
    for unit_id, unit in known_by_id.items():
        if unit_id in parents_with_segments:
            continue
        regions = unit_regions.get(unit_id, [])
        if not regions:
            uncertainties.append(f"命名地层 {unit.get('name')} 没有可用 bbox，无法逐层识别岩性")
            continue
        segment_id = f"{unit_id}_segment_1"
        segments.append(
            {
                "segment_id": segment_id,
                "parent_unit_id": unit_id,
                "parent_unit_name": str(unit.get("name") or unit_id),
                "segment_role": "vertical_layer",
                "order_within_parent": 0,
                "visible_regions": regions,
                "boundary_evidence": "命名地层可见区域的保守单层段回退",
                "confidence": min(_confidence(region.get("confidence")) for region in regions),
            }
        )

    segments.sort(
        key=lambda segment: (
            int(known_by_id[segment["parent_unit_id"]].get("order_bottom_to_top") or 0),
            int(segment.get("order_within_parent") or 0),
        )
    )
    return {
        "schema_version": MULTIPASS_SCHEMA_VERSION,
        "stage": "layer_inventory",
        "source_stage": str(raw.get("stage") or "layer_inventory"),
        "coordinate_space": "original_pixels",
        "input_coordinate_space": coordinate_space,
        "units": [
            {"unit_id": unit_id, "visible_regions": regions}
            for unit_id, regions in unit_regions.items()
        ],
        "visual_layer_segments": segments,
        "uncertainties": uncertainties,
    }


def _choose_inventory(
    initial: Mapping[str, Any],
    reviewed: Mapping[str, Any],
    known_units: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """中文说明：优先选择覆盖命名地层更多且层段不更少的独立审查结果，防止复核反而丢层。"""

    expected = {str(unit.get("id") or "") for unit in known_units if str(unit.get("id") or "")}

    def score(payload: Mapping[str, Any]) -> tuple[int, int]:
        """中文说明：以父地层覆盖数为首要指标、有效层段数为次要指标评估层界清单。"""

        segments = _items(payload.get("visual_layer_segments"))
        covered = {str(item.get("parent_unit_id") or "") for item in segments} & expected
        return len(covered), len(segments)

    initial_score = score(initial)
    reviewed_score = score(reviewed)
    selected = dict(reviewed if reviewed_score >= initial_score else initial)
    uncertainties = [str(item) for item in selected.get("uncertainties", [])]
    if reviewed_score < initial_score:
        uncertainties.append(
            f"独立层界审查覆盖下降 initial={initial_score}, reviewed={reviewed_score}，保留第一次清点"
        )
    selected["inventory_selection"] = {
        "initial_score": list(initial_score),
        "reviewed_score": list(reviewed_score),
        "selected": "reviewed" if reviewed_score >= initial_score else "initial",
    }
    selected["uncertainties"] = list(dict.fromkeys(uncertainties))
    return selected


def _normalize_legend(
    raw: Mapping[str, Any],
    fallback_lithologies: list[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """中文说明：校验图例 ID、名称和像素区域，必要时保留整图初读的可追踪回退项。"""

    coordinate_space = _detect_coordinate_space(raw, width, height)
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    uncertainties = [str(item) for item in raw.get("uncertainties", []) if str(item).strip()] if isinstance(raw.get("uncertainties"), list) else []
    for index, item in enumerate(_items(raw.get("legend_items"))):
        lithology_id = str(item.get("lithology_id") or f"lith_{index + 1}")
        name = str(item.get("normalized_name") or item.get("raw_text") or "").strip()
        if not lithology_id or lithology_id in seen_ids or not name:
            continue
        seen_ids.add(lithology_id)
        items.append(
            {
                "lithology_id": lithology_id,
                "raw_text": str(item.get("raw_text") or name),
                "normalized_name": name,
                "bbox": _normalize_bbox(
                    item.get("bbox"),
                    width,
                    height,
                    coordinate_space=coordinate_space,
                ),
                "color_features": [str(value) for value in item.get("color_features", [])] if isinstance(item.get("color_features"), list) else [],
                "pattern_features": [str(value) for value in item.get("pattern_features", [])] if isinstance(item.get("pattern_features"), list) else [],
                "text_explicit": bool(item.get("text_explicit", True)),
                "catalog_source": str(item.get("catalog_source") or "legend"),
                "evidence": str(item.get("evidence") or f"图例文字 {name}"),
                "confidence": _confidence(item.get("confidence")),
            }
        )
    if not items:
        uncertainties.append("专用图例调用没有返回可用项，回退到整图初读图例")
        for index, item in enumerate(fallback_lithologies):
            lithology_id = str(item.get("id") or f"lith_{index + 1}")
            name = str(item.get("name") or lithology_id)
            items.append(
                {
                    "lithology_id": lithology_id,
                    "raw_text": name,
                    "normalized_name": name,
                    "bbox": [],
                    "color_features": [],
                    "pattern_features": [str(item.get("visual_pattern") or "")],
                    "text_explicit": True,
                    "catalog_source": "base_visual_fallback",
                    "evidence": str(item.get("evidence") or f"整图初读图例 {name}"),
                    "confidence": _confidence(item.get("confidence")),
                }
            )
    return {
        "schema_version": MULTIPASS_SCHEMA_VERSION,
        "stage": "legend_catalog",
        "coordinate_space": "original_pixels",
        "input_coordinate_space": coordinate_space,
        "legend_bbox": _normalize_bbox(
            raw.get("legend_bbox"),
            width,
            height,
            coordinate_space=coordinate_space,
        ),
        "legend_items": items,
        "uncertainties": uncertainties,
    }


def _expanded_crop(bbox: list[int], width: int, height: int, ratio: float = 0.12) -> tuple[int, int, int, int]:
    """中文说明：给目标层段增加少量邻域，既保留层界又避免整图细节过小。"""

    x0, y0, x1, y1 = bbox
    dx = max(8, int((x1 - x0) * ratio))
    dy = max(8, int((y1 - y0) * ratio))
    return max(0, x0 - dx), max(0, y0 - dy), min(width, x1 + dx), min(height, y1 + dy)


def _bbox_containment_ratio(inner: list[int], outer: list[int]) -> float:
    """中文说明：计算一个层段框被另一表面框覆盖的面积比例，用于发现顶面和侧壁误用同一区域。"""

    if len(inner) != 4 or len(outer) != 4:
        return 0.0
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    intersection = max(0, min(ix1, ox1) - max(ix0, ox0)) * max(
        0,
        min(iy1, oy1) - max(iy0, oy0),
    )
    inner_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    return intersection / inner_area if inner_area else 0.0


def _resize_panel(image: Image.Image, max_width: int = 1200, max_height: int = 800) -> Image.Image:
    """中文说明：等比例缩放拼图面板，避免超大画布同时保留层段纹理细节。"""

    panel = image.convert("RGB")
    scale = min(max_width / panel.width, max_height / panel.height, 1.0)
    if scale >= 1.0:
        return panel
    return panel.resize((max(1, int(panel.width * scale)), max(1, int(panel.height * scale))), Image.Resampling.LANCZOS)


def _build_focus_sheet(
    image_path: str,
    segment: Mapping[str, Any],
    legend_bbox: list[int],
    output_path: Path,
) -> None:
    """中文说明：生成全图、每个可见面层段放大图和图例的竖向拼图，供单层调用聚焦。"""

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    panels: list[tuple[str, Image.Image]] = [("A FULL IMAGE", _resize_panel(image))]
    for index, region in enumerate(_items(segment.get("visible_regions")), start=1):
        bbox = _normalize_bbox(region.get("bbox"), width, height)
        if not bbox:
            continue
        crop = image.crop(_expanded_crop(bbox, width, height))
        panels.append((f"{chr(65 + index)} TARGET {region.get('surface', '')}", _resize_panel(crop)))
    if legend_bbox:
        legend_crop = image.crop(_expanded_crop(legend_bbox, width, height, ratio=0.05))
        panels.append((f"{chr(65 + len(panels))} LEGEND", _resize_panel(legend_crop)))
    canvas_width = max(panel.width for _, panel in panels)
    label_height = 34
    canvas_height = sum(panel.height + label_height for _, panel in panels)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, panel in panels:
        draw.text((8, y + 8), label, fill="black")
        y += label_height
        canvas.paste(panel, (0, y))
        y += panel.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def _dominant_region_color_signature(
    image: Image.Image,
    visible_regions: list[Mapping[str, Any]],
    *,
    max_colors: int = 3,
) -> dict[str, Any]:
    """中文说明：从地层区域中排除深色线稿和红色断层后提取主底色，供同色范围高亮使用。"""

    width, height = image.size
    samples: list[np.ndarray] = []
    for region in visible_regions:
        bbox = _normalize_bbox(region.get("bbox"), width, height)
        if not bbox:
            continue
        crop = np.asarray(image.crop(tuple(bbox)).convert("RGB"), dtype=np.uint8).reshape(-1, 3)
        if crop.shape[0] > 120_000:
            step = max(1, crop.shape[0] // 120_000)
            crop = crop[::step]
        brightness = crop.mean(axis=1)
        red_fault = (
            (crop[:, 0] > 145)
            & (crop[:, 0].astype(np.int16) - crop[:, 1].astype(np.int16) > 55)
            & (crop[:, 0].astype(np.int16) - crop[:, 2].astype(np.int16) > 45)
        )
        pink_fault_halo = (
            (crop[:, 0] > 195)
            & (crop[:, 0].astype(np.int16) - crop[:, 1].astype(np.int16) > 15)
            & (crop[:, 2].astype(np.int16) - crop[:, 1].astype(np.int16) > 5)
        )
        valid = (brightness >= 72) & ~red_fault & ~pink_fault_halo
        if np.any(valid):
            samples.append(crop[valid])
    if not samples:
        return {"rgb": [], "hex": [], "sampled_pixel_count": 0, "covered_fraction": 0.0}

    pixels = np.concatenate(samples, axis=0)
    quantized = np.clip(((pixels.astype(np.uint16) + 8) // 16) * 16, 0, 255).astype(np.uint8)
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    chroma = colors.max(axis=1).astype(np.int16) - colors.min(axis=1).astype(np.int16)
    colorful_order = [
        int(index)
        for index in np.argsort(counts)[::-1]
        if chroma[index] >= 24 and counts[index] / max(1, int(counts.sum())) >= 0.025
    ]
    neutral_order = [
        int(index)
        for index in np.argsort(counts)[::-1]
        if chroma[index] < 24
    ]
    selected: list[tuple[np.ndarray, int]] = []
    # 中文说明：优先保留能区分地层范围的彩色填充；没有彩色填充时才使用白色或灰色地层底色。
    candidate_order = [*colorful_order, *neutral_order]
    for color_index in candidate_order:
        color = colors[color_index].astype(np.int16)
        if any(np.linalg.norm(color - prior.astype(np.int16)) < 28 for prior, _ in selected):
            continue
        selected.append((colors[color_index], int(counts[color_index])))
        if len(selected) >= max_colors:
            break
    rgb = [[int(channel) for channel in color] for color, _ in selected]
    covered = sum(count for _, count in selected) / max(1, int(counts.sum()))
    return {
        "rgb": rgb,
        "hex": ["#" + "".join(f"{channel:02X}" for channel in color) for color in rgb],
        "sampled_pixel_count": int(pixels.shape[0]),
        "covered_fraction": round(covered, 4),
    }


def _unit_color_inventory(
    image_path: str,
    known_units: list[Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """中文说明：为每个冻结地层单元组合名称、可见面和程序提取的底色签名，形成分批识别白名单。"""

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    unit_names = {
        str(unit.get("id") or ""): str(unit.get("name") or unit.get("id") or "")
        for unit in known_units
        if str(unit.get("id") or "")
    }
    records: list[dict[str, Any]] = []
    for unit in _items(inventory.get("units")):
        unit_id = str(unit.get("unit_id") or "")
        visible_regions = [dict(region) for region in _items(unit.get("visible_regions"))]
        if not unit_id or not visible_regions:
            continue
        records.append(
            {
                "unit_id": unit_id,
                "unit_name": unit_names.get(unit_id, unit_id),
                "visible_regions": visible_regions,
                "color_signature": _dominant_region_color_signature(image, visible_regions),
            }
        )
    return records


def _dilate_color_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """中文说明：把同色像素掩膜向外扩展少量像素，使覆盖在底色上的黑色岩性线纹仍然可见。"""

    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    expanded = np.zeros_like(mask, dtype=bool)
    for offset_y in range(radius * 2 + 1):
        for offset_x in range(radius * 2 + 1):
            expanded |= padded[
                offset_y : offset_y + mask.shape[0],
                offset_x : offset_x + mask.shape[1],
            ]
    return expanded


def _build_color_mask_panel(
    image: Image.Image,
    bbox: list[int],
    color_signature: Mapping[str, Any],
) -> Image.Image:
    """中文说明：生成保留同色范围及其线纹、淡化其他像素的地层颜色辅助面板。"""

    width, height = image.size
    crop_box = _expanded_crop(bbox, width, height, ratio=0.06)
    crop = np.asarray(image.crop(crop_box).convert("RGB"), dtype=np.uint8)
    colors = np.asarray(color_signature.get("rgb") or [], dtype=np.int16)
    if colors.size == 0:
        return Image.fromarray(crop, mode="RGB")
    color_delta = crop.astype(np.int32)[:, :, None, :] - colors.astype(np.int32)[
        None, None, :, :
    ]
    distances = np.sqrt((color_delta**2).sum(axis=3))
    mask = _dilate_color_mask(np.min(distances, axis=2) <= UNIT_COLOR_DISTANCE_THRESHOLD)
    muted = np.clip(crop.astype(np.float32) * 0.32 + 255 * 0.68, 0, 255).astype(np.uint8)
    highlighted = crop.copy()
    highlighted[mask] = np.clip(
        highlighted[mask].astype(np.float32) * 0.84
        + np.asarray([0, 210, 150], dtype=np.float32) * 0.16,
        0,
        255,
    ).astype(np.uint8)
    muted[mask] = highlighted[mask]
    return Image.fromarray(muted, mode="RGB")


def _build_unit_color_batch_sheet(
    image_path: str,
    unit_batch: list[Mapping[str, Any]],
    legend_bbox: list[int],
    output_path: Path,
) -> None:
    """中文说明：为一批地层单元生成整图、原始区域、同色掩膜和图例拼图，兼顾范围与岩性纹理。"""

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    panels: list[tuple[str, Image.Image]] = [("A FULL IMAGE", _resize_panel(image, 1100, 700))]
    panel_index = 1
    for unit in unit_batch:
        unit_id = str(unit.get("unit_id") or "")
        signature = unit.get("color_signature") if isinstance(unit.get("color_signature"), Mapping) else {}
        for region_index, region in enumerate(_items(unit.get("visible_regions")), start=1):
            bbox = _normalize_bbox(region.get("bbox"), width, height)
            if not bbox:
                continue
            original = image.crop(_expanded_crop(bbox, width, height, ratio=0.06))
            panels.append(
                (
                    f"{chr(65 + panel_index)} UNIT {unit_id} REGION {region_index} ORIGINAL",
                    _resize_panel(original, 1000, 620),
                )
            )
            panel_index += 1
            panels.append(
                (
                    f"{chr(65 + panel_index)} UNIT {unit_id} REGION {region_index} COLOR MASK",
                    _resize_panel(_build_color_mask_panel(image, bbox, signature), 1000, 620),
                )
            )
            panel_index += 1
    if legend_bbox:
        legend_crop = image.crop(_expanded_crop(legend_bbox, width, height, ratio=0.05))
        panels.append((f"{chr(65 + panel_index)} LEGEND", _resize_panel(legend_crop, 1100, 620)))
    canvas_width = max(panel.width for _, panel in panels)
    label_height = 34
    canvas_height = sum(panel.height + label_height for _, panel in panels)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, panel in panels:
        draw.text((8, y + 8), label, fill="black")
        y += label_height
        canvas.paste(panel, (0, y))
        y += panel.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def _composition_ids(result: Mapping[str, Any]) -> tuple[str, ...]:
    """中文说明：提取去重岩性候选签名，用于判断两次独立识别是否一致。"""

    return tuple(
        sorted(
            {
                str(item.get("lithology_id") or "")
                for item in _items(result.get("composition"))
                if str(item.get("lithology_id") or "")
            }
        )
    )


def _normalize_layer_result(
    raw: Mapping[str, Any],
    segment_id: str,
    allowed_lithology_ids: set[str],
) -> dict[str, Any]:
    """中文说明：只保留当前层段和图例白名单内的成分，并汇总模型不确定性。"""

    composition_by_id: dict[str, dict[str, Any]] = {}
    composition_order: list[str] = []
    dropped: list[str] = []
    for item in _items(raw.get("composition")):
        lithology_id = str(item.get("lithology_id") or "")
        if lithology_id not in allowed_lithology_ids:
            if lithology_id:
                dropped.append(lithology_id)
            continue
        evidence_source = str(item.get("evidence_source") or "legend_pattern")
        item_confidence = _confidence(item.get("confidence"))
        if evidence_source == "reference_context":
            item_confidence = min(0.85, item_confidence)
        role = str(item.get("role") or "primary")
        if role not in {"primary", "secondary", "interbed"}:
            role = "secondary"
        candidate = {
            "lithology_id": lithology_id,
            "role": role,
            "visible_fraction": item.get("visible_fraction"),
            "evidence_source": evidence_source,
            "evidence": str(item.get("evidence") or "目标层段纹理与图例匹配"),
            "confidence": item_confidence,
        }
        existing = composition_by_id.get(lithology_id)
        if existing is None:
            composition_order.append(lithology_id)
            composition_by_id[lithology_id] = candidate
        elif item_confidence > _confidence(existing.get("confidence"), 0.0):
            composition_by_id[lithology_id] = candidate
    composition = [composition_by_id[item] for item in composition_order]
    primary_seen = False
    for item in composition:
        if item["role"] != "primary":
            continue
        if primary_seen:
            item["role"] = "secondary"
        else:
            primary_seen = True
    uncertainties = [str(item) for item in raw.get("uncertainties", []) if str(item).strip()] if isinstance(raw.get("uncertainties"), list) else []
    if dropped:
        uncertainties.append(f"丢弃图例白名单外岩性 ID：{sorted(set(dropped))}")
    surface_observations = (
        list(raw.get("surface_observations") or [])
        if isinstance(raw.get("surface_observations"), list)
        else []
    )
    has_conflicts = any(
        bool(candidate.get("conflicting_features"))
        for observation in _items(surface_observations)
        for candidate in _items(observation.get("candidates"))
    )
    decision_confidence = _confidence(
        raw.get("decision_confidence"),
        max((item["confidence"] for item in composition), default=0.0),
    )
    if any(item.get("evidence_source") == "reference_context" for item in composition):
        decision_confidence = min(0.85, decision_confidence)
    if has_conflicts:
        decision_confidence = min(0.75, decision_confidence)
        uncertainties.append("候选存在 conflicting_features，已强制触发独立盲审")
    return {
        "segment_id": segment_id,
        "composition": composition,
        "surface_observations": surface_observations,
        "unknown_patterns": [str(item) for item in raw.get("unknown_patterns", [])] if isinstance(raw.get("unknown_patterns"), list) else [],
        "decision_confidence": decision_confidence,
        "needs_independent_review": bool(raw.get("needs_independent_review", False)) or has_conflicts,
        "uncertainties": uncertainties,
    }


def _normalize_unit_color_batch(
    raw: Mapping[str, Any],
    unit_batch: list[Mapping[str, Any]],
    allowed_lithology_ids: set[str],
    explicit_layer_lithologies: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """中文说明：校验同色批次的地层和岩性白名单，并只接纳具有纹理证据且达到阈值的候选。"""

    batch_by_id = {
        str(unit.get("unit_id") or ""): unit
        for unit in unit_batch
        if str(unit.get("unit_id") or "")
    }
    explicit_layer_lithologies = dict(explicit_layer_lithologies or {})
    raw_by_id = {
        str(result.get("unit_id") or ""): result
        for result in _items(raw.get("unit_results"))
        if str(result.get("unit_id") or "") in batch_by_id
    }
    batch_uncertainties = [
        str(item)
        for item in raw.get("uncertainties", [])
        if str(item).strip()
    ] if isinstance(raw.get("uncertainties"), list) else []
    normalized: list[dict[str, Any]] = []
    for unit_id, unit in batch_by_id.items():
        result = raw_by_id.get(unit_id, {})
        decision_confidence = _confidence(result.get("decision_confidence"), 0.0)
        unresolved_reason = str(result.get("unresolved_reason") or "").strip()
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in _items(result.get("composition")):
            lithology_id = str(item.get("lithology_id") or "")
            matched_patterns = [
                str(pattern)
                for pattern in item.get("matched_patterns", [])
                if str(pattern).strip()
            ] if isinstance(item.get("matched_patterns"), list) else []
            item_confidence = min(
                _confidence(item.get("confidence"), 0.0),
                decision_confidence,
            )
            gate_reason = ""
            if lithology_id not in allowed_lithology_ids:
                gate_reason = "lithology_not_in_legend_whitelist"
            elif not matched_patterns:
                gate_reason = "missing_pattern_evidence"
            elif (
                lithology_id in explicit_layer_lithologies
                and explicit_layer_lithologies[lithology_id]
                not in str(unit.get("unit_name") or "")
            ):
                gate_reason = "explicit_lithology_not_in_unit_label"
            elif unresolved_reason:
                gate_reason = "unit_color_scope_unresolved"
            elif item_confidence < UNIT_COLOR_ACCEPT_THRESHOLD:
                gate_reason = "below_unit_color_accept_threshold"
            elif lithology_id in seen_ids:
                gate_reason = "duplicate_unit_lithology"
            candidate = {
                "lithology_id": lithology_id,
                "role": str(item.get("role") or "secondary"),
                "matched_patterns": matched_patterns,
                "evidence_source": "same_color_unit_pattern_scan",
                "evidence": str(item.get("evidence") or "同色地层范围内的纹理与图例匹配"),
                "confidence": item_confidence,
            }
            if gate_reason:
                rejected.append({**candidate, "gate_reason": gate_reason})
                continue
            seen_ids.add(lithology_id)
            if candidate["role"] not in {"primary", "secondary", "interbed"}:
                candidate["role"] = "secondary"
            accepted.append(candidate)
        if not result:
            unresolved_reason = "本批 VLM 未返回该地层单元"
        uncertainties = [
            str(item)
            for item in result.get("uncertainties", [])
            if str(item).strip()
        ] if isinstance(result.get("uncertainties"), list) else []
        if unresolved_reason:
            uncertainties.append(unresolved_reason)
        if rejected:
            uncertainties.append(
                "同色扫描候选被质量门拒绝："
                + ",".join(
                    f"{item.get('lithology_id')}:{item.get('gate_reason')}"
                    for item in rejected
                )
            )
        normalized.append(
            {
                "unit_id": unit_id,
                "unit_name": str(unit.get("unit_name") or unit_id),
                "color_signature": dict(unit.get("color_signature") or {}),
                "visible_regions": [dict(region) for region in _items(unit.get("visible_regions"))],
                "same_color_region_evidence": str(result.get("same_color_region_evidence") or ""),
                "checked_surfaces": [str(item) for item in result.get("checked_surfaces", [])]
                if isinstance(result.get("checked_surfaces"), list)
                else [],
                "composition": accepted,
                "rejected_candidates": rejected,
                "coverage_complete": bool(result.get("coverage_complete", False)),
                "decision_confidence": decision_confidence,
                "unresolved_reason": unresolved_reason,
                "uncertainties": list(dict.fromkeys(uncertainties)),
            }
        )
    return normalized, batch_uncertainties


def _merge_agreeing_reviews(primary: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    """中文说明：两次候选一致时合并证据并提升为独立复核一致结果。"""

    merged = dict(primary)
    review_by_id = {
        str(item.get("lithology_id") or ""): item for item in _items(review.get("composition"))
    }
    composition: list[dict[str, Any]] = []
    for item in _items(primary.get("composition")):
        record = dict(item)
        peer = review_by_id.get(str(item.get("lithology_id") or ""), {})
        record["confidence"] = round(
            min(0.99, (_confidence(item.get("confidence")) + _confidence(peer.get("confidence"))) / 2 + 0.05),
            3,
        )
        record["evidence_source"] = "independent_review_agreement"
        record["evidence"] = f"第一次：{item.get('evidence', '')}；独立复核：{peer.get('evidence', '')}"
        composition.append(record)
    merged["composition"] = composition
    merged["decision_confidence"] = round(
        min(0.99, (_confidence(primary.get("decision_confidence")) + _confidence(review.get("decision_confidence"))) / 2 + 0.05),
        3,
    )
    merged["needs_independent_review"] = False
    merged["review_status"] = "independent_agreement"
    merged["independent_review"] = dict(review)
    return merged


def _apply_global_corrections(
    results: list[dict[str, Any]],
    audit: Mapping[str, Any],
    allowed_lithology_ids: set[str],
) -> list[dict[str, Any]]:
    """中文说明：按白名单应用最终审查的增删替换建议，无法验证的修正只留在审计记录中。"""

    by_segment = {str(result.get("segment_id") or ""): result for result in results}
    for correction in _items(audit.get("corrections")):
        segment_id = str(correction.get("segment_id") or "")
        result = by_segment.get(segment_id)
        if result is None:
            continue
        action = str(correction.get("action") or "")
        old_id = str(correction.get("old_lithology_id") or "")
        new_id = str(correction.get("new_lithology_id") or "")
        composition = [dict(item) for item in _items(result.get("composition"))]
        if action == "remove" and old_id:
            composition = [item for item in composition if item.get("lithology_id") != old_id]
        elif action == "replace" and old_id and new_id in allowed_lithology_ids:
            for item in composition:
                if item.get("lithology_id") == old_id:
                    item["lithology_id"] = new_id
                    item["role"] = str(correction.get("new_role") or item.get("role") or "primary")
                    item["evidence_source"] = "global_audit_correction"
                    item["evidence"] = str(correction.get("evidence") or correction.get("reason") or "全图一致性修正")
                    item["confidence"] = _confidence(correction.get("confidence"))
        elif action == "add" and new_id in allowed_lithology_ids and all(item.get("lithology_id") != new_id for item in composition):
            composition.append(
                {
                    "lithology_id": new_id,
                    "role": str(correction.get("new_role") or "secondary"),
                    "visible_fraction": None,
                    "evidence_source": "global_audit_correction",
                    "evidence": str(correction.get("evidence") or correction.get("reason") or "全图一致性补充"),
                    "confidence": _confidence(correction.get("confidence")),
                }
            )
        elif action == "change_role" and old_id:
            for item in composition:
                if item.get("lithology_id") == old_id:
                    item["role"] = str(correction.get("new_role") or item.get("role") or "primary")
        result["composition"] = composition
    return results


def _apply_unit_color_corrections(
    results: list[dict[str, Any]],
    audit: Mapping[str, Any],
    allowed_lithology_ids: set[str],
) -> list[dict[str, Any]]:
    """中文说明：对同色地层单元应用全图审查修正，新增候选仍必须满足图例纹理和置信度质量门。"""

    by_unit = {str(result.get("unit_id") or ""): result for result in results}
    for correction in _items(audit.get("unit_corrections")):
        unit_id = str(correction.get("unit_id") or "")
        result = by_unit.get(unit_id)
        if result is None:
            continue
        action = str(correction.get("action") or "")
        old_id = str(correction.get("old_lithology_id") or "")
        new_id = str(correction.get("new_lithology_id") or "")
        composition = [dict(item) for item in _items(result.get("composition"))]
        if action == "remove" and old_id:
            composition = [item for item in composition if item.get("lithology_id") != old_id]
        elif action == "replace" and old_id and new_id in allowed_lithology_ids:
            matched_patterns = [
                str(item)
                for item in correction.get("matched_patterns", [])
                if str(item).strip()
            ] if isinstance(correction.get("matched_patterns"), list) else []
            confidence = _confidence(correction.get("confidence"), 0.0)
            if matched_patterns and confidence >= UNIT_COLOR_ACCEPT_THRESHOLD:
                for item in composition:
                    if item.get("lithology_id") == old_id:
                        item.update(
                            {
                                "lithology_id": new_id,
                                "role": str(correction.get("new_role") or item.get("role") or "secondary"),
                                "matched_patterns": matched_patterns,
                                "evidence_source": "global_audit_color_correction",
                                "evidence": str(correction.get("evidence") or correction.get("reason") or "全图同色审查修正"),
                                "confidence": confidence,
                            }
                        )
        elif action == "add" and new_id in allowed_lithology_ids and all(
            item.get("lithology_id") != new_id for item in composition
        ):
            matched_patterns = [
                str(item)
                for item in correction.get("matched_patterns", [])
                if str(item).strip()
            ] if isinstance(correction.get("matched_patterns"), list) else []
            confidence = _confidence(correction.get("confidence"), 0.0)
            if matched_patterns and confidence >= UNIT_COLOR_ACCEPT_THRESHOLD:
                composition.append(
                    {
                        "lithology_id": new_id,
                        "role": str(correction.get("new_role") or "secondary"),
                        "matched_patterns": matched_patterns,
                        "evidence_source": "global_audit_color_correction",
                        "evidence": str(correction.get("evidence") or correction.get("reason") or "全图同色审查补充"),
                        "confidence": confidence,
                    }
                )
        elif action == "change_role" and old_id:
            for item in composition:
                if item.get("lithology_id") == old_id:
                    item["role"] = str(correction.get("new_role") or item.get("role") or "secondary")
        result["composition"] = composition

    unresolved_ids = {
        str(item.get("unit_id") or "") if isinstance(item, Mapping) else str(item)
        for item in audit.get("unresolved_units", [])
    } if isinstance(audit.get("unresolved_units"), list) else set()
    for unit_id in unresolved_ids:
        result = by_unit.get(unit_id)
        if result is None:
            continue
        rejected = [
            {**dict(item), "gate_reason": "global_audit_unit_unresolved"}
            for item in _items(result.get("composition"))
        ]
        result["rejected_candidates"] = [
            *[dict(item) for item in _items(result.get("rejected_candidates"))],
            *rejected,
        ]
        result["composition"] = []
        result["unresolved_reason"] = "全图联合审查无法确认同色地层范围或岩性归属"
        result["uncertainties"] = list(
            dict.fromkeys(
                [
                    *[str(item) for item in result.get("uncertainties", [])],
                    result["unresolved_reason"],
                ]
            )
        )
    return results


def apply_multipass_quality_gates(
    payload: Mapping[str, Any],
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[str, Any]:
    """中文说明：恢复真实像素层框，并阻止未解决或显式文字越界的岩性候选进入正式三元组。"""

    corrected = deepcopy(dict(payload))
    multipass = corrected.get("multipass_lithology")
    segments = corrected.get("visual_layer_segments")
    units = corrected.get("stratigraphic_units")
    if not isinstance(multipass, dict) or not isinstance(segments, list):
        return corrected
    unit_names = {
        str(unit.get("id") or ""): str(unit.get("name") or "")
        for unit in _items(units)
    }
    if image_width and image_height:
        call_records = _items(multipass.get("call_records"))
        stored_inventory = multipass.get("layer_inventory")
        selection_name = (
            str(stored_inventory.get("inventory_selection", {}).get("selected") or "reviewed")
            if isinstance(stored_inventory, Mapping)
            and isinstance(stored_inventory.get("inventory_selection"), Mapping)
            else "reviewed"
        )
        wanted_stage = "layer_inventory_review" if selection_name == "reviewed" else "layer_inventory"
        selected_call = next(
            (
                record
                for record in reversed(call_records)
                if str(record.get("stage") or "") == wanted_stage
                and isinstance(record.get("response"), Mapping)
            ),
            None,
        )
        if selected_call is not None:
            restored_inventory = _normalize_inventory(
                selected_call["response"],
                _items(units),
                width=image_width,
                height=image_height,
            )
            restored_by_id = {
                str(item.get("segment_id") or ""): item
                for item in _items(restored_inventory.get("visual_layer_segments"))
            }
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                restored = restored_by_id.get(str(segment.get("id") or ""))
                if restored is None:
                    continue
                segment["visible_regions"] = list(restored.get("visible_regions") or [])
                segment["image_region"] = json.dumps(segment["visible_regions"], ensure_ascii=False)
                segment["evidence"] = str(
                    restored.get("boundary_evidence")
                    or segment.get("evidence")
                    or "可见层段边界"
                )
            old_selection = (
                dict(stored_inventory.get("inventory_selection"))
                if isinstance(stored_inventory, Mapping)
                and isinstance(stored_inventory.get("inventory_selection"), Mapping)
                else {}
            )
            restored_inventory["inventory_selection"] = old_selection
            multipass["layer_inventory"] = restored_inventory
    legend_catalog = multipass.get("legend_catalog")
    legend_items = _items(
        legend_catalog.get("legend_items") if isinstance(legend_catalog, Mapping) else []
    )
    explicit_layer_lithologies = {
        str(item.get("lithology_id") or ""): str(item.get("normalized_name") or "")
        for item in legend_items
        if str(item.get("catalog_source") or "") == "explicit_layer_label"
    }
    global_audit = multipass.get("global_audit")
    unresolved_items = (
        global_audit.get("unresolved_segments", [])
        if isinstance(global_audit, Mapping)
        else []
    )
    unresolved_ids = {
        str(item.get("segment_id") or "") if isinstance(item, Mapping) else str(item)
        for item in unresolved_items
    }
    surface_overlap_ids: set[str] = set()
    segment_records = [item for item in segments if isinstance(item, Mapping)]
    for vertical in segment_records:
        if str(vertical.get("segment_role") or "vertical_layer") != "vertical_layer":
            continue
        for lateral in segment_records:
            if str(lateral.get("segment_role") or "") != "lateral_facies_zone":
                continue
            if str(vertical.get("parent_unit_id") or "") != str(lateral.get("parent_unit_id") or ""):
                continue
            if any(
                _bbox_containment_ratio(
                    list(vertical_region.get("bbox") or []),
                    list(lateral_region.get("bbox") or []),
                )
                >= 0.9
                for vertical_region in _items(vertical.get("visible_regions"))
                for lateral_region in _items(lateral.get("visible_regions"))
            ):
                surface_overlap_ids.add(str(vertical.get("id") or ""))
    gate_uncertainties: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_id = str(segment.get("id") or segment.get("segment_id") or "")
        parent_name = unit_names.get(str(segment.get("parent_unit_id") or ""), "")
        segment_role = str(segment.get("segment_role") or "vertical_layer")
        segment["name"] = (
            f"{parent_name or segment.get('parent_unit_id')}"
            f"{'横向岩相区' if segment_role == 'lateral_facies_zone' else '垂向层段'}"
            f"{int(segment.get('order_within_parent') or 0) + 1}"
        )
        assignments = [dict(item) for item in _items(segment.get("lithology_assignments"))]
        rejected: list[dict[str, Any]] = []
        retained: list[dict[str, Any]] = []
        for assignment in assignments:
            lithology_id = str(assignment.get("lithology_id") or "")
            explicit_name = explicit_layer_lithologies.get(lithology_id)
            if explicit_name and explicit_name not in parent_name:
                rejected.append({**assignment, "gate_reason": "explicit_lithology_not_in_parent_label"})
            else:
                retained.append(assignment)
        if segment_id in unresolved_ids:
            rejected.extend({**item, "gate_reason": "global_audit_unresolved"} for item in retained)
            retained = []
        if segment_id in surface_overlap_ids:
            rejected.extend({**item, "gate_reason": "vertical_region_contained_by_top_surface"} for item in retained)
            retained = []
        if rejected:
            segment["rejected_lithology_candidates"] = rejected
            segment["lithology_assignments"] = retained
            segment["lithology_ids"] = [str(item.get("lithology_id") or "") for item in retained]
            segment["review_status"] = (
                "unresolved_after_global_audit"
                if segment_id in unresolved_ids
                else "unresolved_surface_role_overlap"
                if segment_id in surface_overlap_ids
                else "rejected_by_catalog_scope_gate"
            )
            segment["confidence"] = min(_confidence(segment.get("confidence")), 0.49)
            reasons = sorted({str(item.get("gate_reason") or "") for item in rejected})
            gate_uncertainties.append(f"{segment_id} 岩性候选未入图：{','.join(reasons)}")
    raw_uncertainties = corrected.get("uncertainties")
    corrected["uncertainties"] = list(
        dict.fromkeys(
            [
                *(
                    [str(item) for item in raw_uncertainties]
                    if isinstance(raw_uncertainties, list)
                    else []
                ),
                *gate_uncertainties,
            ]
        )
    )
    summary = multipass.get("summary")
    if isinstance(summary, dict):
        summary["segment_count"] = len([item for item in segments if isinstance(item, Mapping)])
        summary["assigned_segment_count"] = sum(
            bool(item.get("lithology_ids")) for item in segments if isinstance(item, Mapping)
        )
        summary["unknown_segment_count"] = sum(
            not bool(item.get("lithology_ids")) for item in segments if isinstance(item, Mapping)
        )
    multipass["quality_gates"] = {
        "applied": True,
        "image_size": [image_width, image_height] if image_width and image_height else [],
        "rejected_candidate_count": sum(
            len(item.get("rejected_lithology_candidates", []))
            for item in segments
            if isinstance(item, Mapping)
        ),
        "uncertainties": gate_uncertainties,
    }
    return corrected


def _merge_into_base_payload(
    base_payload: Mapping[str, Any],
    inventory: Mapping[str, Any],
    legend: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    unit_color_results: list[Mapping[str, Any]],
    unit_color_batch_count: int,
    audit: Mapping[str, Any],
    call_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """中文说明：把逐层与同色批次结果写回审计协议，供质量门校验后投影为地层单元直连岩性。"""

    merged = dict(base_payload)
    merged["lithologies"] = [
        {
            "id": item["lithology_id"],
            "name": item["normalized_name"],
            "visual_pattern": "；".join([*item.get("color_features", []), *item.get("pattern_features", [])]),
            "legend_bbox": item.get("bbox", []),
            "text_explicit": item.get("text_explicit", False),
            "catalog_source": item.get("catalog_source", "legend"),
            "source_kind": "visual",
            "evidence": item.get("evidence", "图例专用调用"),
            "image_region": f"图例 bbox={item.get('bbox', [])}",
            "confidence": item.get("confidence", 0.0),
        }
        for item in legend.get("legend_items", [])
    ]
    raw_units = merged.get("stratigraphic_units")
    color_result_by_unit = {
        str(result.get("unit_id") or ""): result
        for result in unit_color_results
        if str(result.get("unit_id") or "")
    }
    if isinstance(raw_units, list):
        for unit in raw_units:
            if isinstance(unit, dict):
                unit_id = str(unit.get("id") or "")
                color_result = color_result_by_unit.get(unit_id, {})
                color_assignments = []
                for item in _items(color_result.get("composition")):
                    assignment = dict(item)
                    assignment.update(
                        {
                            "source_unit_color_scan": True,
                            "source_visible_regions": list(color_result.get("visible_regions") or []),
                            "source_color_signature": dict(color_result.get("color_signature") or {}),
                            "same_color_region_evidence": str(
                                color_result.get("same_color_region_evidence") or ""
                            ),
                            "visual_anchor_evidence": str(
                                color_result.get("same_color_region_evidence")
                                or "程序颜色掩膜圈定地层范围，VLM 在原始区域核验图例纹理"
                            ),
                        }
                    )
                    color_assignments.append(assignment)
                unit["lithology_ids"] = [
                    str(item.get("lithology_id") or "")
                    for item in color_assignments
                    if str(item.get("lithology_id") or "")
                ]
                unit["lithology_assignments"] = color_assignments
                unit["lithology_assignment_source"] = (
                    "unit_color_scan_and_visual_layer_segments"
                    if color_assignments
                    else "visual_layer_segments"
                )
    result_by_segment = {str(result.get("segment_id") or ""): result for result in results}
    merged_segments: list[dict[str, Any]] = []
    for segment in inventory.get("visual_layer_segments", []):
        if not isinstance(segment, Mapping):
            continue
        segment_id = str(segment.get("segment_id") or "")
        result = result_by_segment.get(segment_id, {})
        composition = [dict(item) for item in _items(result.get("composition"))]
        merged_segments.append(
            {
                "id": segment_id,
                "name": (
                    f"{segment.get('parent_unit_name') or segment.get('parent_unit_id')}"
                    f"{'横向岩相区' if segment.get('segment_role') == 'lateral_facies_zone' else '垂向层段'}"
                    f"{int(segment.get('order_within_parent') or 0) + 1}"
                ),
                "entity_type": "visual_layer_segment",
                "parent_unit_id": segment.get("parent_unit_id"),
                "segment_role": segment.get("segment_role", "vertical_layer"),
                "order_within_parent": segment.get("order_within_parent", 0),
                "visible_regions": list(segment.get("visible_regions") or []),
                "lithology_ids": [item["lithology_id"] for item in composition],
                "lithology_assignments": composition,
                "decision_confidence": result.get("decision_confidence", 0.0),
                "review_status": result.get("review_status", "single_pass"),
                "source_kind": (
                    "visual_context"
                    if any(item.get("evidence_source") == "reference_context" for item in composition)
                    else "visual"
                ),
                "evidence": segment.get("boundary_evidence") or "可见层段边界",
                "image_region": json.dumps(segment.get("visible_regions", []), ensure_ascii=False),
                "confidence": min(_confidence(segment.get("confidence")), _confidence(result.get("decision_confidence"), 0.0)),
            }
        )
    merged["visual_layer_segments"] = merged_segments
    multipass_uncertainties = [
        *[str(item) for item in inventory.get("uncertainties", [])],
        *[str(item) for item in legend.get("uncertainties", [])],
        *[
            f"{result.get('segment_id')}: {item}"
            for result in results
            for item in result.get("uncertainties", [])
        ],
        *[
            f"{result.get('unit_id')} 同色扫描：{item}"
            for result in unit_color_results
            for item in result.get("uncertainties", [])
        ],
        *(
            [f"全图审查未解决：{item}" for item in audit.get("unresolved_segments", [])]
            if isinstance(audit.get("unresolved_segments"), list)
            else []
        ),
    ]
    raw_uncertainties = merged.get("uncertainties")
    merged["uncertainties"] = list(dict.fromkeys([
        *([str(item) for item in raw_uncertainties] if isinstance(raw_uncertainties, list) else []),
        *multipass_uncertainties,
    ]))
    merged["multipass_lithology"] = {
        "schema_version": "three_dimensional_lithology.multipass.v1",
        "layer_inventory": dict(inventory),
        "legend_catalog": dict(legend),
        "layer_results": [dict(result) for result in results],
        "unit_color_scan": {
            "strategy": "dominant_fill_color_mask_plus_legend_pattern",
            "batch_size": UNIT_COLOR_BATCH_SIZE,
            "batch_count": unit_color_batch_count,
            "accept_threshold": UNIT_COLOR_ACCEPT_THRESHOLD,
            "color_distance_threshold": UNIT_COLOR_DISTANCE_THRESHOLD,
            "unit_results": [dict(result) for result in unit_color_results],
        },
        "global_audit": dict(audit),
        "call_records": [dict(record) for record in call_records],
        "summary": {
            "vlm_call_count": len(call_records),
            "segment_count": len(merged_segments),
            "assigned_segment_count": sum(bool(segment.get("lithology_ids")) for segment in merged_segments),
            "reviewed_segment_count": sum(str(result.get("review_status") or "") != "single_pass" for result in results),
            "unknown_segment_count": sum(not bool(segment.get("lithology_ids")) for segment in merged_segments),
            "unit_color_batch_count": unit_color_batch_count,
            "unit_color_scanned_count": len(unit_color_results),
            "unit_color_assigned_count": sum(
                bool(result.get("composition")) for result in unit_color_results
            ),
            "unit_color_rejected_candidate_count": sum(
                len(result.get("rejected_candidates", [])) for result in unit_color_results
            ),
        },
    }
    return merged


def extract_multipass_lithology(
    task: ImageExtractionTask,
    vlm_client: Any,
    classification: StratigraphicSubtypeClassification,
    base_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    """中文说明：执行层界、图例、逐层、盲审、裁决和全图审查，并返回兼容原三维协议的结果。"""

    width, height = _image_size(task.image_path)
    reviewed_base, _ = apply_manual_review_corrections(task, base_payload)
    known_units = _items(reviewed_base.get("stratigraphic_units"))
    if not known_units:
        raise ValueError("多轮岩性识别需要整图阶段先提供地层单元")
    call_records: list[dict[str, Any]] = []
    inventory_raw = _call_stage(
        vlm_client,
        task.image_path,
        build_layer_inventory_prompt(
            task,
            classification,
            known_units,
            width=width,
            height=height,
        ),
        stage="layer_inventory",
        task_name=f"三维地层层界冻结:{task.image_id}",
        call_records=call_records,
    )
    inventory = _normalize_inventory(
        inventory_raw,
        known_units,
        width=width,
        height=height,
    )
    legend_raw = _call_stage(
        vlm_client,
        task.image_path,
        build_legend_catalog_prompt(
            task,
            _items(reviewed_base.get("lithologies")),
            width=width,
            height=height,
        ),
        stage="legend_catalog",
        task_name=f"三维地层图例目录:{task.image_id}",
        call_records=call_records,
    )
    legend = _normalize_legend(
        legend_raw,
        _items(reviewed_base.get("lithologies")),
        width=width,
        height=height,
    )
    inventory_review_raw = _call_stage(
        vlm_client,
        task.image_path,
        build_layer_inventory_review_prompt(
            task,
            known_units,
            inventory,
            legend,
            width=width,
            height=height,
        ),
        stage="layer_inventory_review",
        task_name=f"三维地层层界独立审查:{task.image_id}",
        call_records=call_records,
    )
    reviewed_inventory = _normalize_inventory(
        inventory_review_raw,
        known_units,
        width=width,
        height=height,
    )
    inventory = _choose_inventory(inventory, reviewed_inventory, known_units)
    legend_items = _items(legend.get("legend_items"))
    allowed_lithology_ids = {str(item.get("lithology_id") or "") for item in legend_items}
    explicit_layer_lithologies = {
        str(item.get("lithology_id") or ""): str(item.get("normalized_name") or "")
        for item in legend_items
        if str(item.get("catalog_source") or "") == "explicit_layer_label"
    }
    results: list[dict[str, Any]] = []
    unit_color_inventory = _unit_color_inventory(task.image_path, known_units, inventory)
    unit_color_results: list[dict[str, Any]] = []
    unit_color_batch_count = (
        (len(unit_color_inventory) + UNIT_COLOR_BATCH_SIZE - 1) // UNIT_COLOR_BATCH_SIZE
        if unit_color_inventory
        else 0
    )
    with tempfile.TemporaryDirectory(prefix="three_dimensional_lithology_") as temp_dir:
        temp_root = Path(temp_dir)
        for batch_offset in range(0, len(unit_color_inventory), UNIT_COLOR_BATCH_SIZE):
            batch_index = batch_offset // UNIT_COLOR_BATCH_SIZE + 1
            unit_batch = unit_color_inventory[
                batch_offset : batch_offset + UNIT_COLOR_BATCH_SIZE
            ]
            color_sheet_path = temp_root / f"unit_color_batch_{batch_index:02d}.png"
            _build_unit_color_batch_sheet(
                task.image_path,
                unit_batch,
                legend.get("legend_bbox", []),
                color_sheet_path,
            )
            color_raw = _call_stage(
                vlm_client,
                str(color_sheet_path),
                build_unit_color_lithology_prompt(
                    task,
                    unit_batch,
                    legend_items,
                    batch_index=batch_index,
                    batch_count=unit_color_batch_count,
                ),
                stage="unit_color_lithology",
                task_name=f"三维地层同色岩性批次:{task.image_id}:{batch_index}",
                call_records=call_records,
            )
            normalized_color_results, batch_uncertainties = _normalize_unit_color_batch(
                color_raw,
                unit_batch,
                allowed_lithology_ids,
                explicit_layer_lithologies,
            )
            if batch_uncertainties and normalized_color_results:
                normalized_color_results[0]["uncertainties"] = list(
                    dict.fromkeys(
                        [
                            *normalized_color_results[0].get("uncertainties", []),
                            *[f"批次级：{item}" for item in batch_uncertainties],
                        ]
                    )
                )
            unit_color_results.extend(normalized_color_results)
        for sequence, segment in enumerate(inventory["visual_layer_segments"], start=1):
            focus_path = temp_root / f"{sequence:02d}_{segment['segment_id']}.png"
            _build_focus_sheet(task.image_path, segment, legend.get("legend_bbox", []), focus_path)
            primary_raw = _call_stage(
                vlm_client,
                str(focus_path),
                build_layer_lithology_prompt(task, segment, legend_items, stage="layer_lithology"),
                stage="layer_lithology",
                task_name=f"三维逐层岩性:{task.image_id}:{segment['segment_id']}",
                call_records=call_records,
            )
            primary = _normalize_layer_result(primary_raw, segment["segment_id"], allowed_lithology_ids)
            needs_review = (
                primary["decision_confidence"] < REVIEW_THRESHOLD
                or primary["needs_independent_review"]
                or not primary["composition"]
                or any(
                    (region["bbox"][3] - region["bbox"][1]) < 24
                    for region in segment.get("visible_regions", [])
                    if isinstance(region.get("bbox"), list) and len(region["bbox"]) == 4
                )
            )
            if not needs_review:
                primary["review_status"] = "single_pass"
                results.append(primary)
                continue
            review_raw = _call_stage(
                vlm_client,
                str(focus_path),
                build_layer_lithology_prompt(task, segment, legend_items, stage="independent_review"),
                stage="independent_review",
                task_name=f"三维逐层岩性盲审:{task.image_id}:{segment['segment_id']}",
                call_records=call_records,
            )
            review = _normalize_layer_result(review_raw, segment["segment_id"], allowed_lithology_ids)
            if _composition_ids(primary) == _composition_ids(review):
                results.append(_merge_agreeing_reviews(primary, review))
                continue
            arbitration_raw = _call_stage(
                vlm_client,
                str(focus_path),
                build_arbitration_prompt(segment, legend_items, primary, review),
                stage="arbitration",
                task_name=f"三维逐层岩性裁决:{task.image_id}:{segment['segment_id']}",
                call_records=call_records,
            )
            arbitration = _normalize_layer_result(arbitration_raw, segment["segment_id"], allowed_lithology_ids)
            arbitration["review_status"] = "arbitrated"
            arbitration["primary_result"] = primary
            arbitration["independent_review"] = review
            results.append(arbitration)

    audit_raw = _call_stage(
        vlm_client,
        task.image_path,
        build_global_audit_prompt(inventory, legend, results, unit_color_results),
        stage="global_audit",
        task_name=f"三维逐层岩性全图审查:{task.image_id}",
        call_records=call_records,
    )
    results = _apply_global_corrections(results, audit_raw, allowed_lithology_ids)
    unit_color_results = _apply_unit_color_corrections(
        unit_color_results,
        audit_raw,
        allowed_lithology_ids,
    )
    return (
        _merge_into_base_payload(
            base_payload,
            inventory,
            legend,
            results,
            unit_color_results,
            unit_color_batch_count,
            audit_raw,
            call_records,
        ),
        len(call_records),
    )
