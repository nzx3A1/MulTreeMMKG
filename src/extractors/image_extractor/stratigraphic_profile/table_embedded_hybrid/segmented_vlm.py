"""按轨道组分段调用 VLM，避免复杂地层表格单次长响应超时。"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from src.utils.llm_client import safe_json_loads

from ...schema_models import ImageExtractionTask
from ..subclassifier import StratigraphicSubtypeClassification
from .pipeline import INTERVAL_FIELDS, TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION
from .ppstructure_geometry import (
    apply_ppstructure_geometry,
    extract_ppstructure_geometry,
    geometry_prompt_catalog,
)


SEGMENT_ORDER = ("layout", "stratigraphy_lithology", "facies_reservoir_wells")
NODE_ENRICHMENT_SCHEMA_VERSION = "table_embedded_hybrid.node_enrichment.v1"
NODE_FIELDS = (*INTERVAL_FIELDS, "curve_tracks", "point_markers", "objects")


def _source_image_size(task: ImageExtractionTask) -> tuple[int, int]:
    """中文说明：读取原图尺寸，仅用于向 VLM 说明画布；像素几何由 PP-StructureV3 负责。"""

    image_path = Path(task.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"来源图片不存在：{image_path}")
    with Image.open(image_path) as image:
        return image.width, image.height


def build_segmented_table_prompt(
    task: ImageExtractionTask,
    classification: StratigraphicSubtypeClassification,
    segment: str,
    known_tracks: list[Mapping[str, Any]] | None = None,
    known_intervals: list[Mapping[str, Any]] | None = None,
    geometry: Mapping[str, Any] | None = None,
) -> str:
    """中文说明：为三组语义任务生成 Prompt，所有坐标都改由同一份 PP 几何目录提供。"""

    if segment not in SEGMENT_ORDER:
        raise ValueError(f"未知表格分段：{segment}")
    image_width, image_height = _source_image_size(task)
    track_catalog = ""
    if known_tracks:
        rows = [
            {
                "id": str(track.get("id") or ""),
                "role": str(track.get("role") or ""),
                "header": str(track.get("header") or ""),
            }
            for track in known_tracks
            if str(track.get("id") or "")
        ]
        track_catalog = (
            "\nlayout 分段已经确定以下轨道；本段所有 track_id 必须逐字选用其中的 id，"
            f"不得另造别名：\n{rows}\n"
        )
    interval_catalog = ""
    if known_intervals and segment == "facies_reservoir_wells":
        rows = [
            {
                "id": str(interval.get("id") or ""),
                "name": str(interval.get("name") or ""),
                "geometry_refs": list(interval.get("geometry_refs") or []),
            }
            for interval in known_intervals
            if str(interval.get("id") or "")
        ]
        interval_catalog = (
            "\n左侧语义分段已经选择以下 PP-StructureV3 几何锚点。右侧图元应复用这些 geometry_refs：\n"
            f"{json.dumps(rows, ensure_ascii=False)}\n"
        )
    geometry_catalog = geometry_prompt_catalog(geometry or {})
    geometry_catalog_json = json.dumps(
        geometry_catalog,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    common = f"""
你正在对同一张表格—图片嵌入混合型地层柱状图执行分段视觉 OCR。
图片 ID：{task.image_id}
图题：{task.caption or '无'}
已确认子类：{classification.subtype.value}
原图真实尺寸：width={image_width}，height={image_height}。
PP-StructureV3 几何目录（ID 与像素坐标均由程序生成；只选择 ID，不估算坐标）：
{geometry_catalog_json}

共同约束：
1. 只读取当前图片直接可见内容，不用正文常识补值；无法辨认时写入 uncertainties。
2. 禁止输出 content_bbox、bbox、pixel_y、top_y、bottom_y；轨道只选 pp_track_*，区间/点只用 geometry_refs 选择 pp_cell_* 或 pp_ocr_*。
3. id 使用简短稳定英文或拼音，不得在不同对象间重复；evidence 必须引用图中可见文字、刻度、色条或曲线。
4. 只输出指定 segment 的 JSON，不输出 Markdown、解释或思维过程。
{track_catalog}
{interval_catalog}
""".strip()

    instructions = {
        "layout": """
本次只做轨道语义标注和纵轴类型判断，不抽取地层区间：
- layout_family 从 stratigraphic_column_table、multi_track_well_log、imaging_log_panel、stratigraphic_summary_table 中单选；无法归入时用 other_table_hybrid。
- 若存在连续深度轴，kind=depth 并从目录选择其 track_id 与 calibration_ocr_ids；逐层厚度数字不是连续深度轴，必须使用 kind=thickness，程序会退化到相对层序而不生成伪深度。
- tracks 只能选择 PP 目录中实际存在的 pp_track_*，补充地层层级、深度、岩性、曲线、相、储层等角色；没有的栏不要虚构。
输出：
{
  "schema_version":"table_embedded_hybrid.segment.v1",
  "segment":"layout",
  "layout_family":"stratigraphic_column_table|multi_track_well_log|imaging_log_panel|stratigraphic_summary_table|other_table_hybrid",
  "diagram_id":"",
  "diagram_name":"",
  "image_size":{"width":0,"height":0},
  "coordinate_system":{"vertical_axis":{"kind":"thickness|depth|relative_sequence","unit":"m|无量纲","increases":"downward|upward","track_id":"pp_track_*","calibration_ocr_ids":["pp_ocr_*"]}},
  "tracks":[{"id":"pp_track_*","role":"stratigraphy|depth|lithology|curve|text|facies|reservoir|well","header":"","parser":"","evidence":""}],
  "uncertainties":[]
}
""",
        "stratigraphy_lithology": """
本次只读取左半部的地层层级、代号、厚度轴、岩性剖面和岩性简述：
- stratigraphic_intervals 完整保留当前图片可见的系/统/组/段/亚段、测井小层或表格行层级；parent_id 必须指向已输出父层。图中没有地层层级时返回空数组。
- lithology_intervals 按图中可见岩性文字与对应纵向区间读取；不要仅凭图例花纹猜名称。
- reference_intervals 用于井段、高亮重点段、甜点段、成像测井解释段或其他明确但不属于地层单位的连续区间；没有则返回空数组。
- geological_feature_intervals 只保留图中明确标注的裂缝、孔洞、断层、顶底板等连续特征区间。
- 多轨测井或成像面板不要求创造地层单位；应优先忠实读取深度段、可见岩性和高亮区间。
输出：
{
  "schema_version":"table_embedded_hybrid.segment.v1",
  "segment":"stratigraphy_lithology",
  "primitives":{
    "stratigraphic_intervals":[{"id":"","name":"","parent_id":"","rank":"","track_id":"pp_track_*","geometry_refs":["pp_cell_*"],"evidence":"","confidence":0.0}],
    "reference_intervals":[],
    "lithology_intervals":[{"id":"","name":"","track_id":"pp_track_*","geometry_refs":["pp_cell_*"],"evidence":"","confidence":0.0}],
    "geological_feature_intervals":[]
  },
  "uncertainties":[]
}
""",
        "facies_reservoir_wells": """
本次读取其余曲线、相、储层/高亮、井号、嵌入图像面板及说明栏：
- facies_intervals 记录图片实际可见的沉积相或微相区间；没有则为空。
- curve_tracks 逐条记录可见曲线名称、单位和刻度，包括但不限于 GR、SP、AC、DEN、CNL、电阻率、TOC、矿物含量、脆性指数、含气量或海平面曲线。
- curve_observations 按可稳定对应的层段或深度段记录高低、增减、峰谷或异常响应；刻度不清时只写 qualitative_response，不虚构数值。
- reservoir_intervals 记录图中明确的储层、油气层、甜点、高亮色带或优质页岩段；没有则为空。
- point_markers 只逐个读取井号并选择 geometry_refs；其他点状标签放入 objects。objects 还可保存组合、孔隙类型、成像测井面板、储层成因说明等非连续对象。
- 必须从有效内容顶部一直读取到底部，不能只抽取上半图；每个坐标都应与上面的已知地层/井段像素边界比较。
- 对成像测井双面板，应分别建立面板对象和对应深度段；对纯测井图，不得虚构沉积相、储层成因或井号。
- explicit_relations 只允许图中竖排文字或同一储层行明确表达的关系；其余跨轨关系由程序按纵轴生成。
输出：
{
  "schema_version":"table_embedded_hybrid.segment.v1",
  "segment":"facies_reservoir_wells",
  "primitives":{
    "facies_intervals":[{"id":"","name":"","track_id":"pp_track_*","geometry_refs":["pp_cell_*"],"evidence":"","confidence":0.0}],
    "curve_tracks":[{"id":"","name":"","track_id":"","scale_min":null,"scale_max":null,"unit":"","scale_direction":"","evidence":""}],
    "curve_observations":[{"id":"","name":"","curve_ids":[],"track_id":"pp_track_*","geometry_refs":["pp_cell_*"],"qualitative_response":"","evidence":"","confidence":0.0}],
    "reservoir_intervals":[{"id":"","name":"","track_id":"pp_track_*","geometry_refs":["pp_cell_*"],"evidence":"","confidence":0.0}],
    "oil_layer_intervals":[],
    "point_markers":[{"id":"","name":"","kind":"well","track_id":"pp_track_*","geometry_refs":["pp_cell_*","pp_ocr_*"],"evidence":"","confidence":0.0}],
    "objects":[{"id":"","name":"","entity_type":"oil_layer_group|pore_type|imaging_log|embedded_panel|geological_object","evidence":"","confidence":0.0}],
    "explicit_relations":[{"source_id":"","relation_type":"","target_id":"","evidence":"","confidence":0.0}]
  },
  "uncertainties":[]
}
""",
    }[segment].strip()
    return f"{common}\n\n{instructions}"


def _parse_segment(response: Any, expected_segment: str) -> dict[str, Any]:
    """中文说明：解析并校验单个真实 VLM 分段，防止错段或空响应进入合并结果。"""

    payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"表格分段 {expected_segment} 响应不是非空 JSON 对象")
    if str(payload.get("segment") or "") != expected_segment:
        raise ValueError(
            f"表格分段响应错位：expected={expected_segment}，actual={payload.get('segment')}"
        )
    return dict(payload)


def merge_segmented_table_payloads(segments: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """中文说明：按字段所有权确定性合并三段 OCR，重复字段和缺段会立即报错。"""

    missing = [segment for segment in SEGMENT_ORDER if segment not in segments]
    if missing:
        raise ValueError(f"表格分段 OCR 缺少：{missing}")
    layout = dict(segments["layout"])
    merged_primitives: dict[str, Any] = {
        **{field: [] for field in INTERVAL_FIELDS},
        "curve_tracks": [],
        "point_markers": [],
        "objects": [],
        "explicit_relations": [],
    }
    owners: dict[str, str] = {}
    uncertainties: list[str] = []
    for segment in SEGMENT_ORDER:
        payload = segments[segment]
        raw_uncertainties = payload.get("uncertainties")
        if isinstance(raw_uncertainties, list):
            for item in raw_uncertainties:
                text = str(item).strip()
                if text and text not in uncertainties:
                    uncertainties.append(text)
        primitives = payload.get("primitives")
        if not isinstance(primitives, Mapping):
            continue
        for field, values in primitives.items():
            if field not in merged_primitives:
                continue
            if field in owners:
                raise ValueError(f"分段字段 {field} 同时由 {owners[field]} 和 {segment} 输出")
            if not isinstance(values, list):
                raise ValueError(f"分段 {segment}.primitives.{field} 必须是数组")
            owners[field] = segment
            merged_primitives[field] = list(values)

    return {
        "schema_version": TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION,
        "diagram_id": str(layout.get("diagram_id") or "table_embedded_hybrid_diagram"),
        "diagram_name": str(layout.get("diagram_name") or "表格嵌入混合地层图"),
        "layout_family": str(layout.get("layout_family") or "other_table_hybrid"),
        "image_size": dict(layout.get("image_size") or {}),
        "coordinate_system": dict(layout.get("coordinate_system") or {}),
        "tracks": list(layout.get("tracks") or []),
        "primitives": merged_primitives,
        "uncertainties": uncertainties,
    }


def _track_header_map(payload: Mapping[str, Any]) -> dict[str, str]:
    """中文说明：合并 VLM 表头、PP-OCR 表头和重复轨道语义，生成稳定的轨道表头属性。"""

    result: dict[str, str] = {}
    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        return result
    for track in tracks:
        if not isinstance(track, Mapping):
            continue
        track_id = str(track.get("id") or "").strip()
        if not track_id:
            continue
        values: list[str] = []
        for raw in (
            track.get("header"),
            track.get("ppstructure_header_text"),
            *(track.get("semantic_headers") if isinstance(track.get("semantic_headers"), list) else []),
        ):
            text = str(raw or "").strip()
            if text and text not in values:
                values.append(text)
        result[track_id] = " / ".join(values)
    return result


def _geometry_track_ids(item: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    """中文说明：优先采用显式 track_id，并用 PP cell/OCR 引用补足节点所在轨道。"""

    track_ids: list[str] = []
    explicit = str(item.get("track_id") or "").strip()
    if explicit:
        track_ids.append(explicit)
    refs = item.get("geometry_refs")
    geometry = payload.get("ppstructure_geometry")
    if not isinstance(refs, list) or not isinstance(geometry, Mapping):
        return track_ids
    ref_ids = {str(value) for value in refs if str(value)}
    for field in ("cells", "ocr_lines"):
        records = geometry.get(field)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping) or str(record.get("id") or "") not in ref_ids:
                continue
            candidates = record.get("track_ids")
            if not isinstance(candidates, list):
                candidates = [record.get("track_id")]
            for value in candidates:
                track_id = str(value or "").strip()
                if track_id and track_id not in track_ids:
                    track_ids.append(track_id)
    return track_ids


def _node_candidates(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """中文说明：只枚举前三段已经确认的节点，第四次调用不得新增任何候选。"""

    headers = _track_header_map(payload)
    diagram_id = str(payload.get("diagram_id") or "table_embedded_hybrid_diagram").strip()
    candidates = [
        {
            "id": diagram_id,
            "name": str(payload.get("diagram_name") or "表格嵌入混合地层图"),
            "node_kind": "stratigraphic_profile",
            "track_header": "整图",
            "evidence": "整张图片及图题",
        }
    ]
    primitives = payload.get("primitives")
    if not isinstance(primitives, Mapping):
        raise ValueError("节点规范化前缺少 primitives")
    seen = {diagram_id}
    for field in NODE_FIELDS:
        records = primitives.get(field, [])
        if not isinstance(records, list):
            raise ValueError(f"primitives.{field} 必须是数组")
        for index, raw in enumerate(records):
            if not isinstance(raw, Mapping):
                continue
            fallback = f"{field}_{index}"
            local_id = str(raw.get("id") or fallback).strip()
            if not local_id or local_id in seen:
                raise ValueError(f"节点规范化发现缺失或重复局部 ID：{local_id!r}")
            seen.add(local_id)
            track_ids = _geometry_track_ids(raw, payload)
            track_headers = [headers[value] for value in track_ids if headers.get(value)]
            candidates.append(
                {
                    "id": local_id,
                    "name": str(raw.get("name") or local_id),
                    "node_kind": str(raw.get("entity_type") or field.removesuffix("s")),
                    "track_id": " / ".join(track_ids),
                    "track_header": " / ".join(dict.fromkeys(track_headers)),
                    "evidence": str(raw.get("evidence") or "")[:500],
                }
            )
    return candidates


def build_node_enrichment_prompt(task: ImageExtractionTask, payload: Mapping[str, Any]) -> str:
    """中文说明：携带图题、正文上下文和已确认节点，请 VLM 仅补充领域官方名称。"""

    candidates = _node_candidates(payload)
    context = "\n".join(str(value) for value in task.references if str(value).strip())[:6000]
    return f"""
你正在规范化一张地质表格图中已经抽取完成的知识图谱节点。
图题：{task.caption or '无'}
相邻正文上下文：{context or '无'}
已确认节点（id、当前名称、类型、所在轨道表头和图内证据）：
{json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}

你可以使用地质学、地层学、测井学领域知识以及上面的图题和正文上下文，但必须遵守：
1. 只能返回上面已有的 id，不能新增、删除、合并或拆分节点，不能生成关系或坐标。
2. official_name 填写该节点最规范、完整的官方中文名或行业标准名；缩写可在上下文足够时展开。
3. 若无法可靠规范化，official_name 必须原样保留当前名称，basis 使用 insufficient_context_keep_original，禁止猜测具体组、段、井号或岩性。
4. 每个输入 id 必须且只能返回一次，official_name 不得为空。
5. 只输出 JSON：
{{"schema_version":"{NODE_ENRICHMENT_SCHEMA_VERSION}","nodes":[{{"id":"","official_name":"","basis":"standardized_domain_term|expanded_abbreviation|already_official|insufficient_context_keep_original","confidence":0.0}}]}}
""".strip()


def apply_node_enrichment(payload: Mapping[str, Any], enrichment: Mapping[str, Any]) -> dict[str, Any]:
    """中文说明：按局部 ID 写入官方名，并以确定性轨道映射写入每个节点的 track_header。"""

    if str(enrichment.get("schema_version") or "") != NODE_ENRICHMENT_SCHEMA_VERSION:
        raise ValueError(f"节点规范化响应必须使用 {NODE_ENRICHMENT_SCHEMA_VERSION}")
    raw_nodes = enrichment.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("节点规范化响应的 nodes 必须是数组")
    expected = _node_candidates(payload)
    expected_ids = {item["id"] for item in expected}
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, Mapping):
            raise ValueError(f"节点规范化 nodes[{index}] 不是对象")
        local_id = str(raw.get("id") or "").strip()
        official_name = str(raw.get("official_name") or "").strip()
        if local_id not in expected_ids or local_id in by_id or not official_name:
            raise ValueError(f"节点规范化返回未知、重复或空名称节点：{local_id!r}")
        try:
            confidence = round(max(0.0, min(1.0, float(raw.get("confidence")))), 3)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"节点 {local_id} 缺少有效 confidence") from exc
        by_id[local_id] = {
            "official_name": official_name,
            "official_name_basis": str(raw.get("basis") or "insufficient_context_keep_original"),
            "official_name_confidence": confidence,
        }
    missing = sorted(expected_ids - set(by_id))
    if missing:
        raise ValueError(f"节点规范化响应缺少节点：{missing}")

    enriched = deepcopy(dict(payload))
    candidate_by_id = {item["id"]: item for item in expected}
    diagram_id = str(enriched.get("diagram_id") or "table_embedded_hybrid_diagram")
    enriched["diagram_official_name"] = by_id[diagram_id]["official_name"]
    enriched["diagram_official_name_basis"] = by_id[diagram_id]["official_name_basis"]
    enriched["diagram_official_name_confidence"] = by_id[diagram_id]["official_name_confidence"]
    enriched["diagram_track_header"] = "整图"
    primitives = enriched["primitives"]
    for field in NODE_FIELDS:
        for index, item in enumerate(primitives.get(field, [])):
            local_id = str(item.get("id") or f"{field}_{index}").strip()
            item["id"] = local_id
            item.update(by_id[local_id])
            item["track_header"] = candidate_by_id[local_id]["track_header"]
    enriched["node_enrichment"] = deepcopy(dict(enrichment))
    return enriched


def enrich_table_node_names(
    task: ImageExtractionTask,
    vlm_client: Any,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """中文说明：执行第 4 次受约束 VLM 调用，只为现有节点补充官方名。"""

    response = vlm_client.describe_image(
        task.image_path,
        build_node_enrichment_prompt(task, payload),
        task_name=f"表格嵌入混合节点官方名规范化:{task.image_id}",
        response_format={"type": "json_object"},
        max_tokens=int(os.getenv("STRATIGRAPHIC_TABLE_NODE_ENRICHMENT_MAX_TOKENS", "12288")),
    )
    enrichment = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
    if not isinstance(enrichment, Mapping):
        raise ValueError("节点官方名规范化响应不是 JSON 对象")
    return apply_node_enrichment(payload, enrichment)


def validate_and_repair_pixel_geometry(
    task: ImageExtractionTask,
    payload: dict[str, Any],
    *,
    geometry: Mapping[str, Any] | None = None,
    geometry_provider: Any | None = None,
) -> dict[str, Any]:
    """中文说明：兼容原入口名称，但现在用 PP-StructureV3 全量替换 VLM 像素坐标。"""

    resolved_geometry = (
        dict(geometry)
        if isinstance(geometry, Mapping)
        else extract_ppstructure_geometry(task.image_path, geometry_provider)
    )
    return apply_ppstructure_geometry(payload, resolved_geometry)


def extract_segmented_table_visual(
    task: ImageExtractionTask,
    vlm_client: Any,
    classification: StratigraphicSubtypeClassification,
    *,
    geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """中文说明：PP-StructureV3 先跑一次，三次 VLM 只复用几何 ID 完成语义识别。"""

    geometry_provider = getattr(vlm_client, "ppstructure_geometry_extractor", None)
    resolved_geometry = (
        dict(geometry)
        if isinstance(geometry, Mapping)
        else extract_ppstructure_geometry(task.image_path, geometry_provider)
    )

    max_tokens = {
        # 中文说明：宽幅综合柱状表可能包含十余条轨道，布局 JSON 需要更高上限以避免在最后一条轨道处截断。
        "layout": int(os.getenv("STRATIGRAPHIC_TABLE_LAYOUT_MAX_TOKENS", "4096")),
        "stratigraphy_lithology": int(os.getenv("STRATIGRAPHIC_TABLE_GEOLOGY_MAX_TOKENS", "8192")),
        "facies_reservoir_wells": int(os.getenv("STRATIGRAPHIC_TABLE_RESERVOIR_MAX_TOKENS", "8192")),
    }
    segments: dict[str, dict[str, Any]] = {}
    for segment in SEGMENT_ORDER:
        response = vlm_client.describe_image(
            task.image_path,
            build_segmented_table_prompt(
                task,
                classification,
                segment,
                known_tracks=(segments.get("layout") or {}).get("tracks"),
                known_intervals=(
                    ((segments.get("stratigraphy_lithology") or {}).get("primitives") or {}).get(
                        "stratigraphic_intervals"
                    )
                ),
                geometry=resolved_geometry,
            ),
            task_name=(
                f"表格嵌入混合专用OCR:{segment}_v2:{task.image_id}"
                if segment == "facies_reservoir_wells"
                else f"表格嵌入混合专用OCR:{segment}:{task.image_id}"
            ),
            response_format={"type": "json_object"},
            max_tokens=max_tokens[segment],
        )
        segments[segment] = _parse_segment(response, segment)
    resolved_payload = validate_and_repair_pixel_geometry(
        task,
        merge_segmented_table_payloads(segments),
        geometry=resolved_geometry,
    )
    return enrich_table_node_names(task, vlm_client, resolved_payload)
