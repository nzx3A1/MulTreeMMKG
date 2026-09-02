"""综合柱状图的语义轨道合并、曲线矢量化、图例匹配与轨道内容补全。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from PIL import Image

from src.utils.llm_client import safe_json_loads

from .vlm_options import table_embedded_hybrid_vlm_timeout_secs


VISUAL_TRACK_EXTRACTION_VERSION = "semantic-track-vlm-description-only.v8"
VISUAL_TRACK_SLICE_SCHEMA_VERSION = "table_embedded_hybrid.track_slice_description.v2"
VLM_TRACK_TYPES = ("table_text", "legend", "curve")
TRACK_RECOGNITION_ORDER = ("table_text", "legend", "curve")
VLM_SLICE_TRACK_TYPES = ("legend",)
_TEXT_CLEANER = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff.%\-/]+")
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
_FALLBACK_TEXT_ENTITY_ROLES = frozenset(
    {"stratigraphy", "lithology", "facies", "reservoir", "well", "text"}
)
_SINGLE_CJK_GEOLOGICAL_LABELS = frozenset("煤岩层段组相坪井系统界")
_ROCK_KEYWORDS = (
    "岩",
    "云",
    "灰",
    "泥",
    "砂",
    "砾",
    "煤",
    "膏",
    "盐",
    "白云",
)
_CURVE_STYLE_RULES = (
    ("GR", "red", "continuous_curve", "linear"),
    ("SP", "blue", "continuous_curve", "linear"),
    ("CNL", "green", "continuous_curve", "linear"),
    ("AC", "red", "continuous_curve", "linear"),
    ("DEN", "blue", "continuous_curve", "linear"),
    ("RS", "blue", "continuous_curve", "log10"),
    ("RD", "red", "continuous_curve", "log10"),
    ("测井孔隙度", "red", "filled_profile", "linear"),
    ("岩心孔隙度", "black", "sample_bars", "linear"),
    ("测井渗透率", "cyan", "filled_profile", "log10"),
    ("岩心渗透率", "black", "sample_bars", "log10"),
)
_CURVE_NAME_ALIASES = {
    "测井孔隙度": ("测井孔隙度", "logporosity", "welllogporosity", "loggingporosity"),
    "岩心孔隙度": ("岩心孔隙度", "coreporosity"),
    "测井渗透率": ("测井渗透率", "logpermeability", "welllogpermeability", "loggingpermeability"),
    "岩心渗透率": ("岩心渗透率", "corepermeability"),
}
_LITHOLOGY_PROFILE_HEADERS = ("岩性剖面", "剖面图", "岩性柱", "岩性柱状", "剖面")
_LITHOLOGY_TYPES = (
    "泥岩",
    "页岩",
    "粉砂岩",
    "砂岩",
    "细砂岩",
    "中砂岩",
    "粗砂岩",
    "砾岩",
    "角砾岩",
    "石灰岩",
    "白云岩",
    "泥质灰岩",
    "泥质云岩",
    "灰质云岩",
    "云质灰岩",
    "云质泥岩",
    "含膏云岩",
    "石膏岩",
    "盐岩",
    "煤",
    "火山灰岩/凝灰岩",
)
_LITHOLOGY_PROFILE_REFERENCE_PROMPT = """
当且仅当本轨道是岩性剖面图时，使用下表作为可见纹理候选参考。它只用于图案比对，不能替代图像证据；
纹理不清、混合关系不确定或与候选不符时必须返回 unresolved，并在 uncertainty 中说明。

| 岩性 | 常见图例视觉模式 | 自动识别时的重要特征 |
| --- | --- | --- |
| 泥岩 Mudstone | 密集或稀疏水平短线 | 水平方向占主导，基本无斜线 |
| 页岩 Shale | 很密的平行水平线 | 比泥岩更密、层理感明显 |
| 粉砂岩 Siltstone | 细小短点、短划线 | 点比砂岩小且密 |
| 砂岩 Sandstone | 均匀点状 | 大量离散圆点 |
| 细砂岩 | 密集小点 | 点径较小 |
| 中砂岩 | 中等大小点 | 点径中等 |
| 粗砂岩 | 较大、较疏的点 | 粗颗粒感明显 |
| 砾岩 Conglomerate | 圆形、椭圆形砾石符号 | 不规则大圆/卵石 |
| 角砾岩 Breccia | 棱角状多边形 | 三角形、多边形碎块 |
| 石灰岩 Limestone | 砖墙状 | 横线加错列竖线 |
| 白云岩 Dolomite | 斜线状或斜砖状 | 大量同方向斜线 |
| 泥质灰岩 | 石灰岩砖纹加泥质横纹 | 两种纹理叠加 |
| 泥质云岩 | 白云岩斜纹加泥质横纹 | 斜线加水平短线 |
| 灰质云岩 | 白云岩纹加石灰岩成分纹 | 斜纹为主的组合纹 |
| 云质灰岩 | 石灰岩砖纹加白云岩斜纹 | 砖纹为主并带斜线 |
| 云质泥岩 | 泥岩横纹加白云岩符号 | 横纹为主，局部斜线或点 |
| 含膏云岩 | 白云岩纹加石膏符号 | 斜纹加成组特殊短线 |
| 石膏岩 Gypsum | 特定短线、叉状或晶体状符号 | 重复特殊几何符号 |
| 盐岩 Halite | 方格或晶体状符号 | 规则方形或晶体结构 |
| 煤 Coal | 实黑或高密度黑纹 | 黑色填充最明显 |
| 火山灰岩/凝灰岩 | 点、叉、碎屑组合 | 不规则碎屑及火山符号 |
""".strip()


def _visual_track_slice_cache_root() -> Path:
    """中文说明：返回持久化轨道切片缓存目录，并保证结果始终为可写入的绝对路径。"""

    configured = os.getenv("TABLE_VISUAL_TRACK_SLICE_CACHE_DIR")
    project_root = Path(__file__).resolve().parents[5]
    cache_root = Path(
        configured
        or project_root / "data" / "cache" / "table_embedded_hybrid" / "visual_track_slices"
    ).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _visual_track_slice_cache_path(
    image_path: Path,
    slice_record: Mapping[str, Any],
) -> Path:
    """中文说明：按原图指纹、切片 ID 与像素框生成稳定文件名，原图变化后自动使用新缓存。"""

    resolved_image = image_path.expanduser().resolve(strict=True)
    stat = resolved_image.stat()
    source_fingerprint = hashlib.sha256(
        (
            f"{resolved_image}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"{VISUAL_TRACK_EXTRACTION_VERSION}"
        ).encode("utf-8")
    ).hexdigest()[:20]
    slice_id = re.sub(
        r"[^0-9A-Za-z_.-]+",
        "_",
        str(slice_record.get("slice_id") or "track_slice"),
    ).strip("_.") or "track_slice"
    bbox_key = "_".join(str(int(value)) for value in slice_record.get("bbox", []))
    slice_fingerprint = hashlib.sha256(
        f"{slice_id}|{bbox_key}".encode("utf-8")
    ).hexdigest()[:12]
    return (
        _visual_track_slice_cache_root()
        / source_fingerprint
        / f"{slice_id}_{slice_fingerprint}.png"
    ).resolve()


def _clean_text(value: Any) -> str:
    """中文说明：清理 OCR 空白与无关符号，供表头、曲线名和图例名匹配。"""

    return _TEXT_CLEANER.sub("", str(value or "")).strip()


def _bbox(record: Mapping[str, Any]) -> list[float]:
    """中文说明：读取左上右下像素框，非法框统一返回空列表。"""

    raw = record.get("bbox")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        return []
    try:
        box = [float(value) for value in raw]
    except (TypeError, ValueError):
        return []
    return box if box[2] > box[0] and box[3] > box[1] else []


def _center_x(record: Mapping[str, Any]) -> float:
    """中文说明：返回轨道或单元格的水平中心，用于一维轨道归并。"""

    box = _bbox(record)
    return (box[0] + box[2]) / 2.0 if box else 0.0


def sort_tracks_left_to_right(
    tracks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """中文说明：按轨道像素框从左到右重排，并写入连续序号和相邻轨道 ID。"""

    ordered = sorted(
        (dict(item) for item in tracks if isinstance(item, Mapping) and _bbox(item)),
        key=lambda item: (_bbox(item)[0], _center_x(item), _bbox(item)[2]),
    )
    for order, track in enumerate(ordered):
        track["order"] = order
        track["previous_track_id"] = (
            str(ordered[order - 1].get("id") or "") if order > 0 else ""
        )
        track["next_track_id"] = (
            str(ordered[order + 1].get("id") or "")
            if order + 1 < len(ordered)
            else ""
        )
    return ordered


def validate_vlm_track_types(tracks: Any) -> list[dict[str, Any]]:
    """中文说明：校验 VLM 为每条唯一轨道给出三分类结果，拒绝缺失、重复或越界类型。"""

    if not isinstance(tracks, list) or not tracks:
        raise ValueError("layout.tracks 必须是非空数组")
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(tracks):
        if not isinstance(raw, Mapping):
            raise ValueError(f"layout.tracks[{index}] 必须是对象")
        track = dict(raw)
        track_id = str(track.get("id") or "").strip()
        track_type = str(track.get("track_type") or "").strip()
        if not track_id or track_id in seen_ids:
            raise ValueError(f"layout.tracks[{index}].id 缺失或重复：{track_id!r}")
        if track_type not in VLM_TRACK_TYPES:
            raise ValueError(
                f"layout.tracks[{index}].track_type 必须由 VLM 从 {VLM_TRACK_TYPES} 中选择，"
                f"实际为 {track_type!r}"
            )
        seen_ids.add(track_id)
        track["track_type_source"] = "VLM.layout_track_classification"
        validated.append(track)
    return validated


def _horizontal_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """中文说明：计算两个框相对较窄框的水平覆盖率。"""

    left_box, right_box = _bbox(left), _bbox(right)
    if not left_box or not right_box:
        return 0.0
    overlap = max(0.0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0]))
    shorter = min(left_box[2] - left_box[0], right_box[2] - right_box[0])
    return overlap / shorter if shorter > 0 else 0.0


def _infer_role(header: Any) -> str:
    """中文说明：仅从可见表头推断粗粒度轨道角色，不生成地质事实。"""

    text = _clean_text(header).casefold()
    if not text:
        return "unknown"
    if "深度" in text or text in {"深", "度/m"}:
        return "depth"
    if "亚段" in text or "地层" in text or text in {"段", "系", "统", "组", "代号"}:
        return "stratigraphy"
    if "岩性" in text or "剖面" in text:
        return "lithology"
    if "沉积相" in text or "微相" in text:
        return "facies"
    if "孔隙度" in text:
        return "porosity"
    if "渗透率" in text:
        return "permeability"
    if "储层" in text or "发育层段" in text or "油层" in text:
        return "reservoir"
    if "井" in text:
        return "well"
    if any(token.casefold() in text for token, *_ in _CURVE_STYLE_RULES):
        return "curve"
    return "text"


def _meaningful_header(value: Any) -> bool:
    """中文说明：排除纯刻度数字，只让有文字语义的未命中列成为新轨道锚点。"""

    text = _clean_text(value)
    letters = "".join(character for character in text if character.isalpha() or "\u4e00" <= character <= "\u9fff")
    return len(letters) >= 2 or letters.upper() in {"GR", "SP", "AC", "RS", "RD"}


def _semantic_track_header_values(track: Mapping[str, Any]) -> list[str]:
    """中文说明：汇总语义轨道的可见表头，去除空值和重复值，用于判断相邻岩性列是否属于同一剖面。"""

    values: list[str] = []
    for value in (
        track.get("header"),
        track.get("semantic_header_text"),
        track.get("ppstructure_header_text"),
    ):
        text = str(value or "").strip()
        if text and text not in values:
            values.append(text)
    return values


def _is_profile_semantic_track(track: Mapping[str, Any]) -> bool:
    """中文说明：仅将 VLM 标注为岩性图例的轨道视为剖面合并候选，避免误合并普通图例和曲线列。"""

    return (
        str(track.get("track_type") or "") == "legend"
        and str(track.get("role") or "").strip().casefold() == "lithology"
    )


def _merge_adjacent_profile_tracks(
    tracks: Sequence[Mapping[str, Any]],
    physical_to_semantic: Mapping[str, str],
    geometry_cells: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """中文说明：利用相邻边界、同一岩性语义和 PP 共享单元格证据合并被网格线拆开的剖面物理列。"""

    ordered = sort_tracks_left_to_right(tracks)
    cells = [dict(item) for item in geometry_cells if isinstance(item, Mapping) and _bbox(item)]
    if len(ordered) < 2 or not cells:
        return ordered, dict(physical_to_semantic)

    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        """中文说明：返回相邻剖面轨道的并查集根节点。"""

        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        """中文说明：将有充分共享单元格证据的两条相邻剖面轨道归入同一组。"""

        left_root, right_root = find(left_index), find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root

    def member_ids(track: Mapping[str, Any]) -> set[str]:
        """中文说明：读取语义轨道保留的 PP 物理轨道 ID 集合。"""

        values = track.get("member_track_ids")
        if isinstance(values, list):
            resolved = {str(value) for value in values if str(value)}
            if resolved:
                return resolved
        track_id = str(track.get("id") or "")
        return {track_id} if track_id else set()

    def shared_cells(left_ids: set[str], right_ids: set[str]) -> list[dict[str, Any]]:
        """中文说明：查找同时跨越左右物理轨道的 PP 单元格，作为剖面合并的强几何证据。"""

        return [
            cell
            for cell in cells
            if left_ids
            & {str(value) for value in cell.get("track_ids", [])}
            and right_ids
            & {str(value) for value in cell.get("track_ids", [])}
        ]

    shared_by_pair: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for index in range(len(ordered) - 1):
        left, right = ordered[index], ordered[index + 1]
        if not (_is_profile_semantic_track(left) and _is_profile_semantic_track(right)):
            continue
        left_box, right_box = _bbox(left), _bbox(right)
        gap = right_box[0] - left_box[2]
        vertical_overlap = max(
            0.0,
            min(left_box[3], right_box[3]) - max(left_box[1], right_box[1]),
        )
        shorter_height = min(left_box[3] - left_box[1], right_box[3] - right_box[1])
        if abs(gap) > 2.0 or shorter_height <= 0 or vertical_overlap / shorter_height < 0.9:
            continue
        left_headers = _semantic_track_header_values(left)
        right_headers = _semantic_track_header_values(right)
        has_profile_header = any(
            token in header
            for header in [*left_headers, *right_headers]
            for token in _LITHOLOGY_PROFILE_HEADERS
        )
        if not has_profile_header:
            continue
        evidence = shared_cells(member_ids(left), member_ids(right))
        if not evidence:
            continue
        shared_by_pair[(index, index + 1)] = evidence
        union(index, index + 1)

    groups: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        groups.setdefault(find(index), []).append(index)
    if all(len(indices) == 1 for indices in groups.values()):
        return ordered, dict(physical_to_semantic)

    physical_order = {
        physical_id: order
        for order, track in enumerate(ordered)
        for physical_id in member_ids(track)
    }
    merged_mapping = dict(physical_to_semantic)
    merged_tracks: list[dict[str, Any]] = []
    for indices in groups.values():
        group = [ordered[index] for index in indices]
        if len(group) == 1:
            merged_tracks.append(dict(group[0]))
            continue
        canonical = next(
            (
                track
                for track in group
                if any(
                    token in header
                    for header in _semantic_track_header_values(track)
                    for token in _LITHOLOGY_PROFILE_HEADERS
                )
            ),
            group[0],
        )
        canonical_id = str(canonical.get("id") or "")
        merged_member_ids = sorted(
            {physical_id for track in group for physical_id in member_ids(track)},
            key=lambda value: (physical_order.get(value, 10**9), value),
        )
        evidence_cells = [
            cell
            for cell in cells
            if len(
                set(merged_member_ids)
                & {str(value) for value in cell.get("track_ids", [])}
            )
            >= 2
        ]
        boxes = [*(_bbox(track) for track in group), *(_bbox(cell) for cell in evidence_cells)]
        boxes = [box for box in boxes if box]
        header_values: list[str] = []
        for track in group:
            for header in _semantic_track_header_values(track):
                if header not in header_values:
                    header_values.append(header)
        merged_tracks.append(
            {
                **dict(canonical),
                "id": canonical_id,
                "bbox": [
                    round(min(box[0] for box in boxes)),
                    round(min(box[1] for box in boxes)),
                    round(max(box[2] for box in boxes)),
                    round(max(box[3] for box in boxes)),
                ],
                "header": str(canonical.get("header") or " / ".join(header_values)),
                "semantic_header_text": " / ".join(header_values),
                "member_track_ids": merged_member_ids,
                "merged_from_semantic_track_ids": [str(track.get("id") or "") for track in group],
                "merge_evidence_cell_ids": [str(cell.get("id") or "") for cell in evidence_cells],
                "merge_policy": "adjacent_lithology_profile_shared_pp_cell",
                "geometry_source": "PP-StructureV3.semantic_track_group",
            }
        )
        for physical_id in merged_member_ids:
            merged_mapping[physical_id] = canonical_id
    return sort_tracks_left_to_right(merged_tracks), merged_mapping


def build_semantic_tracks(
    mapped_tracks: Sequence[Mapping[str, Any]],
    geometry_tracks: Sequence[Mapping[str, Any]],
    geometry_cells: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """中文说明：以 VLM 语义列为锚点归并 PP 物理窄列，再依据共享单元格合并被拆开的岩性剖面。"""

    physical = [dict(item) for item in geometry_tracks if isinstance(item, Mapping) and _bbox(item)]
    anchors = [dict(item) for item in mapped_tracks if isinstance(item, Mapping) and _bbox(item)]
    selected_ids = {str(item.get("id") or "") for item in anchors}
    for item in physical:
        item_id = str(item.get("id") or "")
        header = str(item.get("header_text") or "")
        if not item_id or item_id in selected_ids or not _meaningful_header(header):
            continue
        inferred_role = _infer_role(header)
        nearby_same_role = any(
            str(anchor.get("role") or "unknown") == inferred_role
            and abs(_center_x(anchor) - _center_x(item))
            <= max(_bbox(anchor)[2] - _bbox(anchor)[0], _bbox(item)[2] - _bbox(item)[0]) * 2.0
            for anchor in anchors
        )
        if nearby_same_role:
            continue
        anchors.append(
            {
                **item,
                "role": inferred_role,
                "track_type": "unresolved",
                "track_type_source": "missing_vlm_track_classification",
                "header": header,
                "parser": "deterministic_header_role",
                "evidence": f"PP-OCR 表头：{header}",
                "ppstructure_header_text": header,
                "geometry_source": "PP-StructureV3",
            }
        )
        selected_ids.add(item_id)
    if not anchors:
        unresolved = [
            {
                **item,
                "track_type": "unresolved",
                "track_type_source": "missing_vlm_track_classification",
            }
            for item in physical
        ]
        return sort_tracks_left_to_right(unresolved), {
            str(item.get("id") or ""): str(item.get("id") or "") for item in physical
        }
    anchors = sort_tracks_left_to_right(anchors)
    members: dict[str, list[dict[str, Any]]] = {str(anchor.get("id") or ""): [] for anchor in anchors}
    physical_to_semantic: dict[str, str] = {}
    for item in physical:
        chosen = min(anchors, key=lambda anchor: abs(_center_x(anchor) - _center_x(item)))
        semantic_id = str(chosen.get("id") or "")
        physical_id = str(item.get("id") or "")
        members[semantic_id].append(item)
        if physical_id:
            physical_to_semantic[physical_id] = semantic_id
    semantic_tracks: list[dict[str, Any]] = []
    for order, anchor in enumerate(anchors):
        anchor_id = str(anchor.get("id") or "")
        owned = members.get(anchor_id) or [anchor]
        boxes = [_bbox(item) for item in owned if _bbox(item)]
        header_values: list[str] = []
        for value in (
            anchor.get("header"),
            anchor.get("ppstructure_header_text"),
            *(item.get("header_text") for item in owned),
        ):
            text = str(value or "").strip()
            if text and text not in header_values:
                header_values.append(text)
        semantic_tracks.append(
            {
                **anchor,
                "id": anchor_id,
                "order": order,
                "bbox": [
                    round(min(box[0] for box in boxes)),
                    round(min(box[1] for box in boxes)),
                    round(max(box[2] for box in boxes)),
                    round(max(box[3] for box in boxes)),
                ],
                "header": str(anchor.get("header") or " / ".join(header_values)),
                "semantic_header_text": " / ".join(header_values),
                "member_track_ids": [str(item.get("id") or "") for item in owned if item.get("id")],
                "geometry_source": "PP-StructureV3.semantic_track_group",
            }
        )
    return _merge_adjacent_profile_tracks(
        semantic_tracks,
        physical_to_semantic,
        geometry_cells,
    )


def _track_for_cell(cell: Mapping[str, Any], tracks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """中文说明：按物理成员 ID 和水平覆盖率为单元格选择唯一语义轨道。"""

    physical_ids = {str(value) for value in cell.get("track_ids", [])} if isinstance(cell.get("track_ids"), list) else set()
    candidates: list[tuple[float, Mapping[str, Any]]] = []
    for track in tracks:
        member_ids = {str(value) for value in track.get("member_track_ids", [])}
        membership = 2.0 if physical_ids & member_ids else 0.0
        score = membership + _horizontal_overlap(cell, track)
        if score > 0:
            candidates.append((score, track))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _body_top(track: Mapping[str, Any], geometry: Mapping[str, Any]) -> float:
    """中文说明：以表头词命中单元格和约 15% 高度的长横线确定数据区起点。"""

    content = geometry.get("content_bbox")
    content_box = [float(value) for value in content] if isinstance(content, list) and len(content) == 4 else _bbox(track)
    if not content_box:
        return _bbox(track)[1]
    top, bottom = content_box[1], content_box[3]
    limit = top + (bottom - top) * 0.35
    cells = [
        item
        for item in geometry.get("cells", [])
        if isinstance(item, Mapping) and _bbox(item) and _horizontal_overlap(item, track) >= 0.35
    ]
    header_text = " / ".join(
        str(value or "")
        for value in (
            track.get("header"),
            track.get("semantic_header_text"),
            track.get("ppstructure_header_text"),
        )
        if str(value or "")
    )
    header_tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{1,}", header_text)
        if token.strip()
    }
    matched_bottoms = []
    for cell in cells:
        box = _bbox(cell)
        if box[1] > limit or box[3] > limit:
            continue
        cell_text = _clean_text(cell.get("refined_text") or cell.get("text")).casefold()
        if cell_text and any(token in cell_text for token in header_tokens):
            matched_bottoms.append(box[3])
    if matched_bottoms:
        return max(matched_bottoms)
    horizontal = ((geometry.get("rule_lines") or {}).get("horizontal_lines") or [])
    candidates = [float(value) for value in horizontal if top + (bottom - top) * 0.08 <= float(value) <= limit]
    nominal = top + (bottom - top) * 0.15
    return min(candidates, key=lambda value: abs(value - nominal)) if candidates else nominal


def _visual_slice_source_entities(
    payload: Mapping[str, Any],
    geometry: Mapping[str, Any],
    track: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """中文说明：收集相邻表格轨道中已识别实体的原图纵向像素范围，供视觉轨道比较数量并投影切片。"""

    track_id = str(track.get("id") or "")
    body_top = _body_top(track, geometry)
    primitives = payload.get("primitives")
    primitives = primitives if isinstance(primitives, Mapping) else {}
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for field in (
        "stratigraphic_intervals",
        "reference_intervals",
        "lithology_intervals",
        "facies_intervals",
        "reservoir_intervals",
        "oil_layer_intervals",
        "geological_feature_intervals",
        "track_intervals",
    ):
        values = primitives.get(field)
        if not isinstance(values, list):
            continue
        for index, raw in enumerate(values):
            if not isinstance(raw, Mapping) or str(raw.get("track_id") or "") != track_id:
                continue
            entity_id = str(raw.get("id") or f"{field}_{index}")
            if entity_id in seen_ids:
                continue
            box = raw.get("geometry_bbox")
            if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
                try:
                    box = [
                        float(_bbox(track)[0]),
                        float(raw["top_y"]),
                        float(_bbox(track)[2]),
                        float(raw["bottom_y"]),
                    ]
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
            try:
                normalized_box = [float(value) for value in box]
            except (TypeError, ValueError):
                continue
            if normalized_box[3] <= normalized_box[1] or normalized_box[3] <= body_top + 1.0:
                continue
            normalized_box[1] = max(normalized_box[1], body_top)
            refs = [
                str(value)
                for value in raw.get("geometry_refs", [])
                if str(value)
            ] if isinstance(raw.get("geometry_refs"), list) else []
            records.append(
                {
                    "source_entity_id": entity_id,
                    "source_cell_id": refs[0] if refs else entity_id,
                    "source_geometry_refs": refs,
                    "source_label": str(raw.get("name") or entity_id).strip(),
                    "bbox": normalized_box,
                }
            )
            seen_ids.add(entity_id)
    return sorted(records, key=lambda item: (item["bbox"][1], item["bbox"][3], item["source_entity_id"]))


def _fallback_slice_source_cells(
    geometry: Mapping[str, Any],
    tracks: Sequence[Mapping[str, Any]],
    track: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """中文说明：相邻轨道尚无可定位实体时，回退到该轨道正文单元格，保证旧缓存和退化识别仍可切片。"""

    track_id = str(track.get("id") or "")
    body_top = _body_top(track, geometry)
    records: list[dict[str, Any]] = []
    seen_cells: set[str] = set()
    for cell in geometry.get("cells", []):
        if not isinstance(cell, Mapping):
            continue
        cell_id = str(cell.get("id") or "")
        box = _bbox(cell)
        owner = _track_for_cell(cell, tracks)
        if (
            not cell_id
            or cell_id in seen_cells
            or not box
            or not owner
            or str(owner.get("id") or "") != track_id
            or box[3] <= body_top + 1.0
            or box[1] < body_top - 2.0
        ):
            continue
        seen_cells.add(cell_id)
        records.append(
            {
                "source_entity_id": cell_id,
                "source_cell_id": cell_id,
                "source_geometry_refs": [cell_id],
                "source_label": str(
                    cell.get("refined_text") or cell.get("text") or cell_id
                ).strip(),
                "bbox": box,
            }
        )
    return sorted(records, key=lambda item: (item["bbox"][1], item["bbox"][3], item["source_entity_id"]))


def _nearest_table_text_track_candidates(
    tracks: Sequence[Mapping[str, Any]],
    target_order: int,
) -> list[dict[str, Any]]:
    """中文说明：从目标视觉轨道两侧分别寻找最近的 table_text 轨道，跳过中间视觉轨道。"""

    candidates: list[dict[str, Any]] = []
    for side, orders in (
        ("left", range(target_order - 1, -1, -1)),
        ("right", range(target_order + 1, len(tracks))),
    ):
        for candidate_order in orders:
            candidate = tracks[candidate_order]
            if str(candidate.get("track_type") or "") != "table_text":
                continue
            candidates.append(
                {
                    "side": side,
                    "track": candidate,
                    "distance": abs(candidate_order - target_order),
                }
            )
            break
    return candidates


def _is_lithology_profile_track(track: Mapping[str, Any]) -> bool:
    """中文说明：仅根据轨道表头或岩性角色判断是否启用剖面岩性纹理参考，不据图案预判岩性。"""

    header = " / ".join(
        str(value or "")
        for value in (
            track.get("header"),
            track.get("semantic_header_text"),
            track.get("ppstructure_header_text"),
        )
        if str(value or "").strip()
    )
    role = str(track.get("role") or "").strip().casefold()
    return any(token in header for token in _LITHOLOGY_PROFILE_HEADERS) or (
        role == "lithology" and "剖面" in header
    )


def build_adjacent_visual_track_slices(
    payload: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """中文说明：仅为 legend 轨道按相邻 table_text 实体投影切片，curve 轨道不再生成 VLM 切片。"""

    tracks = sort_tracks_left_to_right(
        [item for item in payload.get("tracks", []) if isinstance(item, Mapping)]
    )
    content_box = geometry.get("content_bbox")
    content = (
        [float(value) for value in content_box]
        if isinstance(content_box, list) and len(content_box) == 4
        else []
    )
    slices: list[dict[str, Any]] = []
    visual_targets = sorted(
        (
            (order, track)
            for order, track in enumerate(tracks)
            if str(track.get("track_type") or "") in VLM_SLICE_TRACK_TYPES
        ),
        key=lambda item: (
            TRACK_RECOGNITION_ORDER.index(str(item[1].get("track_type") or "")),
            item[0],
        ),
    )
    for order, track in visual_targets:
        track_type = str(track.get("track_type") or "")
        current_id = str(track.get("id") or "")
        current_box = _bbox(track)
        if not current_id or not current_box:
            continue
        candidates: list[dict[str, Any]] = []
        for candidate in _nearest_table_text_track_candidates(tracks, order):
            table_text_track = candidate["track"]
            entities = _visual_slice_source_entities(payload, geometry, table_text_track)
            candidates.append(
                {
                    **candidate,
                    "entity_count": len(entities),
                    "sources": entities,
                }
            )
        if not candidates:
            continue
        # 中文说明：左右最近文字轨道实体数相同时固定选左侧，使纵向基准稳定且可复现。
        selected = max(
            candidates,
            key=lambda item: (
                int(item["entity_count"]),
                1 if item["side"] == "left" else 0,
            ),
        )
        source_track = selected["track"]
        source_id = str(source_track.get("id") or "")
        source_records = selected["sources"] or _fallback_slice_source_cells(
            geometry,
            tracks,
            source_track,
        )
        adjacent_counts = {
            str(item["side"]): {
                "track_id": str(item["track"].get("id") or ""),
                "entity_count": int(item["entity_count"]),
                "track_distance": int(item["distance"]),
                "track_type": "table_text",
            }
            for item in candidates
        }
        raw_body_top = track.get("body_top")
        try:
            header_bottom = float(raw_body_top)
        except (TypeError, ValueError):
            header_bottom = current_box[1]
        header_bottom = max(current_box[1], min(header_bottom, current_box[3]))
        header_bbox = [
            round(current_box[0]),
            round(current_box[1]),
            round(current_box[2]),
            round(header_bottom),
        ]
        for source in source_records:
            box = source["bbox"]
            top = max(box[1], content[1] if content else box[1])
            bottom = min(box[3], content[3] if content else box[3])
            if bottom - top < 2.0:
                continue
            cell_id = str(source["source_cell_id"])
            source_entity_id = str(source["source_entity_id"])
            safe_track = re.sub(r"[^0-9A-Za-z_]+", "_", current_id).strip("_")
            safe_source = re.sub(r"[^0-9A-Za-z_]+", "_", source_entity_id).strip("_")
            slice_id = f"track_slice_{safe_track}_{safe_source}"
            source_label = str(source["source_label"])
            target_bbox = [
                round(current_box[0]),
                round(top),
                round(current_box[2]),
                round(bottom),
            ]
            slices.append(
                {
                    "slice_id": slice_id,
                    "track_id": current_id,
                    "track_order": order,
                    "track_type": track_type,
                    "recognition_stage": TRACK_RECOGNITION_ORDER.index(track_type) + 1,
                    "track_role": str(track.get("role") or ""),
                    "is_lithology_profile": _is_lithology_profile_track(track),
                    "track_header": str(
                        track.get("header")
                        or track.get("semantic_header_text")
                        or track.get("ppstructure_header_text")
                        or current_id
                    ),
                    "source_track_id": source_id,
                    "source_track_order": int(source_track.get("order", 0)),
                    "source_track_side": str(selected["side"]),
                    "source_track_distance": int(selected["distance"]),
                    "source_track_entity_count": int(selected["entity_count"]),
                    "adjacent_track_entity_counts": adjacent_counts,
                    "candidate_table_text_track_entity_counts": adjacent_counts,
                    "source_track_type": str(source_track.get("track_type") or ""),
                    "source_track_header": str(
                        source_track.get("header")
                        or source_track.get("semantic_header_text")
                        or source_track.get("ppstructure_header_text")
                        or source_id
                    ),
                    "source_entity_id": source_entity_id,
                    "source_cell_id": cell_id,
                    "source_geometry_refs": list(source["source_geometry_refs"]),
                    "source_label": source_label,
                    "header_bbox": header_bbox,
                    "header_pixel_range": {
                        "left_x": header_bbox[0],
                        "top_y": header_bbox[1],
                        "right_x": header_bbox[2],
                        "bottom_y": header_bbox[3],
                        "coordinate_space": "original_pixels",
                    },
                    "bbox": target_bbox,
                    "top_y": round(top, 3),
                    "bottom_y": round(bottom, 3),
                    "pixel_range": {
                        "left_x": target_bbox[0],
                        "top_y": round(top, 3),
                        "right_x": target_bbox[2],
                        "bottom_y": round(bottom, 3),
                        "coordinate_space": "original_pixels",
                    },
                    "coordinate_source": "PP-StructureV3.table_text_entity_y_projection",
                }
            )
    return slices


def build_visual_track_slice_prompt(slice_record: Mapping[str, Any]) -> str:
    """中文说明：结合目标轨道表头和切片生成曲线数值或图例类型识别 Prompt，禁止 VLM 生成坐标。"""

    track_type = str(slice_record.get("track_type") or "")
    if track_type == "curve":
        task_instruction = """
输入图上半部分是同一曲线轨道的完整表头图，下半部分是当前纵向区间的曲线切片，中间灰线仅用于分隔两部分。
先读取目标轨道表头中的每条曲线名、单位、左右刻度值和线性/对数刻度，再结合当前切片中曲线的水平位置，
分别估读每条可见曲线在切片顶部、中部、底部的数值，以及切片内最小值、最大值和从上到下的变化。
表头刻度不清、曲线重叠或某个值无法可靠估读时，对应数值必须为 null，并在 uncertainty 中说明；不得猜值。
curve_readings 每条可见曲线一项；无法辨认曲线名时也返回一项，curve_name=unresolved。
curve_change 必须从 rising、falling、stable、peak、trough、fluctuating、uncertain 中选择。
""".strip()
        type_specific_json = """
  "curve_readings":[{
    "curve_name":"GR或unresolved",
    "unit":"",
    "left_scale_value":null,
    "right_scale_value":null,
    "scale_transform":"linear|log10|unknown",
    "top_value":null,
    "middle_value":null,
    "bottom_value":null,
    "minimum_value":null,
    "maximum_value":null,
    "change":"rising|falling|stable|peak|trough|fluctuating|uncertain",
    "value_basis":"表头刻度与切片曲线位置的可见依据",
    "confidence":0.0,
    "uncertainty":""
  }],
  "legend_interpretations":[],
""".strip()
    else:
        profile_reference = (
            _LITHOLOGY_PROFILE_REFERENCE_PROMPT
            if bool(slice_record.get("is_lithology_profile"))
            else "本轨道不是岩性剖面，不使用岩性纹理候选表；只按实际可见内容识别图例类型。"
        )
        task_instruction = f"""
描述该图例或图像块实际可见的颜色、纹理、形状和标记，并输出对应的图例类型。
legend_interpretations 至少返回一项；无法辨认时 legend_type=unresolved、category=unknown，并说明不确定性。
若同一切片存在多种清晰图案，可分别返回多项，但不能把混合纹理强行拆成没有视觉证据的地质事实。
{profile_reference}
""".strip()
        type_specific_json = """
  "curve_readings":[],
  "legend_interpretations":[{
    "legend_type":"可见图例类型或unresolved",
    "category":"lithology|reservoir|facies|symbol|color_band|pattern|other|unknown",
    "visual_basis":"颜色、纹理、形状或标记依据",
    "confidence":0.0,
    "uncertainty":""
  }],
""".strip()
    # 中文说明：严格限制逐块描述长度，避免纵向跨度较大的图例块因逐项复述纹理而截断 JSON。
    return f"""
你正在识别综合柱状图中已经由程序裁剪好的单个轨道块。
目标轨道 ID：{slice_record.get('track_id')}
目标轨道表头：{slice_record.get('track_header')}
目标轨道类型：{track_type}
轨道识别阶段：{slice_record.get('recognition_stage')}（1=table_text，2=legend，3=curve）
切分基准 table_text 轨道方向：{slice_record.get('source_track_side')}
切分基准 table_text 轨道 ID：{slice_record.get('source_track_id')}
切分基准实体 ID：{slice_record.get('source_entity_id')}
切分基准单元格 ID：{slice_record.get('source_cell_id')}
切分基准实体文字：{slice_record.get('source_label') or '无可读文字'}

程序已经在目标轨道左右两侧分别寻找最近的 table_text 轨道，并选择已识别实体数较多的一侧作为基准。
该裁剪块的横向范围来自目标轨道，纵向范围严格继承所选基准实体。你只描述裁剪图内容，
不得输出或估算 bbox、pixel_y、top_y、bottom_y，不得扩展到其他深度区间。
description 必须是不超过 120 个汉字的总体摘要；visual_features 最多 6 项，每项不超过 20 个汉字。
若存在大量重复图案，只概括主要颜色、纹理及变化，不得逐项枚举或重复描述；整个 JSON 必须完整闭合。
{task_instruction}

只返回以下 JSON，不输出 Markdown：
{{
  "schema_version":"{VISUAL_TRACK_SLICE_SCHEMA_VERSION}",
  "slice_id":"{slice_record.get('slice_id')}",
  "track_id":"{slice_record.get('track_id')}",
  "track_type":"{track_type}",
  "source_cell_id":"{slice_record.get('source_cell_id')}",
  "description":"中文可见内容描述",
  "curve_change":"rising|falling|stable|peak|trough|fluctuating|uncertain|not_applicable",
{type_specific_json}
  "visual_features":["可见特征"],
  "confidence":0.0,
  "uncertainty":""
}}
""".strip()


def build_visual_track_slice_retry_prompt(
    slice_record: Mapping[str, Any],
    error: Exception,
) -> str:
    """中文说明：按 curve 或 legend 的 v2 专用字段重试同一切片，保持几何和目标类型不变。"""

    track_type = str(slice_record.get("track_type") or "")
    type_fields = (
        "curve_readings 必须为非空数组、legend_interpretations=[]；每条曲线的不可读数值写 null。"
        if track_type == "curve"
        else "curve_readings=[]、legend_interpretations 必须为非空数组；不可辨认时 legend_type=unresolved。"
    )

    return f"""
上一条对同一裁剪图的响应无效：{str(error)[:160]}。
请只重新返回一个完整、紧凑的 JSON 对象；不要解释，不要 Markdown，不要换行列表。
不得改变以下固定字段：
schema_version={VISUAL_TRACK_SLICE_SCHEMA_VERSION}
slice_id={slice_record.get('slice_id')}
track_id={slice_record.get('track_id')}
track_type={slice_record.get('track_type')}
source_cell_id={slice_record.get('source_cell_id')}

字段必须包含 schema_version、slice_id、track_id、track_type、source_cell_id、description、curve_change、curve_readings、legend_interpretations、visual_features、confidence、uncertainty。
{type_fields}
description 不超过 60 个汉字；visual_features 最多 3 项；不得输出任何像素坐标。
""".strip()


def cache_visual_track_slices(
    image_path: str | Path,
    slices: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """中文说明：缓存正文切片；曲线轨道另把同轨表头与正文纵向拼接为 VLM 输入图。"""

    resolved_image = Path(image_path).expanduser().resolve()
    if not resolved_image.is_file():
        raise FileNotFoundError(f"轨道裁剪来源图片不存在：{resolved_image}")
    records = [dict(item) for item in slices if isinstance(item, Mapping)]
    if not records:
        return []
    cached: list[dict[str, Any]] = []
    with Image.open(resolved_image) as source:
        source_rgb = source.convert("RGB")
        for record in records:
            crop_path = _visual_track_slice_cache_path(resolved_image, record)
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            existed = crop_path.is_file()
            if not existed:
                crop_box = tuple(int(value) for value in record["bbox"])
                source_rgb.crop(crop_box).save(crop_path, format="PNG")
            header_path = crop_path.with_name(f"{crop_path.stem}_header.png")
            vlm_input_path = crop_path
            header_box = record.get("header_bbox")
            header_available = (
                str(record.get("track_type") or "") == "curve"
                and isinstance(header_box, list)
                and len(header_box) == 4
                and int(header_box[2]) - int(header_box[0]) >= 2
                and int(header_box[3]) - int(header_box[1]) >= 2
            )
            composite_existed = False
            if header_available:
                header_existed = header_path.is_file()
                if not header_existed:
                    source_rgb.crop(tuple(int(value) for value in header_box)).save(
                        header_path,
                        format="PNG",
                    )
                vlm_input_path = crop_path.with_name(f"{crop_path.stem}_header_slice.png")
                composite_existed = vlm_input_path.is_file()
                if not composite_existed:
                    with Image.open(header_path) as header_image, Image.open(crop_path) as body_image:
                        header_rgb = header_image.convert("RGB")
                        body_rgb = body_image.convert("RGB")
                        width = max(header_rgb.width, body_rgb.width)
                        gap = 4
                        composite = Image.new(
                            "RGB",
                            (width, header_rgb.height + gap + body_rgb.height),
                            "white",
                        )
                        composite.paste(header_rgb, ((width - header_rgb.width) // 2, 0))
                        separator_y = header_rgb.height
                        for offset in range(gap):
                            for x in range(width):
                                composite.putpixel((x, separator_y + offset), (160, 160, 160))
                        composite.paste(
                            body_rgb,
                            ((width - body_rgb.width) // 2, header_rgb.height + gap),
                        )
                        composite.save(vlm_input_path, format="PNG")
            cached.append(
                {
                    **record,
                    "cropped_image_cache_path": str(crop_path.resolve()),
                    "track_header_image_cache_path": (
                        str(header_path.resolve()) if header_available else ""
                    ),
                    "vlm_input_image_cache_path": str(vlm_input_path.resolve()),
                    "vlm_input_layout": (
                        "track_header_above_slice" if header_available else "slice_only"
                    ),
                    "crop_cache_status": "disk_hit" if existed else "written",
                    "vlm_input_cache_status": (
                        "disk_hit" if header_available and composite_existed else "written"
                        if header_available
                        else "same_as_slice"
                    ),
                }
            )
    return cached


def _optional_finite_float(value: Any, *, field: str) -> float | None:
    """中文说明：把 VLM 可选数值规范为有限浮点数，不可读值必须显式使用 null。"""

    if value is None or str(value).strip().casefold() in {"", "null", "none", "unknown"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须为数值或 null：{value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} 必须为有限数值或 null：{value!r}")
    return round(number, 6)


def _bounded_confidence(value: Any, *, field: str) -> float:
    """中文说明：校验并截断 VLM 置信度，统一保留三位小数。"""

    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 的 confidence 非法") from exc
    if not math.isfinite(confidence):
        raise ValueError(f"{field} 的 confidence 必须为有限数值")
    return round(max(0.0, min(1.0, confidence)), 3)


def _normalize_curve_change(value: Any, *, default: str = "uncertain") -> str:
    """中文说明：把模型常见的同义变化词归一到受控枚举，不把未知状态伪装成确定趋势。"""

    normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "increase": "rising",
        "increasing": "rising",
        "up": "rising",
        "decrease": "falling",
        "decreasing": "falling",
        "down": "falling",
        "flat": "stable",
        "unchanged": "stable",
        "unknown": "uncertain",
        "indeterminate": "uncertain",
        "none": "not_applicable",
        "n/a": "not_applicable",
        "na": "not_applicable",
    }
    return aliases.get(normalized, normalized or default)


def _unresolved_curve_reading(*, uncertainty: str) -> dict[str, Any]:
    """中文说明：曲线无法估读时生成全 null 审计记录，明确表达缺失而不虚构数值。"""

    return {
        "curve_name": "unresolved",
        "unit": "",
        "scale_transform": "unknown",
        "change": "uncertain",
        "value_basis": "",
        "confidence": 0.0,
        "uncertainty": uncertainty,
        "left_scale_value": None,
        "right_scale_value": None,
        "top_value": None,
        "middle_value": None,
        "bottom_value": None,
        "minimum_value": None,
        "maximum_value": None,
    }


def _parse_curve_readings(raw: Any, *, slice_id: str) -> list[dict[str, Any]]:
    """中文说明：规范曲线估读值；空数组或 null 项降为全 null 的 unresolved 记录。"""

    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, list):
        raw = []
    allowed_changes = {
        "rising",
        "falling",
        "stable",
        "peak",
        "trough",
        "fluctuating",
        "uncertain",
    }
    allowed_transforms = {"linear", "log10", "unknown"}
    numeric_fields = (
        "left_scale_value",
        "right_scale_value",
        "top_value",
        "middle_value",
        "bottom_value",
        "minimum_value",
        "maximum_value",
    )
    readings: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        curve_name = str(item.get("curve_name") or "unresolved").strip() or "unresolved"
        change = _normalize_curve_change(item.get("change"))
        if change == "not_applicable":
            change = "uncertain"
        if change not in allowed_changes:
            raise ValueError(f"曲线切片 {slice_id} 的 {curve_name}.change 非法：{change!r}")
        transform = str(item.get("scale_transform") or "unknown").strip().casefold()
        transform = {"log": "log10", "logarithmic": "log10"}.get(transform, transform)
        if transform not in allowed_transforms:
            raise ValueError(f"曲线切片 {slice_id} 的 {curve_name}.scale_transform 非法：{transform!r}")
        reading = {
            "curve_name": curve_name,
            "unit": str(item.get("unit") or "").strip(),
            "scale_transform": transform,
            "change": change,
            "value_basis": str(item.get("value_basis") or "").strip(),
            "confidence": _bounded_confidence(
                item.get("confidence", 0.0),
                field=f"曲线切片 {slice_id} 的 {curve_name}",
            ),
            "uncertainty": str(item.get("uncertainty") or "").strip(),
        }
        for field in numeric_fields:
            reading[field] = _optional_finite_float(
                item.get(field),
                field=f"曲线切片 {slice_id} 的 {curve_name}.{field}",
            )
        if all(reading[field] is None for field in numeric_fields) and not reading["uncertainty"]:
            reading["uncertainty"] = "表头刻度或切片曲线位置不足以可靠估读数值"
        readings.append(reading)
    return readings or [
        _unresolved_curve_reading(
            uncertainty="模型未返回可解析的曲线读数；该切片数值保持未知"
        )
    ]


def _normalize_profile_lithology(value: Any) -> tuple[str, str]:
    """中文说明：把剖面图岩性限制到用户给定候选；越界名称降为 unresolved 并保留原始输出。"""

    raw = str(value or "").strip()
    compact = _clean_text(raw).casefold()
    english_aliases = {
        "mudstone": "泥岩",
        "shale": "页岩",
        "siltstone": "粉砂岩",
        "sandstone": "砂岩",
        "conglomerate": "砾岩",
        "breccia": "角砾岩",
        "limestone": "石灰岩",
        "dolomite": "白云岩",
        "gypsum": "石膏岩",
        "halite": "盐岩",
        "coal": "煤",
    }
    for lithology in sorted(_LITHOLOGY_TYPES, key=len, reverse=True):
        if compact == _clean_text(lithology).casefold() or compact.startswith(
            _clean_text(lithology).casefold()
        ):
            return lithology, raw
    for alias, lithology in english_aliases.items():
        if alias in compact:
            return lithology, raw
    return "unresolved", raw


def _parse_legend_interpretations(
    raw: Any,
    *,
    slice_id: str,
    is_lithology_profile: bool,
) -> list[dict[str, Any]]:
    """中文说明：规范图例类型；剖面图仅接受岩性参考表候选，其他图像轨道保留通用图例类别。"""

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"图例切片 {slice_id} 的 legend_interpretations 必须是非空数组")
    allowed_categories = {
        "lithology",
        "reservoir",
        "facies",
        "symbol",
        "color_band",
        "pattern",
        "other",
        "unknown",
    }
    interpretations: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"图例切片 {slice_id} 的 legend_interpretations[{index}] 必须是对象")
        raw_type = str(item.get("legend_type") or "unresolved").strip() or "unresolved"
        category = str(item.get("category") or "unknown").strip()
        if category not in allowed_categories:
            raise ValueError(f"图例切片 {slice_id} 的 category 非法：{category!r}")
        uncertainty = str(item.get("uncertainty") or "").strip()
        normalized_type = raw_type
        if is_lithology_profile and raw_type != "unresolved":
            normalized_type, original = _normalize_profile_lithology(raw_type)
            category = "lithology" if normalized_type != "unresolved" else "unknown"
            if normalized_type == "unresolved":
                uncertainty = uncertainty or f"模型类型不在剖面岩性候选表中：{original}"
        interpretations.append(
            {
                "legend_type": normalized_type,
                "raw_legend_type": raw_type,
                "category": category,
                "visual_basis": str(item.get("visual_basis") or "").strip(),
                "confidence": _bounded_confidence(
                    item.get("confidence"),
                    field=f"图例切片 {slice_id} 的第 {index + 1} 项",
                ),
                "uncertainty": uncertainty,
            }
        )
    return interpretations


def _parse_visual_track_slice_response(
    response: Any,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """中文说明：校验裁剪块响应只能描述指定切片，且不得改变轨道、单元格或类型。"""

    parsed = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
    if not isinstance(parsed, Mapping):
        raise ValueError("轨道裁剪块 VLM 响应不是 JSON 对象")
    payload = dict(parsed)
    if str(payload.get("schema_version") or "") != VISUAL_TRACK_SLICE_SCHEMA_VERSION:
        raise ValueError("轨道裁剪块 VLM 响应 schema_version 不正确")
    for field in ("slice_id", "track_id", "track_type", "source_cell_id"):
        if str(payload.get(field) or "") != str(expected.get(field) or ""):
            raise ValueError(
                f"轨道裁剪块 VLM 响应 {field} 错位："
                f"expected={expected.get(field)!r}, actual={payload.get(field)!r}"
            )
    description = str(payload.get("description") or "").strip()
    if not description:
        raise ValueError(f"轨道裁剪块 {expected.get('slice_id')} 缺少 description")
    curve_change = _normalize_curve_change(payload.get("curve_change"))
    allowed_changes = {
        "rising",
        "falling",
        "stable",
        "peak",
        "trough",
        "fluctuating",
        "uncertain",
        "not_applicable",
    }
    if curve_change not in allowed_changes:
        raise ValueError(
            f"轨道裁剪块 {expected.get('slice_id')} 的 curve_change 非法：{curve_change!r}"
        )
    track_type = str(expected.get("track_type") or "")
    # 中文说明：极窄、空白或被网格遮挡的曲线切片允许明确标记不可读，避免把不确定内容强行伪造成数值变化。
    if track_type == "legend" and curve_change != "not_applicable":
        raise ValueError(f"图例切片 {expected.get('slice_id')} 的 curve_change 必须为 not_applicable")
    if track_type == "curve":
        payload["curve_readings"] = _parse_curve_readings(
            payload.get("curve_readings"),
            slice_id=str(expected.get("slice_id") or ""),
        )
        if payload.get("legend_interpretations") not in (None, []):
            raise ValueError(f"曲线切片 {expected.get('slice_id')} 的 legend_interpretations 必须为空")
        payload["legend_interpretations"] = []
    else:
        if payload.get("curve_readings") not in (None, []):
            raise ValueError(f"图例切片 {expected.get('slice_id')} 的 curve_readings 必须为空")
        payload["curve_readings"] = []
        payload["legend_interpretations"] = _parse_legend_interpretations(
            payload.get("legend_interpretations"),
            slice_id=str(expected.get("slice_id") or ""),
            is_lithology_profile=bool(expected.get("is_lithology_profile")),
        )
    payload["description"] = description
    payload["curve_change"] = curve_change
    payload["confidence"] = _bounded_confidence(
        payload.get("confidence"),
        field=f"轨道裁剪块 {expected.get('slice_id')}",
    )
    payload["visual_features"] = [
        str(item).strip()
        for item in payload.get("visual_features", [])
        if str(item).strip()
    ] if isinstance(payload.get("visual_features"), list) else []
    payload["uncertainty"] = str(payload.get("uncertainty") or "").strip()
    return payload


def apply_visual_track_slice_responses(
    payload: Mapping[str, Any],
    responses: Sequence[Mapping[str, Any]],
    *,
    slices: Sequence[Mapping[str, Any]] | None = None,
    response_errors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """中文说明：把 VLM 切片描述和永久缓存绝对路径写回视觉轨道区间，并保留单片失败原因。"""

    enriched = deepcopy(dict(payload))
    geometry = enriched.get("ppstructure_geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("应用轨道裁剪块响应前缺少 ppstructure_geometry")
    resolved_slices = (
        [dict(item) for item in slices if isinstance(item, Mapping)]
        if slices is not None
        else build_adjacent_visual_track_slices(enriched, geometry)
    )
    if resolved_slices and any(
        not str(item.get("cropped_image_cache_path") or "") for item in resolved_slices
    ):
        resolved_slices = cache_visual_track_slices(
            str(geometry.get("source_image_path") or ""),
            resolved_slices,
        )
    expected_by_id = {str(item["slice_id"]): item for item in resolved_slices}
    parsed_by_id: dict[str, dict[str, Any]] = {}
    errors = [str(item) for item in (response_errors or []) if str(item).strip()]
    for raw in responses:
        if not isinstance(raw, Mapping):
            continue
        slice_id = str(raw.get("slice_id") or "")
        expected = expected_by_id.get(slice_id)
        if not expected:
            errors.append(f"未知轨道裁剪块响应：{slice_id or '<empty>'}")
            continue
        if slice_id in parsed_by_id:
            errors.append(f"轨道裁剪块响应重复：{slice_id}")
            continue
        try:
            parsed_by_id[slice_id] = _parse_visual_track_slice_response(raw, expected)
        except ValueError as exc:
            errors.append(str(exc))
    primitives = enriched.get("primitives")
    if not isinstance(primitives, Mapping):
        raise ValueError("应用轨道裁剪块响应前缺少 primitives")
    primitives = deepcopy(dict(primitives))
    enriched["primitives"] = primitives
    curve_tracks = [
        item for item in primitives.get("curve_tracks", []) if isinstance(item, Mapping)
    ]
    curve_observations = [
        dict(item)
        for item in primitives.get("curve_observations", [])
        if isinstance(item, Mapping)
    ]
    track_intervals = [
        dict(item)
        for item in primitives.get("track_intervals", [])
        if isinstance(item, Mapping)
    ]
    existing_ids = {
        str(item.get("id") or "") for item in [*curve_observations, *track_intervals]
    }
    audit_slices: list[dict[str, Any]] = []
    for slice_record in resolved_slices:
        slice_id = str(slice_record["slice_id"])
        response = parsed_by_id.get(slice_id)
        audit = {**slice_record, "status": "completed" if response else "missing"}
        if not response:
            errors.append(f"轨道裁剪块缺少 VLM 描述：{slice_id}")
            audit_slices.append(audit)
            continue
        audit.update(
            {
                "description": response["description"],
                "curve_change": response["curve_change"],
                "curve_readings": response["curve_readings"],
                "legend_interpretations": response["legend_interpretations"],
                "visual_features": response["visual_features"],
                "confidence": response["confidence"],
                "uncertainty": response["uncertainty"],
            }
        )
        audit_slices.append(audit)
        node_id = f"visual_{slice_id}"
        if node_id in existing_ids:
            continue
        existing_ids.add(node_id)
        label = str(slice_record.get("source_label") or slice_record.get("source_cell_id") or "")
        header = str(slice_record.get("track_header") or slice_record.get("track_id") or "")
        common = {
            "id": node_id,
            "track_id": str(slice_record["track_id"]),
            "geometry_refs": [str(slice_record["source_cell_id"])],
            "geometry_bbox": list(slice_record["bbox"]),
            "pixel_range": dict(slice_record["pixel_range"]),
            "top_y": float(slice_record["top_y"]),
            "bottom_y": float(slice_record["bottom_y"]),
            "description": response["description"],
            "visual_features": response["visual_features"],
            "cropped_image_cache_path": str(
                Path(str(slice_record["cropped_image_cache_path"])).resolve()
            ),
            "track_header_image_cache_path": str(
                slice_record.get("track_header_image_cache_path") or ""
            ),
            "vlm_input_image_cache_path": str(
                slice_record.get("vlm_input_image_cache_path")
                or slice_record["cropped_image_cache_path"]
            ),
            "vlm_input_layout": str(slice_record.get("vlm_input_layout") or "slice_only"),
            "source_cell_id": str(slice_record["source_cell_id"]),
            "source_entity_id": str(slice_record["source_entity_id"]),
            "source_track_side": str(slice_record["source_track_side"]),
            "source_track_distance": int(slice_record["source_track_distance"]),
            "source_track_entity_count": int(slice_record["source_track_entity_count"]),
            "adjacent_track_entity_counts": dict(
                slice_record["adjacent_track_entity_counts"]
            ),
            "candidate_table_text_track_entity_counts": dict(
                slice_record["candidate_table_text_track_entity_counts"]
            ),
            "source_track_type": str(slice_record["source_track_type"]),
            "aligned_from_track_id": str(slice_record["source_track_id"]),
            "track_header": header,
            "track_role": str(slice_record.get("track_role") or ""),
            "recognition_stage": int(slice_record["recognition_stage"]),
            "is_lithology_profile": bool(slice_record.get("is_lithology_profile")),
            "recognition_source": "VLM.adjacent_track_slice_description",
            "coordinate_source": str(slice_record["coordinate_source"]),
            "evidence": (
                f"目标轨道 {header} 比较左右最近 table_text 轨道实体数后，按"
                f"{slice_record['source_track_side']}侧文字基准实体 {label} 的纵向范围裁剪，"
                f"VLM 描述为：{response['description']}"
            ),
            "confidence": response["confidence"],
        }
        if str(slice_record.get("track_type")) == "curve":
            curve_ids = [
                str(item.get("id") or "")
                for item in curve_tracks
                if str(item.get("track_id") or "") == str(slice_record["track_id"])
                and str(item.get("id") or "")
            ]
            curve_observations.append(
                {
                    **common,
                    "name": f"{label}对应{header}变化",
                    "curve_ids": curve_ids,
                    "qualitative_response": response["description"],
                    "curve_change": response["curve_change"],
                    "curve_readings": response["curve_readings"],
                    "curve_value_source": "VLM.track_header_and_cropped_slice",
                }
            )
        else:
            legend_types = [
                str(item.get("legend_type") or "unresolved")
                for item in response["legend_interpretations"]
            ]
            track_intervals.append(
                {
                    **common,
                    "name": f"{label}对应{header}：{'、'.join(legend_types)}",
                    "entity_type": "legend_slice_interval",
                    "curve_change": "not_applicable",
                    "legend_type": legend_types[0] if legend_types else "unresolved",
                    "legend_interpretations": response["legend_interpretations"],
                    "legend_type_source": "VLM.track_header_and_cropped_slice",
                    "lithology_types": (
                        legend_types if bool(slice_record.get("is_lithology_profile")) else []
                    ),
                }
            )
    primitives["curve_observations"] = curve_observations
    primitives["track_intervals"] = track_intervals
    visual = enriched.get("visual_track_extraction")
    visual = deepcopy(dict(visual)) if isinstance(visual, Mapping) else {}
    visual["adjacent_slice_description"] = {
        "schema_version": VISUAL_TRACK_SLICE_SCHEMA_VERSION,
        "recognition_mode": "vlm_legend_slice_interpretation_curve_slice_disabled",
        "track_recognition_order": list(TRACK_RECOGNITION_ORDER),
        "vlm_slice_track_types": list(VLM_SLICE_TRACK_TYPES),
        "curve_slice_vlm_enabled": False,
        "vertical_baseline_policy": "nearest_table_text_track_with_more_entities_left_on_tie",
        "cache_directory": str(_visual_track_slice_cache_root()),
        "slice_count": len(resolved_slices),
        "completed_count": len(parsed_by_id),
        "missing_count": max(0, len(resolved_slices) - len(parsed_by_id)),
        "ok": not errors,
        "errors": list(dict.fromkeys(errors)),
        "slices": audit_slices,
    }
    enriched["visual_track_extraction"] = visual
    uncertainties = [
        str(item) for item in enriched.get("uncertainties", [])
    ] if isinstance(enriched.get("uncertainties"), list) else []
    for error in dict.fromkeys(errors):
        uncertainty = f"adjacent_visual_track_slice: {error}"
        if uncertainty not in uncertainties:
            uncertainties.append(uncertainty)
    enriched["uncertainties"] = uncertainties
    return enriched


def describe_adjacent_visual_track_slices(
    task: Any,
    vlm_client: Any,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """中文说明：持久化并调用 VLM 描述 legend 切片，显式跳过所有 curve 轨道切片。"""

    geometry = payload.get("ppstructure_geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("轨道裁剪描述缺少 ppstructure_geometry")
    slices = build_adjacent_visual_track_slices(payload, geometry)
    if not slices:
        return apply_visual_track_slice_responses(payload, [])
    image_path = Path(
        str(getattr(task, "image_path", "") or geometry.get("source_image_path") or "")
    ).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"轨道裁剪来源图片不存在：{image_path}")
    responses: list[dict[str, Any]] = []
    response_errors: list[str] = []
    cached_slices = [
        item
        for item in cache_visual_track_slices(
            image_path,
            [
                item
                for item in slices
                if str(item.get("track_type") or "") in VLM_SLICE_TRACK_TYPES
            ],
        )
        if str(item.get("track_type") or "") in VLM_SLICE_TRACK_TYPES
    ]
    max_tokens = int(os.getenv("STRATIGRAPHIC_TABLE_VISUAL_SLICE_MAX_TOKENS", "4096"))
    for cached_slice in cached_slices:
        # 中文说明：双重校验切片类型，即使上游或旧缓存混入 curve 记录也不会发起 VLM 请求。
        if str(cached_slice.get("track_type") or "") not in VLM_SLICE_TRACK_TYPES:
            continue
        crop_path = str(
            cached_slice.get("vlm_input_image_cache_path")
            or cached_slice["cropped_image_cache_path"]
        )
        response = vlm_client.describe_image(
            crop_path,
            build_visual_track_slice_prompt(cached_slice),
            task_name=(
                f"表格嵌入混合专用OCR:track_slice:{cached_slice['track_type']}:"
                f"{cached_slice['slice_id']}:"
                f"y{cached_slice['top_y']}-{cached_slice['bottom_y']}:"
                f"{getattr(task, 'image_id', '')}"
            ),
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            timeout=table_embedded_hybrid_vlm_timeout_secs(),
        )
        try:
            responses.append(_parse_visual_track_slice_response(response, cached_slice))
        except ValueError as first_error:
            # 中文说明：同一切片只额外重试一次，避免截断 JSON 阻断整图，又不把模型偶发错误无限放大为重复请求。
            retry_response = vlm_client.describe_image(
                crop_path,
                build_visual_track_slice_retry_prompt(cached_slice, first_error),
                task_name=(
                    f"表格嵌入混合专用OCR:track_slice_retry:{cached_slice['track_type']}:"
                    f"{cached_slice['slice_id']}:"
                    f"y{cached_slice['top_y']}-{cached_slice['bottom_y']}:"
                    f"{getattr(task, 'image_id', '')}"
                ),
                response_format={"type": "json_object"},
                max_tokens=min(max_tokens, 2048),
                timeout=table_embedded_hybrid_vlm_timeout_secs(),
            )
            try:
                responses.append(_parse_visual_track_slice_response(retry_response, cached_slice))
            except ValueError as retry_error:
                # 中文说明：单个低清或空白切片两次格式失败时记录审计错误并继续，不能阻断后续轨道切片。
                response_errors.extend([str(first_error), str(retry_error)])
                continue
    return apply_visual_track_slice_responses(
        payload,
        responses,
        slices=cached_slices,
        response_errors=response_errors,
    )


def _primitive_cell_ids(primitives: Mapping[str, Any], geometry: Mapping[str, Any]) -> set[str]:
    """中文说明：收集已被类型化图元消费的单元格，避免通用轨道节点重复建点。"""

    ocr_to_cell = {
        str(item.get("id") or ""): str(item.get("cell_id") or "")
        for item in geometry.get("ocr_lines", [])
        if isinstance(item, Mapping) and item.get("id")
    }
    consumed: set[str] = set()
    for values in primitives.values():
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, Mapping):
                continue
            refs = item.get("geometry_refs")
            if not isinstance(refs, list):
                continue
            for value in refs:
                ref = str(value or "")
                if ref.startswith("pp_cell_") or ref in {
                    str(cell.get("id") or "") for cell in geometry.get("cells", []) if isinstance(cell, Mapping)
                }:
                    consumed.add(ref)
                elif ocr_to_cell.get(ref):
                    consumed.add(ocr_to_cell[ref])
    return consumed


def _cell_confidence(cell: Mapping[str, Any], geometry: Mapping[str, Any]) -> float:
    """中文说明：以单元格内 OCR 置信度的保守下界作为自动轨道节点置信度。"""

    by_id = {
        str(item.get("id") or ""): item
        for item in geometry.get("ocr_lines", [])
        if isinstance(item, Mapping) and item.get("id")
    }
    values = []
    for ref in cell.get("ocr_ids", []) if isinstance(cell.get("ocr_ids"), list) else []:
        try:
            values.append(float(by_id.get(str(ref), {}).get("confidence")))
        except (TypeError, ValueError):
            pass
    return round(max(0.45, min(values) if values else 0.72), 3)


def _fallback_entity_type(role: str) -> str:
    """中文说明：将轨道角色映射为保守的通用区间节点类型。"""

    return {
        "stratigraphy": "stratigraphic_label_interval",
        "lithology": "lithology_description_interval",
        "facies": "sedimentary_facies_label_interval",
        "reservoir": "reservoir_description_interval",
        "well": "well_annotation_interval",
        "porosity": "porosity_annotation_interval",
        "permeability": "permeability_annotation_interval",
        "text": "track_text_interval",
    }.get(role, "track_interval")


def _meaningful_fallback_text(value: Any, role: str) -> bool:
    """中文说明：只把具有地质标签信息量的 OCR 文本升级为兜底实体，拒绝线条误识别出的数字、标点和短字母。"""

    text = _clean_text(value)
    if not text:
        return False
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if len(cjk_chars) >= 2:
        return True
    if len(cjk_chars) == 1:
        return cjk_chars[0] in _SINGLE_CJK_GEOLOGICAL_LABELS
    compact = re.sub(r"[^0-9A-Za-z]+", "", text)
    # 中文说明：O1m、T3x、W1 等地层代号或井号可保留，H、ANV、1 等孤立 OCR 噪声不建实体。
    return (
        role in {"stratigraphy", "well"}
        and len(compact) >= 2
        and bool(re.search(r"[A-Za-z]", compact))
        and bool(re.search(r"\d", compact))
    )


def _unconsumed_track_intervals(
    primitives: Mapping[str, Any],
    geometry: Mapping[str, Any],
    tracks: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """中文说明：为每个语义轨道补齐未被 VLM 类型化的非空文本单元格。"""

    consumed = _primitive_cell_ids(primitives, geometry)
    records: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen_cells: set[str] = set()
    for track in tracks:
        track_id = str(track.get("id") or "")
        role = str(track.get("role") or "unknown")
        fallback_enabled = role in _FALLBACK_TEXT_ENTITY_ROLES
        body_top = _body_top(track, geometry)
        total = existing = fallback = rejected_noise = 0
        if role not in {"depth", "curve"}:
            for cell in geometry.get("cells", []):
                if not isinstance(cell, Mapping):
                    continue
                cell_id = str(cell.get("id") or "")
                box = _bbox(cell)
                owner = _track_for_cell(cell, tracks)
                text = str(cell.get("refined_text") or cell.get("text") or "").strip()
                if (
                    not cell_id
                    or cell_id in seen_cells
                    or not box
                    or not owner
                    or str(owner.get("id") or "") != track_id
                    or box[1] < body_top - 2.0
                    or not _clean_text(text)
                ):
                    continue
                total += 1
                seen_cells.add(cell_id)
                if cell_id in consumed:
                    existing += 1
                    continue
                if not fallback_enabled or not _meaningful_fallback_text(text, role):
                    rejected_noise += 1
                    continue
                fallback += 1
                safe_id = re.sub(r"[^0-9A-Za-z_]+", "_", f"track_{track_id}_{cell_id}")
                records.append(
                    {
                        "id": safe_id,
                        "name": text,
                        "entity_type": _fallback_entity_type(role),
                        "track_id": track_id,
                        "geometry_refs": [cell_id],
                        "recognition_source": "PP-StructureV3.refined_cell_text",
                        "evidence": f"{track.get('header') or track_id} 轨道单元格可见文字：{text}",
                        "confidence": _cell_confidence(cell, geometry),
                    }
                )
        audit.append(
            {
                "track_id": track_id,
                "role": role,
                "header": str(track.get("header") or ""),
                "body_top": round(body_top, 3),
                "text_cell_count": total,
                "typed_entity_cell_count": existing,
                "fallback_entity_cell_count": fallback,
                "rejected_ocr_cell_count": rejected_noise,
                "fallback_entity_enabled": fallback_enabled,
                "fallback_skipped": role not in {"depth", "curve"} and not fallback_enabled,
                "ocr_noise_only": (
                    fallback_enabled
                    and total > 0
                    and existing == 0
                    and fallback == 0
                    and rejected_noise == total
                ),
                "covered": (
                    role in {"depth", "curve"}
                    or not fallback_enabled
                    or total == existing + fallback + rejected_noise
                ),
            }
        )
    return records, {
        "track_count": len(tracks),
        "tracks": audit,
        "uncovered_track_ids": [str(item["track_id"]) for item in audit if not item["covered"]],
    }


def _canonical_curve_name(value: Any) -> str:
    """中文说明：统一中英文曲线别名，避免 VLM 定义与表头兜底生成重复曲线。"""

    text = _clean_text(value).casefold()
    compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
    for canonical, aliases in _CURVE_NAME_ALIASES.items():
        if any(alias.casefold() in compact for alias in aliases):
            return canonical.casefold()
    upper = _clean_text(value).upper()
    components = [part for part in re.split(r"[^0-9A-Z\u4e00-\u9fff]+", upper) if part]
    for token, *_ in _CURVE_STYLE_RULES[:7]:
        if token in components or upper == token or upper.startswith(f"{token}曲线") or upper.endswith(token):
            return token.casefold()
    return compact


def _style_for_curve(name: Any) -> tuple[str, str, str]:
    """中文说明：按曲线规范名补全颜色、图元形态和横轴变换。"""

    canonical = _canonical_curve_name(name)
    for token, color, visual_form, transform in _CURVE_STYLE_RULES:
        if canonical == token.casefold():
            return color, visual_form, transform
    return "unknown", "continuous_curve", "linear"


def _parse_number(value: Any) -> float | None:
    """中文说明：从单个 OCR 刻度文本中安全读取有限数值。"""

    matched = _NUMBER_PATTERN.search(str(value or "").replace(",", ""))
    if not matched:
        return None
    try:
        number = float(matched.group(0))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _curve_descriptors(
    primitives: dict[str, Any],
    tracks: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    physical_to_semantic: Mapping[str, str],
) -> list[dict[str, Any]]:
    """中文说明：规范 VLM 曲线定义，并从明确表头补充被漏掉的常见曲线。"""

    raw_items = primitives.get("curve_tracks")
    descriptors = [dict(item) for item in raw_items if isinstance(item, Mapping)] if isinstance(raw_items, list) else []
    track_by_id = {str(item.get("id") or ""): item for item in tracks}
    for item in descriptors:
        raw_track_id = str(item.get("track_id") or "")
        item["track_id"] = physical_to_semantic.get(raw_track_id, raw_track_id)
        color, visual_form, transform = _style_for_curve(item.get("name"))
        # 中文说明：VLM 明确返回 unknown/空值时仍采用名称规则，避免可见曲线因样式字段缺省被丢弃。
        if str(item.get("color") or "unknown").casefold() == "unknown":
            item["color"] = color
        if not item.get("visual_form"):
            item["visual_form"] = visual_form
        if not item.get("scale_transform"):
            item["scale_transform"] = transform
        if item.get("left_value") is None and item.get("scale_min") is not None:
            item["left_value"] = item.get("scale_min")
        if item.get("right_value") is None and item.get("scale_max") is not None:
            item["right_value"] = item.get("scale_max")
    known = {_canonical_curve_name(item.get("name")) for item in descriptors}
    for track in tracks:
        role = str(track.get("role") or "")
        if role not in {"curve", "porosity", "permeability"}:
            continue
        header = " / ".join(
            str(value or "")
            for value in (track.get("header"), track.get("semantic_header_text"))
            if str(value or "")
        )
        for token, color, visual_form, transform in _CURVE_STYLE_RULES:
            if token.casefold() not in _clean_text(header).casefold() or token.casefold() in known:
                continue
            curve_id = re.sub(r"[^0-9A-Za-z_]+", "_", f"curve_{track.get('id')}_{token}")
            descriptors.append(
                {
                    "id": curve_id,
                    "name": token,
                    "track_id": str(track.get("id") or ""),
                    "scale_min": None,
                    "scale_max": None,
                    "left_value": None,
                    "right_value": None,
                    "unit": "",
                    "color": color,
                    "visual_form": visual_form,
                    "scale_transform": transform,
                    "evidence": f"语义轨道表头可见曲线名：{token}",
                    "confidence": 0.82,
                    "recognition_source": "deterministic_track_header",
                }
            )
            known.add(token.casefold())
    for item in descriptors:
        if item.get("left_value") is not None and item.get("right_value") is not None:
            continue
        name = _clean_text(item.get("name"))
        track = track_by_id.get(str(item.get("track_id") or ""))
        if not track:
            continue
        track_box = _bbox(track)
        header_bottom = _body_top(track, geometry)
        label_lines = [
            line
            for line in geometry.get("ocr_lines", [])
            if isinstance(line, Mapping)
            and name.casefold() in _clean_text(line.get("text")).casefold()
            and _bbox(line)
            and track_box[0] <= (_bbox(line)[0] + _bbox(line)[2]) / 2.0 <= track_box[2]
            and _bbox(line)[3] <= header_bottom + 3
        ]
        if not label_lines:
            continue
        label_box = _bbox(label_lines[0])
        label_y = (label_box[1] + label_box[3]) / 2.0
        numeric = []
        for line in geometry.get("ocr_lines", []):
            if not isinstance(line, Mapping) or not _bbox(line):
                continue
            box = _bbox(line)
            center_x, center_y = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            value = _parse_number(line.get("text"))
            if value is None or not (track_box[0] - 5 <= center_x <= track_box[2] + 5):
                continue
            if abs(center_y - label_y) <= max(18.0, label_box[3] - label_box[1]):
                numeric.append((center_x, value, str(line.get("id") or "")))
        if len(numeric) >= 2:
            numeric.sort()
            item["left_value"] = numeric[0][1]
            item["right_value"] = numeric[-1][1]
            item["scale_ocr_ids"] = [numeric[0][2], numeric[-1][2]]
    primitives["curve_tracks"] = descriptors
    return descriptors


def _color_mask(array: Any, color: str) -> Any:
    """中文说明：用通道差分离红、蓝、绿和青色细曲线，避免灰色网格进入轨迹。"""

    import numpy as np

    red = array[:, :, 0].astype(np.int16)
    green = array[:, :, 1].astype(np.int16)
    blue = array[:, :, 2].astype(np.int16)
    if color == "red":
        return (red - green >= 35) & (red - blue >= 35) & (red >= 100)
    if color == "green":
        return (green - red >= 25) & (green - blue >= 8) & (green >= 80)
    if color == "cyan":
        return (blue - red >= 25) & (green - red >= 15) & (blue >= 90)
    if color == "blue":
        return (blue - red >= 30) & (blue - green >= 5) & (blue >= 90)
    return (red <= 85) & (green <= 85) & (blue <= 85)


def _trace_colored_curve(mask: Any, visual_form: str) -> list[tuple[int, int]]:
    """中文说明：逐行跟踪颜色骨架，对断线优先选择与上一行连续的分量。"""

    import numpy as np

    points: list[tuple[int, int]] = []
    previous_x: float | None = None
    for y in range(mask.shape[0]):
        xs = np.flatnonzero(mask[y])
        if xs.size == 0:
            continue
        groups = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
        candidates = [((float(group[0]) + float(group[-1])) / 2.0, len(group)) for group in groups if len(group)]
        if not candidates:
            continue
        if visual_form in {"filled_profile", "sample_bars"}:
            chosen_x = max(float(group[-1]) for group in groups if len(group))
        elif previous_x is None:
            chosen_x = max(candidates, key=lambda item: item[1])[0]
        else:
            chosen_x = min(candidates, key=lambda item: (abs(item[0] - previous_x), -item[1]))[0]
        previous_x = chosen_x
        points.append((round(chosen_x), y))
    return points


def _trace_black_bars(mask: Any) -> list[tuple[int, int]]:
    """中文说明：用水平形态学提取黑色岩心离散测量棒，不把它们误连成连续曲线。"""

    import cv2
    import numpy as np

    source = (mask.astype(np.uint8) * 255)
    kernel_width = max(4, round(mask.shape[1] * 0.045))
    horizontal = cv2.morphologyEx(
        source,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(horizontal, 8)
    points = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if width < kernel_width or height > max(7, round(mask.shape[0] * 0.03)) or area < kernel_width:
            continue
        points.append((x + width - 1, round(float(centroids[index][1]))))
    return sorted(points, key=lambda item: item[1])


def _axis_value(descriptor: Mapping[str, Any], normalized_x: float) -> float | None:
    """中文说明：按线性或对数横轴把曲线像素 x 换算为测井值。"""

    try:
        left = float(descriptor.get("left_value"))
        right = float(descriptor.get("right_value"))
    except (TypeError, ValueError):
        return None
    position = max(0.0, min(1.0, float(normalized_x)))
    if str(descriptor.get("scale_transform") or "linear") == "log10":
        if left <= 0 or right <= 0:
            return None
        return 10 ** (math.log10(left) + position * (math.log10(right) - math.log10(left)))
    return left + position * (right - left)


def _curve_traces(
    image_path: Path,
    descriptors: Sequence[Mapping[str, Any]],
    tracks: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """中文说明：在各语义曲线 ROI 内输出可审计像素采样、刻度值和连续覆盖率。"""

    try:
        import cv2
        import numpy as np
    except ImportError:
        return [], {"available": False, "reason": "opencv_or_numpy_unavailable"}
    if not image_path.is_file():
        return [], {"available": False, "reason": f"source_image_not_found:{image_path}"}
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    track_by_id = {str(item.get("id") or ""): item for item in tracks}
    horizontal_lines = [int(round(float(value))) for value in ((geometry.get("rule_lines") or {}).get("horizontal_lines") or [])]
    traces: list[dict[str, Any]] = []
    rejected = []
    max_samples = max(32, int(os.getenv("TABLE_CURVE_TRACE_MAX_SAMPLES", "320")))
    min_coverage = float(os.getenv("TABLE_CURVE_TRACE_MIN_COVERAGE", "0.025"))
    for descriptor in descriptors:
        track = track_by_id.get(str(descriptor.get("track_id") or ""))
        color = str(descriptor.get("color") or "unknown").casefold()
        if not track or color == "unknown":
            rejected.append({"curve_id": descriptor.get("id"), "reason": "missing_track_or_color"})
            continue
        box = [int(round(value)) for value in _bbox(track)]
        body_top = max(box[1], int(round(_body_top(track, geometry))))
        left, right = max(0, box[0] + 1), min(image.shape[1], box[2] - 1)
        top, bottom = max(0, body_top + 1), min(image.shape[0], box[3] - 1)
        if right - left < 4 or bottom - top < 8:
            rejected.append({"curve_id": descriptor.get("id"), "reason": "empty_curve_roi"})
            continue
        roi = image[top:bottom, left:right]
        mask = _color_mask(roi, color)
        if color == "black":
            for line_y in horizontal_lines:
                local_y = line_y - top
                if 0 <= local_y < mask.shape[0]:
                    mask[max(0, local_y - 1) : min(mask.shape[0], local_y + 2), :] = False
            points = _trace_black_bars(mask)
        else:
            binary = (mask.astype(np.uint8) * 255)
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
            )
            points = _trace_colored_curve(binary > 0, str(descriptor.get("visual_form") or "continuous_curve"))
        coverage = len({point[1] for point in points}) / max(1, bottom - top)
        if not points or coverage < min_coverage:
            rejected.append(
                {
                    "curve_id": descriptor.get("id"),
                    "reason": "insufficient_colored_pixels",
                    "coverage": round(coverage, 4),
                }
            )
            continue
        stride = max(1, math.ceil(len(points) / max_samples))
        selected = points[::stride]
        samples = []
        for local_x, local_y in selected:
            pixel_x, pixel_y = left + local_x, top + local_y
            normalized_x = (pixel_x - left) / max(1.0, right - left - 1)
            value = _axis_value(descriptor, normalized_x)
            samples.append(
                {
                    "pixel_x": round(float(pixel_x), 3),
                    "pixel_y": round(float(pixel_y), 3),
                    "normalized_x": round(normalized_x, 6),
                    "axis_value": round(value, 6) if value is not None else None,
                }
            )
        traces.append(
            {
                "id": f"trace_{descriptor.get('id')}",
                "curve_id": str(descriptor.get("id") or ""),
                "name": str(descriptor.get("name") or descriptor.get("id") or "曲线"),
                "track_id": str(descriptor.get("track_id") or ""),
                "color": color,
                "visual_form": str(descriptor.get("visual_form") or "continuous_curve"),
                "scale_transform": str(descriptor.get("scale_transform") or "linear"),
                "left_value": descriptor.get("left_value"),
                "right_value": descriptor.get("right_value"),
                "unit": str(descriptor.get("unit") or ""),
                "roi_bbox": [left, top, right, bottom],
                "samples": samples,
                "raw_point_count": len(points),
                "trace_coverage": round(coverage, 4),
                "confidence": round(min(0.98, 0.58 + min(0.4, coverage)), 3),
                "coordinate_source": "OpenCV.color_or_bar_trace",
                "evidence": f"{descriptor.get('name') or descriptor.get('id')} 在轨道 ROI 中的{color}像素轨迹",
            }
        )
    return traces, {
        "available": True,
        "curve_descriptor_count": len(descriptors),
        "trace_count": len(traces),
        "rejected": rejected,
    }


def _legend_label_candidates(primitives: Mapping[str, Any], geometry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """中文说明：优先使用 VLM 选择的图例文字，否则从主表外的岩性 OCR 自动召回。"""

    ocr_by_id = {
        str(item.get("id") or ""): item
        for item in geometry.get("ocr_lines", [])
        if isinstance(item, Mapping) and item.get("id")
    }
    candidates: list[dict[str, Any]] = []
    raw_entries = primitives.get("legend_entries")
    if isinstance(raw_entries, list):
        for index, item in enumerate(raw_entries):
            if not isinstance(item, Mapping):
                continue
            refs = item.get("geometry_refs")
            refs = [str(value) for value in refs] if isinstance(refs, list) else []
            label = next((ocr_by_id[value] for value in refs if value in ocr_by_id), None)
            if label:
                candidates.append(
                    {
                        "id": str(item.get("id") or f"legend_{index:03d}"),
                        "name": str(item.get("name") or label.get("text") or ""),
                        "label_ocr_id": str(label.get("id") or ""),
                        "label_bbox": list(label.get("bbox") or []),
                    }
                )
    if candidates:
        return candidates
    content = geometry.get("content_bbox")
    content_bottom = float(content[3]) if isinstance(content, list) and len(content) == 4 else 0.0
    for line in geometry.get("ocr_lines", []):
        if not isinstance(line, Mapping) or not _bbox(line):
            continue
        text = str(line.get("text") or "").strip()
        box = _bbox(line)
        if not text or not any(keyword in text for keyword in _ROCK_KEYWORDS) or box[1] < content_bottom - 2:
            continue
        candidates.append(
            {
                "id": f"legend_{len(candidates):03d}",
                "name": text,
                "label_ocr_id": str(line.get("id") or ""),
                "label_bbox": list(line.get("bbox") or []),
            }
        )
    return candidates


def _rectangle_candidates(gray: Any) -> list[list[int]]:
    """中文说明：检测图例文字左侧的矩形纹理样方外框。"""

    import cv2

    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[int]] = []
    height, width = gray.shape
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 16 or box_height < 10 or box_width / max(1, box_height) > 5.5:
            continue
        if box_width > width * 0.35 or box_height > height * 0.2:
            continue
        top = binary[y : y + 2, x : x + box_width]
        bottom = binary[y + box_height - 2 : y + box_height, x : x + box_width]
        left = binary[y : y + box_height, x : x + 2]
        right = binary[y : y + box_height, x + box_width - 2 : x + box_width]
        border_support = sum(float(part.mean() > 45) for part in (top, bottom, left, right))
        if border_support >= 3:
            boxes.append([x, y, x + box_width, y + box_height])
    return boxes


def _pattern_feature(gray_crop: Any) -> tuple[Any, float]:
    """中文说明：将纹理归一化为低分辨率像素、方向直方图和投影联合特征。"""

    import cv2
    import numpy as np

    if gray_crop.size == 0:
        return np.asarray([], dtype=np.float32), 0.0
    binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    pad_y = min(max(1, binary.shape[0] // 12), max(1, binary.shape[0] // 3))
    pad_x = min(max(1, binary.shape[1] // 12), max(1, binary.shape[1] // 3))
    inner = binary[pad_y : binary.shape[0] - pad_y, pad_x : binary.shape[1] - pad_x]
    if inner.size == 0:
        inner = binary
    ink_density = float((inner > 0).mean())
    small = cv2.resize(inner, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gx = cv2.Sobel(inner, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(inner, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = ((angle % 180.0) / 22.5).astype(np.int32)
    orientation = np.asarray(
        [float(magnitude[bins == index].sum()) for index in range(8)],
        dtype=np.float32,
    )
    row_projection = cv2.resize(inner.mean(axis=1).reshape(-1, 1), (1, 8)).reshape(-1) / 255.0
    col_projection = cv2.resize(inner.mean(axis=0).reshape(1, -1), (8, 1)).reshape(-1) / 255.0
    feature = np.concatenate([small.reshape(-1), orientation, row_projection, col_projection]).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    return (feature / norm if norm > 0 else feature), ink_density


def _legend_pattern_matches(
    image_path: Path,
    primitives: Mapping[str, Any],
    geometry: Mapping[str, Any],
    tracks: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """中文说明：将岩性剖面单元格与已标名图例样方比较，只把高置信匹配写成岩性事实。"""

    try:
        import cv2
        import numpy as np
    except ImportError:
        return [], [], [], {"available": False, "reason": "opencv_or_numpy_unavailable"}
    labels = _legend_label_candidates(primitives, geometry)
    pattern_tracks = [
        track
        for track in tracks
        if str(track.get("role") or "") == "lithology"
        and (
            "pattern" in str(track.get("parser") or "").casefold()
            or "剖面" in str(track.get("header") or "")
            or "柱" in str(track.get("header") or "")
        )
    ]
    if not image_path.is_file() or not labels or not pattern_tracks:
        return [], [], labels, {
            "available": bool(image_path.is_file()),
            "legend_entry_count": len(labels),
            "pattern_track_count": len(pattern_tracks),
            "reason": "legend_or_pattern_track_missing",
        }
    with Image.open(image_path) as source:
        gray = np.asarray(source.convert("L"))
    rectangles = _rectangle_candidates(gray)
    prototypes: list[dict[str, Any]] = []
    for label in labels:
        label_box = [float(value) for value in label.get("label_bbox", [])]
        if len(label_box) != 4:
            continue
        label_height = max(1.0, label_box[3] - label_box[1])
        candidates = []
        for box in rectangles:
            gap = label_box[0] - box[2]
            center_gap = abs((box[1] + box[3]) / 2.0 - (label_box[1] + label_box[3]) / 2.0)
            if -5 <= gap <= max(48.0, label_height * 4.0) and center_gap <= max(label_height, box[3] - box[1]):
                candidates.append((gap + center_gap, box))
        if not candidates:
            continue
        swatch = min(candidates, key=lambda item: item[0])[1]
        feature, density = _pattern_feature(gray[swatch[1] : swatch[3], swatch[0] : swatch[2]])
        if feature.size == 0:
            continue
        prototypes.append({**label, "swatch_bbox": swatch, "feature": feature, "ink_density": density})
    min_score = float(os.getenv("TABLE_LEGEND_MATCH_MIN_SCORE", "0.72"))
    min_margin = float(os.getenv("TABLE_LEGEND_MATCH_MIN_MARGIN", "0.035"))
    matched: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    seen_cells: set[str] = set()
    for track in pattern_tracks:
        body_top = _body_top(track, geometry)
        for cell in geometry.get("cells", []):
            if not isinstance(cell, Mapping) or not _bbox(cell) or _horizontal_overlap(cell, track) < 0.45:
                continue
            cell_id = str(cell.get("id") or "")
            box = [int(round(value)) for value in _bbox(cell)]
            if not cell_id or cell_id in seen_cells or box[1] < body_top - 2:
                continue
            seen_cells.add(cell_id)
            feature, density = _pattern_feature(gray[box[1] : box[3], box[0] : box[2]])
            if feature.size == 0 or density < 0.008:
                continue
            scores = sorted(
                [
                    (float(np.dot(feature, prototype["feature"])), prototype)
                    for prototype in prototypes
                    if feature.shape == prototype["feature"].shape
                ],
                key=lambda item: item[0],
                reverse=True,
            )
            if not scores:
                continue
            best_score, best = scores[0]
            second_score = scores[1][0] if len(scores) > 1 else 0.0
            margin = best_score - second_score
            common = {
                "id": f"legend_pattern_{track.get('id')}_{cell_id}",
                "track_id": str(track.get("id") or ""),
                "geometry_refs": [cell_id],
                "legend_entry_id": str(best.get("id") or ""),
                "pattern_similarity": round(best_score, 4),
                "pattern_match_margin": round(margin, 4),
                "candidate_names": [str(item[1].get("name") or "") for item in scores[:3]],
                "recognition_source": "OpenCV.legend_pattern_similarity",
            }
            if best_score >= min_score and margin >= min_margin:
                matched.append(
                    {
                        **common,
                        "name": str(best.get("name") or "未命名岩性"),
                        "evidence": (
                            f"剖面单元格 {cell_id} 与图例 {best.get('name')} 纹理相似度 "
                            f"{best_score:.3f}，与次优候选差 {margin:.3f}"
                        ),
                        "confidence": round(min(0.96, 0.55 + best_score * 0.4), 3),
                    }
                )
            else:
                uncertain.append(
                    {
                        **common,
                        "name": "未判定岩性纹理",
                        "entity_type": "unresolved_lithology_pattern_interval",
                        "match_status": "uncertain",
                        "evidence": (
                            f"剖面单元格 {cell_id} 图例匹配未过门限："
                            f"score={best_score:.3f}, margin={margin:.3f}"
                        ),
                        "confidence": round(max(0.3, min(0.69, best_score)), 3),
                    }
                )
    public_entries = [
        {
            key: value
            for key, value in item.items()
            if key != "feature"
        }
        for item in prototypes
    ]
    return matched, uncertain, public_entries, {
        "available": True,
        "legend_entry_count": len(public_entries),
        "pattern_interval_count": len(matched) + len(uncertain),
        "matched_interval_count": len(matched),
        "uncertain_interval_count": len(uncertain),
        "minimum_score": min_score,
        "minimum_margin": min_margin,
    }


def enrich_visual_track_primitives(
    payload: Mapping[str, Any],
    geometry: Mapping[str, Any],
    *,
    physical_to_semantic: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """中文说明：只做语义轨道映射和文本兜底，不再用本地视觉算法解释曲线或图例。"""

    enriched = deepcopy(dict(payload))
    primitives = enriched.get("primitives")
    if not isinstance(primitives, Mapping):
        return enriched
    primitives = deepcopy(dict(primitives))
    enriched["primitives"] = primitives
    tracks = [dict(item) for item in enriched.get("tracks", []) if isinstance(item, Mapping)]
    mapping = dict(physical_to_semantic or {})
    for values in primitives.values():
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict):
                raw_track_id = str(item.get("track_id") or "")
                if raw_track_id in mapping:
                    item["track_id"] = mapping[raw_track_id]
    track_intervals = primitives.get("track_intervals")
    track_intervals = [dict(item) for item in track_intervals if isinstance(item, Mapping)] if isinstance(track_intervals, list) else []
    track_by_id = {str(item.get("id") or ""): item for item in tracks}
    # 中文说明：恢复旧中间结果时同步清除此前由宽松 PP 文本兜底生成的测量轨道噪声节点。
    track_intervals = [
        item
        for item in track_intervals
        if not (
            str(item.get("recognition_source") or "") == "PP-StructureV3.refined_cell_text"
            and (
                str(track_by_id.get(str(item.get("track_id") or ""), {}).get("role") or "unknown")
                not in _FALLBACK_TEXT_ENTITY_ROLES
                or not _meaningful_fallback_text(
                    item.get("name"),
                    str(track_by_id.get(str(item.get("track_id") or ""), {}).get("role") or "unknown"),
                )
            )
        )
    ]
    primitives["track_intervals"] = track_intervals
    fallback_intervals, coverage = _unconsumed_track_intervals(primitives, geometry, tracks)
    known_track_interval_ids = {str(item.get("id") or "") for item in track_intervals}
    for item in fallback_intervals:
        if str(item.get("id") or "") not in known_track_interval_ids:
            track_intervals.append(item)
            known_track_interval_ids.add(str(item.get("id") or ""))
    primitives["track_intervals"] = track_intervals
    # 中文说明：曲线像素追踪和图例纹理匹配暂时关闭，视觉轨道只接受后续 VLM 切片描述。
    primitives["curve_traces"] = []
    primitives["legend_entries"] = []
    enriched["visual_track_extraction"] = {
        "version": VISUAL_TRACK_EXTRACTION_VERSION,
        "recognition_mode": "vlm_legend_slice_interpretation_curve_slice_disabled",
        "track_recognition_order": list(TRACK_RECOGNITION_ORDER),
        "vlm_slice_track_types": list(VLM_SLICE_TRACK_TYPES),
        "curve_slice_vlm_enabled": False,
        "vertical_baseline_policy": "nearest_table_text_track_with_more_entities_left_on_tie",
        "coordinate_source": "PP-StructureV3_crop_geometry",
        "curve": {
            "available": False,
            "reason": "local_curve_vectorization_disabled",
        },
        "legend": {
            "available": False,
            "reason": "local_legend_pattern_matching_disabled",
        },
        "track_entity_coverage": coverage,
    }
    return enriched
