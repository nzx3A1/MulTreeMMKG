"""使用 PaddleOCR PP-StructureV3 生成表格混合图的可信像素几何。"""
from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from PIL import Image

from src.utils.json_io import read_json, write_json

PPSTRUCTURE_GEOMETRY_SCHEMA_VERSION = "ppstructurev3.table_geometry.v1"
TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION = "table_embedded_hybrid.v1"
INTERVAL_FIELDS = (
    "stratigraphic_intervals",
    "reference_intervals",
    "lithology_intervals",
    "facies_intervals",
    "reservoir_intervals",
    "oil_layer_intervals",
    "geological_feature_intervals",
    "curve_observations",
)
_GEOMETRY_CACHE: dict[str, dict[str, Any]] = {}
_PIPELINE_CACHE: dict[tuple[str, str, str, bool], Any] = {}
_NUMERIC_TEXT = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:m|米)?$", re.IGNORECASE)


def _as_bool(value: str | None, default: bool) -> bool:
    """中文说明：把环境变量中的常见真假写法转换为布尔值。"""

    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _to_python(value: Any) -> Any:
    """中文说明：递归把 NumPy/Paddle 结果转换为可缓存的标准 Python 数据。"""

    if isinstance(value, Mapping):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_python(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _finite_float(value: Any) -> float:
    """中文说明：读取 PP-StructureV3 坐标并拒绝 NaN 或无穷值。"""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"PP-StructureV3 返回非有限数值：{value!r}")
    return number


def _box(value: Any, width: int, height: int) -> list[int]:
    """中文说明：把两点框或四点多边形统一裁剪为原图左上右下像素框。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    raw = list(value)
    if len(raw) == 4 and all(not isinstance(item, Sequence) for item in raw):
        x0, y0, x1, y1 = (_finite_float(item) for item in raw)
    else:
        points: list[tuple[float, float]] = []
        if len(raw) == 8 and all(not isinstance(item, Sequence) for item in raw):
            points = [(_finite_float(raw[index]), _finite_float(raw[index + 1])) for index in range(0, 8, 2)]
        else:
            for point in raw:
                if isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) >= 2:
                    points.append((_finite_float(point[0]), _finite_float(point[1])))
        if not points:
            return []
        x0, x1 = min(point[0] for point in points), max(point[0] for point in points)
        y0, y1 = min(point[1] for point in points), max(point[1] for point in points)
    clipped = [
        max(0, min(width, round(x0))),
        max(0, min(height, round(y0))),
        max(0, min(width, round(x1))),
        max(0, min(height, round(y1))),
    ]
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else []


def _union_box(boxes: Sequence[Sequence[int]]) -> list[int]:
    """中文说明：计算一组有效像素框的最小外接矩形。"""

    valid = [list(box) for box in boxes if len(box) == 4]
    if not valid:
        return []
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def _area(box: Sequence[int]) -> int:
    """中文说明：计算像素框面积，用于选择主表格和最小包含单元格。"""

    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1])) if len(box) == 4 else 0


def _center(box: Sequence[int]) -> tuple[float, float]:
    """中文说明：返回像素框中心，作为 OCR 文本归属单元格和轨道的判据。"""

    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _cluster(values: Sequence[int], tolerance: int) -> list[dict[str, Any]]:
    """中文说明：把相邻表格边界聚成一条中心线并保留支持次数。"""

    if not values:
        return []
    groups: list[list[int]] = []
    for value in sorted(int(item) for item in values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [
        {"position": round(sum(group) / len(group)), "support": len(group)}
        for group in groups
    ]


def _ocr_lines(raw: Mapping[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    """中文说明：从 overall_ocr_res 提取带稳定 ID、置信度和原图框的文本行。"""

    overall = raw.get("overall_ocr_res")
    overall = overall if isinstance(overall, Mapping) else {}
    texts = list(overall.get("rec_texts") or [])
    scores = list(overall.get("rec_scores") or [])
    boxes = list(overall.get("rec_boxes") or overall.get("rec_polys") or [])
    lines: list[dict[str, Any]] = []
    for index, (text, raw_box) in enumerate(zip(texts, boxes)):
        bbox = _box(raw_box, width, height)
        clean_text = str(text or "").strip()
        if not bbox or not clean_text:
            continue
        try:
            score = float(scores[index]) if index < len(scores) else 0.0
        except (TypeError, ValueError):
            score = 0.0
        lines.append(
            {
                "id": f"pp_ocr_{len(lines):04d}",
                "text": clean_text,
                "confidence": round(max(0.0, min(1.0, score)), 4),
                "bbox": bbox,
            }
        )
    return lines


def _dominant_cells(raw: Mapping[str, Any], width: int, height: int) -> tuple[list[list[int]], dict[str, Any]]:
    """中文说明：按覆盖面积与单元格数选择主表格，避免多个小表的边界互相污染。"""

    tables = raw.get("table_res_list")
    tables = tables if isinstance(tables, list) else []
    candidates: list[tuple[list[list[int]], list[int], int]] = []
    for table_index, table in enumerate(tables):
        if not isinstance(table, Mapping):
            continue
        cells: list[list[int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for raw_box in table.get("cell_box_list") or []:
            bbox = _box(raw_box, width, height)
            key = tuple(bbox) if bbox else ()
            if bbox and key not in seen:
                seen.add(key)
                cells.append(bbox)
        table_bbox = _union_box(cells)
        if cells and table_bbox:
            candidates.append((cells, table_bbox, table_index))
    if not candidates:
        return [], {"detected_table_count": len(tables), "selected_table_index": None}
    cells, table_bbox, table_index = max(
        candidates,
        key=lambda item: (_area(item[1]), len(item[0])),
    )
    return cells, {
        "detected_table_count": len(candidates),
        "selected_table_index": table_index,
        "selected_table_bbox": table_bbox,
        "ignored_table_count": max(0, len(candidates) - 1),
    }


def _image_rule_cells(image_path: str | Path, width: int, height: int) -> tuple[list[list[int]], dict[str, Any]]:
    """中文说明：当 PP 表格分支失效时，用原图横纵表线恢复含合并单元格在内的完整边界。"""

    try:
        import cv2
        import numpy as np
    except ImportError:
        return [], {"available": False, "reason": "opencv_or_numpy_unavailable"}

    with Image.open(image_path) as source:
        grayscale = np.asarray(source.convert("L"))
    binary = cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        -2,
    )
    horizontal_length = max(12, round(width * 0.02))
    vertical_length = max(12, round(height * 0.025))
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length, 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length)),
    )
    rule_mask = cv2.morphologyEx(
        cv2.bitwise_or(horizontal, vertical),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    contours, hierarchy = cv2.findContours(rule_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def strongest_vertical_coverage(x: int, top: int, bottom: int) -> float:
        """中文说明：在候选边界附近寻找支持度最高的真实纵向表线。"""

        start, stop = max(0, top), min(height, bottom + 1)
        if stop <= start:
            return 0.0
        return max(
            float(np.mean(vertical[start:stop, candidate] > 0))
            for candidate in range(max(0, x - 3), min(width, x + 4))
        )

    def strongest_horizontal_coverage(y: int, left: int, right: int) -> float:
        """中文说明：在候选边界附近寻找支持度最高的真实横向表线。"""

        start, stop = max(0, left), min(width, right + 1)
        if stop <= start:
            return 0.0
        return max(
            float(np.mean(horizontal[candidate, start:stop] > 0))
            for candidate in range(max(0, y - 3), min(height, y + 4))
        )

    minimum_width = max(10, round(width * 0.01))
    minimum_height = max(10, round(height * 0.012))
    candidates: list[tuple[int, int, list[int]]] = []
    hierarchy_rows = hierarchy[0] if hierarchy is not None and len(hierarchy) else []
    for contour_index, contour in enumerate(contours):
        # 中文说明：父级为空的是整张表线网络外框，单元格应是其内部的闭合孔洞。
        if len(hierarchy_rows) and int(hierarchy_rows[contour_index][3]) < 0:
            continue
        x, y, cell_width, cell_height = cv2.boundingRect(contour)
        if cell_width < minimum_width or cell_height < minimum_height:
            continue
        if cell_width >= width * 0.99 or cell_height >= height * 0.99:
            continue
        right, bottom = x + cell_width - 1, y + cell_height - 1
        coverages = (
            strongest_vertical_coverage(x, y, bottom),
            strongest_vertical_coverage(right, y, bottom),
            strongest_horizontal_coverage(y, x, right),
            strongest_horizontal_coverage(bottom, x, right),
        )
        if min(coverages) >= 0.5:
            parent = int(hierarchy_rows[contour_index][3]) if len(hierarchy_rows) else -1
            candidates.append((contour_index, parent, [x, y, right, bottom]))

    # 中文说明：同一表线网络的直接子轮廓才是并列单元格；更深层轮廓多为文字或曲线形成的伪框。
    groups: dict[int, list[list[int]]] = {}
    for _, parent, cell in candidates:
        groups.setdefault(parent, []).append(cell)
    cells = max(
        groups.values(),
        key=lambda group: (len(group), sum(_area(cell) for cell in group)),
        default=[],
    )
    cells = sorted({tuple(cell) for cell in cells}, key=lambda item: (item[1], item[0], item[3], item[2]))
    return [list(cell) for cell in cells], {
        "available": bool(cells),
        "method": "opencv_adaptive_rule_morphology",
        "horizontal_kernel_length": horizontal_length,
        "vertical_kernel_length": vertical_length,
        "candidate_count": len(candidates),
        "cell_count": len(cells),
    }


def _build_tracks(cells: Sequence[Sequence[int]], content_bbox: Sequence[int], ocr: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    """中文说明：用高支持度单元格左右边界重建列轨道，坐标不再来自 VLM。"""

    if len(content_bbox) != 4:
        return [], []
    tolerance = max(2, round((content_bbox[2] - content_bbox[0]) * 0.003))
    clustered = _cluster([value for box in cells for value in (box[0], box[2])], tolerance)
    # 中文说明：合并单元格列的边界出现次数较少，1% 支持度可保留窄地层栏而不依赖 OCR 字框。
    minimum_support = max(2, math.ceil(len(cells) * 0.01))
    boundary_values = {
        int(content_bbox[0]),
        int(content_bbox[2]),
        *(
            int(item["position"])
            for item in clustered
            if int(item["support"]) >= minimum_support
        ),
    }
    boundaries = sorted(value for value in boundary_values if content_bbox[0] <= value <= content_bbox[2])
    tracks: list[dict[str, Any]] = []
    header_limit = content_bbox[1] + max(24, round((content_bbox[3] - content_bbox[1]) * 0.12))
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left < 5:
            continue
        bbox = [left, int(content_bbox[1]), right, int(content_bbox[3])]
        header_texts = []
        for line in ocr:
            center_x, center_y = _center(line["bbox"])
            if left <= center_x <= right and content_bbox[1] <= center_y <= header_limit:
                header_texts.append(str(line["text"]))
        tracks.append(
            {
                "id": f"pp_track_{len(tracks):03d}",
                "order": len(tracks),
                "bbox": bbox,
                "header_text": " / ".join(header_texts),
                "geometry_source": "PP-StructureV3.cell_box_list",
            }
        )
    if not tracks:
        tracks.append(
            {
                "id": "pp_track_000",
                "order": 0,
                "bbox": list(content_bbox),
                "header_text": "",
                "geometry_source": "PP-StructureV3.content_bbox_fallback",
            }
        )
        boundaries = [int(content_bbox[0]), int(content_bbox[2])]
    return tracks, boundaries


def _assign_geometry(cells: Sequence[Sequence[int]], tracks: Sequence[Mapping[str, Any]], ocr: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """中文说明：把 OCR 行归入最小包含单元格，并给单元格标注其跨越的 PP 轨道。"""

    normalized_cells = [
        {
            "id": f"pp_cell_{index:04d}",
            "bbox": list(bbox),
            "ocr_ids": [],
            "text": "",
            "track_ids": [],
        }
        for index, bbox in enumerate(sorted(cells, key=lambda item: (item[1], item[0], item[3], item[2])))
    ]
    for line in ocr:
        center_x, center_y = _center(line["bbox"])
        containing = [
            cell
            for cell in normalized_cells
            if cell["bbox"][0] <= center_x <= cell["bbox"][2]
            and cell["bbox"][1] <= center_y <= cell["bbox"][3]
        ]
        if containing:
            cell = min(containing, key=lambda item: _area(item["bbox"]))
            cell["ocr_ids"].append(line["id"])
            line["cell_id"] = cell["id"]
        line["track_id"] = next(
            (
                str(track["id"])
                for track in tracks
                if track["bbox"][0] <= center_x <= track["bbox"][2]
            ),
            "",
        )
    text_by_id = {line["id"]: line["text"] for line in ocr}
    for cell in normalized_cells:
        cell["text"] = " ".join(text_by_id[item] for item in cell["ocr_ids"])
        cell_width = max(1, cell["bbox"][2] - cell["bbox"][0])
        cell["track_ids"] = [
            str(track["id"])
            for track in tracks
            if max(0, min(cell["bbox"][2], track["bbox"][2]) - max(cell["bbox"][0], track["bbox"][0]))
            / cell_width
            >= 0.2
        ]
    return normalized_cells


def normalize_ppstructure_result(raw_result: Mapping[str, Any], *, image_path: str | Path) -> dict[str, Any]:
    """中文说明：把 PP-StructureV3 原始结果标准化为本模块唯一使用的几何目录。"""

    image_file = Path(image_path).resolve()
    if not image_file.is_file():
        raise FileNotFoundError(f"PP-StructureV3 来源图片不存在：{image_file}")
    with Image.open(image_file) as image:
        image_width, image_height = image.size
    raw = _to_python(raw_result)
    width = int(raw.get("width") or image_width)
    height = int(raw.get("height") or image_height)
    if width != image_width or height != image_height:
        raise ValueError(
            f"PP-StructureV3 输出画布 {width}x{height} 与原图 {image_width}x{image_height} 不一致"
        )
    ocr = _ocr_lines(raw, width, height)
    raw_cells, table_meta = _dominant_cells(raw, width, height)
    cell_geometry_source = "PP-StructureV3.cell_box_list"
    image_rule_meta: dict[str, Any] = {"available": False}
    if not raw_cells:
        raw_cells, image_rule_meta = _image_rule_cells(image_file, width, height)
        if raw_cells:
            cell_geometry_source = "image_rule_morphology_fallback"
            table_meta["cell_detection_fallback"] = True
    content_bbox = table_meta.get("selected_table_bbox") or _union_box([line["bbox"] for line in ocr])
    if raw_cells:
        content_bbox = _union_box(raw_cells)
    if not content_bbox:
        content_bbox = [0, 0, width, height]
    tracks, vertical_lines = _build_tracks(raw_cells, content_bbox, ocr)
    cells = _assign_geometry(raw_cells, tracks, ocr)
    horizontal_lines = [
        int(item["position"])
        for item in _cluster([value for box in raw_cells for value in (box[1], box[3])], max(2, round(height * 0.002)))
    ]
    return {
        "schema_version": PPSTRUCTURE_GEOMETRY_SCHEMA_VERSION,
        "engine": "PaddleOCR.PPStructureV3",
        "coordinate_space": "original_pixels",
        "source_image_path": str(image_file),
        "image_size": {"width": width, "height": height},
        "content_bbox": list(content_bbox),
        "tracks": tracks,
        "cells": cells,
        "ocr_lines": ocr,
        "rule_lines": {
            "available": bool(raw_cells),
            "method": cell_geometry_source,
            "vertical_lines": vertical_lines,
            "horizontal_lines": horizontal_lines,
            "image_rule_detection": image_rule_meta,
        },
        "quality": {
            **table_meta,
            "ocr_line_count": len(ocr),
            "cell_count": len(cells),
            "track_count": len(tracks),
            "coordinate_source": "PP-StructureV3",
            "cell_geometry_source": cell_geometry_source,
        },
    }


def _repair_cached_cell_geometry(geometry: Mapping[str, Any], image_file: Path) -> tuple[dict[str, Any], bool]:
    """中文说明：旧缓存若只有 OCR 字框，则直接从原图补建单元格与轨道，避免重新运行 PP 模型。"""

    repaired = deepcopy(dict(geometry))
    if repaired.get("cells"):
        return repaired, False
    image_size = repaired.get("image_size")
    if not isinstance(image_size, Mapping):
        return repaired, False
    width, height = int(image_size.get("width") or 0), int(image_size.get("height") or 0)
    if width <= 0 or height <= 0:
        return repaired, False
    raw_cells, detection_meta = _image_rule_cells(image_file, width, height)
    if not raw_cells:
        return repaired, False
    ocr = [dict(item) for item in repaired.get("ocr_lines", []) if isinstance(item, Mapping)]
    for line in ocr:
        line.pop("cell_id", None)
        line.pop("track_id", None)
    content_bbox = _union_box(raw_cells)
    tracks, vertical_lines = _build_tracks(raw_cells, content_bbox, ocr)
    cells = _assign_geometry(raw_cells, tracks, ocr)
    horizontal_lines = [
        int(item["position"])
        for item in _cluster(
            [value for box in raw_cells for value in (box[1], box[3])],
            max(2, round(height * 0.002)),
        )
    ]
    repaired["content_bbox"] = content_bbox
    repaired["tracks"] = tracks
    repaired["cells"] = cells
    repaired["ocr_lines"] = ocr
    repaired["rule_lines"] = {
        "available": True,
        "method": "image_rule_morphology_fallback",
        "vertical_lines": vertical_lines,
        "horizontal_lines": horizontal_lines,
        "image_rule_detection": detection_meta,
    }
    quality = repaired.setdefault("quality", {})
    quality.update(
        {
            "cell_count": len(cells),
            "track_count": len(tracks),
            "cell_detection_fallback": True,
            "cell_geometry_source": "image_rule_morphology_fallback",
        }
    )
    repaired.setdefault("runtime", {})["cached_cell_geometry_repaired"] = True
    repaired.setdefault("uncertainties", []).append(
        "ppstructure_cell_geometry_repaired: PP 表格结构不可用，已从原图表线恢复完整单元格"
    )
    return repaired, True


def _result_mapping(result: Any) -> Mapping[str, Any]:
    """中文说明：兼容 PaddleX Result 对象和测试桩字典，提取其中的 res 主体。"""

    if isinstance(result, Mapping):
        payload: Any = result
    else:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
    if isinstance(payload, Mapping) and isinstance(payload.get("res"), Mapping):
        payload = payload["res"]
    if not isinstance(payload, Mapping):
        raise TypeError(f"无法读取 PP-StructureV3 结果：{type(result).__name__}")
    return payload


class PPStructureV3GeometryExtractor:
    """懒加载并缓存 PP-StructureV3，把每张图只转换一次。"""

    def __init__(self, pipeline: Any | None = None, *, cache_dir: str | Path | None = None) -> None:
        """中文说明：允许测试注入轻量预测器，生产环境默认使用官方 PP-StructureV3。"""

        self._pipeline = pipeline
        project_root = Path(__file__).resolve().parents[5]
        configured_cache = os.getenv("TABLE_PPSTRUCTURE_CACHE_DIR")
        self.cache_dir = Path(cache_dir or configured_cache or project_root / "data" / "cache" / "ppstructurev3")
        self.device = os.getenv("TABLE_PPSTRUCTURE_DEVICE", "cpu")
        self.layout_model = os.getenv("TABLE_PPSTRUCTURE_LAYOUT_MODEL", "PP-DocLayout-M")
        self.det_model = os.getenv("TABLE_PPSTRUCTURE_TEXT_DET_MODEL", "PP-OCRv5_mobile_det")
        self.rec_model = os.getenv("TABLE_PPSTRUCTURE_TEXT_REC_MODEL", "PP-OCRv5_mobile_rec")
        self.enable_mkldnn = _as_bool(os.getenv("TABLE_PPSTRUCTURE_ENABLE_MKLDNN"), False)

    def _cache_key(self, image_file: Path) -> tuple[str, dict[str, Any]]:
        """中文说明：用图片状态和模型配置生成可失效的稳定缓存键。"""

        stat = image_file.stat()
        fingerprint = {
            "schema_version": PPSTRUCTURE_GEOMETRY_SCHEMA_VERSION,
            "source_image_path": str(image_file.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "device": self.device,
            "layout_model": self.layout_model,
            "text_detection_model": self.det_model,
            "text_recognition_model": self.rec_model,
            "enable_mkldnn": self.enable_mkldnn,
        }
        encoded = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), fingerprint

    def _get_pipeline(self) -> Any:
        """中文说明：创建进程内共享模型，CPU 默认关闭已知有兼容问题的 oneDNN 路径。"""

        if self._pipeline is not None:
            return self._pipeline
        key = (self.device, self.layout_model, self.det_model + ":" + self.rec_model, self.enable_mkldnn)
        if key in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[key]
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            # 中文说明：当前环境中先加载 torch 可避免 Paddle 与 ModelScope 间的 Windows DLL 加载顺序冲突。
            try:
                __import__("torch")
            except Exception:
                pass
            from paddleocr import PPStructureV3
        except ImportError as exc:
            raise RuntimeError(
                '缺少 PP-StructureV3 依赖，请在 treeSchemeKG 中安装 paddlepaddle==3.3.1 与 "paddleocr[doc-parser]==3.7.0"'
            ) from exc
        pipeline = PPStructureV3(
            device=self.device,
            enable_mkldnn=self.enable_mkldnn,
            layout_detection_model_name=self.layout_model,
            text_detection_model_name=self.det_model,
            text_recognition_model_name=self.rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_seal_recognition=False,
            use_table_recognition=True,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_region_detection=False,
        )
        _PIPELINE_CACHE[key] = pipeline
        return pipeline

    def extract(self, image_path: str | Path) -> dict[str, Any]:
        """中文说明：命中缓存或执行一次 PP-StructureV3，并返回原图像素几何。"""

        image_file = Path(image_path).resolve()
        if not image_file.is_file():
            raise FileNotFoundError(f"PP-StructureV3 来源图片不存在：{image_file}")
        cache_key, fingerprint = self._cache_key(image_file)
        if cache_key in _GEOMETRY_CACHE:
            geometry, _ = _repair_cached_cell_geometry(_GEOMETRY_CACHE[cache_key], image_file)
            _GEOMETRY_CACHE[cache_key] = deepcopy(geometry)
            geometry.setdefault("runtime", {})["cache_status"] = "memory_hit"
            return geometry
        cache_path = self.cache_dir / f"{cache_key}.json"
        use_cache = _as_bool(os.getenv("TABLE_PPSTRUCTURE_CACHE"), True)
        force_refresh = _as_bool(os.getenv("TABLE_PPSTRUCTURE_FORCE_REFRESH"), False)
        if use_cache and not force_refresh and cache_path.is_file():
            cached = read_json(cache_path)
            if isinstance(cached, Mapping) and cached.get("fingerprint") == fingerprint and isinstance(cached.get("geometry"), Mapping):
                geometry, repaired = _repair_cached_cell_geometry(cached["geometry"], image_file)
                if repaired:
                    write_json(cache_path, {"fingerprint": fingerprint, "geometry": geometry})
                _GEOMETRY_CACHE[cache_key] = geometry
                geometry = deepcopy(geometry)
                geometry.setdefault("runtime", {})["cache_status"] = "disk_hit"
                return geometry
        pipeline = self._get_pipeline()
        predict_options = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "use_seal_recognition": False,
            "use_table_recognition": True,
            "use_formula_recognition": False,
            "use_chart_recognition": False,
            "use_region_detection": False,
            "use_table_orientation_classify": False,
        }
        table_fallback_error = ""
        try:
            results = pipeline.predict(str(image_file), **predict_options)
        except AttributeError as exc:
            # 中文说明：PaddleX 3.7.0 在部分纯表格图上会进入未初始化的表格 OCR 分支；
            # 此时仅关闭单元格结构识别，仍保留布局检测和通用 OCR 原图坐标。
            if "text_rec_model" not in str(exc):
                raise
            table_fallback_error = str(exc)
            predict_options["use_table_recognition"] = False
            results = pipeline.predict(str(image_file), **predict_options)
        if not isinstance(results, list) or len(results) != 1:
            raise ValueError(f"PP-StructureV3 单图应返回一个页面，实际为 {len(results) if isinstance(results, list) else type(results).__name__}")
        geometry = normalize_ppstructure_result(_result_mapping(results[0]), image_path=image_file)
        geometry["runtime"] = {
            "device": self.device,
            "layout_model": self.layout_model,
            "text_detection_model": self.det_model,
            "text_recognition_model": self.rec_model,
            "enable_mkldnn": self.enable_mkldnn,
            "cache_status": "fresh_inference",
            "table_recognition_fallback": bool(table_fallback_error),
        }
        if table_fallback_error:
            geometry.setdefault("uncertainties", []).append(
                "ppstructure_table_fallback: 表格结构分支缺少 text_rec_model，已退化为布局检测和通用 OCR 几何"
            )
        _GEOMETRY_CACHE[cache_key] = geometry
        if use_cache:
            write_json(cache_path, {"fingerprint": fingerprint, "geometry": geometry})
        return deepcopy(geometry)


def extract_ppstructure_geometry(image_path: str | Path, provider: Any | None = None) -> dict[str, Any]:
    """中文说明：统一调用生产提取器或测试注入的几何提供者。"""

    if provider is None:
        return PPStructureV3GeometryExtractor().extract(image_path)
    if hasattr(provider, "extract"):
        geometry = provider.extract(image_path)
    elif callable(provider):
        geometry = provider(image_path)
    else:
        raise TypeError("ppstructure_geometry_extractor 必须可调用或具有 extract 方法")
    if not isinstance(geometry, Mapping):
        raise TypeError("PP-StructureV3 几何提供者必须返回对象")
    return deepcopy(dict(geometry))


def geometry_prompt_catalog(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """中文说明：生成不给 VLM 重新估算坐标、只供选择稳定几何 ID 的紧凑目录。"""

    return {
        "coordinate_source": "PP-StructureV3",
        "tracks": [
            {
                "id": item.get("id"),
                "order": item.get("order"),
                "header_text": item.get("header_text", ""),
            }
            for item in geometry.get("tracks", [])
            if isinstance(item, Mapping)
        ],
        "cells": [
            {
                "id": item.get("id"),
                "track_ids": item.get("track_ids", []),
                "text": item.get("text", ""),
            }
            for item in geometry.get("cells", [])
            if isinstance(item, Mapping)
        ],
        "ocr_lines": [
            {
                "id": item.get("id"),
                "track_id": item.get("track_id", ""),
                "cell_id": item.get("cell_id", ""),
                "text": item.get("text", ""),
            }
            for item in geometry.get("ocr_lines", [])
            if isinstance(item, Mapping)
        ],
    }


def _clean_text(value: Any) -> str:
    """中文说明：移除空白与标点以匹配 VLM 名称和 PP-OCR 文本。"""

    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _parse_numeric_text(value: Any) -> float | None:
    """中文说明：只把完整的深度刻度文本解析为数值，拒绝从任意说明文字中抓数字。"""

    text = str(value or "").strip().translate(str.maketrans("０１２３４５６７８９．，－", "0123456789.,-"))
    text = text.replace(",", "").replace(" ", "")
    if not _NUMERIC_TEXT.fullmatch(text):
        return None
    try:
        return float(re.sub(r"(?:m|米)$", "", text, flags=re.IGNORECASE))
    except ValueError:
        return None


def _longest_monotonic(points: Sequence[tuple[float, float, str]], increases: str) -> list[tuple[float, float, str]]:
    """中文说明：从含少量 OCR 误读的刻度中选择纵向跨度最大的单调子序列。"""

    ordered = sorted(points, key=lambda item: item[0])
    if len(ordered) <= 2:
        return ordered
    paths: list[list[tuple[float, float, str]]] = [[point] for point in ordered]
    for index, point in enumerate(ordered):
        for previous in range(index):
            value_ok = point[1] > ordered[previous][1] if increases == "downward" else point[1] < ordered[previous][1]
            if value_ok and len(paths[previous]) + 1 > len(paths[index]):
                paths[index] = [*paths[previous], point]
    return max(paths, key=lambda path: (len(path), path[-1][0] - path[0][0]))


def _axis_points(raw_axis: Mapping[str, Any], tracks: Sequence[Mapping[str, Any]], ocr: Sequence[Mapping[str, Any]], content_bbox: Sequence[int]) -> tuple[dict[str, Any], list[str]]:
    """中文说明：从 PP-OCR 数字框生成纵轴锚点；无可靠连续深度轴时退化为相对层序。"""

    uncertainties: list[str] = []
    kind = str(raw_axis.get("kind") or "relative_sequence")
    increases = str(raw_axis.get("increases") or "downward")
    unit = str(raw_axis.get("unit") or "")
    track_id = str(raw_axis.get("track_id") or "")
    if not track_id:
        track_id = next(
            (str(track.get("id") or "") for track in tracks if str(track.get("role") or "") == "depth"),
            "",
        )
    selected_ids = {
        str(item)
        for item in raw_axis.get("calibration_ocr_ids", [])
        if str(item)
    } if isinstance(raw_axis.get("calibration_ocr_ids"), list) else set()
    candidates: list[tuple[float, float, str]] = []
    for line in ocr:
        line_id = str(line.get("id") or "")
        if selected_ids and line_id not in selected_ids:
            continue
        if not selected_ids and track_id and str(line.get("track_id") or "") != track_id:
            continue
        value = _parse_numeric_text(line.get("text"))
        bbox = line.get("bbox")
        if value is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        candidates.append((_center(bbox)[1], value, line_id))
    points = _longest_monotonic(candidates, increases)
    continuous_axis = kind in {"depth", "elevation", "time"}
    if continuous_axis and len(points) >= 2 and abs(points[-1][0] - points[0][0]) >= 20:
        return {
            "kind": kind,
            "unit": unit,
            "increases": increases,
            "track_id": track_id,
            "coordinate_source": "PP-StructureV3.overall_ocr_res.rec_boxes",
            "calibration_points": [
                {
                    "pixel_y": round(pixel_y, 3),
                    "value": value,
                    "ocr_id": ocr_id,
                    "evidence": "PP-OCR 可见纵轴刻度",
                }
                for pixel_y, value, ocr_id in points
            ],
        }, uncertainties
    uncertainties.append(
        "ppstructure_axis_fallback: 未发现至少两个单调且跨越足够像素的连续深度刻度，使用相对层序，禁止输出伪绝对深度"
    )
    top, bottom = float(content_bbox[1]), float(content_bbox[3])
    return {
        "kind": "relative_sequence",
        "unit": "relative",
        "increases": "downward",
        "track_id": track_id,
        "coordinate_source": "PP-StructureV3.content_bbox",
        "calibration_points": [
            {"pixel_y": top, "value": 0.0, "evidence": "PP-StructureV3 内容顶部"},
            {"pixel_y": bottom, "value": 1.0, "evidence": "PP-StructureV3 内容底部"},
        ],
    }, uncertainties


def _match_tracks(raw_tracks: Any, geometry_tracks: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """中文说明：把 VLM 语义角色映射到 PP 列 ID，并强制使用 PP 像素框。"""

    raw_list = raw_tracks if isinstance(raw_tracks, list) else []
    by_id = {str(track.get("id") or ""): track for track in geometry_tracks if str(track.get("id") or "")}
    used: set[str] = set()
    mapped: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}
    for index, raw in enumerate(raw_list):
        if not isinstance(raw, Mapping):
            continue
        raw_id = str(raw.get("id") or "")
        chosen = by_id.get(raw_id)
        if chosen is None:
            header = _clean_text(raw.get("header"))
            candidates = [track for track in geometry_tracks if str(track.get("id") or "") not in used]
            if header and candidates:
                chosen = max(
                    candidates,
                    key=lambda track: SequenceMatcher(None, header, _clean_text(track.get("header_text"))).ratio(),
                )
            elif index < len(geometry_tracks):
                chosen = geometry_tracks[index]
        if chosen is None:
            continue
        chosen_id = str(chosen["id"])
        if raw_id:
            id_map[raw_id] = chosen_id
        if chosen_id in used:
            # 中文说明：当 PP 只能给出一个整表轨道而 VLM 为它标注多个语义角色时，
            # 合并语义而不复制几何 ID，避免下游把同一像素列误当成多条独立轨道。
            existing = next(track for track in mapped if str(track.get("id") or "") == chosen_id)
            for source_key, target_key in (
                ("role", "semantic_roles"),
                ("header", "semantic_headers"),
                ("evidence", "semantic_evidence"),
            ):
                values = [str(value) for value in existing.get(target_key, []) if str(value)]
                for value in (existing.get(source_key), raw.get(source_key)):
                    text = str(value or "").strip()
                    if text and text not in values:
                        values.append(text)
                if values:
                    existing[target_key] = values
            continue
        used.add(chosen_id)
        mapped.append(
            {
                **dict(raw),
                "id": chosen_id,
                "order": int(chosen.get("order", len(mapped))),
                "bbox": list(chosen["bbox"]),
                "ppstructure_header_text": str(chosen.get("header_text") or ""),
                "geometry_source": "PP-StructureV3",
            }
        )
    if not mapped:
        mapped = [
            {
                **dict(track),
                "role": "unknown",
                "header": str(track.get("header_text") or ""),
                "parser": "semantic_role_unresolved",
                "evidence": "PP-StructureV3 检测到列，但 VLM 未返回语义角色",
            }
            for track in geometry_tracks
        ]
    return mapped, id_map


def _geometry_index(geometry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """中文说明：建立 cell/ocr ID 到原图像素框的统一索引。"""

    return {
        str(item["id"]): item
        for field in ("cells", "ocr_lines")
        for item in geometry.get(field, [])
        if isinstance(item, Mapping) and item.get("id") and isinstance(item.get("bbox"), list)
    }


def _scored_semantic_cells(item: Mapping[str, Any], geometry: Mapping[str, Any], track_id: str) -> list[tuple[float, str]]:
    """中文说明：按单元格全文匹配语义名称；轨道仅作加分项，避免旧单轨缓存阻断正确单元格。"""

    target = _clean_text(item.get("name"))
    if not target:
        return []
    scored: list[tuple[float, str]] = []
    for cell in geometry.get("cells", []):
        if not isinstance(cell, Mapping):
            continue
        text = _clean_text(cell.get("text"))
        if not text:
            continue
        if target == text:
            score = 1.0
        elif min(len(target), len(text)) >= 2 and (target in text or text in target):
            score = 0.9
        elif (
            len(target) == len(text)
            and target[:-1] == text[:-1]
            and target[-1] == "4"
            and text[-1] == "a"
        ):
            # 中文说明：地层下标 4 在小字号图中常被 OCR 成 A，此处只在同前缀末字符上纠正。
            score = 0.93
        else:
            score = SequenceMatcher(None, target, text).ratio()
        if score >= 0.62:
            if track_id and track_id in {str(value) for value in cell.get("track_ids", [])}:
                score += 0.01
            scored.append((score, str(cell.get("id") or "")))
    return scored


def _semantic_geometry_refs(item: Mapping[str, Any], geometry: Mapping[str, Any], track_id: str) -> list[str]:
    """中文说明：兼容旧语义响应时用名称匹配 PP 单元格；绝不回退到旧 VLM 像素坐标。"""

    scored = _scored_semantic_cells(item, geometry, track_id)
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return [item_id for score, item_id in scored if score >= max(0.8, best - 0.03)]


def _promote_ocr_refs_to_cells(refs: Sequence[str], index: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """中文说明：把文字 OCR 引用提升为所属完整单元格，防止字框被误当成区间框。"""

    promoted: list[str] = []
    for ref in refs:
        record = index.get(ref)
        if not isinstance(record, Mapping):
            continue
        cell_id = ref if ref.startswith("pp_cell_") else str(record.get("cell_id") or "")
        if cell_id in index and cell_id not in promoted:
            promoted.append(cell_id)
    return promoted


def _preferred_geometry_refs(
    item: Mapping[str, Any],
    geometry: Mapping[str, Any],
    track_id: str,
    explicit_refs: Sequence[str],
    index: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """中文说明：唯一全文匹配优先，其次采用 OCR 所属单元格，最后才保留裸 OCR 框。"""

    promoted = _promote_ocr_refs_to_cells(explicit_refs, index)
    target = _clean_text(item.get("name"))
    scored = _scored_semantic_cells(item, geometry, track_id)
    exact = [
        cell_id
        for _, cell_id in scored
        if target and _clean_text(index.get(cell_id, {}).get("text")) == target
    ]
    if len(exact) == 1:
        return exact
    if exact:
        overlapping = [cell_id for cell_id in promoted if cell_id in exact]
        if overlapping:
            return overlapping
    if scored:
        best_score = max(score for score, _ in scored)
        best_cells = [cell_id for score, cell_id in scored if score >= best_score - 0.001]
        overlapping = [cell_id for cell_id in promoted if cell_id in best_cells]
        if best_score >= 0.89 and overlapping:
            return overlapping
        if best_score >= 0.89 and len(best_cells) == 1:
            return best_cells
    if promoted:
        return promoted
    semantic_refs = _semantic_geometry_refs(item, geometry, track_id)
    return semantic_refs or list(explicit_refs)


def _track_from_geometry_refs(
    refs: Sequence[str], index: Mapping[str, Mapping[str, Any]], fallback: str
) -> str:
    """中文说明：当选中的完整单元格只属于一个轨道时，同步纠正旧语义响应里的错误轨道。"""

    track_ids: list[str] = []
    for ref in refs:
        record = index.get(ref)
        if not isinstance(record, Mapping):
            continue
        values = record.get("track_ids")
        if not isinstance(values, list):
            values = [record.get("track_id")]
        for value in values:
            track_id = str(value or "")
            if track_id and track_id not in track_ids:
                track_ids.append(track_id)
    return track_ids[0] if len(track_ids) == 1 else fallback


def _resolve_primitives(payload: dict[str, Any], geometry: Mapping[str, Any], id_map: Mapping[str, str]) -> list[str]:
    """中文说明：用 VLM 选择的 PP 几何 ID 生成区间/点坐标，并删除无法定位的语义图元。"""

    uncertainties: list[str] = []
    primitives = payload.get("primitives")
    if not isinstance(primitives, Mapping):
        raise ValueError("表格语义响应缺少 primitives")
    index = _geometry_index(geometry)
    for field in INTERVAL_FIELDS:
        raw_items = primitives.get(field, [])
        if not isinstance(raw_items, list):
            raise ValueError(f"primitives.{field} 必须是数组")
        resolved: list[dict[str, Any]] = []
        for item_index, raw in enumerate(raw_items):
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            old_track_id = str(item.get("track_id") or "")
            track_id = id_map.get(old_track_id, old_track_id)
            item["track_id"] = track_id
            refs = [str(value) for value in item.get("geometry_refs", []) if str(value)] if isinstance(item.get("geometry_refs"), list) else []
            refs = [value for value in refs if value in index]
            refs = _preferred_geometry_refs(item, geometry, track_id, refs, index)
            bbox = _union_box([index[value]["bbox"] for value in refs if value in index])
            if not bbox:
                uncertainties.append(
                    f"ppstructure_unresolved_geometry: primitives.{field}[{item_index}] {item.get('name') or item.get('id') or ''} 已删除"
                )
                continue
            item["top_y"] = float(bbox[1])
            item["bottom_y"] = float(bbox[3])
            item["track_id"] = _track_from_geometry_refs(refs, index, track_id)
            item["geometry_refs"] = refs
            item["geometry_bbox"] = bbox
            item["coordinate_source"] = "PP-StructureV3"
            resolved.append(item)
        primitives[field] = resolved
    point_markers = primitives.get("point_markers", [])
    if not isinstance(point_markers, list):
        raise ValueError("primitives.point_markers 必须是数组")
    resolved_points: list[dict[str, Any]] = []
    for item_index, raw in enumerate(point_markers):
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        old_track_id = str(item.get("track_id") or "")
        track_id = id_map.get(old_track_id, old_track_id)
        item["track_id"] = track_id
        refs = [str(value) for value in item.get("geometry_refs", []) if str(value)] if isinstance(item.get("geometry_refs"), list) else []
        refs = [value for value in refs if value in index]
        refs = _preferred_geometry_refs(item, geometry, track_id, refs, index)
        bbox = _union_box([index[value]["bbox"] for value in refs if value in index])
        if not bbox:
            uncertainties.append(
                f"ppstructure_unresolved_geometry: primitives.point_markers[{item_index}] {item.get('name') or item.get('id') or ''} 已删除"
            )
            continue
        item["pixel_y"] = round((bbox[1] + bbox[3]) / 2.0, 3)
        item["track_id"] = _track_from_geometry_refs(refs, index, track_id)
        item["geometry_refs"] = refs
        item["geometry_bbox"] = bbox
        item["coordinate_source"] = "PP-StructureV3"
        resolved_points.append(item)
    primitives["point_markers"] = resolved_points
    return uncertainties


def apply_ppstructure_geometry(payload: Mapping[str, Any], geometry: Mapping[str, Any]) -> dict[str, Any]:
    """中文说明：用 PP-StructureV3 全量替换 VLM 几何，同时保留 VLM 语义字段。"""

    if str(payload.get("schema_version") or "") != TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION:
        raise ValueError(f"语义响应必须使用 {TABLE_EMBEDDED_HYBRID_SCHEMA_VERSION}")
    if str(geometry.get("schema_version") or "") != PPSTRUCTURE_GEOMETRY_SCHEMA_VERSION:
        raise ValueError(f"几何响应必须使用 {PPSTRUCTURE_GEOMETRY_SCHEMA_VERSION}")
    enriched = deepcopy(dict(payload))
    geometry_tracks = [dict(item) for item in geometry.get("tracks", []) if isinstance(item, Mapping)]
    tracks, id_map = _match_tracks(enriched.get("tracks"), geometry_tracks)
    enriched["tracks"] = tracks
    coordinate_system = enriched.get("coordinate_system")
    coordinate_system = dict(coordinate_system) if isinstance(coordinate_system, Mapping) else {}
    raw_axis = coordinate_system.get("vertical_axis")
    raw_axis = dict(raw_axis) if isinstance(raw_axis, Mapping) else {}
    if str(raw_axis.get("track_id") or "") in id_map:
        raw_axis["track_id"] = id_map[str(raw_axis["track_id"])]
    axis, axis_uncertainties = _axis_points(
        raw_axis,
        tracks,
        [item for item in geometry.get("ocr_lines", []) if isinstance(item, Mapping)],
        geometry["content_bbox"],
    )
    coordinate_system.update(
        {
            "content_bbox": list(geometry["content_bbox"]),
            "vertical_axis": axis,
            "coordinate_source": "PP-StructureV3",
            "coordinate_space": "original_pixels",
        }
    )
    enriched["coordinate_system"] = coordinate_system
    enriched["image_size"] = dict(geometry.get("image_size") or {})
    primitive_uncertainties = _resolve_primitives(enriched, geometry, id_map)
    raw_uncertainties = enriched.get("uncertainties")
    uncertainties = [str(item) for item in raw_uncertainties] if isinstance(raw_uncertainties, list) else []
    for item in [*axis_uncertainties, *primitive_uncertainties]:
        if item not in uncertainties:
            uncertainties.append(item)
    enriched["uncertainties"] = uncertainties
    enriched["ppstructure_geometry"] = deepcopy(dict(geometry))
    enriched["geometry_policy"] = {
        "coordinate_generator": "PaddleOCR.PPStructureV3",
        "vlm_pixel_coordinates_used": False,
        "vlm_role": "semantic_labeling_and_geometry_id_selection",
    }
    return enriched


def ensure_ppstructure_geometry(payload: Mapping[str, Any], image_path: str | Path, provider: Any | None = None) -> dict[str, Any]:
    """中文说明：已有 PP 几何时复用，否则提取后替换所有 VLM 坐标。"""

    geometry = payload.get("ppstructure_geometry")
    if isinstance(geometry, Mapping) and str(geometry.get("schema_version") or "") == PPSTRUCTURE_GEOMETRY_SCHEMA_VERSION:
        policy = payload.get("geometry_policy")
        if isinstance(policy, Mapping) and policy.get("vlm_pixel_coordinates_used") is False:
            return deepcopy(dict(payload))
    return apply_ppstructure_geometry(payload, extract_ppstructure_geometry(image_path, provider))
