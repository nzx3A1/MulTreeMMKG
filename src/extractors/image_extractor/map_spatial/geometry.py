"""地图几何规范化、坐标重建与空间关系计算。

本文件把 VLM 给出的点、线、面转换为统一的 0–1000 归一化坐标，按需生成
像素坐标和世界坐标，并通过确定性几何算法计算拓扑、方向与距离关系。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from ..schema_models import ImageExtractionTask


NORMALIZED_EXTENT = 1000.0
_LITHOLOGY_TYPES = {"lithology", "lithofacies"}
_FACIES_TYPES = {"sedimentary_facies", "subfacies", "microfacies", "paleogeographic_unit"}
_STRUCTURAL_TYPES = {"structural_unit", "depression", "sag", "uplift", "slope", "structural_belt"}


def _number(value: Any) -> float | None:
    """把有限数值转换为浮点数，非法输入返回空值。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _point(value: Any) -> list[float] | None:
    """读取一个二维点并过滤非数值坐标。"""

    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = _number(value[0]), _number(value[1])
    return [x, y] if x is not None and y is not None else None


def _points(value: Any) -> list[list[float]]:
    """读取点序列，兼容单点和多点两种 JSON 表达。"""

    if point := _point(value):
        return [point]
    if not isinstance(value, (list, tuple)):
        return []
    return [point for item in value if (point := _point(item))]


def _image_size(task: ImageExtractionTask, georeference: Mapping[str, Any]) -> tuple[int, int]:
    """优先读取原图尺寸，测试或远程路径不可用时采用模型声明尺寸。"""

    path = Path(task.image_path)
    if path.is_file():
        try:
            with Image.open(path) as image:
                return int(image.width), int(image.height)
        except (OSError, ValueError):
            pass
    size = georeference.get("image_size")
    if isinstance(size, Mapping):
        width, height = _number(size.get("width")), _number(size.get("height"))
        if width and height and width > 0 and height > 0:
            return int(width), int(height)
    return 0, 0


def _normalize_points(
    points: Iterable[list[float]],
    coordinate_space: str,
    width: int,
    height: int,
) -> list[list[float]]:
    """将像素或百分比坐标统一到 0–1000 坐标系并裁剪越界值。"""

    normalized: list[list[float]] = []
    for x, y in points:
        if coordinate_space == "pixel" and width and height:
            x, y = x * NORMALIZED_EXTENT / width, y * NORMALIZED_EXTENT / height
        elif coordinate_space in {"normalized_1", "ratio"}:
            x, y = x * NORMALIZED_EXTENT, y * NORMALIZED_EXTENT
        elif coordinate_space == "percent":
            x, y = x * 10.0, y * 10.0
        normalized.append([
            round(max(0.0, min(NORMALIZED_EXTENT, x)), 3),
            round(max(0.0, min(NORMALIZED_EXTENT, y)), 3),
        ])
    return normalized


def _bbox(points: list[list[float]]) -> list[float]:
    """计算点集的轴对齐包围盒。"""

    if not points:
        return []
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    return [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)]


def _centroid(points: list[list[float]], kind: str) -> list[float]:
    """计算点、线或多边形的质心；退化多边形使用顶点均值。"""

    if not points:
        return []
    if kind != "polygon" or len(points) < 3:
        return [round(sum(p[0] for p in points) / len(points), 3), round(sum(p[1] for p in points) / len(points), 3)]
    twice_area = sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )
    if abs(twice_area) < 1e-9:
        return _centroid(points, "line")
    cx = sum(
        (points[index][0] + points[(index + 1) % len(points)][0])
        * (points[index][0] * points[(index + 1) % len(points)][1] - points[(index + 1) % len(points)][0] * points[index][1])
        for index in range(len(points))
    ) / (3.0 * twice_area)
    cy = sum(
        (points[index][1] + points[(index + 1) % len(points)][1])
        * (points[index][0] * points[(index + 1) % len(points)][1] - points[(index + 1) % len(points)][0] * points[index][1])
        for index in range(len(points))
    ) / (3.0 * twice_area)
    return [round(cx, 3), round(cy, 3)]


def _polygon_area(points: list[list[float]]) -> float:
    """用鞋带公式计算归一化多边形面积。"""

    if len(points) < 3:
        return 0.0
    return round(abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2.0, 3)


def _pixel_points(points: list[list[float]], width: int, height: int) -> list[list[float]]:
    """把归一化点转换为原图像素坐标。"""

    if not width or not height:
        return []
    return [[round(x * width / NORMALIZED_EXTENT, 2), round(y * height / NORMALIZED_EXTENT, 2)] for x, y in points]


def _geometry_points(raw: Mapping[str, Any], kind: str) -> list[list[float]]:
    """从统一 coordinates 或旧 bbox 字段恢复几何顶点。"""

    points = _points(raw.get("coordinates") or raw.get("points"))
    if points:
        return points
    bbox = _points(raw.get("bbox"))
    if len(bbox) == 2:
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
    elif isinstance(raw.get("bbox"), (list, tuple)) and len(raw["bbox"]) >= 4:
        values = [_number(value) for value in raw["bbox"][:4]]
        if any(value is None for value in values):
            return []
        x0, y0, x1, y1 = values  # type: ignore[misc]
    else:
        return []
    if kind == "point":
        return [[(x0 + x1) / 2.0, (y0 + y1) / 2.0]]
    if kind == "line":
        return [[x0, y0], [x1, y1]]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def normalize_geometry(value: Any, width: int = 0, height: int = 0) -> dict[str, Any]:
    """规范一个几何对象并生成包围盒、质心、面积和像素坐标。"""

    raw = dict(value) if isinstance(value, Mapping) else {}
    kind = str(raw.get("kind") or "unknown").strip().lower()
    if kind not in {"point", "line", "polygon"}:
        return {"kind": "unknown", "coordinate_space": "normalized_1000", "coordinates": []}
    coordinate_space = str(raw.get("coordinate_space") or "normalized_1000").strip().lower()
    points = _normalize_points(_geometry_points(raw, kind), coordinate_space, width, height)
    minimum = {"point": 1, "line": 2, "polygon": 3}[kind]
    if len(points) < minimum:
        return {"kind": "unknown", "coordinate_space": "normalized_1000", "coordinates": []}
    geometry = {
        "kind": kind,
        "coordinate_space": "normalized_1000",
        "coordinates": points[0] if kind == "point" else points,
        "bbox": _bbox(points),
        "centroid": _centroid(points, kind),
        "area_normalized": _polygon_area(points) if kind == "polygon" else 0.0,
        "position": str(raw.get("position") or "").strip(),
        "geometry_confidence": round(max(0.0, min(1.0, _number(raw.get("confidence")) or 0.7)), 3),
    }
    pixels = _pixel_points(points, width, height)
    if pixels:
        geometry.update({
            "pixel_coordinates": pixels[0] if kind == "point" else pixels,
            "pixel_bbox": _bbox(pixels),
            "pixel_centroid": _centroid(pixels, kind),
        })
    return geometry


def _geometry_vertex_list(geometry: Mapping[str, Any]) -> list[list[float]]:
    """把规范几何的坐标恢复为统一点列表。"""

    return _points(geometry.get("coordinates"))


def _point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    """用射线法判断点是否位于多边形内部或边界上。"""

    x, y = point
    inside = False
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        x1, y1 = previous
        x2, y2 = current
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-7 and min(x1, x2) - 1e-7 <= x <= max(x1, x2) + 1e-7 and min(y1, y2) - 1e-7 <= y <= max(y1, y2) + 1e-7:
            return True
        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x <= intersection_x:
                inside = not inside
    return inside


def _on_segment(point: list[float], start: list[float], end: list[float]) -> bool:
    """判断点是否位于指定闭线段上。"""

    return (
        abs(_orientation(start, end, point)) <= 1e-7
        and min(start[0], end[0]) - 1e-7 <= point[0] <= max(start[0], end[0]) + 1e-7
        and min(start[1], end[1]) - 1e-7 <= point[1] <= max(start[1], end[1]) + 1e-7
    )


def _strictly_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    """判断点是否严格位于多边形内部并排除边界点。"""

    if any(_on_segment(point, polygon[index - 1], polygon[index]) for index in range(len(polygon))):
        return False
    return _point_in_polygon(point, polygon)


def _orientation(a: list[float], b: list[float], c: list[float]) -> float:
    """计算三个点的有向面积，用于线段相交判断。"""

    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: list[float], b: list[float], c: list[float], d: list[float]) -> bool:
    """判断两条闭线段是否相交。"""

    o1, o2, o3, o4 = _orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b)
    if o1 * o2 < -1e-9 and o3 * o4 < -1e-9:
        return True
    return (
        (abs(o1) <= 1e-7 and _on_segment(c, a, b))
        or (abs(o2) <= 1e-7 and _on_segment(d, a, b))
        or (abs(o3) <= 1e-7 and _on_segment(a, c, d))
        or (abs(o4) <= 1e-7 and _on_segment(b, c, d))
    )


def _segments_cross(a: list[float], b: list[float], c: list[float], d: list[float]) -> bool:
    """判断两条线段是否在各自内部发生严格交叉。"""

    o1, o2, o3, o4 = _orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b)
    return o1 * o2 < -1e-9 and o3 * o4 < -1e-9


def _segments(points: list[list[float]], closed: bool) -> list[tuple[list[float], list[float]]]:
    """将点序列展开为线段列表。"""

    pairs = list(zip(points, points[1:]))
    if closed and len(points) > 2:
        pairs.append((points[-1], points[0]))
    return pairs


def _point_segment_distance(point: list[float], start: list[float], end: list[float]) -> float:
    """计算点到线段的最短欧氏距离。"""

    dx, dy = end[0] - start[0], end[1] - start[1]
    if dx == 0 and dy == 0:
        return math.dist(point, start)
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)))
    projection = [start[0] + ratio * dx, start[1] + ratio * dy]
    return math.dist(point, projection)


def _edge_distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    """计算两个几何边界之间的近似最短距离。"""

    first_points, second_points = _geometry_vertex_list(first), _geometry_vertex_list(second)
    first_segments = _segments(first_points, first.get("kind") == "polygon")
    second_segments = _segments(second_points, second.get("kind") == "polygon")
    if any(_segments_intersect(a, b, c, d) for a, b in first_segments for c, d in second_segments):
        return 0.0
    distances = [math.dist(a, b) for a in first_points for b in second_points]
    distances += [_point_segment_distance(point, start, end) for point in first_points for start, end in second_segments]
    distances += [_point_segment_distance(point, start, end) for point in second_points for start, end in first_segments]
    return min(distances) if distances else math.inf


def _within(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """判断第一个几何是否完全位于第二个多边形内。"""

    if second.get("kind") != "polygon":
        return False
    points = _geometry_vertex_list(first)
    polygon = _geometry_vertex_list(second)
    if not points or not all(_point_in_polygon(point, polygon) for point in points):
        return False
    return not any(
        _segments_cross(a, b, c, d)
        for a, b in _segments(points, first.get("kind") == "polygon")
        for c, d in _segments(polygon, True)
    )


def _overlaps(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """判断两个多边形是否发生面积重叠且不存在完全包含。"""

    if first.get("kind") != "polygon" or second.get("kind") != "polygon" or _within(first, second) or _within(second, first):
        return False
    first_points, second_points = _geometry_vertex_list(first), _geometry_vertex_list(second)
    if any(_strictly_in_polygon(point, second_points) for point in first_points):
        return True
    if any(_strictly_in_polygon(point, first_points) for point in second_points):
        return True
    return any(
        _segments_cross(a, b, c, d)
        for a, b in _segments(first_points, True)
        for c, d in _segments(second_points, True)
    )


def _crosses_line_and_polygon(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """判断一条线是否穿过多边形边界。"""

    line, polygon = (first, second) if first.get("kind") == "line" else (second, first)
    if line.get("kind") != "line" or polygon.get("kind") != "polygon":
        return False
    crossings = sum(
        _segments_intersect(a, b, c, d)
        for a, b in _segments(_geometry_vertex_list(line), False)
        for c, d in _segments(_geometry_vertex_list(polygon), True)
    )
    return crossings >= 1 and not _within(line, polygon)


def _direction(first: Mapping[str, Any], second: Mapping[str, Any], north_rotation: float | None) -> str:
    """根据质心差计算八方位；无指北依据时返回页面方向。"""

    first_center, second_center = _point(first.get("centroid")), _point(second.get("centroid"))
    if not first_center or not second_center:
        return ""
    dx, dy = first_center[0] - second_center[0], first_center[1] - second_center[1]
    if abs(dx) + abs(dy) < 1e-7:
        return ""
    if north_rotation is None:
        if abs(dx) > abs(dy) * 1.5:
            return "PAGE_RIGHT_OF" if dx > 0 else "PAGE_LEFT_OF"
        if abs(dy) > abs(dx) * 1.5:
            return "PAGE_BELOW" if dy > 0 else "PAGE_ABOVE"
        vertical = "BELOW" if dy > 0 else "ABOVE"
        horizontal = "RIGHT" if dx > 0 else "LEFT"
        return f"PAGE_{vertical}_{horizontal}_OF"
    radians = math.radians(north_rotation)
    east = dx * math.cos(radians) + dy * math.sin(radians)
    north = dx * math.sin(radians) - dy * math.cos(radians)
    angle = math.degrees(math.atan2(north, east)) % 360
    labels = ["EAST_OF", "NORTHEAST_OF", "NORTH_OF", "NORTHWEST_OF", "WEST_OF", "SOUTHWEST_OF", "SOUTH_OF", "SOUTHEAST_OF"]
    return labels[int((angle + 22.5) // 45) % 8]


def _semantic_containment_relation(entity_type: str) -> tuple[str, str]:
    """按源实体类型把几何包含翻译为地质语义或纯空间关系。"""

    if entity_type in _LITHOLOGY_TYPES:
        return "DISTRIBUTED_IN", "semantic"
    if entity_type in _FACIES_TYPES:
        return "DEVELOPED_IN", "semantic"
    if entity_type in _STRUCTURAL_TYPES:
        return "PART_OF", "structural"
    return "WITHIN", "spatial"


def _affine_transform(georeference: Mapping[str, Any], width: int, height: int) -> list[list[float]]:
    """使用至少三个控制点拟合像素到世界坐标的二维仿射变换。"""

    rows: list[list[float]] = []
    world_x: list[float] = []
    world_y: list[float] = []
    for raw in georeference.get("control_points") or []:
        if not isinstance(raw, Mapping):
            continue
        image_point = _point(raw.get("normalized_point") or raw.get("pixel_point"))
        world_point = _point(raw.get("world_point") or raw.get("coordinate"))
        if not image_point or not world_point:
            continue
        if raw.get("pixel_point") is not None:
            px, py = image_point
        else:
            px, py = image_point[0] * width / NORMALIZED_EXTENT, image_point[1] * height / NORMALIZED_EXTENT
        rows.append([px, py, 1.0])
        world_x.append(world_point[0])
        world_y.append(world_point[1])
    if len(rows) < 3 or not width or not height:
        return []
    matrix = np.asarray(rows, dtype=float)
    if np.linalg.matrix_rank(matrix) < 3:
        return []
    x_coeff = np.linalg.lstsq(matrix, np.asarray(world_x), rcond=None)[0]
    y_coeff = np.linalg.lstsq(matrix, np.asarray(world_y), rcond=None)[0]
    return [[round(float(value), 12) for value in x_coeff], [round(float(value), 12) for value in y_coeff]]


def _add_world_coordinates(geometry: dict[str, Any], transform: list[list[float]]) -> None:
    """利用仿射矩阵为一个几何对象补充世界坐标。"""

    pixels = _points(geometry.get("pixel_coordinates"))
    if not pixels or len(transform) != 2:
        return
    world = [
        [
            round(transform[0][0] * x + transform[0][1] * y + transform[0][2], 8),
            round(transform[1][0] * x + transform[1][1] * y + transform[1][2], 8),
        ]
        for x, y in pixels
    ]
    geometry["world_coordinates"] = world[0] if geometry.get("kind") == "point" else world
    geometry["world_bbox"] = _bbox(world)
    geometry["world_centroid"] = _centroid(world, str(geometry.get("kind")))


def _scale_bar_ratio(georeference: Mapping[str, Any]) -> tuple[float | None, str]:
    """从比例尺端点计算每个归一化坐标单位代表的实际距离。"""

    scale_bar = georeference.get("scale_bar")
    if not isinstance(scale_bar, Mapping):
        return None, ""
    start, end = _point(scale_bar.get("start")), _point(scale_bar.get("end"))
    distance = _number(scale_bar.get("distance"))
    if not start or not end or not distance or distance <= 0:
        return None, ""
    bar_length = math.dist(start, end)
    return (distance / bar_length, str(scale_bar.get("unit") or "").strip()) if bar_length > 0 else (None, "")


def reconstruct_map_geometry(task: ImageExtractionTask, result: dict[str, Any]) -> dict[str, Any]:
    """重建所有可定位实体的几何，并返回可审计的测量与推导关系。"""

    georeference = dict(result.get("georeference")) if isinstance(result.get("georeference"), Mapping) else {}
    width, height = _image_size(task, georeference)
    georeference["image_size"] = {"width": width, "height": height}
    georeference["coordinate_space"] = "normalized_1000"
    rotation = _number(georeference.get("north_rotation_degrees_clockwise_from_page_up"))
    transform = _affine_transform(georeference, width, height)
    if transform:
        georeference["pixel_to_world_affine"] = transform

    records: list[dict[str, Any]] = []
    for group in ("entities", "secondary_features"):
        for item in result.get(group) or []:
            if not isinstance(item, dict):
                continue
            item["geometry"] = normalize_geometry(item.get("geometry"), width, height)
            if transform:
                _add_world_coordinates(item["geometry"], transform)
            if item["geometry"].get("kind") != "unknown":
                records.append(item)

    measurements: list[dict[str, Any]] = []
    computed_relations: list[dict[str, Any]] = []
    ratio, distance_unit = _scale_bar_ratio(georeference)
    for index, first in enumerate(records):
        for second in records[index + 1:]:
            first_geometry, second_geometry = first["geometry"], second["geometry"]
            centroid_distance = math.dist(first_geometry["centroid"], second_geometry["centroid"])
            edge_distance = _edge_distance(first_geometry, second_geometry)
            direction = _direction(first_geometry, second_geometry, rotation)
            measurement = {
                "source_id": first["id"],
                "target_id": second["id"],
                "centroid_distance_normalized": round(centroid_distance, 3),
                "edge_distance_normalized": round(edge_distance, 3),
                "direction": direction,
            }
            if width and height:
                measurement["centroid_distance_pixels"] = round(math.dist(first_geometry["pixel_centroid"], second_geometry["pixel_centroid"]), 3)
            if ratio is not None:
                measurement["edge_distance_map"] = round(edge_distance * ratio, 6)
                measurement["distance_unit"] = distance_unit
            measurements.append(measurement)

            common = {
                "source_id": first["id"],
                "target_id": second["id"],
                "explicit": False,
                "computed": True,
                "calculation_basis": "normalized_geometry",
                "confidence": round(min(float(first.get("confidence") or 0.7), float(second.get("confidence") or 0.7), float(first_geometry.get("geometry_confidence") or 0.7), float(second_geometry.get("geometry_confidence") or 0.7)), 3),
                **measurement,
            }
            if _within(first_geometry, second_geometry):
                relation_type, dimension = _semantic_containment_relation(str(first.get("type") or ""))
                computed_relations.append({**common, "type": relation_type, "dimension": dimension, "evidence": "由归一化几何计算：源几何位于目标多边形内"})
            elif _within(second_geometry, first_geometry):
                relation_type, dimension = _semantic_containment_relation(str(second.get("type") or ""))
                computed_relations.append({**common, "source_id": second["id"], "target_id": first["id"], "type": relation_type, "dimension": dimension, "evidence": "由归一化几何计算：源几何位于目标多边形内"})
            elif _overlaps(first_geometry, second_geometry):
                computed_relations.append({**common, "type": "OVERLAPS", "dimension": "spatial", "evidence": "由归一化多边形计算：两区域相交"})
            elif _crosses_line_and_polygon(first_geometry, second_geometry):
                line_first = first_geometry.get("kind") == "line"
                computed_relations.append({
                    **common,
                    "source_id": first["id"] if line_first else second["id"],
                    "target_id": second["id"] if line_first else first["id"],
                    "type": "CROSSES",
                    "dimension": "spatial",
                    "evidence": "由归一化几何计算：线对象穿过面对象边界",
                })
            elif edge_distance <= 8.0 and first_geometry.get("kind") == second_geometry.get("kind") == "polygon":
                computed_relations.append({**common, "type": "ADJACENT_TO", "dimension": "spatial", "evidence": "由归一化多边形计算：两区域边界相邻"})
            elif edge_distance <= 1e-7:
                computed_relations.append({**common, "type": "TOUCHES", "dimension": "spatial", "evidence": "由归一化几何计算：两个几何边界接触"})
            # 中文说明：距离和方向完整保留在 measurements，避免把所有相近标签都提升为图谱边。
            # 只有模型给出明确证据时，NEAR 和方向关系才进入 spatial_relations。

    return {
        "schema_version": "map_geometry.v1",
        "coordinate_space": "normalized_1000",
        "axis": {"x": "right", "y": "down", "extent": [0, 0, 1000, 1000]},
        "georeference": georeference,
        "geometry_count": len(records),
        "measurements": measurements,
        "computed_relations": computed_relations,
    }


__all__ = ["NORMALIZED_EXTENT", "normalize_geometry", "reconstruct_map_geometry"]
