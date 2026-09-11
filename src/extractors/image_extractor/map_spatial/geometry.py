"""地图对象几何规范化与确定性空间判定工具。

本文件只处理坐标、距离、拓扑和方向，不承担地质语义判断。
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence


NORMALIZED_EXTENT = 1000.0
_EPSILON = 1e-7


def _number(value: Any) -> float | None:
    """把有限数值安全转换为浮点数。"""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flat_numbers(value: Any) -> list[float]:
    """从列表或空格分隔字符串中提取有限数值。"""

    if isinstance(value, str):
        raw_values: Sequence[Any] = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = value
    else:
        return []
    numbers = [_number(item) for item in raw_values]
    return [number for number in numbers if number is not None]


def _point(value: Any) -> list[float] | None:
    """把一个坐标值规范为二维点。"""

    numbers = _flat_numbers(value)
    return numbers[:2] if len(numbers) >= 2 else None


def _points(value: Any) -> list[list[float]]:
    """把点或点列表规范为二维点列表。"""

    if isinstance(value, str):
        numbers = _flat_numbers(value)
        return [numbers[index:index + 2] for index in range(0, len(numbers) - 1, 2)]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    single = _point(value)
    if single and all(not isinstance(item, Sequence) or isinstance(item, str) for item in value):
        return [single]
    return [point for item in value if (point := _point(item))]


def _normalized_point(point: Sequence[float]) -> list[float]:
    """兼容 0~1 与 0~1000 坐标，并约束到图像范围。"""

    scale = NORMALIZED_EXTENT if max(abs(float(point[0])), abs(float(point[1]))) <= 1.000001 else 1.0
    return [
        round(max(0.0, min(NORMALIZED_EXTENT, float(point[0]) * scale)), 3),
        round(max(0.0, min(NORMALIZED_EXTENT, float(point[1]) * scale)), 3),
    ]


def _bbox_from_points(points: Sequence[Sequence[float]]) -> list[float]:
    """计算点集的轴对齐包围盒。"""

    return [
        round(min(point[0] for point in points), 3),
        round(min(point[1] for point in points), 3),
        round(max(point[0] for point in points), 3),
        round(max(point[1] for point in points), 3),
    ]


def _polygon_centroid(points: Sequence[Sequence[float]]) -> list[float]:
    """计算多边形质心，退化多边形回退为顶点平均值。"""

    if len(points) < 3:
        return [round(sum(point[0] for point in points) / len(points), 3), round(sum(point[1] for point in points) / len(points), 3)]
    twice_area = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        cross = current[0] * following[1] - following[0] * current[1]
        twice_area += cross
        x_sum += (current[0] + following[0]) * cross
        y_sum += (current[1] + following[1]) * cross
    if abs(twice_area) <= _EPSILON:
        return [round(sum(point[0] for point in points) / len(points), 3), round(sum(point[1] for point in points) / len(points), 3)]
    return [round(x_sum / (3.0 * twice_area), 3), round(y_sum / (3.0 * twice_area), 3)]


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """用鞋带公式计算归一化多边形面积。"""

    if len(points) < 3:
        return 0.0
    return round(abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2.0, 3)


def normalize_geometry(value: Any) -> dict[str, Any]:
    """统一 center、bbox、polygon 和旧 coordinates 表达。"""

    raw = dict(value) if isinstance(value, Mapping) else {}
    polygon = _points(raw.get("polygon") or raw.get("points"))
    kind = str(raw.get("kind") or "").strip().lower()
    coordinates = raw.get("coordinates")
    if not polygon and kind in {"line", "polygon"}:
        polygon = _points(coordinates)

    bbox_values = _flat_numbers(raw.get("bbox"))
    bbox: list[float] = []
    if len(bbox_values) >= 4:
        first = _normalized_point(bbox_values[:2])
        second = _normalized_point(bbox_values[2:4])
        bbox = [min(first[0], second[0]), min(first[1], second[1]), max(first[0], second[0]), max(first[1], second[1])]

    center = _point(raw.get("center") or raw.get("centroid") or raw.get("center_approx"))
    if center:
        center = _normalized_point(center)
    elif kind == "point" and (coordinate_point := _point(coordinates)):
        center = _normalized_point(coordinate_point)

    normalized_polygon = [_normalized_point(point) for point in polygon]
    if not bbox and normalized_polygon:
        bbox = _bbox_from_points(normalized_polygon)
    if not center and normalized_polygon:
        center = _polygon_centroid(normalized_polygon)
    if not center and bbox:
        center = [round((bbox[0] + bbox[2]) / 2.0, 3), round((bbox[1] + bbox[3]) / 2.0, 3)]

    resolved_kind = kind if kind in {"point", "line", "polygon"} else ""
    if not resolved_kind:
        resolved_kind = "polygon" if len(normalized_polygon) >= 3 or bbox else "point" if center else "unknown"
    if resolved_kind == "polygon" and not normalized_polygon and bbox:
        x0, y0, x1, y1 = bbox
        normalized_polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    if resolved_kind == "line" and len(normalized_polygon) < 2:
        resolved_kind = "unknown"

    result: dict[str, Any] = {"kind": resolved_kind, "coordinate_space": "normalized_1000"}
    if center:
        result["center"] = center
    if bbox:
        result["bbox"] = [round(number, 3) for number in bbox]
    if normalized_polygon:
        key = "points" if resolved_kind == "line" else "polygon"
        result[key] = normalized_polygon
    if resolved_kind == "polygon" and normalized_polygon:
        result["area"] = polygon_area(normalized_polygon)
    if raw.get("position"):
        result["position"] = str(raw["position"]).strip()
    confidence = _number(raw.get("confidence"))
    if confidence is not None:
        result["confidence"] = round(max(0.0, min(1.0, confidence)), 3)
    return result


def geometry_center(value: Any) -> list[float] | None:
    """读取规范几何的中心点。"""

    geometry = normalize_geometry(value)
    return _point(geometry.get("center"))


def _polygon(value: Any) -> list[list[float]]:
    """读取规范几何的多边形顶点。"""

    geometry = normalize_geometry(value)
    return _points(geometry.get("polygon")) if geometry.get("kind") == "polygon" else []


def _line(value: Any) -> list[list[float]]:
    """读取规范几何的折线顶点。"""

    geometry = normalize_geometry(value)
    return _points(geometry.get("points")) if geometry.get("kind") == "line" else []


def _orientation(first: Sequence[float], second: Sequence[float], third: Sequence[float]) -> float:
    """计算三点有向面积。"""

    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])


def _on_segment(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> bool:
    """判断点是否位于闭线段上。"""

    return (
        abs(_orientation(start, end, point)) <= _EPSILON
        and min(start[0], end[0]) - _EPSILON <= point[0] <= max(start[0], end[0]) + _EPSILON
        and min(start[1], end[1]) - _EPSILON <= point[1] <= max(start[1], end[1]) + _EPSILON
    )


def _segments(points: Sequence[Sequence[float]], *, closed: bool) -> list[tuple[Sequence[float], Sequence[float]]]:
    """把点序列展开成线段列表。"""

    segments = list(zip(points, points[1:]))
    if closed and len(points) > 2:
        segments.append((points[-1], points[0]))
    return segments


def _segments_intersect(first_start: Sequence[float], first_end: Sequence[float], second_start: Sequence[float], second_end: Sequence[float]) -> bool:
    """判断两条闭线段是否相交。"""

    first_a = _orientation(first_start, first_end, second_start)
    first_b = _orientation(first_start, first_end, second_end)
    second_a = _orientation(second_start, second_end, first_start)
    second_b = _orientation(second_start, second_end, first_end)
    if first_a * first_b < -_EPSILON and second_a * second_b < -_EPSILON:
        return True
    return (
        abs(first_a) <= _EPSILON and _on_segment(second_start, first_start, first_end)
        or abs(first_b) <= _EPSILON and _on_segment(second_end, first_start, first_end)
        or abs(second_a) <= _EPSILON and _on_segment(first_start, second_start, second_end)
        or abs(second_b) <= _EPSILON and _on_segment(first_end, second_start, second_end)
    )


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """用射线法判断点是否位于多边形内部或边界。"""

    if len(polygon) < 3:
        return False
    if any(_on_segment(point, polygon[index - 1], polygon[index]) for index in range(len(polygon))):
        return True
    x, y = point
    inside = False
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        if (current[1] > y) != (previous[1] > y):
            crossing_x = (previous[0] - current[0]) * (y - current[1]) / (previous[1] - current[1]) + current[0]
            if x < crossing_x:
                inside = not inside
    return inside


def _point_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    """计算点到线段的最短欧氏距离。"""

    dx, dy = end[0] - start[0], end[1] - start[1]
    if abs(dx) + abs(dy) <= _EPSILON:
        return math.dist(point, start)
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)))
    projection = [start[0] + ratio * dx, start[1] + ratio * dy]
    return math.dist(point, projection)


def geometry_distance(first: Any, second: Any) -> float:
    """计算两个几何对象边界之间的近似最短距离。"""

    first_geometry = normalize_geometry(first)
    second_geometry = normalize_geometry(second)
    first_center = geometry_center(first_geometry)
    second_center = geometry_center(second_geometry)
    first_points = _polygon(first_geometry) or _line(first_geometry) or ([first_center] if first_center else [])
    second_points = _polygon(second_geometry) or _line(second_geometry) or ([second_center] if second_center else [])
    if not first_points or not second_points:
        return math.inf
    first_segments = _segments(first_points, closed=first_geometry.get("kind") == "polygon")
    second_segments = _segments(second_points, closed=second_geometry.get("kind") == "polygon")
    if any(_segments_intersect(a, b, c, d) for a, b in first_segments for c, d in second_segments):
        return 0.0
    distances = [math.dist(a, b) for a in first_points for b in second_points]
    distances.extend(_point_segment_distance(point, start, end) for point in first_points for start, end in second_segments)
    distances.extend(_point_segment_distance(point, start, end) for point in second_points for start, end in first_segments)
    return min(distances) if distances else math.inf


def geometry_within(first: Any, second: Any) -> bool:
    """判断第一个几何对象是否位于第二个多边形内。"""

    second_polygon = _polygon(second)
    if not second_polygon:
        return False
    first_geometry = normalize_geometry(first)
    first_points = _polygon(first_geometry) or _line(first_geometry)
    if not first_points and (center := geometry_center(first_geometry)):
        first_points = [center]
    return bool(first_points) and all(point_in_polygon(point, second_polygon) for point in first_points)


def polygons_overlap(first: Any, second: Any) -> bool:
    """判断两个多边形是否有面积交叠且不是单纯包含。"""

    first_polygon, second_polygon = _polygon(first), _polygon(second)
    if not first_polygon or not second_polygon or geometry_within(first, second) or geometry_within(second, first):
        return False
    if any(point_in_polygon(point, second_polygon) for point in first_polygon):
        return True
    if any(point_in_polygon(point, first_polygon) for point in second_polygon):
        return True
    return any(
        _segments_intersect(a, b, c, d)
        for a, b in _segments(first_polygon, closed=True)
        for c, d in _segments(second_polygon, closed=True)
    )


def line_crosses_polygon(line_geometry: Any, polygon_geometry: Any) -> bool:
    """判断折线是否穿过多边形边界。"""

    line, polygon = _line(line_geometry), _polygon(polygon_geometry)
    if len(line) < 2 or len(polygon) < 3:
        return False
    intersections = sum(
        _segments_intersect(a, b, c, d)
        for a, b in _segments(line, closed=False)
        for c, d in _segments(polygon, closed=True)
    )
    return intersections >= 1


def relation_matches_geometry(relation_type: str, source_geometry: Any, target_geometry: Any) -> bool:
    """按关系类型使用几何证据进行确定性确认。"""

    relation = str(relation_type or "").strip().lower()
    if relation in {"located_in", "contained_in"}:
        return geometry_within(source_geometry, target_geometry)
    if relation == "surrounds":
        return geometry_within(target_geometry, source_geometry)
    if relation == "overlaps":
        return polygons_overlap(source_geometry, target_geometry)
    if relation == "crosses":
        return line_crosses_polygon(source_geometry, target_geometry)
    if relation == "adjacent_to":
        return not polygons_overlap(source_geometry, target_geometry) and geometry_distance(source_geometry, target_geometry) <= 12.0
    if relation in {"located_near", "near_boundary_of", "near_or_within", "along_boundary_of"}:
        return geometry_within(source_geometry, target_geometry) or geometry_distance(source_geometry, target_geometry) <= 25.0
    if relation == "outside_erosion_boundary":
        return bool(geometry_center(source_geometry)) and normalize_geometry(target_geometry).get("kind") in {"line", "polygon"}
    return True


def calculate_direction(source_center: Sequence[float], target_center: Sequence[float], *, significance_ratio: float = 2.0) -> str:
    """按图像坐标差与显著性阈值计算源相对目标的八方位。"""

    dx = float(source_center[0]) - float(target_center[0])
    dy = float(source_center[1]) - float(target_center[1])
    if abs(dx) + abs(dy) <= _EPSILON:
        return ""
    if abs(dx) > significance_ratio * abs(dy):
        return "east_of" if dx > 0 else "west_of"
    if abs(dy) > significance_ratio * abs(dx):
        return "south_of" if dy > 0 else "north_of"
    vertical = "south" if dy > 0 else "north"
    horizontal = "east" if dx > 0 else "west"
    return f"{vertical}{horizontal}_of"


def build_spatial_chain(
    entities: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    *,
    anchor_id: str = "",
) -> list[dict[str, Any]]:
    """以主锚点构造覆盖候选实体的最近邻稀疏有向链。"""

    by_id = {str(entity.get("id") or ""): entity for entity in entities}
    ordered_ids = list(dict.fromkeys(str(value) for value in candidate_ids if str(value) in by_id))
    candidates = [by_id[entity_id] for entity_id in ordered_ids if geometry_center(by_id[entity_id].get("geometry"))]
    if len(candidates) < 2:
        return []

    def priority(entity: Mapping[str, Any]) -> float:
        """读取并约束空间候选优先级。"""

        value = _number(entity.get("spatial_mapping_priority"))
        if value is None and isinstance(entity.get("attributes"), Mapping):
            value = _number(entity["attributes"].get("spatial_mapping_priority"))
        return max(0.0, min(1.0, value if value is not None else 0.5))

    anchor = by_id.get(anchor_id)
    if anchor not in candidates:
        anchor = max(candidates, key=lambda entity: (priority(entity), -ordered_ids.index(str(entity.get("id")))))
    remaining = [entity for entity in candidates if entity is not anchor]

    def weighted_distance(source: Mapping[str, Any], target: Mapping[str, Any]) -> tuple[float, float, int]:
        """按距离优先，并用候选优先级和原顺序稳定打破平局。"""

        source_center = geometry_center(source.get("geometry")) or [0.0, 0.0]
        target_center = geometry_center(target.get("geometry")) or [0.0, 0.0]
        return (math.dist(source_center, target_center), -priority(target), ordered_ids.index(str(target.get("id"))))

    first = min(remaining, key=lambda entity: weighted_distance(anchor, entity))
    chain = [first, anchor]
    remaining.remove(first)
    tail = anchor
    while remaining:
        following = min(remaining, key=lambda entity: weighted_distance(tail, entity))
        chain.append(following)
        remaining.remove(following)
        tail = following

    edges: list[dict[str, Any]] = []
    for source, target in zip(chain, chain[1:]):
        source_center = geometry_center(source.get("geometry"))
        target_center = geometry_center(target.get("geometry"))
        if not source_center or not target_center:
            continue
        relation = calculate_direction(source_center, target_center)
        if relation:
            edges.append({
                "source_id": str(source["id"]),
                "target_id": str(target["id"]),
                "type": relation,
                "confidence": round(min(priority(source), priority(target)), 3),
                "mapping_basis": "entity_center_coordinates",
            })
    return edges


__all__ = [
    "NORMALIZED_EXTENT",
    "build_spatial_chain",
    "calculate_direction",
    "geometry_center",
    "geometry_distance",
    "geometry_within",
    "line_crosses_polygon",
    "normalize_geometry",
    "point_in_polygon",
    "polygons_overlap",
    "relation_matches_geometry",
]
