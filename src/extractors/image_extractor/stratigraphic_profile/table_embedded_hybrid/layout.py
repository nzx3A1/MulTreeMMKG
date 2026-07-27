"""混合表格图片的坐标标定、表格线检测与轨道重建。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


def _float(value: Any, field_name: str) -> float:
    """中文说明：把坐标字段转换为有限浮点数，并在输入错误时给出明确字段名。"""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 不是有效数值：{value!r}") from exc
    if not np.isfinite(number):
        raise ValueError(f"{field_name} 必须是有限数值：{value!r}")
    return number


def _bbox(value: Any, field_name: str) -> tuple[float, float, float, float]:
    """中文说明：校验并规范化左上右下像素框，避免负宽度轨道进入对齐阶段。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{field_name} 必须是 [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (_float(item, field_name) for item in value)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{field_name} 的右下坐标必须大于左上坐标：{value!r}")
    return x0, y0, x1, y1


@dataclass(frozen=True)
class LinearDepthTransform:
    """以多个像素—深度锚点拟合得到的线性纵轴变换。"""

    slope: float
    intercept: float
    unit: str
    increases: str
    rmse: float
    calibration_points: tuple[tuple[float, float], ...]

    def value_at(self, pixel_y: float) -> float:
        """中文说明：把图片纵坐标换算为深度或地层柱厚度值。"""

        return self.slope * float(pixel_y) + self.intercept

    def to_dict(self) -> dict[str, Any]:
        """中文说明：导出可写入结构化中间结果的坐标模型。"""

        return {
            "model": "linear_pixel_y_to_vertical_value",
            "formula": "value = slope * pixel_y + intercept",
            "slope": round(self.slope, 8),
            "intercept": round(self.intercept, 8),
            "unit": self.unit,
            "increases": self.increases,
            "rmse": round(self.rmse, 6),
            "calibration_points": [
                {"pixel_y": y, "value": value} for y, value in self.calibration_points
            ],
        }


def fit_vertical_axis(raw_axis: Mapping[str, Any]) -> LinearDepthTransform:
    """中文说明：用最小二乘从至少两个可见刻度锚点重建纵轴，并检查方向一致性。"""

    raw_points = raw_axis.get("calibration_points")
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        raise ValueError("coordinate_system.vertical_axis 至少需要两个 calibration_points")
    points: list[tuple[float, float]] = []
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, Mapping):
            raise ValueError(f"calibration_points[{index}] 必须是对象")
        points.append(
            (
                _float(raw.get("pixel_y"), f"calibration_points[{index}].pixel_y"),
                _float(raw.get("value"), f"calibration_points[{index}].value"),
            )
        )
    y_values = np.asarray([point[0] for point in points], dtype=float)
    axis_values = np.asarray([point[1] for point in points], dtype=float)
    if float(np.ptp(y_values)) == 0.0:
        raise ValueError("纵轴标定点不能全部位于同一 pixel_y")
    matrix = np.column_stack([y_values, np.ones_like(y_values)])
    slope, intercept = np.linalg.lstsq(matrix, axis_values, rcond=None)[0]
    predicted = slope * y_values + intercept
    rmse = float(np.sqrt(np.mean((predicted - axis_values) ** 2)))
    increases = str(raw_axis.get("increases") or "downward")
    if increases == "downward" and slope <= 0:
        raise ValueError("纵轴声明向下增大，但标定点拟合斜率不为正")
    if increases == "upward" and slope >= 0:
        raise ValueError("纵轴声明向上增大，但标定点拟合斜率不为负")
    return LinearDepthTransform(
        slope=float(slope),
        intercept=float(intercept),
        unit=str(raw_axis.get("unit") or ""),
        increases=increases,
        rmse=rmse,
        calibration_points=tuple(points),
    )


def _cluster_indices(indices: np.ndarray) -> list[int]:
    """中文说明：把连续的深色投影像素合并为一条中心表格线。"""

    if indices.size == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index <= groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [round(sum(group) / len(group)) for group in groups]


def detect_rule_lines(
    image_path: str | Path,
    content_bbox: Sequence[float],
    *,
    dark_threshold: int = 170,
    vertical_coverage: float = 0.45,
    horizontal_coverage: float = 0.55,
) -> dict[str, Any]:
    """中文说明：用灰度投影检测长直表格线，为模型识别的轨道边界提供独立像素证据。"""

    image_file = Path(image_path)
    if not image_file.is_file():
        return {
            "available": False,
            "reason": f"source_image_not_found:{image_file}",
            "vertical_lines": [],
            "horizontal_lines": [],
        }
    with Image.open(image_file) as image:
        gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    x0, y0, x1, y1 = _bbox(content_bbox, "content_bbox")
    left = max(0, min(width - 1, round(x0)))
    right = max(left + 1, min(width, round(x1)))
    top = max(0, min(height - 1, round(y0)))
    bottom = max(top + 1, min(height, round(y1)))
    dark = gray[top:bottom, left:right] < dark_threshold
    vertical = _cluster_indices(np.flatnonzero(dark.mean(axis=0) >= vertical_coverage))
    horizontal = _cluster_indices(np.flatnonzero(dark.mean(axis=1) >= horizontal_coverage))
    return {
        "available": True,
        "method": "grayscale_dark_pixel_projection",
        "parameters": {
            "dark_threshold": dark_threshold,
            "vertical_coverage": vertical_coverage,
            "horizontal_coverage": horizontal_coverage,
        },
        "vertical_lines": [left + value for value in vertical],
        "horizontal_lines": [top + value for value in horizontal],
    }


def rebuild_tracks(
    raw_tracks: Any,
    *,
    image_width: int,
    image_height: int,
    detected_lines: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """中文说明：排序并校验语义轨道，同时标记左右边界是否得到表格线检测支持。"""

    if not isinstance(raw_tracks, list) or not raw_tracks:
        raise ValueError("table_embedded_hybrid 响应缺少 tracks")
    vertical_lines = [float(value) for value in detected_lines.get("vertical_lines", [])]
    rebuilt: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_tracks):
        if not isinstance(raw, Mapping):
            raise ValueError(f"tracks[{index}] 必须是对象")
        track_id = str(raw.get("id") or "").strip()
        if not track_id or track_id in seen_ids:
            raise ValueError(f"tracks[{index}].id 缺失或重复：{track_id!r}")
        seen_ids.add(track_id)
        x0, y0, x1, y1 = _bbox(raw.get("bbox"), f"tracks[{index}].bbox")
        if x0 < 0 or y0 < 0 or x1 > image_width or y1 > image_height:
            raise ValueError(f"轨道 {track_id} 超出图片边界")

        def supported(x_value: float) -> bool:
            """中文说明：判断语义轨道边界附近是否存在独立检测到的长直表格线。"""

            return any(abs(line - x_value) <= 4.0 for line in vertical_lines)

        rebuilt.append(
            {
                **dict(raw),
                "id": track_id,
                "order": int(raw.get("order", index)),
                "bbox": [round(x0), round(y0), round(x1), round(y1)],
                "normalized_bbox": [
                    round(x0 / image_width, 6),
                    round(y0 / image_height, 6),
                    round(x1 / image_width, 6),
                    round(y1 / image_height, 6),
                ],
                "left_rule_supported": supported(x0),
                "right_rule_supported": supported(x1),
            }
        )
    rebuilt.sort(key=lambda item: (item["order"], item["bbox"][0]))
    return rebuilt
