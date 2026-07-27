"""判断图片中是否包含表格区域。

脚本优先使用 PaddleOCR 3.x 的表格专用 PicoDet 模型做版面检测；模型不可用时，
自动使用基于 Pillow 和 NumPy 的横纵网格线检测，保证脚本仍可直接运行。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_IMAGE = Path(__file__).with_name("Snipaste_2026-07-15_22-39-44.png")


def _merge_adjacent_positions(positions: np.ndarray) -> list[tuple[int, int]]:
    """把相邻的像素坐标合并成线段，避免将一条粗线重复计数。"""
    if positions.size == 0:
        return []

    groups: list[tuple[int, int]] = []
    start = previous = int(positions[0])
    for value in positions[1:]:
        current = int(value)
        if current > previous + 1:
            groups.append((start, previous))
            start = current
        previous = current
    groups.append((start, previous))
    return groups


def detect_table_by_grid(image_path: Path) -> tuple[bool, dict[str, Any]]:
    """通过长横线和长纵线数量判断规则网格型图片是否包含表格。"""
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"))

    # 深色像素视为文字或表格线；表格线会在整行/整列形成较高覆盖率。
    dark_pixels = gray < 190
    vertical_positions = np.flatnonzero(dark_pixels.mean(axis=0) >= 0.50)
    horizontal_positions = np.flatnonzero(dark_pixels.mean(axis=1) >= 0.25)
    vertical_lines = _merge_adjacent_positions(vertical_positions)
    horizontal_lines = _merge_adjacent_positions(horizontal_positions)

    # 至少 4 条纵线和 4 条横线可构成 3×3 网格，降低普通文字误判为表格的概率。
    has_table = len(vertical_lines) >= 4 and len(horizontal_lines) >= 4
    details = {
        "backend": "grid",
        "image_size": [int(gray.shape[1]), int(gray.shape[0])],
        "vertical_line_count": len(vertical_lines),
        "horizontal_line_count": len(horizontal_lines),
        "vertical_lines": vertical_lines,
        "horizontal_lines": horizontal_lines,
    }
    return has_table, details


def _normalize_regions(raw_result: Any) -> list[dict[str, Any]]:
    """兼容 PaddleOCR 3.x 与旧版 ppstructure 的结果结构，提取版面区域。"""
    pages = [raw_result] if isinstance(raw_result, Mapping) else list(raw_result)
    detected_regions: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, Mapping):
            continue

        page_data = page.get("res", page)
        items = page_data.get("boxes", []) if isinstance(page_data, Mapping) else []
        # 旧版 ppstructure 直接返回区域列表，此处保持向后兼容。
        if not items and "label" in page:
            items = [page]
        for item in items:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label", item.get("type", ""))).strip().lower()
            score = float(item.get("score", item.get("confidence", 0.0)))
            bbox = item.get("coordinate", item.get("bbox", item.get("box")))
            # 将 NumPy 标量转成普通浮点数，让终端输出和后续 JSON 序列化更清晰。
            normalized_bbox = [float(value) for value in bbox] if bbox is not None else None
            detected_regions.append(
                {
                    "label": label,
                    "bbox": normalized_bbox,
                    "score": score,
                }
            )
    return detected_regions


def detect_table_by_layout_model(
    image_path: Path,
    model_name: str,
    score_threshold: float,
) -> tuple[bool, dict[str, Any]]:
    """调用 PaddleOCR 版面模型，根据 table 类别及置信度判断是否存在表格。"""
    try:
        # treeSchemeKG 使用 PaddleOCR 3.x，对应官方 LayoutDetection 接口。
        from paddleocr import LayoutDetection

        # treeSchemeKG 的 PaddlePaddle 3.3.1 在 Windows CPU 的 oneDNN 路径存在
        # PIR 属性转换错误，显式关闭 MKLDNN 后改用标准 Paddle CPU 算子。
        layout_model = LayoutDetection(model_name=model_name, enable_mkldnn=False)
        raw_result = layout_model.predict(str(image_path), batch_size=1, layout_nms=True)
        backend_name = "paddleocr-layout"
    except ImportError:
        # 仅为兼容安装了旧版独立 ppstructure 包的 Python 环境。
        from ppstructure.layout import LayoutPredictor

        layout_model = LayoutPredictor(model_name=model_name)
        raw_result = layout_model.predict(str(image_path))
        backend_name = "pp-layout"

    detected_regions = _normalize_regions(raw_result)

    has_table = any(
        region["label"] == "table" and region["score"] >= score_threshold
        for region in detected_regions
    )
    return has_table, {"backend": backend_name, "regions": detected_regions}


def detect_table(
    image_path: Path,
    backend: str = "auto",
    model_name: str = "PicoDet_layout_1x_table",
    score_threshold: float = 0.50,
) -> tuple[bool, dict[str, Any]]:
    """校验输入并选择检测后端；auto 模式在模型不可用时自动回退。"""
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    if backend == "grid":
        return detect_table_by_grid(image_path)

    try:
        return detect_table_by_layout_model(image_path, model_name, score_threshold)
    except (ImportError, ModuleNotFoundError) as exc:
        if backend in {"paddleocr", "pp-layout"}:
            raise RuntimeError(
                "PaddleOCR 版面检测依赖未安装，请在 treeSchemeKG 环境安装 paddleocr 和 paddle。"
            ) from exc
        print(f"提示：PaddleOCR 版面模型不可用（{exc}），自动改用网格线检测。")
        return detect_table_by_grid(image_path)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，并默认检测 DEFAULT_IMAGE 指定的同目录图片。"""
    parser = argparse.ArgumentParser(description="判断一张图片中是否包含表格")
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--backend",
        choices=("auto", "paddleocr", "pp-layout", "grid"),
        default="auto",
        help="检测方式，默认优先 PaddleOCR，依赖缺失时回退到网格线检测",
    )
    parser.add_argument(
        "--model",
        default="PicoDet_layout_1x_table",
        help="PaddleOCR 版面模型名称，默认使用表格区域专用模型",
    )
    parser.add_argument("--threshold", type=float, default=0.50, help="模型置信度阈值")
    return parser.parse_args()


def main() -> int:
    """执行单张图片检测，打印判断依据，并用退出码表示检测结果。"""
    args = parse_args()
    try:
        has_table, details = detect_table(
            args.image,
            backend=args.backend,
            model_name=args.model,
            score_threshold=args.threshold,
        )
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        print(f"检测失败：{exc}")
        return 2

    print(f"检测图片：{args.image.resolve()}")
    print(f"检测后端：{details['backend']}")
    if details["backend"] in {"paddleocr-layout", "pp-layout"}:
        for region in details["regions"]:
            print(
                f"类别：{region['label'] or 'unknown'}，坐标：{region['bbox']}，"
                f"置信度：{region['score']:.3f}"
            )
    else:
        print(
            f"检测到长纵线 {details['vertical_line_count']} 条，"
            f"长横线 {details['horizontal_line_count']} 条。"
        )

    print("判断结果：该图片包含表格。" if has_table else "判断结果：该图片不包含表格。")
    return 0 if has_table else 1


if __name__ == "__main__":
    raise SystemExit(main())
