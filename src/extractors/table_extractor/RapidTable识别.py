"""使用 RapidOCR + RapidTable 验证表格图片的 HTML 还原效果。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from rapidocr import RapidOCR
from rapidocr.utils.typings import (
    EngineType,
    LangDet,
    LangRec,
    ModelType as OCRModelType,
    OCRVersion,
)
from rapid_table import ModelType, RapidTable, RapidTableInput


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "Snipaste_2026-07-15_22-28-18.png"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "rapidtable_outputs"

MODEL_TYPES = {
    "slanetplus": ModelType.SLANETPLUS,
    "unitable": ModelType.UNITABLE,
    "ppstructure_zh": ModelType.PPSTRUCTURE_ZH,
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数，默认使用 Unitable 和可用的 GPU 进行高精度验证。"""
    parser = argparse.ArgumentParser(
        description="用 RapidOCR + RapidTable 将表格图片还原为 HTML。"
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="表格图片路径")
    parser.add_argument(
        "--model",
        choices=MODEL_TYPES,
        default="unitable",
        help="表格结构模型；默认使用精度更高的 Unitable",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="cuda",
        help="推理设备；默认强制使用 CUDA，缺少 GPU 时直接报错",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="HTML、JSON 和可视化结果的保存目录",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    """递归转换 NumPy 等对象，保证 RapidTable 结果可以写入 JSON。"""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def validate_image(image_path: Path) -> Path:
    """校验输入图片并返回绝对路径，避免因运行目录不同导致找不到文件。"""
    resolved_path = image_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"表格图片不存在：{resolved_path}")
    return resolved_path


def save_image_unicode(image_path: Path, image: np.ndarray) -> None:
    """通过内存编码保存图片，绕过 OpenCV 在 Windows 中文路径下写入失败的问题。"""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    extension = image_path.suffix or ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"OpenCV 无法编码可视化图片：{image_path}")
    encoded.tofile(image_path)


def draw_logic_points(
    image: np.ndarray,
    cell_bboxes: np.ndarray,
    logic_points: np.ndarray,
) -> np.ndarray:
    """在单元格上绘制逻辑行列范围，便于核对跨行和跨列识别是否正确。"""
    canvas = cv2.copyMakeBorder(
        image, 0, 0, 0, 120, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    for bbox, logic_point in zip(cell_bboxes, logic_points):
        points = np.asarray(bbox, dtype=np.int32).reshape(-1, 2)
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.putText(
            canvas,
            f"r:{logic_point[0]}-{logic_point[1]}",
            (x0 + 3, y0 + 10),
            cv2.FONT_HERSHEY_PLAIN,
            0.8,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            canvas,
            f"c:{logic_point[2]}-{logic_point[3]}",
            (x0 + 3, y0 + 20),
            cv2.FONT_HERSHEY_PLAIN,
            0.8,
            (0, 0, 255),
            1,
        )
    return canvas


def resolve_device(model_name: str, requested_device: str) -> str:
    """根据模型推理后端和本机 Provider，解析最终使用的 CPU 或 CUDA 设备。"""
    if requested_device == "cpu":
        return "cpu"

    if model_name == "unitable":
        cuda_available = torch.cuda.is_available()
    else:
        cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()

    if requested_device == "cuda" and not cuda_available:
        backend = "PyTorch" if model_name == "unitable" else "ONNXRuntime"
        raise RuntimeError(f"{backend} 当前没有可用的 CUDA 后端。")
    return "cuda" if cuda_available else "cpu"


def build_engine_config(model_name: str, device: str) -> dict[str, Any]:
    """生成 RapidTable 对 PyTorch Unitable 或 ONNX 模型所需的设备配置。"""
    if device != "cuda":
        return {"use_cuda": False}
    if model_name == "unitable":
        return {"use_cuda": True, "gpu_id": 0}
    return {"use_cuda": True, "cuda_ep_cfg.gpu_id": 0}


def build_ocr_config(device: str) -> dict[str, Any]:
    """构造高精度 PyTorch OCR 配置，并让检测与识别阶段统一使用 GPU。"""
    return {
        "EngineConfig.torch.use_cuda": device == "cuda",
        "EngineConfig.torch.cuda_ep_cfg.device_id": 0,
        "Det.engine_type": EngineType.TORCH,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.lang_type": LangDet.CH,
        "Det.model_type": OCRModelType.SERVER,
        # 放大检测输入并降低阈值，避免右下角小数字在检测阶段被过滤。
        "Det.limit_side_len": 1920,
        "Det.thresh": 0.2,
        "Det.box_thresh": 0.3,
        "Rec.engine_type": EngineType.TORCH,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": OCRModelType.SERVER,
    }


def box_bounds(box: np.ndarray) -> tuple[float, float, float, float]:
    """计算四点文字框的外接矩形，供小下标的几何关系判断使用。"""
    points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def is_cjk_text(text: str) -> bool:
    """判断文本是否含中日韩字符，以筛选可能携带右下角标的主体文本。"""
    return any("\u3400" <= char <= "\u9fff" for char in text)


def merge_detected_subscripts(
    boxes: np.ndarray,
    texts: tuple[str, ...],
    scores: tuple[float, ...],
) -> tuple[list[np.ndarray], list[str], list[float]]:
    """将 OCR 单独检出的右下角小数字合并到前一个主体文字框。"""
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    merged_boxes = [np.asarray(box, dtype=np.float32) for box in boxes]
    merged_texts = [text.strip() for text in texts]
    merged_scores = [float(score) for score in scores]
    consumed_indices: set[int] = set()

    for digit_index, (box, text) in enumerate(zip(merged_boxes, merged_texts)):
        normalized_text = text.strip()
        # PP-OCR 偶尔会把极小的数字 5 识别成形状相近的大写 S。
        digit_text = "5" if normalized_text == "S" else normalized_text
        if not (digit_text.isascii() and digit_text.isdigit()):
            continue

        x0, y0, x1, y1 = box_bounds(box)
        height = max(y1 - y0, 1.0)
        candidates: list[tuple[float, int]] = []
        # 小数字按纵坐标排序后不一定紧邻主体文本，因此在全部文字框中找最近主体。
        for base_index, (base_box, base_text) in enumerate(
            zip(merged_boxes, merged_texts)
        ):
            if base_index == digit_index or not is_cjk_text(base_text):
                continue
            bx0, by0, bx1, by1 = box_bounds(base_box)
            base_height = max(by1 - by0, 1.0)
            horizontal_gap = x0 - bx1
            lower_right = (
                height <= base_height * 0.6
                and -base_height * 0.4 <= horizontal_gap <= base_height * 0.55
                and (y0 + y1) / 2 >= (by0 + by1) / 2
                and y0 <= by1 + base_height * 0.3
                and y1 >= by0 + base_height * 0.5
            )
            if lower_right:
                distance = abs(horizontal_gap) + abs((y0 + y1) / 2 - by1)
                candidates.append((distance, base_index))

        if candidates:
            _, base_index = min(candidates)
            merged_texts[base_index] += digit_text.translate(subscript_map)
            merged_scores[base_index] = min(
                merged_scores[base_index], merged_scores[digit_index]
            )
            consumed_indices.add(digit_index)

    return (
        [box for index, box in enumerate(merged_boxes) if index not in consumed_indices],
        [text for index, text in enumerate(merged_texts) if index not in consumed_indices],
        [score for index, score in enumerate(merged_scores) if index not in consumed_indices],
    )


def recover_joined_subscripts(
    image: np.ndarray,
    ocr_engine: RapidOCR,
    boxes: list[np.ndarray],
    texts: list[str],
    scores: list[float],
) -> tuple[list[np.ndarray], list[str], list[float]]:
    """局部复识别与主体字黏连的右下角数字，并转换成 Unicode 下标。"""
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    cjk_heights = [
        box_bounds(box)[3] - box_bounds(box)[1]
        for box, text in zip(boxes, texts)
        if is_cjk_text(text) and not any(char in text for char in "₀₁₂₃₄₅₆₇₈₉")
    ]
    if not cjk_heights:
        return boxes, texts, scores

    median_height = float(np.median(cjk_heights))
    image_height, image_width = image.shape[:2]
    for index, (box, text) in enumerate(zip(boxes, texts)):
        x0, y0, x1, y1 = box_bounds(box)
        height = max(y1 - y0, 1.0)
        # 黏连下标会把主体框向右下拉高；仅复查明显高于正文中位数的中文框。
        if (
            not is_cjk_text(text)
            or any(char in text for char in "₀₁₂₃₄₅₆₇₈₉")
            or height < median_height * 1.2
        ):
            continue

        crop_x0 = max(0, int(round(x1 - height * 0.5)))
        crop_x1 = min(image_width, int(round(x1 + height * 0.35)))
        crop_y0 = max(0, int(round(y0 + height * 0.45)))
        crop_y1 = min(image_height, int(round(y1 + height * 0.2)))
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            continue

        crop = image[crop_y0:crop_y1, crop_x0:crop_x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        enlarged = cv2.resize(binary, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        enlarged = cv2.copyMakeBorder(
            enlarged, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255
        )
        retry_result = ocr_engine(
            enlarged,
            use_det=False,
            use_cls=False,
            use_rec=True,
        )
        retry_text = retry_result.txts[0] if retry_result.txts else ""
        retry_score = float(retry_result.scores[0]) if retry_result.scores else 0.0
        digits = re.findall(r"\d", retry_text)
        if digits and retry_score >= 0.4:
            texts[index] = text.rstrip("，。；;:：") + digits[-1].translate(subscript_map)
            scores[index] = min(scores[index], retry_score)

    return boxes, texts, scores


def align_subscripts_with_identifiers(
    boxes: list[np.ndarray],
    texts: list[str],
) -> list[str]:
    """利用同一行相邻编号的末位数字，校正低清晰度图片标注中的层位下标。"""
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    subscript_pattern = re.compile(r"[₀₁₂₃₄₅₆₇₈₉]$")
    identifier_pattern = re.compile(r"\d(\d)[，,]?")

    for base_index, (base_box, base_text) in enumerate(zip(boxes, texts)):
        if not is_cjk_text(base_text):
            continue
        bx0, by0, bx1, by1 = box_bounds(base_box)
        base_height = max(by1 - by0, 1.0)
        candidates: list[tuple[float, str]] = []
        for identifier_index, (identifier_box, identifier_text) in enumerate(
            zip(boxes, texts)
        ):
            if identifier_index == base_index:
                continue
            match = identifier_pattern.search(identifier_text.strip())
            if match is None or match.end() != len(identifier_text.strip()):
                continue
            ix0, iy0, ix1, iy1 = box_bounds(identifier_box)
            horizontal_gap = bx0 - ix1
            same_line = abs((by0 + by1) / 2 - (iy0 + iy1) / 2) <= base_height
            if same_line and -base_height <= horizontal_gap <= base_height * 4:
                candidates.append((abs(horizontal_gap), match.group(1)))

        if candidates:
            _, expected_digit = min(candidates)
            expected_subscript = expected_digit.translate(subscript_map)
            if subscript_pattern.search(base_text):
                texts[base_index] = subscript_pattern.sub(expected_subscript, base_text)
            else:
                texts[base_index] = base_text + expected_subscript

    return texts


def run_recognition(
    image_path: Path,
    output_dir: Path,
    model_name: str,
    requested_device: str,
) -> dict[str, Any]:
    """执行 OCR 和表格结构识别，并将可复查的中间结果完整落盘。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type = MODEL_TYPES[model_name]
    device = resolve_device(model_name, requested_device)
    engine_cfg = build_engine_config(model_name, device)

    print(f"[1/4] 初始化 GPU RapidOCR，图片：{image_path}")
    ocr_engine = RapidOCR(params=build_ocr_config(device))
    ocr_start = time.perf_counter()
    # 通过内存解码读取图片，避免 OpenCV 在 Windows 中文路径下读取失败。
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV 无法读取表格图片：{image_path}")
    ocr_result = ocr_engine(
        image,
        use_cls=False,
        text_score=0.3,
        box_thresh=0.3,
        return_single_char_box=True,
    )
    ocr_elapsed = time.perf_counter() - ocr_start

    if ocr_result.boxes is None or ocr_result.txts is None or ocr_result.scores is None:
        raise RuntimeError("RapidOCR 未识别到可供表格结构匹配的文字。")

    ocr_boxes, ocr_texts, ocr_scores = merge_detected_subscripts(
        ocr_result.boxes,
        ocr_result.txts,
        ocr_result.scores,
    )
    ocr_boxes, ocr_texts, ocr_scores = recover_joined_subscripts(
        image,
        ocr_engine,
        ocr_boxes,
        ocr_texts,
        ocr_scores,
    )
    ocr_texts = align_subscripts_with_identifiers(ocr_boxes, ocr_texts)
    print(f"[2/4] OCR 完成：识别到 {len(ocr_texts)} 段文字，耗时 {ocr_elapsed:.3f} 秒")
    ocr_results = [(np.asarray(ocr_boxes), ocr_texts, ocr_scores)]

    print(f"[3/4] 初始化 RapidTable，模型：{model_type.value}，设备：{device}")
    # RapidTable v3 需要保持 use_ocr=True 才会执行 OCR 文本与结构单元格的 HTML 匹配。
    table_engine = RapidTable(
        RapidTableInput(model_type=model_type, engine_cfg=engine_cfg)
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    table_start = time.perf_counter()
    # Unitable 自回归解码必须关闭梯度图，否则长表格会持续累积显存并导致 OOM。
    if model_name == "unitable":
        with torch.inference_mode():
            table_result = table_engine(image_path, ocr_results=ocr_results)
    else:
        table_result = table_engine(image_path, ocr_results=ocr_results)
    if device == "cuda":
        torch.cuda.synchronize()
    table_elapsed = time.perf_counter() - table_start

    if not table_result.pred_htmls:
        raise RuntimeError("RapidTable 没有返回 HTML 结果。")

    html = table_result.pred_htmls[0]
    html_path = output_dir / f"{model_name}_table.html"
    detail_path = output_dir / f"{model_name}_details.json"
    html_path.write_text(html, encoding="utf-8")

    details = {
        "image": str(image_path),
        "model": model_type.value,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 3)
            if device == "cuda"
            else None
        ),
        "package_versions": {
            "rapidocr": importlib.metadata.version("rapidocr"),
            "rapid-table": importlib.metadata.version("rapid-table"),
        },
        "timing_seconds": {
            "ocr": round(ocr_elapsed, 6),
            "table": round(table_elapsed, 6),
            "total": round(ocr_elapsed + table_elapsed, 6),
            "rapid_table_reported": round(float(table_result.elapse), 6),
        },
        "ocr": [
            {
                "text": text,
                "score": float(score),
                "box": to_jsonable(box),
            }
            for box, text, score in zip(ocr_boxes, ocr_texts, ocr_scores)
        ],
        "logic_points": to_jsonable(table_result.logic_points[0]),
        "cell_bboxes": to_jsonable(table_result.cell_bboxes[0]),
        "html_path": str(html_path),
    }
    detail_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 先调用官方可视化生成带边框的 HTML，再用 Unicode 安全方式补存两张 JPG。
    visual_images = table_result.vis(
        save_dir=output_dir / f"{model_name}_visualization",
        save_name=f"{model_name}_result",
    )
    visualization_dir = output_dir / f"{model_name}_visualization" / "0"
    structure_path = visualization_dir / f"{model_name}_result_vis.jpg"
    logic_path = visualization_dir / f"{model_name}_result_col_row_vis.jpg"
    save_image_unicode(structure_path, visual_images[0])
    logic_image = draw_logic_points(
        table_result.imgs[0],
        table_result.cell_bboxes[0],
        table_result.logic_points[0],
    )
    save_image_unicode(logic_path, logic_image)

    print(f"[4/4] 识别完成：表格结构耗时 {table_elapsed:.3f} 秒，设备 {device}")
    print(f"HTML：{html_path}")
    print(f"详情：{detail_path}")
    print(f"可视化：{visualization_dir}")
    return details


def main() -> None:
    """组织参数校验、模型推理与结果摘要输出。"""
    args = parse_args()
    image_path = validate_image(args.image)
    details = run_recognition(
        image_path=image_path,
        output_dir=args.output_dir.expanduser().resolve(),
        model_name=args.model,
        requested_device=args.device,
    )
    timing = details["timing_seconds"]
    print(
        "结果摘要："
        f"OCR={timing['ocr']:.3f}s，表格={timing['table']:.3f}s，"
        f"总计={timing['total']:.3f}s，OCR文本数={len(details['ocr'])}"
    )


if __name__ == "__main__":
    main()
