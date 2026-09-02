"""对通用小文字单元格执行局部 OCR，并提供亚段跨列校验和覆盖率审计。"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from PIL import Image, ImageOps


SMALL_CELL_REFINEMENT_VERSION = "small-cell-ocr-with-submember-crosscheck.v3"
CellRecognizer = Callable[
    [Path, Sequence[Mapping[str, Any]]],
    Mapping[str, Sequence[Mapping[str, Any]]],
]
_LOCAL_OCR_PIPELINE: Any | None = None
_CLEAN_PATTERN = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_CHINESE_SUBMEMBER_PATTERN = re.compile(r"^(?P<base>[\u4e00-\u9fff]{2,}?)(?P<suffix>\d{1,2})?$")


def _clean_text(value: Any) -> str:
    """中文说明：移除空白和标点，使小下标、普通数字与 OCR 变体能够统一比较。"""

    digit_translation = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉", "01234567890123456789")
    return _CLEAN_PATTERN.sub("", str(value or "").translate(digit_translation)).strip()


def _bbox(record: Mapping[str, Any]) -> list[float]:
    """中文说明：读取标准左上右下框，非法记录返回空列表供上层安全跳过。"""

    raw = record.get("bbox")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        return []
    try:
        return [float(value) for value in raw]
    except (TypeError, ValueError):
        return []


def _vertical_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """中文说明：计算两个单元格相对较短高度的纵向重叠率，用于跨列逐行配对。"""

    left_box, right_box = _bbox(left), _bbox(right)
    if not left_box or not right_box:
        return 0.0
    overlap = max(0.0, min(left_box[3], right_box[3]) - max(left_box[1], right_box[1]))
    shorter = min(left_box[3] - left_box[1], right_box[3] - right_box[1])
    return overlap / shorter if shorter > 0 else 0.0


def _track_map(geometry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """中文说明：建立 PP 轨道 ID 索引，后续按“亚段”和“代号”表头定位目标列。"""

    return {
        str(track.get("id")): track
        for track in geometry.get("tracks", [])
        if isinstance(track, Mapping) and track.get("id")
    }


def _cells_on_track(geometry: Mapping[str, Any], track_id: str) -> list[dict[str, Any]]:
    """中文说明：按纵向顺序返回完整位于目标轨道的单元格副本。"""

    records = []
    for raw in geometry.get("cells", []):
        if not isinstance(raw, Mapping):
            continue
        track_ids = raw.get("track_ids")
        if isinstance(track_ids, list) and track_id in {str(value) for value in track_ids}:
            record = dict(raw)
            if _bbox(record):
                records.append(record)
    return sorted(records, key=lambda item: (_bbox(item)[1], _bbox(item)[0]))


def _submember_base(text: Any) -> str:
    """中文说明：从“马五”“马五1”等亚段文字取得共同主体，带“段”的父层不参与分组。"""

    cleaned = _clean_text(text)
    if not cleaned or cleaned.endswith("段") or cleaned == "亚段":
        return ""
    matched = _CHINESE_SUBMEMBER_PATTERN.fullmatch(cleaned)
    return str(matched.group("base")) if matched else ""


def _submember_suffix(text: Any, base: str) -> int | None:
    """中文说明：只接受紧随亚段主体的一至两位数字，避免把其他数字误当下标。"""

    cleaned = _clean_text(text)
    matched = re.fullmatch(re.escape(base) + r"(?P<suffix>\d{1,2})", cleaned)
    if not matched:
        return None
    value = int(matched.group("suffix"))
    return value if 1 <= value <= 99 else None


def _code_suffix(text: Any) -> int | None:
    """中文说明：从相邻地层代号末尾读取亚段序号，兼容 O/0、逗号和上下标归一化结果。"""

    cleaned = _clean_text(text).casefold().replace("o", "0")
    if "m" not in cleaned:
        return None
    matched = re.search(r"(?P<suffix>\d{1,2})$", cleaned)
    if not matched:
        return None
    value = int(matched.group("suffix"))
    return value if 1 <= value <= 99 else None


def _result_payload(result: Any) -> Mapping[str, Any]:
    """中文说明：兼容 PaddleX Result.json 属性、方法和测试字典三种局部 OCR 返回形态。"""

    if isinstance(result, Mapping):
        payload: Any = result
    else:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
    if isinstance(payload, Mapping) and isinstance(payload.get("res"), Mapping):
        payload = payload["res"]
    return payload if isinstance(payload, Mapping) else {}


def _get_local_ocr_pipeline() -> Any:
    """中文说明：惰性复用高精度 PaddleOCR，避免无亚段图片加载额外模型。"""

    global _LOCAL_OCR_PIPELINE
    if _LOCAL_OCR_PIPELINE is not None:
        return _LOCAL_OCR_PIPELINE
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        # 中文说明：Windows 下先加载 Torch 可稳定 Paddle/PaddleX 的动态库导入顺序。
        try:
            __import__("torch")
        except Exception:
            pass
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("局部亚段二次 OCR 需要 paddleocr 3.x") from exc
    _LOCAL_OCR_PIPELINE = PaddleOCR(
        device=os.getenv("TABLE_SUBMEMBER_OCR_DEVICE", "cpu"),
        enable_mkldnn=False,
        text_detection_model_name=os.getenv(
            "TABLE_SUBMEMBER_TEXT_DET_MODEL", "PP-OCRv5_server_det"
        ),
        text_recognition_model_name=os.getenv(
            "TABLE_SUBMEMBER_TEXT_REC_MODEL", "PP-OCRv5_server_rec"
        ),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    return _LOCAL_OCR_PIPELINE


def recognize_cell_crops(
    image_path: Path,
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """中文说明：对目标单元格裁剪、放大并以原色和增强灰度两种视图批量执行二次 OCR。"""

    if not cells:
        return {}
    pipeline = _get_local_ocr_pipeline()
    padding = max(0, int(os.getenv("TABLE_SUBMEMBER_OCR_PADDING", "2")))
    canvas_width = max(128, int(os.getenv("TABLE_SMALL_CELL_OCR_CANVAS_WIDTH", "768")))
    canvas_height = max(48, int(os.getenv("TABLE_SMALL_CELL_OCR_CANVAS_HEIGHT", "160")))
    arrays: list[Any] = []
    owners: list[tuple[str, str]] = []
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("局部亚段二次 OCR 需要 numpy") from exc
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        for cell in cells:
            cell_id = str(cell.get("id") or "")
            box = _bbox(cell)
            if not cell_id or not box:
                continue
            left = max(0, int(box[0]) - padding)
            top = max(0, int(box[1]) - padding)
            right = min(image.width, int(box[2]) + padding)
            bottom = min(image.height, int(box[3]) + padding)
            if right <= left or bottom <= top:
                continue
            crop = image.crop((left, top, right, bottom))
            enlarged = ImageOps.pad(
                crop,
                (canvas_width, canvas_height),
                method=Image.Resampling.LANCZOS,
                color="white",
            )
            enhanced = ImageOps.autocontrast(ImageOps.grayscale(enlarged)).convert("RGB")
            for variant, candidate in (("enlarged_rgb", enlarged), ("autocontrast_gray", enhanced)):
                arrays.append(np.asarray(candidate))
                owners.append((cell_id, variant))
    if not arrays:
        return {}
    batch_size = max(1, int(os.getenv("TABLE_SMALL_CELL_OCR_BATCH_SIZE", "24")))
    results: list[Any] = []
    for start in range(0, len(arrays), batch_size):
        # 中文说明：小单元格数量可能较多，分批预测可控制 CPU 内存峰值且保持输入顺序不变。
        results.extend(
            pipeline.predict(
                arrays[start : start + batch_size],
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_thresh=0.08,
                text_det_box_thresh=0.12,
                text_rec_score_thresh=0.0,
            )
        )
    recognized: dict[str, list[dict[str, Any]]] = {}
    for (cell_id, variant), result in zip(owners, results):
        payload = _result_payload(result)
        texts = payload.get("rec_texts")
        scores = payload.get("rec_scores")
        if not isinstance(texts, list):
            continue
        clean_parts = [str(value).strip() for value in texts if str(value).strip()]
        if not clean_parts:
            continue
        numeric_scores = []
        if isinstance(scores, list):
            for value in scores[: len(clean_parts)]:
                try:
                    numeric_scores.append(float(value))
                except (TypeError, ValueError):
                    pass
        recognized.setdefault(cell_id, []).append(
            {
                "text": "".join(clean_parts),
                "confidence": round(min(numeric_scores) if numeric_scores else 0.0, 4),
                "variant": variant,
                "padding": padding,
                "canvas_size": [canvas_width, canvas_height],
            }
        )
    return recognized


def _best_candidate(
    raw_text: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    base: str = "",
    code: bool = False,
) -> tuple[str, int | None, float]:
    """中文说明：优先选择能恢复有效数字后缀的局部 OCR，原始 PP 文本始终作为可追溯候选保留。"""

    options = [{"text": raw_text, "confidence": 0.0, "variant": "ppstructure_full_image"}]
    options.extend(dict(item) for item in candidates if isinstance(item, Mapping))
    scored: list[tuple[int, float, str, int | None]] = []
    for option in options:
        text = str(option.get("text") or "").strip()
        suffix = _code_suffix(text) if code else _submember_suffix(text, base)
        try:
            confidence = float(option.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        scored.append((int(suffix is not None), confidence, text, suffix))
    _, confidence, text, suffix = max(scored, key=lambda item: (item[0], item[1], len(item[2])))
    return text or raw_text, suffix, round(confidence, 4)


def _small_cell_targets(geometry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """中文说明：按单元格高度、OCR 字高和原始置信度筛选需要放大重识别的通用小文字单元格。"""

    max_cell_height = max(8, int(os.getenv("TABLE_SMALL_CELL_MAX_HEIGHT", "52")))
    max_text_height = max(6, int(os.getenv("TABLE_SMALL_TEXT_MAX_HEIGHT", "24")))
    confidence_threshold = float(os.getenv("TABLE_SMALL_CELL_CONFIDENCE_THRESHOLD", "0.93"))
    ocr_by_id = {
        str(line.get("id")): line
        for line in geometry.get("ocr_lines", [])
        if isinstance(line, Mapping) and line.get("id")
    }
    targets: list[dict[str, Any]] = []
    for raw in geometry.get("cells", []):
        if not isinstance(raw, Mapping) or not str(raw.get("text") or "").strip():
            continue
        box = _bbox(raw)
        if not box:
            continue
        lines = [
            ocr_by_id[str(ocr_id)]
            for ocr_id in raw.get("ocr_ids", [])
            if str(ocr_id) in ocr_by_id
        ] if isinstance(raw.get("ocr_ids"), list) else []
        text_heights = [
            line_box[3] - line_box[1]
            for line in lines
            if (line_box := _bbox(line))
        ]
        confidences = []
        for line in lines:
            try:
                confidences.append(float(line.get("confidence")))
            except (TypeError, ValueError):
                pass
        cell_height = box[3] - box[1]
        cell_width = box[2] - box[0]
        small_geometry = cell_height <= max_cell_height
        low_confidence = bool(confidences) and min(confidences) < confidence_threshold
        compact_mixed_text = bool(re.search(r"[A-Za-z0-9]", str(raw.get("text") or "")))
        compact_cell = cell_width <= 120
        tiny_text = bool(text_heights) and max(text_heights) <= max_text_height
        if not small_geometry or not (compact_cell or low_confidence or tiny_text or compact_mixed_text):
            continue
        record = dict(raw)
        record["small_cell_trigger"] = {
            "cell_height": round(cell_height, 3),
            "cell_width": round(cell_width, 3),
            "max_text_height": round(max(text_heights), 3) if text_heights else None,
            "min_full_image_ocr_confidence": round(min(confidences), 4) if confidences else None,
            "small_geometry": small_geometry,
            "low_confidence": low_confidence,
            "compact_mixed_text": compact_mixed_text,
            "compact_cell": compact_cell,
        }
        targets.append(record)
    return targets


def _choose_generic_refinement(
    cell: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[str, str, float]:
    """中文说明：仅在局部结果补出缺失字符或显著提高置信度时替换原文字，避免放大噪声覆盖正确 OCR。"""

    raw_text = str(cell.get("text") or "").strip()
    raw_clean = _clean_text(raw_text)
    trigger = cell.get("small_cell_trigger")
    trigger = trigger if isinstance(trigger, Mapping) else {}
    try:
        raw_confidence = float(trigger.get("min_full_image_ocr_confidence") or 0.0)
    except (TypeError, ValueError):
        raw_confidence = 0.0
    best_text, best_confidence = raw_text, raw_confidence
    reason = "keep_full_image_ocr"
    for candidate in candidates:
        text = str(candidate.get("text") or "").strip()
        clean = _clean_text(text)
        try:
            confidence = float(candidate.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if not clean:
            continue
        adds_suffix = (
            bool(raw_clean)
            and clean.startswith(raw_clean)
            and len(clean) > len(raw_clean)
            and any(character.isdigit() for character in clean[len(raw_clean) :])
        )
        materially_better = (
            raw_confidence < 0.8
            and confidence >= 0.95
            and confidence >= raw_confidence + 0.08
        )
        if adds_suffix or materially_better:
            score = (int(adds_suffix), confidence, len(clean))
            current_score = (
                int(reason == "local_ocr_added_suffix"),
                best_confidence,
                len(_clean_text(best_text)),
            )
            if score > current_score:
                best_text = text
                best_confidence = confidence
                reason = "local_ocr_added_suffix" if adds_suffix else "local_ocr_higher_confidence"
    return best_text, reason, round(best_confidence, 4)


def _candidate_groups(geometry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """中文说明：从“亚段”轨道中寻找两个以上连续同主体单元格，并与“代号”列逐行配对。"""

    tracks = _track_map(geometry)
    submember_track_ids = [
        track_id
        for track_id, track in tracks.items()
        if "亚段" in _clean_text(track.get("header_text"))
    ]
    code_track_ids = [
        track_id
        for track_id, track in tracks.items()
        if "代号" in _clean_text(track.get("header_text"))
    ]
    groups: list[dict[str, Any]] = []
    for track_id in submember_track_ids:
        cells = _cells_on_track(geometry, track_id)
        cursor = 0
        while cursor < len(cells):
            base = _submember_base(cells[cursor].get("refined_text") or cells[cursor].get("text"))
            if not base:
                cursor += 1
                continue
            end = cursor + 1
            while end < len(cells) and _submember_base(
                cells[end].get("refined_text") or cells[end].get("text")
            ) == base:
                end += 1
            run = cells[cursor:end]
            cursor = end
            if len(run) < 2:
                continue
            code_track_id = code_track_ids[0] if code_track_ids else ""
            code_cells = _cells_on_track(geometry, code_track_id) if code_track_id else []
            rows = []
            for order, cell in enumerate(run, 1):
                aligned = [
                    (candidate, _vertical_overlap(cell, candidate))
                    for candidate in code_cells
                    if _vertical_overlap(cell, candidate) >= 0.5
                ]
                code_cell = max(aligned, key=lambda item: item[1])[0] if aligned else None
                rows.append(
                    {
                        "order": order,
                        "cell_id": str(cell.get("id") or ""),
                        "raw_text": str(cell.get("text") or ""),
                        "generic_refined_text": str(cell.get("refined_text") or cell.get("text") or ""),
                        "bbox": list(cell.get("bbox") or []),
                        "code_cell_id": str(code_cell.get("id") or "") if code_cell else "",
                        "code_raw_text": str(code_cell.get("text") or "") if code_cell else "",
                        "code_generic_refined_text": str(
                            code_cell.get("refined_text") or code_cell.get("text") or ""
                        ) if code_cell else "",
                    }
                )
            groups.append(
                {
                    "id": f"submember_group_{len(groups):03d}",
                    "base_name": base,
                    "track_id": track_id,
                    "code_track_id": code_track_id,
                    "rows": rows,
                }
            )
    return groups


def refine_small_cell_geometry(
    image_path: str | Path,
    geometry: Mapping[str, Any],
    *,
    recognizer: CellRecognizer | None = None,
) -> dict[str, Any]:
    """中文说明：先对所有小尺寸疑难单元格执行局部 OCR，再对亚段与代号列做结构化交叉校验。"""

    refined = deepcopy(dict(geometry))
    quality = refined.setdefault("quality", {})
    if quality.get("small_cell_refinement_version") == SMALL_CELL_REFINEMENT_VERSION:
        return refined
    cell_by_id = {
        str(cell.get("id")): cell
        for cell in refined.get("cells", [])
        if isinstance(cell, Mapping) and cell.get("id")
    }
    for cell in cell_by_id.values():
        # 中文说明：版本升级或缓存复用时先回到原始 PP 文本，防止旧的序列推断被误当成新一轮 OCR 证据。
        cell["refined_text"] = str(cell.get("text") or "")
        cell.pop("submember_resolution", None)
    targets = _small_cell_targets(refined)
    local_results: dict[str, Sequence[Mapping[str, Any]]] = {
        str(target.get("id")): [dict(item) for item in target.get("secondary_ocr", [])]
        for target in targets
        if isinstance(target.get("secondary_ocr"), list) and target.get("secondary_ocr")
    }
    local_error = ""
    try:
        missing_targets = [
            target for target in targets if str(target.get("id") or "") not in local_results
        ]
        local_results.update(
            (recognizer or recognize_cell_crops)(Path(image_path), missing_targets)
        )
    except Exception as exc:
        # 中文说明：局部模型不可用时保留全图 PP OCR，并把异常显式写入质量报告供质量门审计。
        local_error = f"{type(exc).__name__}: {exc}"
    refined_count = 0
    for target in targets:
        cell_id = str(target.get("id") or "")
        cell = cell_by_id.get(cell_id)
        if not isinstance(cell, dict):
            continue
        candidates = [dict(item) for item in local_results.get(cell_id, [])]
        refined_text, decision, confidence = _choose_generic_refinement(target, candidates)
        cell["secondary_ocr"] = candidates
        cell["small_cell_ocr"] = {
            "trigger": dict(target.get("small_cell_trigger") or {}),
            "decision": decision,
            "confidence": confidence,
        }
        cell["refined_text"] = refined_text
        if decision != "keep_full_image_ocr":
            refined_count += 1

    # 中文说明：亚段—代号校验读取通用局部 OCR 的回写文字，不再自行限定或重复调用某两列 OCR。
    groups = _candidate_groups(refined)
    for group in groups:
        base = str(group["base_name"])
        anchors: list[int] = []
        for row in group["rows"]:
            cell_id = str(row["cell_id"])
            code_cell_id = str(row["code_cell_id"])
            direct_text, direct_suffix, direct_confidence = _best_candidate(
                str(row["generic_refined_text"]),
                local_results.get(cell_id, []),
                base=base,
            )
            code_text, code_suffix, code_confidence = _best_candidate(
                str(row["code_generic_refined_text"]),
                local_results.get(code_cell_id, []),
                code=True,
            )
            row.update(
                {
                    "local_ocr_candidates": [dict(item) for item in local_results.get(cell_id, [])],
                    "recognized_text": direct_text,
                    "recognized_suffix": direct_suffix,
                    "recognized_confidence": direct_confidence,
                    "code_local_ocr_candidates": [
                        dict(item) for item in local_results.get(code_cell_id, [])
                    ],
                    "code_recognized_text": code_text,
                    "code_recognized_suffix": code_suffix,
                    "code_recognized_confidence": code_confidence,
                }
            )
            order = int(row["order"])
            if direct_suffix == order or code_suffix == order:
                anchors.append(order)
        row_count = len(group["rows"])
        sequence_supported = (
            len(set(anchors)) >= 2
            and 1 in anchors
            and row_count in anchors
            and all(row.get("code_cell_id") for row in group["rows"])
        )
        group["anchor_orders"] = sorted(set(anchors))
        group["ordered_sequence_supported"] = sequence_supported
        for row in group["rows"]:
            order = int(row["order"])
            direct_suffix = row.get("recognized_suffix")
            code_suffix = row.get("code_recognized_suffix")
            if direct_suffix == code_suffix == order:
                status, suffix, confidence = "local_ocr_and_code_agree", order, 0.98
            elif direct_suffix == order or code_suffix == order:
                status, suffix, confidence = "single_ocr_matches_row_sequence", order, 0.92
            elif sequence_supported:
                status, suffix, confidence = "cross_column_anchored_row_sequence", order, 0.84
            else:
                status, suffix, confidence = "unresolved", None, 0.0
            row.update(
                {
                    "resolved_suffix": suffix,
                    "resolved_name": f"{base}{suffix}" if suffix is not None else base,
                    "resolution_status": status,
                    "resolution_confidence": confidence,
                    "conflicting_suffixes": sorted(
                        {
                            int(value)
                            for value in (direct_suffix, code_suffix)
                            if value is not None and int(value) != order
                        }
                    ),
                }
            )
            cell = cell_by_id.get(str(row["cell_id"]))
            if isinstance(cell, dict):
                cell["refined_text"] = str(row["resolved_name"])
                cell["submember_resolution"] = {
                    key: row[key]
                    for key in (
                        "resolved_suffix",
                        "resolution_status",
                        "resolution_confidence",
                        "code_cell_id",
                        "conflicting_suffixes",
                    )
                }
    refined["submember_groups"] = groups
    quality.update(
        {
            "small_cell_refinement_version": SMALL_CELL_REFINEMENT_VERSION,
            "small_cell_target_count": len(targets),
            "small_cell_local_ocr_result_count": len(local_results),
            "small_cell_refined_count": refined_count,
            "small_cell_local_ocr_error": local_error,
            "submember_group_count": len(groups),
            "submember_row_count": sum(len(group["rows"]) for group in groups),
            "submember_resolved_row_count": sum(
                row.get("resolved_suffix") is not None
                for group in groups
                for row in group["rows"]
            ),
        }
    )
    if local_error:
        refined.setdefault("uncertainties", []).append(
            f"small_cell_local_ocr_failed: {local_error}"
        )
    return refined


def recover_submember_intervals(payload: Mapping[str, Any]) -> dict[str, Any]:
    """中文说明：将已由局部 OCR 和代号列校验的亚段行补为父段下独立实体，不处理未决行。"""

    recovered = deepcopy(dict(payload))
    primitives = recovered.get("primitives")
    geometry = recovered.get("ppstructure_geometry")
    if not isinstance(primitives, Mapping) or not isinstance(geometry, Mapping):
        return recovered
    intervals = primitives.get("stratigraphic_intervals")
    if not isinstance(intervals, list):
        return recovered
    cell_by_id = {
        str(cell.get("id")): cell
        for cell in geometry.get("cells", [])
        if isinstance(cell, Mapping) and cell.get("id")
    }
    consumed = {
        str(ref)
        for item in intervals
        if isinstance(item, Mapping) and isinstance(item.get("geometry_refs"), list)
        for ref in item["geometry_refs"]
    }
    ids = {str(item.get("id")) for item in intervals if isinstance(item, Mapping) and item.get("id")}
    additions: list[dict[str, Any]] = []
    for group in geometry.get("submember_groups", []):
        if not isinstance(group, Mapping):
            continue
        base = str(group.get("base_name") or "")
        rows = group.get("rows")
        if not base or not isinstance(rows, list):
            continue
        row_boxes = [
            _bbox(cell_by_id.get(str(row.get("cell_id")), {}))
            for row in rows
            if isinstance(row, Mapping)
        ]
        row_boxes = [box for box in row_boxes if box]
        if not row_boxes:
            continue
        group_top, group_bottom = min(box[1] for box in row_boxes), max(box[3] for box in row_boxes)
        parents = []
        for item in intervals:
            if not isinstance(item, Mapping) or _clean_text(item.get("name")) != f"{base}段":
                continue
            refs = item.get("geometry_refs")
            if not isinstance(refs, list):
                continue
            boxes = [
                _bbox(cell_by_id.get(str(ref), {}))
                for ref in refs
            ]
            boxes = [box for box in boxes if box]
            # 中文说明：相邻列独立表线可能相差数个像素，允许小容差但仍要求父段覆盖整组亚段。
            if (
                boxes
                and min(box[1] for box in boxes) <= group_top + 4.0
                and max(box[3] for box in boxes) >= group_bottom - 4.0
            ):
                parents.append(item)
        if len(parents) != 1:
            continue
        parent_id = str(parents[0].get("id") or "")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            cell_id = str(row.get("cell_id") or "")
            suffix = row.get("resolved_suffix")
            if not cell_id or cell_id in consumed or suffix is None:
                continue
            base_slug = _CLEAN_PATTERN.sub("_", base).strip("_") or "submember"
            local_id = f"recovered_{base_slug}_{int(suffix):02d}"
            if local_id in ids:
                continue
            status = str(row.get("resolution_status") or "unresolved")
            code_cell_id = str(row.get("code_cell_id") or "")
            evidence = (
                f"亚段单元格 {cell_id} 与代号单元格 {code_cell_id or '无'} 纵向配对；"
                f"局部OCR={row.get('recognized_text')!r}，代号OCR={row.get('code_recognized_text')!r}，"
                f"序号解析={status}"
            )
            additions.append(
                {
                    "id": local_id,
                    "name": str(row.get("resolved_name") or f"{base}{suffix}"),
                    "parent_id": parent_id,
                    "rank": "submember",
                    "track_id": str(group.get("track_id") or ""),
                    "geometry_refs": [cell_id],
                    "evidence": evidence,
                    "confidence": float(row.get("resolution_confidence") or 0.0),
                    "recognition_source": status,
                    "code_geometry_ref": code_cell_id,
                    "raw_ocr_text": str(row.get("raw_text") or ""),
                    "code_raw_ocr_text": str(row.get("code_raw_text") or ""),
                    "conflicting_suffixes": list(row.get("conflicting_suffixes") or []),
                }
            )
            ids.add(local_id)
            consumed.add(cell_id)
    intervals.extend(additions)
    recovered["submember_recovery"] = {
        "version": SMALL_CELL_REFINEMENT_VERSION,
        "added_count": len(additions),
        "added_ids": [item["id"] for item in additions],
    }
    return recovered


def build_submember_coverage(
    primitives: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """中文说明：比较亚段检测单元格、实体引用和未消费单元格，生成可直接门控的审计结果。"""

    detected = [
        str(row.get("cell_id"))
        for group in geometry.get("submember_groups", [])
        if isinstance(group, Mapping)
        for row in group.get("rows", [])
        if isinstance(row, Mapping) and row.get("cell_id")
    ]
    intervals = primitives.get("stratigraphic_intervals")
    interval_records = intervals if isinstance(intervals, list) else []
    consumed_by_entity: dict[str, list[str]] = {}
    for item in interval_records:
        if not isinstance(item, Mapping):
            continue
        refs = item.get("geometry_refs")
        matched = [str(ref) for ref in refs if str(ref) in detected] if isinstance(refs, list) else []
        if matched:
            consumed_by_entity[str(item.get("id") or "")] = matched
    consumed = sorted({cell_id for values in consumed_by_entity.values() for cell_id in values})
    unconsumed = [cell_id for cell_id in detected if cell_id not in consumed]
    duplicate_consumption = {
        cell_id: sorted(entity_id for entity_id, refs in consumed_by_entity.items() if cell_id in refs)
        for cell_id in consumed
        if sum(cell_id in refs for refs in consumed_by_entity.values()) > 1
    }
    errors = []
    if unconsumed:
        errors.append(f"亚段单元格未生成实体：{unconsumed}")
    if duplicate_consumption:
        errors.append(f"亚段单元格被多个实体重复消费：{duplicate_consumption}")
    return {
        "ok": not errors,
        "detected_cell_count": len(detected),
        "entity_count": len(consumed_by_entity),
        "consumed_cell_count": len(consumed),
        "unconsumed_cell_count": len(unconsumed),
        "detected_cell_ids": detected,
        "consumed_cell_ids": consumed,
        "unconsumed_cell_ids": unconsumed,
        "duplicate_consumption": duplicate_consumption,
        "errors": errors,
    }
