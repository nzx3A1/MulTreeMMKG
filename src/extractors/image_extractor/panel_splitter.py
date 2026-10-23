"""由视觉模型检测多子图，确定性裁剪/遮蔽，并递归复核单子图与单类型。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageOps

from src.utils.json_io import write_json
from src.utils.llm_client import safe_json_loads
from src.utils.logger import get_logger

from .schema_models import ImageExtractionTask


PANEL_PREPROCESSING_VERSION = 1
logger = get_logger("extractors.image_extractor.panel_splitter")
Box = tuple[int, int, int, int]


@dataclass
class PanelPreparation:
    """保存单张原图的叶子任务、检测审计及模型调用次数。"""

    tasks: list[ImageExtractionTask]
    audit: list[dict[str, Any]]
    call_count: int = 0


def build_panel_detection_prompt(task: ImageExtractionTask, size: tuple[int, int]) -> str:
    """要求模型穷举独立子图和嵌套图，以当前输入图片的归一化坐标返回边界。"""

    origin = task.panel_metadata.get("bbox_pixels", [0, 0, *size])
    existing_masks = [
        [round((box[0] - origin[0]) * 1000 / size[0]), round((box[1] - origin[1]) * 1000 / size[1]),
         round((box[2] - origin[0]) * 1000 / size[0]), round((box[3] - origin[1]) * 1000 / size[1])]
        for box in task.panel_metadata.get("masked_bboxes_pixels", [])
    ]
    return f"""你是科研图片版面检测员。检测当前图片是否包含多个独立子图，随后检查是否只有一种视觉类型。
当前图片 ID：{task.image_id}；尺寸：{size[0]}×{size[1]}。
父图题（仅辅助定位）：{task.caption}
已独立切出并涂白的区域（当前图坐标 0..1000）：{json.dumps(existing_masks)}。
复核时上述区域残留的少量刻度文字、边框和空白不再是独立子图；必须看到一个有实际主体的新面板才允许继续拆分。
规则：
1. 并排、上下、多行图版、同类型的多个照片/坐标图，以及嵌套在主图内的位置图/放大图，都必须拆开；没有 a/b 编号也要检测。
2. 单个综合地层柱状表的地层/岩性/沉积相/曲线等列属于同一子图，不要按列拆开；同一坐标轴内的多条曲线也不是多个子图。
   独立岩心实物照片与旁边的测井/柱状图属于不同视觉类型，即使同属 a 或 b 标签，也需要分开；照片中的样品条带不按每块岩心切分。
3. 图例、标尺、方向标、标题是附属信息，不是独立子图。bbox 必须保留当前子图的轴、标签、子图标题和图例。
   特别检查整图底部或边缘的共享图例，不得只沿绘图区边框裁剪而丢失它们。图例在主图框外时，扩展 bbox 覆盖图例，
   并在 exclude_bboxes 中遮蔽扩展范围内混入的邻图主体。多个子图可以各自保留同一个共享图例。
4. 若一个主图包含嵌套小图，panels 同时列出主图和嵌套图。主图 bbox 可以包含嵌套图，但必须在主图 exclude_bboxes 中指定嵌套图区域，程序会将其涂白。嵌套图自身单独裁剪。
   嵌套图 bbox 和遮蔽区域必须包含外侧的刻度数字、坐标轴标题和边框，不能只框内部绘图区；残留的嵌套图刻度不能再被识别成新子图。
5. 所有坐标均为当前输入图片上的 [left, top, right, bottom]，范围 0..1000，原点左上，right>left、bottom>top。禁止输出像素坐标或 0..1 坐标。
6. bbox 必须穷举全部子图且不重复；不能只保留感兴趣的类型。不同子图尽量不重叠，必须重叠时通过 exclude_bboxes 排除其他子图。
7. 只有一个独立面板且能确认属于单一类型，才允许 has_subfigures=false 且 single_type=true。
   看不清、裁剪残缺、没有可用主体或无法确认时 single_type=false；仍有独立小图时 has_subfigures=true。
8. 遮蔽产生的空白区域不算新子图。每个子图 label 写图中编号或简短名称，不要把父图其他内容写入 label。
9. 如果多个子图 bbox 为保留同一共享图例而重叠，必须在根对象 shared_bboxes 中列出共享图例区域。
   shared_bboxes 仅能包含图例/符号说明等附属信息，不能包含任何子图主体。排除邻图时禁止把当前子图自身的主体涂白。
只返回 JSON：
{{"has_subfigures":true,"single_type":false,"reason":"可见多个独立面板","shared_bboxes":[],
"panels":[{{"label":"a","bbox":[0,0,600,1000],"exclude_bboxes":[[400,0,600,300]]}},
{{"label":"嵌套位置图","bbox":[400,0,600,300],"exclude_bboxes":[]}},
{{"label":"b","bbox":[650,0,1000,1000],"exclude_bboxes":[]}}]}}
若确认为单图，返回 has_subfigures=false、single_type=true、panels=[] 和 reason。
""".strip()


def _pixel_box(value: Any, size: tuple[int, int]) -> Box:
    """严格校验 0..1000 坐标并转换到实际像素，不静默修正越界或退化框。"""

    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("子图 bbox 必须包含四个归一化坐标")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError("子图 bbox 坐标必须是数值")
    if any(not math.isfinite(item) or not 0 <= item <= 1000 for item in value):
        raise ValueError("子图 bbox 坐标必须在 0..1000 范围内")
    left, top, right, bottom = value
    if left >= right or top >= bottom:
        raise ValueError("子图 bbox 为空或方向错误")
    box = (
        math.floor(left * size[0] / 1000), math.floor(top * size[1] / 1000),
        math.ceil(right * size[0] / 1000), math.ceil(bottom * size[1] / 1000),
    )
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        raise ValueError("子图 bbox 过小，无法可靠抽取")
    return box


def _intersection(first: Box, second: Box) -> Box | None:
    """计算两个矩形的交集，供嵌套检测、重叠检查和裁剪内遮蔽使用。"""

    box = (max(first[0], second[0]), max(first[1], second[1]),
           min(first[2], second[2]), min(first[3], second[3]))
    return box if box[0] < box[2] and box[1] < box[3] else None


def _area(box: Box) -> int:
    """返回矩形像素面积，避免全白或大面积丢失的子图继续进入抽取。"""

    return (box[2] - box[0]) * (box[3] - box[1])


def _visible_overlap(box: Box, masks: list[Box]) -> int:
    """精确计算矩形扣除多个遮蔽框后的面积，兼容重叠遮蔽区域。"""

    clipped = [part for mask in masks if (part := _intersection(box, mask))]
    xs = sorted({box[0], box[2], *(x for part in clipped for x in (part[0], part[2]))})
    hidden = 0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((part[1], part[3]) for part in clipped if part[0] <= left and part[2] >= right)
        end = box[1]
        for top, bottom in intervals:
            hidden += (right - left) * max(0, bottom - max(top, end))
            end = max(end, bottom)
    return _area(box) - hidden


class VLMPanelSplitter:
    """先看图检测，再用 Pillow 裁剪和涂白；只放行经过视觉复核的叶子图。"""

    def __init__(
        self, vlm_client: Any, output_dir: str | Path, *,
        max_depth: int = 3, max_panels: int = 32, max_calls: int = 64,
    ) -> None:
        """保存客户端及递归限额，防止错误模型响应导致无限拆分。"""

        if min(max_depth, max_panels, max_calls) < 1:
            raise ValueError("子图递归深度、数量和调用限额必须大于零")
        self.vlm_client = vlm_client
        self.output_dir = Path(output_dir)
        self.max_depth = max_depth
        self.max_panels = max_panels
        self.max_calls = max_calls

    def prepare(self, task: ImageExtractionTask) -> PanelPreparation:
        """准备一张原图，逐叶保存失败原因，并写入可供复查的检测清单。"""

        result = PanelPreparation([], [])
        # 中文说明：目录同时绑定来源身份和文件内容，避免同名图或原图更新后覆盖其他运行。
        try:
            source = Path(task.image_path)
            digest = hashlib.sha256(task.image_id.encode() + source.read_bytes()).hexdigest()[:20]
            folder = self.output_dir / digest
            with Image.open(source) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            folder.mkdir(parents=True, exist_ok=True)
            # 中文说明：统一 EXIF 朝向后再检测，确保模型所见图像与裁剪像素坐标一致。
            normalized_path = folder / "source.png"
            image.save(normalized_path)
            current = replace(task, image_path=str(normalized_path.resolve()), panel_metadata={
                "source_image_path": task.image_path, "source_image_id": task.image_id,
                "source_image_index": task.image_index, "source_size": list(image.size),
                "bbox_pixels": [0, 0, *image.size], "masked_bboxes_pixels": [],
                "panel_path": [], "status": "pending", "coordinate_frame": "exif_transposed_source_pixels",
            })
            self._visit(current, image, folder, result, depth=0)
            write_json(folder / "panels.json", {
                "version": PANEL_PREPROCESSING_VERSION, "source_image_path": task.image_path,
                "source_image_id": task.image_id, "vlm_call_count": result.call_count,
                "audit": result.audit,
                "panels": [{"image_id": leaf.image_id, "image_path": leaf.image_path,
                            **dict(leaf.panel_metadata)} for leaf in result.tasks],
            })
        except Exception as exc:
            result.tasks = [self._failed(task, str(exc))]
        return result

    def _failed(self, task: ImageExtractionTask, error: str) -> ImageExtractionTask:
        """为不确定、无效或超过递归限额的图片保留失败任务，禁止原图兜底抽取。"""

        logger.warning(f"[图片抽取/子图] 未放行：image={task.image_id}，error={error}")
        return replace(task, panel_metadata={
            **dict(task.panel_metadata), "source_image_path": task.panel_metadata.get("source_image_path", task.image_path),
            "source_image_id": task.panel_metadata.get("source_image_id", task.image_id),
            "status": "failed", "error": error,
        })

    def _detect(
        self, task: ImageExtractionTask, size: tuple[int, int], result: PanelPreparation, *, correction: str = "",
    ) -> Mapping[str, Any]:
        """调用同一检测协议完成原图识别和裁剪后复核，审计记录包括失败调用。"""

        if result.call_count >= self.max_calls:
            raise ValueError("子图检测超过单张原图的模型调用限额")
        entry: dict[str, Any] = {"image_id": task.image_id, "image_path": task.image_path, "status": "failed"}
        result.audit.append(entry)
        result.call_count += 1
        logger.info(f"[图片抽取/子图] 检测/复核：image={task.image_id}，size={size}")
        try:
            response = self.vlm_client.describe_image(
                task.image_path, build_panel_detection_prompt(task, size) + correction,
                task_name=f"图片子图检测与单类型复核:{task.image_id}",
                response_format={"type": "json_object"}, max_tokens=4096,
            )
            payload = response if isinstance(response, Mapping) else safe_json_loads(str(response or ""))
            if not isinstance(payload, Mapping):
                raise ValueError("子图检测响应不是 JSON 对象")
            if type(payload.get("has_subfigures")) is not bool or type(payload.get("single_type")) is not bool:
                raise ValueError("子图检测必须明确返回 has_subfigures 和 single_type 布尔值")
            if not str(payload.get("reason") or "").strip() or not isinstance(payload.get("panels"), list):
                raise ValueError("子图检测缺少 reason 或 panels 数组")
            entry.update(status="completed", detection=dict(payload))
            return payload
        except Exception as exc:
            entry["error"] = str(exc)
            raise

    def _visit(
        self, task: ImageExtractionTask, image: Image.Image, folder: Path,
        result: PanelPreparation, *, depth: int,
    ) -> None:
        """递归拆分当前图；各分支独立失败，经过单图单类型复核的分支独立放行。"""

        try:
            payload = self._detect(task, image.size, result)
            if not payload["has_subfigures"]:
                if payload["panels"] or not payload["single_type"]:
                    raise ValueError("视觉复核未确认单个且单一类型的子图")
                result.tasks.append(replace(task, panel_metadata={
                    **dict(task.panel_metadata), "status": "verified_single",
                    "verification_reason": payload["reason"],
                }))
                return
            if depth >= self.max_depth:
                raise ValueError("达到最大切分深度后仍包含多个子图")
            try:
                panels = self._parse_panels(payload["panels"], image.size, image=image,
                                            shared_bboxes=payload.get("shared_bboxes", []))
            except ValueError as exc:
                # 中文说明：模型框与附属图例扩展区域可能冲突，携带明确几何错误仅重检一次。
                correction = (
                    f"\n上次结果的程序校验失败：{exc}。请重新看图并修正完整 panels。"
                    "扩展 bbox 保留图例时，必须用 exclude_bboxes 遮蔽与邻图主体的整个交集。"
                    "如果剩余重叠部分是两个子图均需保留的共享图例，请通过根对象 shared_bboxes 明确标出它。"
                    f"\n上次结果：{json.dumps(dict(payload), ensure_ascii=False)}"
                )
                payload = self._detect(task, image.size, result, correction=correction)
                if not payload["has_subfigures"]:
                    raise ValueError("几何校验失败后的重检不能直接放行原图")
                panels = self._parse_panels(payload["panels"], image.size, image=image,
                                            shared_bboxes=payload.get("shared_bboxes", []))
            # 中文说明：总节点数有独立限额，不会因层层拆成满额子图而指数增长。
            created = sum(len(entry.get("children", [])) for entry in result.audit)
            if created + len(panels) > self.max_panels:
                raise ValueError("子图总数量超过单张原图限额")
            result.audit[-1]["children"] = [panel["label"] for panel in panels]
            for index, panel in enumerate(panels):
                child, crop = self._crop(task, image, panel, index, folder)
                self._visit(child, crop, folder, result, depth=depth + 1)
        except Exception as exc:
            result.tasks.append(self._failed(task, str(exc)))

    def _parse_panels(
        self, values: list[Any], size: tuple[int, int], *, image: Image.Image | None = None,
        shared_bboxes: Any = None,
    ) -> list[dict[str, Any]]:
        """校验完整面板列表，自动遮蔽包含的小图，拒绝重复或未隔离的部分重叠。"""

        if not 2 <= len(values) <= self.max_panels:
            raise ValueError("多子图检测应返回 2 至 max_panels 个子图")
        if shared_bboxes is not None and not isinstance(shared_bboxes, list):
            raise ValueError("shared_bboxes 必须是共享图例坐标数组")
        shared_boxes = [_pixel_box(value, size) for value in shared_bboxes or []]
        whole = (0, 0, *size)
        if _area(whole) - _visible_overlap(whole, shared_boxes) > _area(whole) * 0.3:
            raise ValueError("共享图例面积过大，不能用共享区域绕过子图主体隔离")
        panels = []
        for index, value in enumerate(values):
            if not isinstance(value, Mapping) or not isinstance(value.get("exclude_bboxes", []), list):
                raise ValueError("子图记录或 exclude_bboxes 格式错误")
            panels.append({
                "label": str(value.get("label") or f"子图{index + 1}"),
                "box": _pixel_box(value.get("bbox"), size),
                "masks": [_pixel_box(box, size) for box in value.get("exclude_bboxes", [])],
                "shared_boxes": shared_boxes,
            })
        for index, panel in enumerate(panels):
            box = panel["box"]
            for other in panels[index + 1:]:
                if box == other["box"]:
                    raise ValueError("模型返回重复的子图边界")
            for other_index, other in enumerate(panels):
                if index == other_index:
                    continue
                overlap = _intersection(box, other["box"])
                if overlap == other["box"]:
                    panel["masks"].append(overlap)
            panel["masks"] = list(dict.fromkeys(
                clipped for mask in panel["masks"] if (clipped := _intersection(box, mask))
            ))
            if _visible_overlap(box, panel["masks"]) < max(64, _area(box) * 0.05):
                raise ValueError("遮蔽后子图没有足够的可见主体")
        for index, panel in enumerate(panels):
            for other in panels[index + 1:]:
                overlap = _intersection(panel["box"], other["box"])
                if not overlap:
                    continue
                masks = [*panel["masks"], *other["masks"], *shared_boxes]
                if not _visible_overlap(overlap, masks):
                    continue
                # 中文说明：共享图例扩展框可能只在空白间隙重叠；检查双方都未遮蔽的实际像素。
                if image is not None:
                    visible = image.crop(overlap).convert("RGB")
                    draw = ImageDraw.Draw(visible)
                    for mask in masks:
                        clipped = _intersection(overlap, mask)
                        if clipped:
                            left, top, right, bottom = clipped
                            draw.rectangle((left - overlap[0], top - overlap[1], right - overlap[0] - 1, bottom - overlap[1] - 1), fill="white")
                    # 中文说明：扫描图/JPEG 的白底常有浅灰压缩噪声，230 以上视为近白背景。
                    if min(channel[0] for channel in visible.getextrema()) >= 230:
                        continue
                residual = visible.convert("L").point([255] * 230 + [0] * 26).getbbox() if image is not None else None
                residual_box = (
                    [round((residual[0] + overlap[0]) * 1000 / size[0]), round((residual[1] + overlap[1]) * 1000 / size[1]),
                     round((residual[2] + overlap[0]) * 1000 / size[0]), round((residual[3] + overlap[1]) * 1000 / size[1])]
                    if residual else None
                )
                raise ValueError(
                    f"子图边界存在未隔离的部分重叠：{panel['label']} 与 {other['label']}，"
                    f"双方仍共同可见内容的坐标={residual_box}（0..1000）。"
                    "请将这部分正确归属于本子图或共享图例；邻图的标题、刻度文字也须遮蔽。"
                )
        return panels

    def _crop(
        self, task: ImageExtractionTask, image: Image.Image, panel: Mapping[str, Any],
        index: int, folder: Path,
    ) -> tuple[ImageExtractionTask, Image.Image]:
        """裁剪单面板并涂白排除区域，将各层坐标累积映射回原图。"""

        box = panel["box"]
        crop = image.crop(box)
        draw = ImageDraw.Draw(crop)
        for left, top, right, bottom in panel["masks"]:
            draw.rectangle((left - box[0], top - box[1], right - box[0] - 1, bottom - box[1] - 1), fill="white")
        parent_meta = task.panel_metadata
        origin = parent_meta["bbox_pixels"]
        absolute_box = [origin[0] + box[0], origin[1] + box[1], origin[0] + box[2], origin[1] + box[3]]
        masks = [
            [origin[0] + left, origin[1] + top, origin[0] + right, origin[1] + bottom]
            for left, top, right, bottom in panel["masks"]
        ]
        masks.extend(
            list(clipped) for value in parent_meta.get("masked_bboxes_pixels", [])
            if (clipped := _intersection(tuple(absolute_box), tuple(value)))
        )
        panel_path = [*parent_meta["panel_path"], index]
        destination = folder / ("panel_" + "_".join(str(part) for part in panel_path) + ".png")
        crop.save(destination)
        metadata = {
            **dict(parent_meta), "parent_image_id": task.image_id, "panel_path": panel_path,
            "panel_label": panel["label"], "bbox_pixels": absolute_box,
            "masked_bboxes_pixels": masks, "status": "pending",
            "shared_bboxes_pixels": [
                [origin[0] + left, origin[1] + top, origin[0] + right, origin[1] + bottom]
                for shared in panel.get("shared_boxes", []) if (clipped := _intersection(box, shared))
                for left, top, right, bottom in [clipped]
            ],
        }
        caption = f"当前仅处理子图：{panel['label']}。父图题仅供参考：{task.caption}"
        # 中文说明：清除整图分类与逐图分类，避免它们被当成当前子图的分类结果。
        raw_chunk = {key: value for key, value in task.raw_chunk.items() if key not in {
            "classification", "image_classification", "image_classifications", "individual_image_classifications",
        }}
        raw_chunk.update(image_path=[str(destination.resolve())], caption=caption)
        return replace(
            task, image_id=f"{task.image_id}:panel:{index}", image_path=str(destination.resolve()),
            caption=caption, classification_code="", classification_type="",
            raw_chunk=raw_chunk, panel_metadata=metadata,
        ), crop
