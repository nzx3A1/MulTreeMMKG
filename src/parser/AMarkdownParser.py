"""MinerU Markdown 论文解析器。

该脚本面向 PDF 论文经 MinerU 导出的 full.md 文件，按论文题名、基本信息、
章节树、表格、图片和块级公式组织为阶段 1 JSON。正文结构主要依赖 Markdown
标题与正则表达式解析；基本信息先用规则提取，并预留可选的大模型补全入口。
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SECTION_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
HTML_TABLE_RE = re.compile(r"<table\b[\s\S]*?</table>", re.IGNORECASE)
IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)[ \t]*(?:\n|$)")
TABLE_CAPTION_START_RE = re.compile(
    r"^(?:表|Table)\s*[．.]?\s*[0-9０-９]+(?:\s*[．.]\s*[0-9０-９]+)*(?=\s|$|[：:]|[\u3400-\u9fff])",
    re.IGNORECASE,
)
FORMULA_RE = re.compile(r"\$\$\s*([\s\S]*?)\s*\$\$")
REFERENCE_HEADING_RE = re.compile(r"^#{2,6}\s+.*?(References|参考文献)", re.IGNORECASE)
MARKUP_TAG_RE = re.compile(r"</?(?:sup|sub|span|b|strong|i|em)[^>]*>", re.IGNORECASE)
TABLE_LAYOUT_MODEL_NAME = "PicoDet_layout_1x_table"
# 默认使用第 0 张 GPU；如需切换设备，可通过 AMARKDOWN_PADDLE_DEVICE 覆盖，例如 cpu 或 gpu:1。
TABLE_LAYOUT_DEVICE = os.getenv("AMARKDOWN_PADDLE_DEVICE", "gpu:0")
TABLE_CONTINUATION_THRESHOLD = 0.95
PDF_IMAGE_RENDER_SCALE = 4.0
PDF_IMAGE_RENDER_PADDING_POINTS = 0.0
PDF_IMAGE_RENDER_JPEG_QUALITY = 95
MINERU_CONTENT_LIST_GLOB = "*_content_list.json"
MINERU_LAYOUT_FILENAME = "layout.json"
MINERU_RENDERABLE_ASSET_TYPES = frozenset({"image", "table", "chart"})
_TABLE_LAYOUT_MODEL: Any | None = None
_CUDA_DLL_HANDLES: list[Any] = []
TableImageDetector = Callable[[Path, float], bool]


@dataclass
class SectionNode:
    """保存章节解析过程中的树节点和原始正文，最后再转为 JSON 字典。"""

    level: int
    id: str
    title: str
    raw_content: str = ""
    children: list["SectionNode"] = field(default_factory=list)
    table: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    formulas: list[dict[str, str]] = field(default_factory=list)
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        """将章节节点递归转换为目标 JSON 结构。"""

        return {
            "level": self.level,
            "id": self.id,
            "title": self.title,
            "children": [child.to_dict() for child in self.children],
            "table": self.table,
            "images": self.images,
            "formulas": self.formulas,
            "content": self.content,
        }


@dataclass(frozen=True)
class MinerUContentImage:
    """保存 content_list 中一个可渲染视觉资源的路径、类型、页码及 MinerU 坐标框。"""

    image_path: Path
    asset_type: str
    page_idx: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class MinerULayoutImage:
    """保存 layout.json 中一个可渲染视觉资源的 PDF 坐标框及版面顺序。"""

    asset_type: str
    page_idx: int
    bbox: tuple[float, float, float, float]
    layout_index: int


@dataclass(frozen=True)
class MinerUImageRenderTask:
    """保存替换一个 MinerU 视觉资源时所需的文件路径和双坐标系溯源信息。"""

    image_path: Path
    asset_type: str
    page_idx: int
    content_bbox: tuple[float, float, float, float]
    layout_bbox: tuple[float, float, float, float]
    layout_index: int


class AMarkdownParser:
    """解析论文 Markdown，并输出按章节组织的结构化 JSON。"""

    produced_by = "src/parser/AMarkdownParser.py"

    def __init__(
        self,
        use_llm_basic_info: bool = False,
        table_image_detector: TableImageDetector | None = None,
        table_continuation_threshold: float = TABLE_CONTINUATION_THRESHOLD,
        source_pdf: str | Path | None = None,
        replace_mineru_images_from_pdf: bool = False,
        pdf_image_render_scale: float = PDF_IMAGE_RENDER_SCALE,
        pdf_image_render_padding_points: float = PDF_IMAGE_RENDER_PADDING_POINTS,
    ) -> None:
        """初始化解析器，并可选配置从原 PDF 重渲染 MinerU 图片的参数。"""

        if pdf_image_render_scale <= 0:
            raise ValueError("PDF 图片渲染倍数必须大于 0")
        if pdf_image_render_padding_points < 0:
            raise ValueError("PDF 图片裁剪外扩距离不能小于 0")

        self.use_llm_basic_info = use_llm_basic_info
        self.table_image_detector = table_image_detector or detect_table_image_with_picodet
        self.table_continuation_threshold = table_continuation_threshold
        self.source_pdf = Path(source_pdf).expanduser() if source_pdf is not None else None
        self.replace_mineru_images_from_pdf = replace_mineru_images_from_pdf
        self.pdf_image_render_scale = pdf_image_render_scale
        self.pdf_image_render_padding_points = pdf_image_render_padding_points

    def parse_file(self, input_file: str | Path) -> dict[str, Any]:
        """读取 Markdown 文件；启用时先用原 PDF 高清替换对应 MinerU 图片。"""

        markdown_path = Path(input_file).expanduser().resolve()
        image_rebuild: dict[str, Any] | None = None
        if self.replace_mineru_images_from_pdf:
            if self.source_pdf is None:
                raise ValueError("启用 MinerU 图片替换时必须提供 source_pdf")
            image_rebuild = replace_mineru_images_from_pdf(
                mineru_output_dir=markdown_path.parent,
                source_pdf=self.source_pdf,
                render_scale=self.pdf_image_render_scale,
                padding_points=self.pdf_image_render_padding_points,
            )
        text = read_text(markdown_path)
        result = self.parse_text(
            text,
            filename=markdown_path.name,
            input_file=str(markdown_path.resolve()),
            asset_base_dir=markdown_path.parent,
        )
        if image_rebuild is not None:
            result["_image_rebuild"] = image_rebuild
        return result

    def parse_text(
        self,
        text: str,
        filename: str = "full.md",
        input_file: str = "",
        asset_base_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """解析 Markdown 文本，组装论文基本信息、目录树和元数据。"""

        text = normalize_newlines(text)
        lines = text.splitlines()
        title, title_index = extract_first_title(lines)
        reference_start = find_reference_heading_index(lines)
        references = "\n".join(lines[reference_start:]).strip() if reference_start is not None else ""

        first_section_index = find_first_section_heading_index(lines, stop_index=reference_start)
        preamble_end = first_section_index if first_section_index is not None else reference_start or len(lines)
        preamble_lines = lines[title_index + 1 : preamble_end]
        basic_lines, intro_lines = split_preamble_lines(preamble_lines)

        basic_information = extract_basic_information(basic_lines, title, references)
        if self.use_llm_basic_info:
            basic_information = merge_llm_basic_information(basic_lines, basic_information)

        toc = self._parse_toc(
            lines,
            first_section_index,
            reference_start,
            intro_lines,
            Path(asset_base_dir) if asset_base_dir is not None else None,
        )
        return {
            "filename": filename,
            "title": title,
            "basicInformation": basic_information,
            "toc": [node.to_dict() for node in toc],
            "_stage": 1,
            "_produced_by": self.produced_by,
            "input_file": input_file,
        }

    def _parse_toc(
        self,
        lines: list[str],
        first_section_index: int | None,
        reference_start: int | None,
        intro_lines: list[str],
        asset_base_dir: Path | None,
    ) -> list[SectionNode]:
        """解析正文标题并构建章节树，同时保留无标题引言为“前言”章节。"""

        nodes: list[SectionNode] = []
        stack: list[SectionNode] = []
        parenthetical_counters: dict[str, int] = {}
        end_index = reference_start if reference_start is not None else len(lines)
        reference_text = "\n".join(lines[:end_index])
        asset_counters = {"formula": 0}

        intro_text = "\n".join(line for line in intro_lines if line.strip()).strip()
        if intro_text:
            intro_node = SectionNode(level=2, id="0", title="前言", raw_content=intro_text)
            populate_section_assets(
                intro_node,
                reference_text,
                asset_counters,
                asset_base_dir,
                self.table_image_detector,
                self.table_continuation_threshold,
            )
            nodes.append(intro_node)

        if first_section_index is None:
            return nodes

        section_records = collect_section_records(lines, first_section_index, end_index)
        for raw_heading, body_lines in section_records:
            heading = parse_heading(raw_heading, stack, parenthetical_counters)
            node = SectionNode(
                level=heading["level"],
                id=heading["id"],
                title=heading["title"],
                raw_content="\n".join(body_lines).strip(),
            )
            populate_section_assets(
                node,
                reference_text,
                asset_counters,
                asset_base_dir,
                self.table_image_detector,
                self.table_continuation_threshold,
            )

            while stack and stack[-1].level >= node.level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                nodes.append(node)
            stack.append(node)
        return nodes


def read_text(path: Path) -> str:
    """按常见编码读取 Markdown 文件，优先使用 UTF-8。"""

    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_newlines(text: str) -> str:
    """统一换行符，避免 Windows/Unix 换行差异影响正则匹配。"""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_inline_text(text: str, strip_markup: bool = True) -> str:
    """清洗行内文本；标题和元数据会去掉常见 HTML 标记，正文保留更多原始信息。"""

    text = html.unescape(text)
    if strip_markup:
        text = MARKUP_TAG_RE.sub("", text)
        text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_first_title(lines: list[str]) -> tuple[str, int]:
    """提取第一个一级标题作为论文题名。"""

    for index, line in enumerate(lines):
        match = HEADING_RE.match(line.strip())
        if match and len(match.group(1)) == 1:
            return clean_inline_text(match.group(2)), index
    fallback = clean_inline_text(next((line for line in lines if line.strip()), ""))
    return fallback, 0


def find_reference_heading_index(lines: list[str]) -> int | None:
    """定位参考文献起始标题，后续内容归入 basicInformation.references。"""

    for index, line in enumerate(lines):
        if REFERENCE_HEADING_RE.match(line.strip()):
            return index
    return None


def find_first_section_heading_index(lines: list[str], stop_index: int | None = None) -> int | None:
    """查找正文第一个二级及以下标题，一级英文题名不会被当成章节。"""

    limit = stop_index if stop_index is not None else len(lines)
    for index, line in enumerate(lines[:limit]):
        if SECTION_HEADING_RE.match(line.strip()):
            return index
    return None


def split_preamble_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    """将章节前内容拆为基本信息区和无标题引言区。"""

    marker_re = re.compile(
        r"(摘\s*要|关键词|Abstract|Key\s*words|Fund\s*support|基金|中图分类号|文章编号|文献标识码)",
        re.IGNORECASE,
    )
    last_marker = -1
    for index, line in enumerate(lines):
        stripped = clean_inline_text(line)
        if line.lstrip().startswith("#") or marker_re.search(stripped):
            last_marker = index
    if last_marker == -1:
        return lines, []
    return lines[: last_marker + 1], lines[last_marker + 1 :]


def extract_basic_information(lines: list[str], title: str, references: str) -> dict[str, Any]:
    """用规则从论文头部提取摘要、作者、关键词、机构和参考文献。"""

    block = "\n".join(lines)
    clean_lines = [clean_inline_text(line) for line in lines if clean_inline_text(line)]
    abstract = extract_labeled_block(block, [r"摘\s*要"], [r"关键词", r"Key\s*words", r"Abstract"])
    if not abstract:
        abstract = extract_labeled_block(block, [r"Abstract"], [r"Key\s*words", r"Fund\s*support"])

    authors = extract_authors(lines, title)
    keywords = extract_keywords(block)
    organization = extract_publish_organization(clean_lines)
    return {
        "abstract": clean_inline_text(abstract, strip_markup=False),
        "title": title,
        "authors": authors,
        "keywords": keywords,
        "publish_organization": organization,
        "references": references,
    }


def extract_labeled_block(text: str, starts: list[str], stops: list[str]) -> str:
    """提取“摘要/Abstract”等标签后的连续文本，直到下一个标签。"""

    start_pattern = r"(?:%s)\s*[:：]?\s*" % "|".join(starts)
    start_match = re.search(start_pattern, text, flags=re.IGNORECASE)
    if not start_match:
        return ""
    tail = text[start_match.end() :]
    stop_pattern = r"\n?\s*(?:%s)\s*[:：]?" % "|".join(stops)
    stop_match = re.search(stop_pattern, tail, flags=re.IGNORECASE)
    return tail[: stop_match.start()].strip() if stop_match else tail.strip()


def extract_authors(lines: list[str], title: str) -> list[str]:
    """从题名后的第一条非机构正文行提取作者列表。"""

    title_seen = False
    for line in lines:
        clean = clean_inline_text(line)
        if not clean:
            continue
        if clean == title:
            title_seen = True
            continue
        if line.lstrip().startswith("#"):
            if title_seen:
                break
            continue
        if clean.startswith(("(", "（")) or "摘" in clean or "Abstract" in clean:
            continue
        if not title_seen and len(clean) > 80:
            continue
        names = split_author_names(clean)
        if names:
            return names
    return []


def split_author_names(text: str) -> list[str]:
    """清理作者行中的上标和连接词，并拆分为作者名称数组。"""

    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\band\b", "、", text, flags=re.IGNORECASE)
    parts = re.split(r"[，,、;；]\s*", text)
    names = []
    for part in parts:
        name = part.strip(" .·")
        if 1 < len(name) <= 40 and not re.search(r"(大学|学院|公司|研究院|Department|Institute|Abstract)", name, re.IGNORECASE):
            names.append(name)
    return names


def extract_keywords(text: str) -> list[str]:
    """提取中文或英文关键词，并按常用分隔符拆分。"""

    match = re.search(r"(关键词|Key\s*words)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if not match:
        return []
    line = match.group(2).splitlines()[0]
    line = re.split(r"(Fund\s*support|中图分类号|文章编号)", line, flags=re.IGNORECASE)[0]
    return [clean_inline_text(item) for item in re.split(r"[;；]\s*", line) if clean_inline_text(item)]


def extract_publish_organization(clean_lines: list[str]) -> str:
    """提取作者单位或发布机构，通常位于作者行之后的括号内。"""

    organizations = []
    for line in clean_lines:
        if line.startswith(("(", "（")) and re.search(r"(大学|学院|公司|研究院|实验室|Department|Institute|Laboratory)", line, re.IGNORECASE):
            organizations.append(line.strip("()（） "))
    return " ".join(organizations)


def merge_llm_basic_information(lines: list[str], basic_information: dict[str, Any]) -> dict[str, Any]:
    """可选使用项目 LLMClient 补全基本信息；失败时保持规则提取结果。"""

    try:
        from src.utils.llm_client import LLMClient

        prompt = (
            "请从以下论文 Markdown 头部提取 JSON，字段为 abstract、title、authors、"
            "keywords、publish_organization。不要编造不存在的信息。\n\n"
            + "\n".join(lines[:80])
        )
        response = LLMClient().chat_json(
            [
                {"role": "system", "content": "你是论文元数据抽取助手，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ]
        )
        if isinstance(response, dict):
            merged = dict(basic_information)
            for key in ("abstract", "title", "authors", "keywords", "publish_organization"):
                if response.get(key):
                    merged[key] = response[key]
            return merged
    except Exception:
        return basic_information
    return basic_information


def collect_section_records(lines: list[str], start_index: int, end_index: int) -> list[tuple[str, list[str]]]:
    """把正文按章节标题切成若干 (标题, 正文行) 记录。"""

    records: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []
    for line in lines[start_index:end_index]:
        if SECTION_HEADING_RE.match(line.strip()):
            if current_heading:
                records.append((current_heading, current_body))
            current_heading = line.strip()
            current_body = []
        else:
            current_body.append(line)
    if current_heading:
        records.append((current_heading, current_body))
    return records


def parse_heading(raw_heading: str, stack: list[SectionNode], parenthetical_counters: dict[str, int]) -> dict[str, Any]:
    """解析章节标题中的编号、层级和纯标题。"""

    match = SECTION_HEADING_RE.match(raw_heading)
    hashes, heading_text = match.groups() if match else ("##", raw_heading)
    clean = clean_inline_text(heading_text)
    normalized = normalize_heading_number_text(clean)

    parenthetical = re.match(r"^[（(]\s*(\d+)\s*[)）]\s*(.*)$", normalized)
    if parenthetical:
        ordinal = int(parenthetical.group(1))
        parent = find_parent_for_parenthetical(stack, ordinal)
        parent_id = parent.id if parent and parent.id else "0"
        parenthetical_counters[parent_id] = max(parenthetical_counters.get(parent_id, 0), ordinal)
        section_id = f"{parent_id}.{ordinal}"
        title = parenthetical.group(2).strip() or f"小节{parenthetical.group(1)}"
        level = (parent.level + 1) if parent else len(hashes)
        return {"level": level, "id": section_id, "title": title}

    numbered = re.match(r"^(\d+(?:\.\d+)*)(?:[.)、\s]+)?(.*)$", normalized)
    if numbered:
        section_id = numbered.group(1)
        title = numbered.group(2).strip() or section_id
        level = len(section_id.split(".")) + 1
        return {"level": level, "id": section_id, "title": title}

    return {"level": len(hashes), "id": slugify_title(clean), "title": clean}


def find_parent_for_parenthetical(stack: list[SectionNode], ordinal: int) -> SectionNode | None:
    """根据括号编号寻找父章节，避免 (2)、(3) 被挂到前一个括号小节下面。"""

    if not stack:
        return None
    if ordinal <= 1:
        return stack[-1]

    previous_suffix = f".{ordinal - 1}"
    for index in range(len(stack) - 1, -1, -1):
        if stack[index].id.endswith(previous_suffix):
            return stack[index - 1] if index > 0 else None
    return stack[-1]


def normalize_heading_number_text(text: str) -> str:
    """规范章节编号中的中文句点、全角点和空格。"""

    text = text.replace("．", ".").replace("。", ".").replace("·", ".")
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"^(\d+(?:\.\d+)*)(?:\.)?\s*", r"\1 ", text)
    return text.strip()


def slugify_title(title: str) -> str:
    """为无显式编号的标题生成稳定 ID。"""

    slug = re.sub(r"\W+", "-", title, flags=re.UNICODE).strip("-").lower()
    return slug or "section"


def prepare_cuda_dll_search_path() -> None:
    """将当前 Conda 环境中的 CUDA 动态库目录加入 Windows 搜索路径。"""

    if os.name != "nt" or not TABLE_LAYOUT_DEVICE.startswith("gpu"):
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    candidate_directories: list[Path] = []
    nvidia_root = site_packages / "nvidia"
    if nvidia_root.is_dir():
        # NVIDIA pip 包可能把 DLL 放在 bin 或 bin/x86_64 子目录中。
        for package_directory in nvidia_root.iterdir():
            for relative_directory in (Path("bin"), Path("bin") / "x86_64"):
                candidate_directories.append(package_directory / relative_directory)
    candidate_directories.append(site_packages / "torch" / "lib")

    for directory in candidate_directories:
        if not directory.is_dir():
            continue
        directory_text = str(directory)
        os.environ["PATH"] = directory_text + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            # 保存句柄，避免 Windows DLL 搜索目录在函数返回后失效。
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(directory_text))


def detect_table_image_with_picodet(image_path: Path, threshold: float = TABLE_CONTINUATION_THRESHOLD) -> bool:
    """使用表格专用 PicoDet 模型判断后续图片是否仍属于当前表格。"""

    global _TABLE_LAYOUT_MODEL
    if not image_path.is_file():
        print(f"表格续图检测跳过：图片不存在 {image_path}")
        return False

    try:
        if _TABLE_LAYOUT_MODEL is None:
            prepare_cuda_dll_search_path()
            from paddleocr import LayoutDetection

            # 显式指定 GPU 设备，确保 PaddleOCR 的表格版面检测使用第 0 张显卡。
            _TABLE_LAYOUT_MODEL = LayoutDetection(
                model_name=TABLE_LAYOUT_MODEL_NAME,
                device=TABLE_LAYOUT_DEVICE,
                enable_mkldnn=False,
            )
            print(f"PaddleOCR 表格检测设备：{TABLE_LAYOUT_DEVICE}")

        results = _TABLE_LAYOUT_MODEL.predict(str(image_path), batch_size=1, layout_nms=True)
        best_score = 0.0
        for result in results:
            if not isinstance(result, dict):
                continue
            result_data = result.get("res", result)
            boxes = result_data.get("boxes", []) if isinstance(result_data, dict) else []
            for box in boxes:
                if str(box.get("label", "")).strip().lower() != "table":
                    continue
                best_score = max(best_score, float(box.get("score", 0.0)))
        is_table = best_score >= threshold
        print(
            f"表格续图检测：{image_path.name}，table置信度={best_score:.3f}，"
            f"阈值={threshold:.2f}，结果={'合并' if is_table else '停止'}"
        )
        return is_table
    except Exception as exc:
        # 模型失败时保守地停止合并，防止普通图片被错误吸收到表格 Chunk。
        print(f"表格续图检测失败：{image_path}，停止合并。原因：{exc}")
        return False


def resolve_markdown_image_path(image_ref: str, asset_base_dir: Path | None) -> Path:
    """将 Markdown 图片引用解析为本地路径，供表格续图模型读取。"""

    image_path = Path(image_ref)
    if image_path.is_absolute() or asset_base_dir is None:
        return image_path
    return (asset_base_dir / image_path).resolve()


def serialize_asset_path(image_ref: str, asset_base_dir: Path | None) -> str:
    """将资源路径转换为 JSON 输出字符串；已知 Markdown 目录时输出绝对路径。"""

    # parse_text 未提供源文件目录时保留原始引用，parse_file 会传入目录并输出绝对路径。
    if asset_base_dir is None:
        return image_ref
    return str(resolve_markdown_image_path(image_ref, asset_base_dir))


def _read_json_file(path: Path, description: str) -> Any:
    """读取 MinerU JSON 文件，并在格式或编码异常时给出带路径的错误信息。"""

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取{description}：{path}；原因：{exc}") from exc


def _parse_bbox(value: Any, description: str) -> tuple[float, float, float, float]:
    """校验并转换四值 bbox，确保裁剪框具有正面积。"""

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{description} 必须是四个数值组成的 bbox：{value!r}")
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} 含有非数值坐标：{value!r}") from exc
    if not x1 > x0 or not y1 > y0:
        raise ValueError(f"{description} 没有正面积：{value!r}")
    return x0, y0, x1, y1


def _ensure_relative_image_path(image_ref: str, mineru_output_dir: Path) -> Path:
    """解析 content_list 图片路径，并拒绝跳出 MinerU 输出目录的引用。"""

    raw_path = Path(image_ref)
    if raw_path.is_absolute():
        candidate = raw_path.resolve()
    else:
        candidate = (mineru_output_dir / raw_path).resolve()
    try:
        candidate.relative_to(mineru_output_dir)
    except ValueError as exc:
        raise ValueError(f"MinerU 图片路径越出了输出目录：{image_ref}") from exc
    return candidate


def find_mineru_content_list_path(mineru_output_dir: str | Path) -> Path:
    """定位当前 MinerU 输出目录唯一的 content_list JSON，优先使用固定文件名。"""

    output_dir = Path(mineru_output_dir).expanduser().resolve()
    fixed_path = output_dir / "content_list.json"
    if fixed_path.is_file():
        return fixed_path
    candidates = sorted(output_dir.glob(MINERU_CONTENT_LIST_GLOB))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"目录 {output_dir} 中应存在唯一的 content_list JSON，实际找到 {len(candidates)} 个："
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def load_mineru_content_images(mineru_output_dir: str | Path) -> tuple[Path, list[MinerUContentImage]]:
    """读取 content_list 内可替换的图、表、图表记录，并保留 MinerU 坐标作为溯源。"""

    output_dir = Path(mineru_output_dir).expanduser().resolve()
    content_list_path = find_mineru_content_list_path(output_dir)
    content = _read_json_file(content_list_path, "MinerU content_list")
    if not isinstance(content, list):
        raise ValueError(f"MinerU content_list 根节点必须是列表：{content_list_path}")

    images: list[MinerUContentImage] = []
    for record_index, item in enumerate(content):
        asset_type = str(item.get("type") or "").lower() if isinstance(item, dict) else ""
        if asset_type not in MINERU_RENDERABLE_ASSET_TYPES:
            continue
        image_ref = str(item.get("img_path") or "").strip()
        if not image_ref:
            raise ValueError(f"content_list 第 {record_index} 条 {asset_type} 缺少 img_path")
        page_idx = item.get("page_idx")
        if isinstance(page_idx, bool) or not isinstance(page_idx, int) or page_idx < 0:
            raise ValueError(f"content_list 第 {record_index} 条 {asset_type} 的 page_idx 无效：{page_idx!r}")
        image_path = _ensure_relative_image_path(image_ref, output_dir)
        if not image_path.is_file():
            raise FileNotFoundError(f"content_list 指向的 MinerU 图片不存在：{image_path}")
        images.append(
            MinerUContentImage(
                image_path=image_path,
                asset_type=asset_type,
                page_idx=page_idx,
                bbox=_parse_bbox(item.get("bbox"), f"content_list 第 {record_index} 条 {asset_type} bbox"),
            )
        )
    if not images:
        raise ValueError(f"content_list 中没有可重渲染的 image/table/chart 记录：{content_list_path}")
    return content_list_path, images


def load_mineru_layout_images(mineru_output_dir: str | Path) -> tuple[Path, list[MinerULayoutImage]]:
    """读取 layout.json 的图、表、图表框；该坐标系将作为 PDF 局部渲染裁剪框。"""

    output_dir = Path(mineru_output_dir).expanduser().resolve()
    layout_path = output_dir / MINERU_LAYOUT_FILENAME
    if not layout_path.is_file():
        raise FileNotFoundError(f"MinerU layout.json 不存在：{layout_path}")
    layout = _read_json_file(layout_path, "MinerU layout")
    if not isinstance(layout, dict) or not isinstance(layout.get("pdf_info"), list):
        raise ValueError(f"MinerU layout 缺少 pdf_info 页面列表：{layout_path}")

    images: list[MinerULayoutImage] = []
    for page_idx, page in enumerate(layout["pdf_info"]):
        if not isinstance(page, dict):
            raise ValueError(f"layout.json 第 {page_idx} 页不是对象")
        blocks = page.get("preproc_blocks")
        if not isinstance(blocks, list):
            raise ValueError(f"layout.json 第 {page_idx} 页缺少 preproc_blocks 列表")
        for block_index, block in enumerate(blocks):
            asset_type = str(block.get("type") or "").lower() if isinstance(block, dict) else ""
            if asset_type not in MINERU_RENDERABLE_ASSET_TYPES:
                continue
            images.append(
                MinerULayoutImage(
                    asset_type=asset_type,
                    page_idx=page_idx,
                    bbox=_parse_bbox(block.get("bbox"), f"layout.json 第 {page_idx} 页第 {block_index} 个 {asset_type} bbox"),
                    layout_index=int(block.get("index", block_index)),
                )
            )
    if not images:
        raise ValueError(f"layout.json 中没有可重渲染的 image/table/chart 版面记录：{layout_path}")
    return layout_path, images


def _bbox_sort_key(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """返回稳定的从上到下、从左到右 bbox 排序键，用于页内图片配对。"""

    return bbox[1], bbox[0], bbox[3], bbox[2]


def _bbox_aspect_ratio(bbox: tuple[float, float, float, float]) -> float:
    """计算 bbox 宽高比，供跨坐标系图片配对的形状校验使用。"""

    return (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])


def _image_aspect_ratio(image_path: Path) -> float:
    """读取现有 MinerU 图片尺寸并返回宽高比，作为 layout 配对的视觉校验依据。"""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("图片布局校验需要 Pillow，请安装 PIL/Pillow 后重试") from exc
    try:
        with Image.open(image_path) as image:
            if image.width <= 0 or image.height <= 0:
                raise ValueError(f"图片尺寸无效：{image.size}")
            return image.width / image.height
    except OSError as exc:
        raise ValueError(f"无法读取 MinerU 图片用于布局校验：{image_path}") from exc


def _ensure_layout_shape_matches(
    content_image: MinerUContentImage,
    layout_image: MinerULayoutImage,
) -> None:
    """校验资源类型和形状均与 layout 候选框一致，避免页内错配后错误覆盖。"""

    if content_image.asset_type != layout_image.asset_type:
        raise ValueError(
            "content_list 与 layout.json 的视觉资源类型不匹配，拒绝覆盖："
            f"图片={content_image.image_path.name}，页码={content_image.page_idx}，"
            f"content_type={content_image.asset_type}，layout_type={layout_image.asset_type}"
        )

    # content_list bbox 属于 MinerU 预处理页面坐标，部分版本的纵横比例会失真，
    # 因此用实际导出图片的像素比例和 layout 的 PDF 坐标比例进行校验。
    content_ratio = _image_aspect_ratio(content_image.image_path)
    layout_ratio = _bbox_aspect_ratio(layout_image.bbox)
    ratio_distance = abs(math.log(content_ratio / layout_ratio))
    if ratio_distance > 0.12:
        raise ValueError(
            "MinerU 图片与 layout.json 的候选框形状不匹配，拒绝覆盖："
            f"图片={content_image.image_path.name}，页码={content_image.page_idx}，"
            f"image_ratio={content_ratio:.4f}，layout_ratio={layout_ratio:.4f}"
        )


def build_mineru_image_render_tasks(mineru_output_dir: str | Path) -> tuple[Path, Path, list[MinerUImageRenderTask]]:
    """按页内阅读顺序配对 content_list 图片与 layout 图片框，构建高清重渲染任务。"""

    content_list_path, content_images = load_mineru_content_images(mineru_output_dir)
    layout_path, layout_images = load_mineru_layout_images(mineru_output_dir)
    content_by_page: dict[int, list[MinerUContentImage]] = {}
    layout_by_page: dict[int, list[MinerULayoutImage]] = {}
    for image in content_images:
        content_by_page.setdefault(image.page_idx, []).append(image)
    for image in layout_images:
        layout_by_page.setdefault(image.page_idx, []).append(image)

    all_pages = sorted(set(content_by_page) | set(layout_by_page))
    tasks: list[MinerUImageRenderTask] = []
    for page_idx in all_pages:
        page_content_images = sorted(content_by_page.get(page_idx, []), key=lambda item: _bbox_sort_key(item.bbox))
        page_layout_images = sorted(layout_by_page.get(page_idx, []), key=lambda item: _bbox_sort_key(item.bbox))
        if len(page_content_images) != len(page_layout_images):
            raise ValueError(
                f"第 {page_idx} 页图片数量不一致，content_list={len(page_content_images)}，"
                f"layout.json={len(page_layout_images)}；拒绝覆盖。"
            )
        for content_image, layout_image in zip(page_content_images, page_layout_images):
            _ensure_layout_shape_matches(content_image, layout_image)
            tasks.append(
                MinerUImageRenderTask(
                    image_path=content_image.image_path,
                    asset_type=content_image.asset_type,
                    page_idx=page_idx,
                    content_bbox=content_image.bbox,
                    layout_bbox=layout_image.bbox,
                    layout_index=layout_image.layout_index,
                )
            )
    if not tasks:
        raise ValueError("没有构建出任何 MinerU 图片高清重渲染任务")
    return content_list_path, layout_path, tasks


def _expanded_pdf_clip(page: Any, bbox: tuple[float, float, float, float], padding_points: float) -> Any:
    """把 layout bbox 外扩并裁剪到 PDF 页面边界，返回可安全渲染的 PyMuPDF Rect。"""

    import fitz

    page_rect = page.rect
    clip = fitz.Rect(
        bbox[0] - padding_points,
        bbox[1] - padding_points,
        bbox[2] + padding_points,
        bbox[3] + padding_points,
    )
    tolerance = 0.5
    if (
        clip.x0 < page_rect.x0 - tolerance
        or clip.y0 < page_rect.y0 - tolerance
        or clip.x1 > page_rect.x1 + tolerance
        or clip.y1 > page_rect.y1 + tolerance
    ):
        raise ValueError(
            f"layout bbox 超出 PDF 页面边界，可能不是 PDF 坐标系："
            f"bbox={list(bbox)}，page_rect={list(page_rect)}"
        )
    clip = clip & page_rect
    if clip.is_empty or clip.is_infinite:
        raise ValueError(f"PDF 图片裁剪框无效：{list(bbox)}")
    return clip


def _temporary_image_path(image_path: Path) -> Path:
    """生成与目标图片同目录、同扩展名的临时文件路径，便于安全原子替换。"""

    return image_path.with_name(f".{image_path.stem}.{uuid4().hex}.pdf-render{image_path.suffix}")


def _save_pixmap_with_original_extension(pixmap: Any, output_path: Path) -> None:
    """以原图片扩展名保存 RGB Pixmap，确保图片名称和下游引用保持不变。"""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("高清图片保存需要 Pillow，请安装 PIL/Pillow 后重试") from exc

    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image_format = "JPEG"
    elif suffix == ".png":
        image_format = "PNG"
    else:
        raise ValueError(f"不支持保持原名称的图片扩展名：{output_path.suffix}")
    if pixmap.n != 3 or pixmap.alpha:
        raise ValueError("PDF 渲染结果不是无透明通道的 RGB 图像")

    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    save_kwargs: dict[str, Any] = {"format": image_format}
    if image_format == "JPEG":
        save_kwargs.update({"quality": PDF_IMAGE_RENDER_JPEG_QUALITY, "subsampling": 0, "optimize": True})
    image.save(output_path, **save_kwargs)


def replace_mineru_images_from_pdf(
    mineru_output_dir: str | Path,
    source_pdf: str | Path,
    render_scale: float = PDF_IMAGE_RENDER_SCALE,
    padding_points: float = PDF_IMAGE_RENDER_PADDING_POINTS,
) -> dict[str, Any]:
    """依据 MinerU layout 的 PDF 坐标局部重渲染，并原子替换同名低清图片。"""

    if render_scale <= 0:
        raise ValueError("PDF 图片渲染倍数必须大于 0")
    if padding_points < 0:
        raise ValueError("PDF 图片裁剪外扩距离不能小于 0")
    output_dir = Path(mineru_output_dir).expanduser().resolve()
    pdf_path = Path(source_pdf).expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"MinerU 输出目录不存在：{output_dir}")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"原始 PDF 不存在：{pdf_path}")
    content_list_path, layout_path, tasks = build_mineru_image_render_tasks(output_dir)

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("高清 PDF 图片重渲染需要 PyMuPDF，请安装 fitz/PyMuPDF 后重试") from exc

    temporary_files: list[tuple[Path, Path]] = []
    rendered_images: list[dict[str, Any]] = []
    try:
        with fitz.open(pdf_path) as document:
            matrix = fitz.Matrix(render_scale, render_scale)
            for task in tasks:
                if task.page_idx >= document.page_count:
                    raise ValueError(
                        f"图片 {task.image_path.name} 指向第 {task.page_idx} 页，"
                        f"但 PDF 仅有 {document.page_count} 页"
                    )
                page = document[task.page_idx]
                if page.rotation != 0:
                    raise ValueError(
                        f"PDF 第 {task.page_idx} 页存在 {page.rotation} 度旋转，"
                        "当前需先完成坐标转换校准，拒绝直接覆盖图片。"
                    )
                clip = _expanded_pdf_clip(page, task.layout_bbox, padding_points)
                pixmap = page.get_pixmap(
                    matrix=matrix,
                    clip=clip,
                    colorspace=fitz.csRGB,
                    alpha=False,
                )
                temporary_path = _temporary_image_path(task.image_path)
                _save_pixmap_with_original_extension(pixmap, temporary_path)
                temporary_files.append((temporary_path, task.image_path))
                rendered_images.append(
                    {
                        "image_path": str(task.image_path),
                        "asset_type": task.asset_type,
                        "page_idx": task.page_idx,
                        "content_bbox": list(task.content_bbox),
                        "layout_bbox": list(task.layout_bbox),
                        "layout_index": task.layout_index,
                        "rendered_size": [pixmap.width, pixmap.height],
                    }
                )
        for temporary_path, destination_path in temporary_files:
            temporary_path.replace(destination_path)
    finally:
        for temporary_path, _ in temporary_files:
            if temporary_path.exists():
                temporary_path.unlink()

    return {
        "mode": "pdf_clip_render",
        "source_pdf": str(pdf_path),
        "content_list": str(content_list_path),
        "layout": str(layout_path),
        "render_scale": render_scale,
        "padding_points": padding_points,
        "image_count": len(rendered_images),
        "images": rendered_images,
    }


def populate_section_assets(
    node: SectionNode,
    reference_text: str = "",
    asset_counters: dict[str, int] | None = None,
    asset_base_dir: Path | None = None,
    table_image_detector: TableImageDetector = detect_table_image_with_picodet,
    table_continuation_threshold: float = TABLE_CONTINUATION_THRESHOLD,
) -> None:
    """提取单个章节内的表格、图片、公式，并生成剥离结构块后的正文。"""

    raw = normalize_newlines(node.raw_content)
    removal_spans: list[tuple[int, int]] = []

    tables, table_spans = extract_tables(
        raw,
        reference_text,
        asset_base_dir,
        table_image_detector,
        table_continuation_threshold,
    )
    images, image_spans = extract_images(
        raw,
        reference_text,
        asset_base_dir=asset_base_dir,
        excluded_spans=table_spans,
    )
    formulas, formula_spans = extract_formulas(raw, reference_text, asset_counters)

    node.table = tables
    node.images = images
    node.formulas = formulas
    removal_spans.extend(table_spans)
    removal_spans.extend(image_spans)
    removal_spans.extend(formula_spans)
    node.content = clean_section_content(remove_spans(raw, removal_spans))


def extract_tables(
    text: str,
    reference_text: str = "",
    asset_base_dir: Path | None = None,
    table_image_detector: TableImageDetector = detect_table_image_with_picodet,
    table_continuation_threshold: float = TABLE_CONTINUATION_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """提取 HTML 表格及“表/Table + 数字”标题后紧接的图片表格。"""

    tables: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for match in HTML_TABLE_RE.finditer(text):
        caption, caption_span = extract_caption_before(text, match.start(), keywords=("表", "Table"))
        table_numbers = extract_labeled_numbers(caption, labels=("表", "Table"))
        contexts = find_table_reference_contexts(reference_text, table_numbers)
        tables.append(
            {
                "content": match.group(0).strip(),
                "caption": caption,
                "context": "\n".join(contexts),
            }
        )
        spans.append((caption_span[0], match.end()))

    image_tables, image_table_spans = extract_captioned_image_tables(
        text,
        reference_text,
        asset_base_dir,
        table_image_detector,
        table_continuation_threshold,
    )
    tables.extend(image_tables)
    spans.extend(image_table_spans)

    ordered = sorted(zip(spans, tables), key=lambda item: item[0][0])
    if not ordered:
        return [], []
    ordered_spans, ordered_tables = zip(*ordered)
    return list(ordered_tables), list(ordered_spans)


def extract_captioned_image_tables(
    text: str,
    reference_text: str = "",
    asset_base_dir: Path | None = None,
    table_image_detector: TableImageDetector = detect_table_image_with_picodet,
    table_continuation_threshold: float = TABLE_CONTINUATION_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """识别表题及连续表格图片，并在首张非表格图片或新表题处结束。"""

    line_matches = list(re.finditer(r"(?m)^.*(?:\n|$)", text))
    tables: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    consumed_until = -1

    for index, line_match in enumerate(line_matches):
        if line_match.start() < consumed_until:
            continue
        first_line = line_match.group(0).strip()
        if not TABLE_CAPTION_START_RE.match(clean_inline_text(first_line)):
            continue

        caption_lines = [first_line]
        caption_numbers = extract_labeled_numbers(first_line, labels=("表", "Table"))
        image_match: re.Match[str] | None = None
        image_line_match: re.Match[str] | None = None
        image_line_index: int | None = None
        nonempty_line_count = 1

        for following_index, following_line_match in enumerate(line_matches[index + 1 :], start=index + 1):
            following_line = following_line_match.group(0)
            stripped = following_line.strip()
            if not stripped:
                continue

            image_match = re.fullmatch(r"!\[[^\]]*]\(([^)]+)\)", stripped)
            if image_match:
                image_line_match = following_line_match
                image_line_index = following_index
                break

            cleaned = clean_inline_text(stripped)
            following_caption_match = TABLE_CAPTION_START_RE.match(cleaned)
            following_numbers = extract_labeled_numbers(cleaned, labels=("表", "Table"))
            is_different_table = bool(
                following_caption_match
                and caption_numbers
                and following_numbers
                and set(caption_numbers).isdisjoint(following_numbers)
            )
            if (
                stripped.startswith(("#", "<table", "$$"))
                or is_different_table
                or nonempty_line_count >= 12
            ):
                image_match = None
                break

            caption_lines.append(stripped)
            nonempty_line_count += 1

        if image_match is None or image_line_match is None or image_line_index is None:
            continue

        table_image_matches = [image_match]
        last_image_line_match = image_line_match
        # 首张图片由明确表题归类；之后只检查中间没有任何非空内容的连续图片。
        for continuation_line_match in line_matches[image_line_index + 1 :]:
            continuation_line = continuation_line_match.group(0).strip()
            if not continuation_line:
                continue
            if TABLE_CAPTION_START_RE.match(clean_inline_text(continuation_line)):
                break

            continuation_image = re.fullmatch(r"!\[[^\]]*]\(([^)]+)\)", continuation_line)
            if continuation_image is None:
                break

            continuation_path = resolve_markdown_image_path(
                continuation_image.group(1).strip(),
                asset_base_dir,
            )
            if not table_image_detector(continuation_path, table_continuation_threshold):
                break
            table_image_matches.append(continuation_image)
            last_image_line_match = continuation_line_match

        caption = clean_inline_text("\n".join(caption_lines), strip_markup=False)
        table_numbers = extract_labeled_numbers(caption, labels=("表", "Table"))
        contexts = find_table_reference_contexts(reference_text, table_numbers)
        # 将图片表格中的 Markdown 相对引用转换为相对于 full.md 的绝对路径后写入 JSON。
        image_paths = [
            serialize_asset_path(item.group(1).strip(), asset_base_dir)
            for item in table_image_matches
        ]
        tables.append(
            {
                # 图片表格只保存干净路径，去掉 Markdown 的 ![](  ) 包装符号。
                "content": "\n".join(image_paths),
                "caption": caption,
                "context": "\n".join(contexts),
                "path": image_paths[0] if len(image_paths) == 1 else image_paths,
                "source_type": "image",
            }
        )
        span = (line_match.start(), last_image_line_match.end())
        spans.append(span)
        consumed_until = span[1]
    return tables, spans


def extract_images(
    text: str,
    reference_text: str = "",
    excluded_spans: Iterable[tuple[int, int]] = (),
    asset_base_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """提取普通图片组；已归类为图片表格的图片不会被重复提取。"""

    excluded = list(excluded_spans)
    matches = [match for match in IMAGE_RE.finditer(text) if not span_overlaps(match.span(), excluded)]
    images: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(matches):
        group = [matches[index]]
        group_start = matches[index].start()
        group_end = matches[index].end()
        index += 1
        while index < len(matches) and not text[group_end : matches[index].start()].strip():
            group.append(matches[index])
            group_end = matches[index].end()
            index += 1

        caption, caption_end = extract_caption_after(text, group_end, keywords=("图", "Fig"))
        # 普通图片与图片表格统一输出绝对路径，便于后续模块直接读取资源文件。
        paths = [
            serialize_asset_path(item.group(1).strip(), asset_base_dir)
            for item in group
        ]
        figure_numbers = extract_labeled_numbers(caption, labels=("图", "Fig"))
        images.append(
            {
                "path": paths,
                "caption": caption,
                "references": find_reference_contexts(reference_text, "图", figure_numbers),
            }
        )
        spans.append((group_start, caption_end))
    return images, spans


def span_overlaps(span: tuple[int, int], excluded_spans: Iterable[tuple[int, int]]) -> bool:
    """判断字符区间是否与任一排除区间重叠。"""

    start, end = span
    return any(start < excluded_end and excluded_start < end for excluded_start, excluded_end in excluded_spans)


def extract_formulas(
    text: str,
    reference_text: str = "",
    asset_counters: dict[str, int] | None = None,
) -> tuple[list[dict[str, str]], list[tuple[int, int]]]:
    """只提取两个 $$ 包裹的公式，并按全文出现顺序查找“公式1”引用上下文。"""

    formulas: list[dict[str, str]] = []
    spans: list[tuple[int, int]] = []
    for match in FORMULA_RE.finditer(text):
        if asset_counters is not None:
            asset_counters["formula"] = asset_counters.get("formula", 0) + 1
            formula_number = str(asset_counters["formula"])
        else:
            formula_number = str(len(formulas) + 1)
        raw = match.group(0).strip()
        content = match.group(1).strip()
        context = "\n".join(find_reference_contexts(reference_text, "公式", [formula_number]))
        formulas.append({"content": content, "raw": raw, "context": context})
        spans.append(match.span())
    return formulas, spans


def extract_caption_before(text: str, offset: int, keywords: Iterable[str]) -> tuple[str, tuple[int, int]]:
    """从结构块前方提取表题，返回表题文本及其在原文中的范围。"""

    prefix = text[:offset]
    lines = [line for line in re.finditer(r"(?m)^.*(?:\n|$)", prefix) if line.group(0).strip()]
    candidates: list[re.Match[str]] = []
    for line_match in reversed(lines):
        line = line_match.group(0).strip()
        if line.startswith(("!", "<table", "$$", "##")):
            break
        candidates.append(line_match)
        if len(candidates) >= 12:
            break

    if not candidates:
        return "", (offset, offset)

    candidates.reverse()
    keyword_index = next(
        (
            index
            for index, item in enumerate(candidates)
            if any(keyword in clean_inline_text(item.group(0)) for keyword in keywords)
        ),
        None,
    )
    if keyword_index is None:
        return "", (offset, offset)

    selected = candidates[keyword_index:]
    caption_lines = [item.group(0).strip() for item in selected if item.group(0).strip()]
    caption = "\n".join(caption_lines).strip()
    start = selected[0].start()
    return clean_inline_text(caption, strip_markup=False), (start, offset)


def extract_caption_after(text: str, offset: int, keywords: Iterable[str]) -> tuple[str, int]:
    """从图片组后方提取图题，直到空行、下一个结构块或标题。"""

    tail = text[offset:]
    consumed = 0
    caption_lines: list[str] = []
    for line in tail.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            if caption_lines:
                consumed += len(line)
                break
            consumed += len(line)
            continue
        if stripped.startswith(("!", "<table", "$$", "##")):
            break
        caption_lines.append(stripped)
        consumed += len(line)
        if len(caption_lines) >= 8:
            break
    caption = "\n".join(caption_lines).strip()
    if caption and not any(keyword in clean_inline_text(caption) for keyword in keywords):
        return "", offset
    return clean_inline_text(caption, strip_markup=False), offset + consumed


def extract_labeled_numbers(text: str, labels: Iterable[str]) -> list[str]:
    """从图题、表题等文本中提取“图1 / 表1 / Fig 1 / Table 1”里的编号。"""

    cleaned = clean_inline_text(text)
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(rf"(?:{label_pattern})[．.:\s]*(\d+(?:\.\d+)*)", re.IGNORECASE)
    numbers: list[str] = []
    for match in pattern.finditer(cleaned):
        number = match.group(1)
        if number not in numbers:
            numbers.append(number)
    return numbers


def find_reference_contexts(reference_text: str, label: str, numbers: Iterable[str]) -> list[str]:
    """按换行切分原文，只返回括号中出现“图1/表1/公式1”的引用段落。"""

    if not reference_text:
        return []

    normalized_numbers = [str(number).strip() for number in numbers if str(number).strip()]
    if not normalized_numbers:
        return []

    contexts: list[str] = []
    for paragraph in normalize_newlines(reference_text).split("\n"):
        cleaned = clean_inline_text(paragraph, strip_markup=False)
        if not cleaned:
            continue
        plain = clean_inline_text(paragraph)
        if any(parenthesized_label_exists(plain, label, number) for number in normalized_numbers):
            if cleaned not in contexts:
                contexts.append(cleaned)
    return contexts


def find_table_reference_contexts(reference_text: str, numbers: Iterable[str]) -> list[str]:
    """同时查找中文“表”和英文“Table”编号引用，并保持原文顺序去重。"""

    contexts = find_reference_contexts(reference_text, "表", numbers)
    contexts.extend(find_reference_contexts(reference_text, "Table", numbers))
    return list(dict.fromkeys(contexts))


def parenthesized_label_exists(text: str, label: str, number: str) -> bool:
    """判断单个段落的括号内容中是否包含指定的图、表或公式编号。"""

    whitespace = r"[\s\u3000]*"
    number_pattern = re.escape(number).replace(r"\.", rf"{whitespace}[．.]{whitespace}")
    pattern = re.compile(
        rf"[（(]{whitespace}[^）)]*{re.escape(label)}{whitespace}{number_pattern}(?!\d)[^）)]*{whitespace}[）)]"
    )
    return bool(pattern.search(text))


def remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """按字符区间删除已抽取的结构块，区间重叠时自动合并。"""

    if not spans:
        return text
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    parts: list[str] = []
    last = 0
    for start, end in merged:
        parts.append(text[last:start])
        last = end
    parts.append(text[last:])
    return "".join(parts)


def clean_section_content(text: str) -> str:
    """整理章节正文空白，保留行内公式和必要 HTML 标记。"""

    text = normalize_newlines(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_json(path: Path, data: dict[str, Any]) -> None:
    """把解析结果写入 UTF-8 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def project_root() -> Path:
    """根据当前脚本位置推导项目根目录，便于 CLI 无参数运行。"""

    return Path(__file__).resolve().parents[2]


def default_output_path() -> Path:
    """返回默认聚合 JSON 输出路径。"""

    return project_root() / "output" / "stage_01_mineru_parse.json"


def resolve_markdown_inputs(input_path: Path) -> list[Path]:
    """把文件或目录输入解析为 Markdown 文件列表，目录模式递归扫描 .md 文件。"""

    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        markdown_files = sorted(input_path.rglob("*.md"))
        if markdown_files:
            return markdown_files
    raise FileNotFoundError(f"没有找到可解析的 Markdown 文件: {input_path}")


def build_cli_result(results: list[dict[str, Any]], input_path: Path) -> dict[str, Any]:
    """单文件保持原 JSON 结构，多文件时生成 documents 聚合结构。"""

    if len(results) == 1:
        return results[0]
    return {
        "_stage": 1,
        "_produced_by": AMarkdownParser.produced_by,
        "input_path": str(input_path.resolve()),
        "document_count": len(results),
        "documents": results,
    }


def main() -> None:
    """命令行入口：解析指定 Markdown；不传 input 时默认扫描 data/mineru_output。"""

    cli = argparse.ArgumentParser(description="解析 MinerU full.md 为章节化论文 JSON")
    cli.add_argument(
        "input",
        nargs="?",
        default=project_root() / "data" / "mineru_output"/"鄂尔多斯盆地奥陶系马家沟组白云岩储层特征及成因机制_吴东旭",
        help="MinerU 导出的 full.md 路径或目录；不传则默认扫描 data/mineru_output",
    )
    cli.add_argument("-o", "--output", help="输出 JSON 文件路径；目录输入且不传时默认写入 output/amarkdown_parser_results.json")
    cli.add_argument("--use-llm-basic-info", action="store_true", help="尝试用项目 LLMClient 补全基本信息")
    cli.add_argument("--source-pdf", type=Path, help="与 MinerU 输出同源的原始 PDF；高清替换图片时必填")
    cli.add_argument(
        "--replace-mineru-images-from-pdf",
        action="store_true",
        help="按 layout.json 的 PDF 坐标以高清局部渲染替换同名 MinerU 图片",
    )
    cli.add_argument(
        "--pdf-image-render-scale",
        type=float,
        default=PDF_IMAGE_RENDER_SCALE,
        help=f"PDF 图片局部渲染倍数，默认 {PDF_IMAGE_RENDER_SCALE:g}",
    )
    cli.add_argument(
        "--pdf-image-render-padding-points",
        type=float,
        default=PDF_IMAGE_RENDER_PADDING_POINTS,
        help=f"图片框四周外扩的 PDF point，默认 {PDF_IMAGE_RENDER_PADDING_POINTS:g}",
    )
    args = cli.parse_args()

    if args.replace_mineru_images_from_pdf and args.source_pdf is None:
        cli.error("--replace-mineru-images-from-pdf 必须同时提供 --source-pdf")
    if args.pdf_image_render_scale <= 0:
        cli.error("--pdf-image-render-scale 必须大于 0")
    if args.pdf_image_render_padding_points < 0:
        cli.error("--pdf-image-render-padding-points 不能小于 0")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root() / input_path

    input_files = resolve_markdown_inputs(input_path)
    if args.replace_mineru_images_from_pdf and len(input_files) != 1:
        cli.error("高清替换图片时一次只能解析一个 full.md，以避免单个 PDF 错配多篇论文")
    parser = AMarkdownParser(
        use_llm_basic_info=args.use_llm_basic_info,
        source_pdf=args.source_pdf,
        replace_mineru_images_from_pdf=args.replace_mineru_images_from_pdf,
        pdf_image_render_scale=args.pdf_image_render_scale,
        pdf_image_render_padding_points=args.pdf_image_render_padding_points,
    )
    result = build_cli_result([parser.parse_file(path) for path in input_files], input_path)

    output_path = Path(args.output) if args.output else None
    if output_path is not None:
        if not output_path.is_absolute():
            output_path = project_root() / output_path
        write_json(output_path, result)
        print(f"已解析 {len(input_files)} 个 Markdown 文件，输出: {output_path}")
    elif input_path.is_dir():
        output_path = default_output_path()
        write_json(output_path, result)
        print(f"已解析 {len(input_files)} 个 Markdown 文件，输出: {output_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
