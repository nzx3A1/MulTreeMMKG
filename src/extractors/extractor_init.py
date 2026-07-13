"""第四阶段多模态抽取器统一调度入口。

读取 ``stage_03_document_summary.json`` 中嵌套章节里的 Chunk，按模态调用同级目录
中的文本、表格、图片和公式抽取器，分别保存结果后再合并为统一 Graph。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping, Sequence

# 中文说明：支持直接执行本文件，将项目根目录加入模块搜索路径以导入 model 和 src。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Graph
from src.utils.json_io import read_json
from src.utils.llm_client import LLMClient
from src.utils.vlm_client import VLMClient

DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "stage_03_document_summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
SUPPORTED_MODALITIES = ("text", "table", "image", "formula")


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """校验输入为映射类型，并提供包含数据位置的清晰错误信息。"""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 JSON 对象，实际为 {type(value).__name__}")
    return value


def _modality(chunk: Mapping[str, Any]) -> str:
    """兼容 modality、chunk_type 和 type 字段，规范化 Chunk 模态名称。"""

    value = chunk.get("modality") or chunk.get("chunk_type") or chunk.get("type") or ""
    modality = str(getattr(value, "value", value)).strip().lower()
    aliases = {"textchunk": "text", "tablechunk": "table", "imagechunk": "image", "formulachunk": "formula"}
    return aliases.get(modality.replace("_", ""), modality)


def collect_chunks(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """递归收集文档章节 Chunk，并补充文档和章节溯源字段。"""

    document = _as_mapping(payload.get("document", payload), "document")
    document_id = str(document.get("id") or document.get("document_id") or "")
    source_file = document.get("source_file")
    document_summary = str(document.get("summary") or "")
    document_schema_keys = list(document.get("schemaKeys") or [])
    result: list[dict[str, Any]] = []

    def visit(section: Mapping[str, Any], inherited_title: str = "") -> None:
        """深度优先遍历单个章节，将章节上下文复制到其 Chunk。"""

        section_id = str(section.get("id") or section.get("section_id") or "")
        section_title = str(section.get("title") or inherited_title or "")
        section_summary = str(section.get("summary") or "")
        section_schema_keys = list(section.get("schemaKeys") or [])
        for raw_chunk in section.get("chunks") or []:
            chunk = dict(_as_mapping(raw_chunk, f"section[{section_id}].chunks[]"))
            chunk.setdefault("document_id", document_id)
            chunk.setdefault("section_id", section_id)
            chunk.setdefault("section_title", section_title)
            chunk.setdefault("document_summary", document_summary)
            chunk.setdefault("document_schema_keys", document_schema_keys)
            chunk.setdefault("section_summary", section_summary)
            chunk.setdefault("schemaKeys", section_schema_keys)
            if source_file is not None:
                chunk.setdefault("source_file", source_file)
            result.append(chunk)
        for child in section.get("children") or []:
            visit(_as_mapping(child, f"section[{section_id}].children[]"), section_title)

    for section in document.get("sections") or []:
        visit(_as_mapping(section, "document.sections[]"))
    return result


class InitExtractor:
    """持有模型客户端与 Schema 选择器，并统一调度四种模态抽取器。"""

    def __init__(
        self,
        llm_client: Any | None = None,
        vlm_client: Any | None = None,
        show_progress: bool = True,
    ) -> None:
        """初始化可注入的客户端；默认客户端均为延迟连接，不会在此处发起请求。"""

        self.llm_client = llm_client or LLMClient()
        self.vlm_client = vlm_client or VLMClient()
        self.show_progress = show_progress
###########################################################核心####################################################
    def extract(
        self,
        chunks: Sequence[Mapping[str, Any]],
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, list[Graph]]:
        """按模态分组全部 Chunk，并将文本分支统一交给 extract_from_text。"""

        grouped: MutableMapping[str, list[Mapping[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            modality = _modality(chunk)
            if modality in SUPPORTED_MODALITIES:
                grouped[modality].append(chunk)

        results: dict[str, list[Graph]] = {}

        # 第一阶段：优先抽取正文文本，为后续多模态结果提供文本语义基础。
        text_chunks = grouped["text"]
        if self.show_progress:
            print(f"正在抽取 text Chunk：{len(text_chunks)} 个")
        if text_chunks:
            from src.extractors.text_extractor import extract_from_text

            results["text"] = extract_from_text(
                text_chunks,
                self.llm_client,
                output_path=(Path(output_dir) / "stage_04_text_extraction.json") if output_dir else None,
                show_progress=self.show_progress,
            )
        else:
            results["text"] = []

        # 第二阶段：抽取表格中的结构化实体、属性与关系。
        # table_chunks = grouped["table"]
        # if self.show_progress:
        #     print(f"正在抽取 table Chunk：{len(table_chunks)} 个")
        # if table_chunks:
        #     from .table_extractor import extract_from_tables
        #
        #     results["table"] = extract_from_tables(
        #         table_chunks,
        #         self.llm_client,
        #     )
        # else:
        #     results["table"] = []
        #
        # # 第三阶段：抽取公式、变量、参数及其领域关系。
        # formula_chunks = grouped["formula"]
        # if self.show_progress:
        #     print(f"正在抽取 formula Chunk：{len(formula_chunks)} 个")
        # if formula_chunks:
        #     from .formula_extractor import extract_from_formulas
        #
        #     results["formula"] = extract_from_formulas(
        #         formula_chunks,
        #         self.llm_client,
        #     )
        # else:
        #     results["formula"] = []
        #
        # # 第四阶段：最后调用视觉模型抽取图片内容，避免提前占用较重的视觉请求资源。
        # image_chunks = grouped["image"]
        # if self.show_progress:
        #     print(f"正在抽取 image Chunk：{len(image_chunks)} 个")
        # if image_chunks:
        #     from .image_extractor import extract_from_images
        #
        #     results["image"] = extract_from_images(
        #         image_chunks,
        #         self.llm_client,
        #         self.vlm_client,
        #     )
        # else:
        #     results["image"] = []

        return results



def main() -> None:
    """读取第三阶段输入，初始化 InitExtractor 并调用 extract 处理全部 Chunk。"""

    parser = argparse.ArgumentParser(description="执行第四阶段抽取，当前只启用文本模态")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="第三阶段摘要 JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="第四阶段输出目录")
    parser.add_argument("--quiet", action="store_true", help="不显示模态调度进度")
    args = parser.parse_args()

    source = _as_mapping(read_json(args.input), "第三阶段输入")
    chunks = collect_chunks(source)
    extractor = InitExtractor(show_progress=not args.quiet)
    results = extractor.extract(chunks, output_dir=args.output_dir)
    print(f"第四阶段文本抽取完成，共生成 {len(results['text'])} 个 Chunk Graph")


if __name__ == "__main__":
    main()
